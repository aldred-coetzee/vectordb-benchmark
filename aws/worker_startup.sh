#!/bin/bash
# Worker startup script for vectordb-benchmark
# This script is passed as user-data when launching worker EC2 instances
#
# Features:
# - Auto-termination on completion, error, or timeout
# - Fetches KDB.AI license and Docker credentials from S3
# - Pulls latest code before running
# - Uploads results to S3

# =============================================================================
# Configuration (populated by orchestrator)
# =============================================================================
# Set HOME for user-data environment (runs as root)
export HOME=/root

DATABASE="{{DATABASE}}"           # e.g., qdrant, milvus, kdbai
DATASET="{{DATASET}}"             # e.g., sift, gist
S3_BUCKET="{{S3_BUCKET}}"         # e.g., vectordb-benchmark-590780615264
RUN_ID="{{RUN_ID}}"               # e.g., 2026-02-04-1430
BENCHMARK_TYPE="{{BENCHMARK_TYPE}}" # e.g., competitive, kdbai-tuning
PULL_LATEST="{{PULL_LATEST}}"     # e.g., "" or "kdbai" or "all"
MAX_RUNTIME_MINUTES=120           # Safety timeout

# Tuning-specific variables (empty for competitive benchmarks)
HNSW_M="{{HNSW_M}}"                       # e.g., 32
HNSW_EFC="{{HNSW_EFC}}"                   # e.g., 128
HNSW_NAME="{{HNSW_NAME}}"                 # e.g., M32_efC128
DOCKER_CONFIG_NAME="{{DOCKER_CONFIG_NAME}}" # e.g., 1wrk_16thr
DOCKER_THREADS="{{DOCKER_THREADS}}"       # e.g., 16
DOCKER_NUM_WRK="{{DOCKER_NUM_WRK}}"       # e.g., 1

# =============================================================================
# Logging (set up FIRST so we capture everything)
# =============================================================================
LOGFILE="/tmp/benchmark.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "Worker startup: $(date)"
echo "Database: $DATABASE"
echo "Dataset: $DATASET"
echo "Run ID: $RUN_ID"
echo "Benchmark type: $BENCHMARK_TYPE"
if [ -n "$HNSW_NAME" ]; then
    echo "Mode: TUNING"
    echo "HNSW: M=$HNSW_M efC=$HNSW_EFC ($HNSW_NAME)"
    echo "Docker: THREADS=$DOCKER_THREADS NUM_WRK=$DOCKER_NUM_WRK ($DOCKER_CONFIG_NAME)"
fi
echo "========================================"

# =============================================================================
# Auto-termination setup (set up EARLY to catch all exits)
# =============================================================================
# Build job name — includes tuning params for tuning runs
if [ -n "$HNSW_NAME" ] && [ -n "$DOCKER_CONFIG_NAME" ]; then
    JOB_NAME="${DATABASE}-${DATASET}-${HNSW_NAME}-${DOCKER_CONFIG_NAME}"
else
    JOB_NAME="${DATABASE}-${DATASET}"
fi
S3_RESULT_PATH="s3://${S3_BUCKET}/runs/${BENCHMARK_TYPE}/${RUN_ID}/jobs/${JOB_NAME}"

cleanup() {
    EXIT_CODE=$?
    echo "Cleaning up... (exit code: $EXIT_CODE)"

    # Upload log file even on failure
    aws s3 cp "$LOGFILE" ${S3_RESULT_PATH}/worker.log --region us-west-2 || true

    # Only set failed status if no status.json was uploaded yet (i.e., crash before
    # the results upload section). If results were already uploaded, don't overwrite.
    if [ $EXIT_CODE -ne 0 ]; then
        aws s3api head-object --bucket ${S3_BUCKET} --key "runs/${BENCHMARK_TYPE}/${RUN_ID}/jobs/${JOB_NAME}/status.json" --region us-west-2 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "{\"status\": \"failed\", \"exit_code\": $EXIT_CODE}" | \
                aws s3 cp - ${S3_RESULT_PATH}/status.json --region us-west-2 || true
        fi
    fi

    echo "Shutting down..."
    sudo shutdown -c 2>/dev/null || true
    sudo shutdown -h now
}
trap cleanup EXIT

# Safety net - terminate after max runtime no matter what
sudo shutdown -h +${MAX_RUNTIME_MINUTES}

# Now enable exit on error (after trap is set)
set -e

# =============================================================================
# Fetch credentials from S3 (for KDB.AI)
# =============================================================================
echo "Fetching credentials from S3..."

# KDB.AI license
aws s3 cp s3://${S3_BUCKET}/config/kc.lic /tmp/kc.lic --region us-west-2 || true
if [ -f /tmp/kc.lic ]; then
    export KDB_LICENSE_B64=$(base64 -w 0 /tmp/kc.lic)
    echo "KDB.AI license loaded"
fi

# Docker credentials (for pulling fresh KDB.AI images)
if [ -n "$PULL_LATEST" ]; then
    aws s3 cp s3://${S3_BUCKET}/config/docker-config.json ~/.docker/config.json --region us-west-2 || true
    echo "Docker credentials loaded"
fi

# =============================================================================
# Pull latest code
# =============================================================================
echo "Pulling latest code..."
cd /app/vectordb-benchmark

# Fix git ownership issue (repo created by ec2-user, user-data runs as root)
git config --global --add safe.directory /app/vectordb-benchmark

git pull origin main

# Create symlink so relative paths in benchmark.yaml work
# (config uses "data/sift", AMI has datasets at "/data/sift")
ln -sf /data /app/vectordb-benchmark/data

# =============================================================================
# Pull fresh Docker images and upgrade client SDKs if requested
# =============================================================================
# Maps database name to Docker image and Python client package
upgrade_client() {
    local db=$1
    echo "Upgrading Python client for $db..."
    case $db in
        kdbai)    sudo -u ec2-user pip3.12 install --upgrade kdbai-client ;;
        qdrant)   sudo -u ec2-user pip3.12 install --upgrade qdrant-client ;;
        milvus)   sudo -u ec2-user pip3.12 install --upgrade pymilvus ;;
        weaviate) sudo -u ec2-user pip3.12 install --upgrade weaviate-client ;;
        chroma)   sudo -u ec2-user pip3.12 install --upgrade chromadb ;;
        redis)    sudo -u ec2-user pip3.12 install --upgrade redis ;;
        pgvector) sudo -u ec2-user pip3.12 install --upgrade psycopg2-binary ;;
    esac
}

if [ "$PULL_LATEST" = "all" ]; then
    echo "Pulling all Docker images..."
    docker pull qdrant/qdrant:latest
    docker pull milvusdb/milvus:latest
    docker pull semitechnologies/weaviate:latest
    docker pull chromadb/chroma:latest
    docker pull redis/redis-stack:latest
    docker pull pgvector/pgvector:pg16
    docker pull portal.dl.kx.com/kdbai-db:latest
    # Upgrade the client SDK for the database this worker is running
    upgrade_client "$DATABASE"
elif [ -n "$PULL_LATEST" ]; then
    # Pull specific images (comma-separated)
    IFS=',' read -ra IMAGES <<< "$PULL_LATEST"
    for img in "${IMAGES[@]}"; do
        echo "Pulling $img..."
        case $img in
            kdbai)   docker pull portal.dl.kx.com/kdbai-db:latest ;;
            qdrant)  docker pull qdrant/qdrant:latest ;;
            milvus)  docker pull milvusdb/milvus:latest ;;
            weaviate) docker pull semitechnologies/weaviate:latest ;;
            chroma)  docker pull chromadb/chroma:latest ;;
            redis)   docker pull redis/redis-stack:latest ;;
            pgvector) docker pull pgvector/pgvector:pg16 ;;
        esac
    done
    # Upgrade client SDK if this worker's database was in the pull list
    for img in "${IMAGES[@]}"; do
        if [ "$img" = "$DATABASE" ]; then
            upgrade_client "$DATABASE"
            break
        fi
    done
fi

# =============================================================================
# Run benchmark
# =============================================================================
echo "Running benchmark: $DATABASE on $DATASET"
cd /app/vectordb-benchmark

# Build benchmark command
BENCHMARK_CMD="run_benchmark.py --config configs/${DATABASE}.yaml --dataset ${DATASET} --output results"

# Add tuning arguments for known tuning benchmark types
if [ "$BENCHMARK_TYPE" = "kdbai-tuning" ]; then
    TUNING_CONFIG="configs/tuning/kdbai-tuning.yaml"
    echo "Tuning config: $TUNING_CONFIG"
    BENCHMARK_CMD="$BENCHMARK_CMD --tuning-config ${TUNING_CONFIG}"

    if [ -n "$HNSW_M" ]; then
        BENCHMARK_CMD="$BENCHMARK_CMD --hnsw-m ${HNSW_M}"
    fi
    if [ -n "$HNSW_EFC" ]; then
        BENCHMARK_CMD="$BENCHMARK_CMD --hnsw-efc ${HNSW_EFC}"
    fi
    if [ -n "$DOCKER_CONFIG_NAME" ]; then
        BENCHMARK_CMD="$BENCHMARK_CMD --docker-config-name ${DOCKER_CONFIG_NAME}"
    fi
fi

# Override THREADS/NUM_WRK if tuning docker params are set
if [ -n "$DOCKER_THREADS" ]; then
    export THREADS="$DOCKER_THREADS"
fi
if [ -n "$DOCKER_NUM_WRK" ]; then
    export NUM_WRK="$DOCKER_NUM_WRK"
fi

echo "Running: $BENCHMARK_CMD"

# Run as ec2-user (who has the Python dependencies installed)
# -E preserves environment (needed for KDB_LICENSE_B64, THREADS, NUM_WRK)
# Capture exit code without letting set -e kill the script — results may
# exist even if cleanup (e.g., docker container.stop()) fails with exit code 1
set +e
sudo -E -u ec2-user python3.12 $BENCHMARK_CMD
BENCHMARK_EXIT_CODE=$?
set -e

echo "Benchmark completed with exit code: $BENCHMARK_EXIT_CODE"

# =============================================================================
# Upload results to S3 (BEFORE any cleanup that might fail)
# =============================================================================
echo "Uploading results to S3..."

# Verify AWS credentials are available (IMDSv2 can be flaky under load)
for i in 1 2 3 4 5; do
    if aws sts get-caller-identity --region us-west-2 >/dev/null 2>&1; then
        break
    fi
    echo "  AWS credentials not available, retrying in ${i}0s... (attempt $i/5)"
    sleep $((i * 10))
done

# Upload result files regardless of exit code — benchmark.db may exist
# even when exit code is non-zero (e.g., docker stop timeout after benchmark completes)
if [ -d results/ ] && [ "$(ls -A results/)" ]; then
    aws s3 cp results/ ${S3_RESULT_PATH}/ --recursive --region us-west-2
    if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
        echo '{"status": "completed"}' | aws s3 cp - ${S3_RESULT_PATH}/status.json --region us-west-2
    else
        echo "{\"status\": \"completed\", \"warning\": \"exit code $BENCHMARK_EXIT_CODE but results uploaded\"}" | \
            aws s3 cp - ${S3_RESULT_PATH}/status.json --region us-west-2
        echo "WARNING: Benchmark exited with code $BENCHMARK_EXIT_CODE but results were produced and uploaded"
    fi
else
    if [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
        echo "{\"status\": \"failed\", \"exit_code\": $BENCHMARK_EXIT_CODE}" | \
            aws s3 cp - ${S3_RESULT_PATH}/status.json --region us-west-2
    else
        echo "No results produced (database may not support requested index types)"
        echo '{"status": "skipped", "reason": "no results produced"}' | aws s3 cp - ${S3_RESULT_PATH}/status.json --region us-west-2
    fi
fi

echo "========================================"
echo "Worker complete: $(date)"
echo "========================================"

# Exit cleanly - trap will handle shutdown and final log upload
