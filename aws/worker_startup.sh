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
DATABASE="{{DATABASE}}"           # e.g., qdrant, milvus, kdbai
DATASET="{{DATASET}}"             # e.g., sift, gist
S3_BUCKET="{{S3_BUCKET}}"         # e.g., vectordb-benchmark-590780615264
RUN_ID="{{RUN_ID}}"               # e.g., 2026-02-04-1430
PULL_LATEST="{{PULL_LATEST}}"     # e.g., "" or "kdbai" or "all"
MAX_RUNTIME_MINUTES=120           # Safety timeout

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
echo "========================================"

# =============================================================================
# Auto-termination setup (set up EARLY to catch all exits)
# =============================================================================
S3_RESULT_PATH="s3://${S3_BUCKET}/runs/${RUN_ID}/jobs/${DATABASE}-${DATASET}"

cleanup() {
    EXIT_CODE=$?
    echo "Cleaning up... (exit code: $EXIT_CODE)"

    # Upload log file even on failure
    aws s3 cp "$LOGFILE" ${S3_RESULT_PATH}/worker.log --region us-west-2 || true

    # Mark as failed if non-zero exit
    if [ $EXIT_CODE -ne 0 ]; then
        echo "{\"status\": \"failed\", \"exit_code\": $EXIT_CODE}" | \
            aws s3 cp - ${S3_RESULT_PATH}/status.json --region us-west-2 || true
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
git pull origin main

# =============================================================================
# Pull fresh Docker images if requested
# =============================================================================
if [ "$PULL_LATEST" = "all" ]; then
    echo "Pulling all Docker images..."
    docker pull qdrant/qdrant:latest
    docker pull milvusdb/milvus:latest
    docker pull semitechnologies/weaviate:latest
    docker pull chromadb/chroma:latest
    docker pull redis/redis-stack:latest
    docker pull pgvector/pgvector:pg16
    docker pull portal.dl.kx.com/kdbai-db:latest
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
fi

# =============================================================================
# Run benchmark
# =============================================================================
echo "Running benchmark: $DATABASE on $DATASET"
cd /app/vectordb-benchmark

python3.12 run_benchmark.py \
    --config configs/${DATABASE}.yaml \
    --dataset ${DATASET} \
    --output results

BENCHMARK_EXIT_CODE=$?
echo "Benchmark completed with exit code: $BENCHMARK_EXIT_CODE"

# =============================================================================
# Upload results to S3
# =============================================================================
echo "Uploading results to S3..."

# Upload result files
aws s3 cp results/ ${S3_RESULT_PATH}/ --recursive --region us-west-2

# Mark job as complete
echo '{"status": "completed"}' | aws s3 cp - ${S3_RESULT_PATH}/status.json --region us-west-2

echo "========================================"
echo "Worker complete: $(date)"
echo "========================================"

# Exit cleanly - trap will handle shutdown and final log upload
