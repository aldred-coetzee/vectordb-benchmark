#!/bin/bash
# Orchestrator startup script for vectordb-benchmark
# This script is passed as user-data when launching the orchestrator EC2 instance
#
# Uses the same Worker AMI but runs orchestrator.py instead of a single benchmark.
# Launches worker instances, monitors progress, aggregates results.
#
# CONFIGURATION VIA INSTANCE TAGS:
#   Databases   - Comma-separated list (default: all 9)
#   Datasets    - Comma-separated list (default: sift,gist)
#   PullLatest  - Docker images to refresh (default: none)
#
# Example tags when launching:
#   Databases = qdrant,milvus,kdbai
#   Datasets = sift

# =============================================================================
# Configuration
# =============================================================================
export HOME=/root

S3_BUCKET="vectordb-benchmark-590780615264"
MAX_RUNTIME_MINUTES=240  # 4 hours max for full suite
AWS_REGION="us-west-2"

# Get instance ID for reading tags
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
echo "Instance ID: $INSTANCE_ID"

# Read configuration from instance tags (with defaults)
get_tag() {
    aws ec2 describe-tags \
        --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=$1" \
        --query 'Tags[0].Value' --output text --region $AWS_REGION 2>/dev/null
}

DATABASES=$(get_tag "Databases")
DATASETS=$(get_tag "Datasets")
PULL_LATEST=$(get_tag "PullLatest")

# Apply defaults if tags not set or returned "None"
[ -z "$DATABASES" ] || [ "$DATABASES" = "None" ] && DATABASES="qdrant,milvus,weaviate,chroma,redis,pgvector,kdbai,faiss,lancedb"
[ -z "$DATASETS" ] || [ "$DATASETS" = "None" ] && DATASETS="sift,gist"
[ "$PULL_LATEST" = "None" ] && PULL_LATEST=""

# =============================================================================
# Logging
# =============================================================================
LOGFILE="/tmp/orchestrator.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "Orchestrator startup: $(date)"
echo "Databases: $DATABASES"
echo "Datasets: $DATASETS"
echo "Pull latest: ${PULL_LATEST:-none}"
echo "========================================"

# =============================================================================
# Auto-termination setup
# =============================================================================
RUN_ID=$(date +%Y-%m-%d-%H%M)
S3_RESULT_PATH="s3://${S3_BUCKET}/runs/${RUN_ID}"

cleanup() {
    EXIT_CODE=$?
    echo "Orchestrator cleanup... (exit code: $EXIT_CODE)"

    # Upload orchestrator log
    aws s3 cp "$LOGFILE" ${S3_RESULT_PATH}/orchestrator.log --region us-west-2 || true

    echo "Shutting down orchestrator..."
    sudo shutdown -c 2>/dev/null || true
    sudo shutdown -h now
}
trap cleanup EXIT

# Safety net - terminate after max runtime
sudo shutdown -h +${MAX_RUNTIME_MINUTES}

set -e

# =============================================================================
# Pull latest code
# =============================================================================
echo "Pulling latest code..."
cd /app/vectordb-benchmark

git config --global --add safe.directory /app/vectordb-benchmark
git pull origin main

# Create symlink so relative paths work
ln -sf /data /app/vectordb-benchmark/data

# =============================================================================
# Run orchestrator
# =============================================================================
echo "Starting orchestrator..."
cd /app/vectordb-benchmark

# Build command with options
ORCHESTRATOR_CMD="python3.12 aws/orchestrator.py"
ORCHESTRATOR_CMD="$ORCHESTRATOR_CMD --databases $DATABASES"
ORCHESTRATOR_CMD="$ORCHESTRATOR_CMD --datasets $DATASETS"
ORCHESTRATOR_CMD="$ORCHESTRATOR_CMD --run-id $RUN_ID"

if [ -n "$PULL_LATEST" ]; then
    ORCHESTRATOR_CMD="$ORCHESTRATOR_CMD --pull-latest $PULL_LATEST"
fi

echo "Running: $ORCHESTRATOR_CMD"
$ORCHESTRATOR_CMD

ORCHESTRATOR_EXIT_CODE=$?
echo "Orchestrator completed with exit code: $ORCHESTRATOR_EXIT_CODE"

echo "========================================"
echo "Orchestrator complete: $(date)"
echo "Results: ${S3_RESULT_PATH}/"
echo "========================================"

# Exit cleanly - trap will handle shutdown
