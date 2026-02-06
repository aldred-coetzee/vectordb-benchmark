#!/bin/bash
# Orchestrator startup script for vectordb-benchmark
# This script is passed as user-data when launching the orchestrator EC2 instance
#
# Runs on the Orchestrator AMI (Python 3.12, boto3, git — no Docker/datasets).
# Launches worker instances, monitors progress, aggregates results.
#
# CONFIGURATION VIA INSTANCE TAGS (all optional — defaults from orchestrator.py):
#   Databases      - Comma-separated list (default: all in DATABASES)
#   Datasets       - Comma-separated list (default: all in DATASETS)
#   BenchmarkType  - S3 organization type (default: competitive)
#   PullLatest     - Docker images to refresh (default: none)
#
# Example tags when launching:
#   Databases = qdrant,milvus,kdbai
#   Datasets = sift
#   BenchmarkType = kdbai-tuning

# =============================================================================
# Configuration
# =============================================================================
export HOME=/root

S3_BUCKET="vectordb-benchmark-590780615264"
MAX_RUNTIME_MINUTES=240  # 4 hours max for full suite
AWS_REGION="us-west-2"

# Get instance ID for reading tags (IMDSv2 - requires token on AL2023)
IMDS_TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
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
BENCHMARK_TYPE=$(get_tag "BenchmarkType")

# Normalize "None" (AWS CLI returns literal "None" for missing tags)
[ "$DATABASES" = "None" ] && DATABASES=""
[ "$DATASETS" = "None" ] && DATASETS=""
[ "$PULL_LATEST" = "None" ] && PULL_LATEST=""
[ "$BENCHMARK_TYPE" = "None" ] && BENCHMARK_TYPE=""
# Default benchmark type to competitive
[ -z "$BENCHMARK_TYPE" ] && BENCHMARK_TYPE="competitive"

# =============================================================================
# Logging
# =============================================================================
LOGFILE="/tmp/orchestrator.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "Orchestrator startup: $(date)"
echo "Benchmark type: $BENCHMARK_TYPE"
echo "Databases: $DATABASES"
echo "Datasets: $DATASETS"
echo "Pull latest: ${PULL_LATEST:-none}"
echo "========================================"

# =============================================================================
# Auto-termination setup
# =============================================================================
RUN_ID=$(date +%Y-%m-%d-%H%M)
S3_RESULT_PATH="s3://${S3_BUCKET}/runs/${BENCHMARK_TYPE}/${RUN_ID}"

# Update instance Name tag to include run ID
aws ec2 create-tags --resources $INSTANCE_ID \
    --tags "Key=Name,Value=vectordb-orchestrator-$RUN_ID" \
    --region $AWS_REGION

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
# Setup
# =============================================================================
# git pull already done by user-data before this script runs
cd /app/vectordb-benchmark

# Create symlink so relative paths work (datasets at /data on worker AMI)
ln -sf /data /app/vectordb-benchmark/data 2>/dev/null || true

# Install Python packages needed for report generation
# (report_generator → config (yaml))
echo "Installing report generation dependencies..."
pip3.12 install pyyaml 2>&1 | tail -1

# =============================================================================
# Run orchestrator
# =============================================================================
echo "Starting orchestrator..."
cd /app/vectordb-benchmark

# Build command with options
# Only pass flags when tags are set — orchestrator.py defaults to all databases/datasets
ORCHESTRATOR_ARGS="aws/orchestrator.py --run-id $RUN_ID --benchmark-type $BENCHMARK_TYPE"

if [ -n "$DATABASES" ]; then
    ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --databases $DATABASES"
fi
if [ -n "$DATASETS" ]; then
    ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --datasets $DATASETS"
fi
if [ -n "$PULL_LATEST" ]; then
    ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --pull-latest $PULL_LATEST"
fi

echo "Running: sudo -u ec2-user python3.12 $ORCHESTRATOR_ARGS"
sudo -u ec2-user python3.12 $ORCHESTRATOR_ARGS

ORCHESTRATOR_EXIT_CODE=$?
echo "Orchestrator completed with exit code: $ORCHESTRATOR_EXIT_CODE"

echo "========================================"
echo "Orchestrator complete: $(date)"
echo "Results: ${S3_RESULT_PATH}/"
echo "========================================"

# Exit cleanly - trap will handle shutdown
