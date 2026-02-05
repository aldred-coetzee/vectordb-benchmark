#!/usr/bin/env python3
"""
Orchestrator for AWS-based vector database benchmarks.

Launches worker EC2 instances, monitors progress, and aggregates results.
"""

import argparse
import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3

# AWS Configuration
AWS_REGION = "us-west-2"
S3_BUCKET = "vectordb-benchmark-590780615264"

# EC2 Configuration
WORKER_AMI = "ami-0f9bf04496aedd923"  # vectordb-benchmark-worker-v1
WORKER_INSTANCE_TYPE = "m5.2xlarge"
SUBNET_ID = "subnet-08871c182a622a9e9"
SECURITY_GROUP_ID = "sg-0fe6723dd1004558d"
IAM_INSTANCE_PROFILE = "vectordb-benchmark-role"
KEY_NAME = "vectordb-benchmark"

# Available databases and datasets
DATABASES = [
    "qdrant", "milvus", "weaviate", "chroma",
    "redis", "pgvector", "kdbai", "faiss", "lancedb"
]
DATASETS = ["sift", "gist"]

# Databases that don't need Docker (embedded)
EMBEDDED_DBS = ["faiss", "lancedb"]


def load_worker_script() -> str:
    """Load the worker startup script template."""
    script_path = Path(__file__).parent / "worker_startup.sh"
    return script_path.read_text()


def create_user_data(
    database: str,
    dataset: str,
    run_id: str,
    pull_latest: str = "",
) -> str:
    """Create user-data script for a worker instance."""
    template = load_worker_script()

    # Substitute placeholders
    script = template.replace("{{DATABASE}}", database)
    script = script.replace("{{DATASET}}", dataset)
    script = script.replace("{{S3_BUCKET}}", S3_BUCKET)
    script = script.replace("{{RUN_ID}}", run_id)
    script = script.replace("{{PULL_LATEST}}", pull_latest)

    # Base64 encode for EC2 user-data
    return base64.b64encode(script.encode()).decode()


def launch_worker(
    ec2_client,
    database: str,
    dataset: str,
    run_id: str,
    pull_latest: str = "",
    dry_run: bool = False,
) -> Optional[str]:
    """Launch a worker EC2 instance."""

    user_data = create_user_data(database, dataset, run_id, pull_latest)

    instance_name = f"vectordb-worker-{database}-{dataset}-{run_id}"

    launch_params = {
        "ImageId": WORKER_AMI,
        "InstanceType": WORKER_INSTANCE_TYPE,
        "KeyName": KEY_NAME,
        "MaxCount": 1,
        "MinCount": 1,
        "UserData": user_data,
        "NetworkInterfaces": [{
            "DeviceIndex": 0,
            "SubnetId": SUBNET_ID,
            "Groups": [SECURITY_GROUP_ID],
            "AssociatePublicIpAddress": True,
        }],
        "IamInstanceProfile": {"Name": IAM_INSTANCE_PROFILE},
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": instance_name},
                {"Key": "Owner", "Value": "acoetzee"},
                {"Key": "Project", "Value": "vectordb-benchmark"},
                {"Key": "RunId", "Value": run_id},
                {"Key": "Database", "Value": database},
                {"Key": "Dataset", "Value": dataset},
            ],
        }],
        "InstanceInitiatedShutdownBehavior": "terminate",
    }

    if dry_run:
        print(f"  [DRY RUN] Would launch: {instance_name}")
        return None

    try:
        response = ec2_client.run_instances(**launch_params)
        instance_id = response["Instances"][0]["InstanceId"]
        print(f"  Launched: {instance_name} ({instance_id})")
        return instance_id
    except Exception as e:
        print(f"  ERROR launching {instance_name}: {e}")
        return None


def check_job_status(s3_client, run_id: str, database: str, dataset: str) -> str:
    """Check the status of a job from S3."""
    key = f"runs/{run_id}/jobs/{database}-{dataset}/status.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        status = json.loads(response["Body"].read().decode())
        return status.get("status", "unknown")
    except s3_client.exceptions.NoSuchKey:
        return "pending"
    except Exception:
        return "unknown"


def wait_for_completion(
    s3_client,
    run_id: str,
    jobs: list[tuple[str, str]],
    timeout_minutes: int = 180,
) -> dict:
    """Wait for all jobs to complete and return status summary."""

    print(f"\nWaiting for {len(jobs)} jobs to complete (timeout: {timeout_minutes}m)...")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    results = {job: "pending" for job in jobs}

    while time.time() - start_time < timeout_seconds:
        all_done = True

        for database, dataset in jobs:
            if results[(database, dataset)] in ("completed", "failed"):
                continue

            status = check_job_status(s3_client, run_id, database, dataset)
            results[(database, dataset)] = status

            if status == "pending":
                all_done = False

        if all_done:
            break

        # Print status update
        completed = sum(1 for s in results.values() if s == "completed")
        failed = sum(1 for s in results.values() if s == "failed")
        pending = sum(1 for s in results.values() if s == "pending")
        print(f"  Status: {completed} completed, {failed} failed, {pending} pending")

        time.sleep(30)  # Check every 30 seconds

    return results


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y-%m-%d-%H%M")


def main():
    parser = argparse.ArgumentParser(description="Launch vectordb benchmark workers")

    parser.add_argument(
        "--databases", "-d",
        help="Comma-separated list of databases (default: all)",
        default=",".join(DATABASES),
    )
    parser.add_argument(
        "--datasets", "-s",
        help="Comma-separated list of datasets (default: all)",
        default=",".join(DATASETS),
    )
    parser.add_argument(
        "--pull-latest",
        help="Pull fresh Docker images: 'all' or comma-separated list (e.g., 'kdbai,qdrant')",
        default="",
    )
    parser.add_argument(
        "--run-id",
        help="Custom run ID (default: auto-generated timestamp)",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be launched without actually launching",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Launch workers and exit without waiting for completion",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in minutes for waiting (default: 180)",
    )

    args = parser.parse_args()

    # Parse inputs
    databases = [d.strip() for d in args.databases.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    run_id = args.run_id or generate_run_id()

    # Validate
    for db in databases:
        if db not in DATABASES:
            print(f"Unknown database: {db}")
            return 1
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}")
            return 1

    print("=" * 60)
    print("VECTORDB BENCHMARK ORCHESTRATOR")
    print("=" * 60)
    print(f"Run ID:     {run_id}")
    print(f"Databases:  {', '.join(databases)}")
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Pull latest: {args.pull_latest or 'none'}")
    print(f"Dry run:    {args.dry_run}")
    print("=" * 60)

    # Create job list
    jobs = [(db, ds) for db in databases for ds in datasets]
    print(f"\nTotal jobs: {len(jobs)}")

    # Initialize AWS clients
    # On EC2: uses IAM role (no profile needed)
    # Locally: uses 'vectordb' profile if available, otherwise default
    try:
        session = boto3.Session(region_name=AWS_REGION, profile_name="vectordb")
        session.client("sts").get_caller_identity()  # Test if profile works
    except Exception:
        # Profile not available (running on EC2), use default credentials
        session = boto3.Session(region_name=AWS_REGION)
    ec2_client = session.client("ec2")
    s3_client = session.client("s3")

    # Create run config in S3
    if not args.dry_run:
        run_config = {
            "run_id": run_id,
            "databases": databases,
            "datasets": datasets,
            "pull_latest": args.pull_latest,
            "started_at": datetime.now().isoformat(),
            "jobs": [{"database": db, "dataset": ds} for db, ds in jobs],
        }
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"runs/{run_id}/config.json",
            Body=json.dumps(run_config, indent=2),
            ContentType="application/json",
        )
        print(f"\nRun config saved to s3://{S3_BUCKET}/runs/{run_id}/config.json")

    # Launch workers
    print("\nLaunching workers...")
    instance_ids = []

    for database, dataset in jobs:
        instance_id = launch_worker(
            ec2_client,
            database,
            dataset,
            run_id,
            pull_latest=args.pull_latest,
            dry_run=args.dry_run,
        )
        if instance_id:
            instance_ids.append(instance_id)

    if args.dry_run:
        print("\n[DRY RUN] No instances launched.")
        return 0

    print(f"\nLaunched {len(instance_ids)} workers")

    if args.no_wait:
        print("\n--no-wait specified, exiting without waiting.")
        print(f"Check results at: s3://{S3_BUCKET}/runs/{run_id}/")
        return 0

    # Wait for completion
    results = wait_for_completion(s3_client, run_id, jobs, args.timeout)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    completed = [(db, ds) for (db, ds), s in results.items() if s == "completed"]
    failed = [(db, ds) for (db, ds), s in results.items() if s == "failed"]
    pending = [(db, ds) for (db, ds), s in results.items() if s == "pending"]

    print(f"Completed: {len(completed)}")
    print(f"Failed:    {len(failed)}")
    print(f"Pending:   {len(pending)} (timed out)")

    if failed:
        print("\nFailed jobs:")
        for db, ds in failed:
            print(f"  - {db}/{ds}")

    print(f"\nResults: s3://{S3_BUCKET}/runs/{run_id}/")

    return 0 if not failed and not pending else 1


if __name__ == "__main__":
    exit(main())
