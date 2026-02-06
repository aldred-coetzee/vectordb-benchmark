#!/usr/bin/env python3
"""
Orchestrator for AWS-based vector database benchmarks.

Launches worker EC2 instances, monitors progress, and aggregates results.
"""

import argparse
import base64
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3

# AWS Configuration
AWS_REGION = "us-west-2"
S3_BUCKET = "vectordb-benchmark-590780615264"

# EC2 Configuration
WORKER_AMI = "ami-0be95bd332a011cc2"  # vectordb-benchmark-worker-v2
WORKER_INSTANCE_TYPE = "m5.4xlarge"
SUBNET_ID = "subnet-08871c182a622a9e9"
SECURITY_GROUP_ID = "sg-0fe6723dd1004558d"
IAM_INSTANCE_PROFILE = "vectordb-benchmark-role"
KEY_NAME = "vectordb-benchmark"

# Default databases for AWS runs (pgvector excluded: HNSW build too slow for GIST/high-dim;
# lancedb excluded: no supported index types). Both can still be run explicitly via --databases.
DATABASES = [
    "qdrant", "milvus", "weaviate", "chroma",
    "redis", "kdbai", "faiss",
]
DATASETS = ["sift", "gist", "glove-100", "dbpedia-openai"]

# Databases that don't need Docker (embedded)
EMBEDDED_DBS = ["faiss", "lancedb"]


def load_tuning_config(benchmark_type: str) -> dict | None:
    """Load a tuning config derived from the benchmark type.

    Convention: configs/tuning/{benchmark_type}.yaml
    Returns None if no tuning config exists for this benchmark type.
    """
    import yaml

    config_path = Path(__file__).parent.parent / "configs" / "tuning" / f"{benchmark_type}.yaml"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_tuning_jobs(
    tuning_config: dict,
    datasets: list[str],
) -> list[dict]:
    """Generate cross-product of (dataset x hnsw_config x docker_config) jobs.

    Returns list of job dicts with keys:
        database, dataset, hnsw_name, hnsw_m, hnsw_efc, docker_name,
        docker_threads, docker_num_wrk
    """
    method_params = tuning_config.get("method_params", {})
    docker_params = tuning_config.get("docker_params", {})

    hnsw_configs = method_params.get("hnsw_configs", [])
    docker_configs = docker_params.get("configs", [{"name": "default", "env": {}}])

    jobs = []
    for dataset in datasets:
        for hnsw_cfg in hnsw_configs:
            for docker_cfg in docker_configs:
                env = docker_cfg.get("env", {})
                jobs.append({
                    "database": "kdbai",
                    "dataset": dataset,
                    "hnsw_name": hnsw_cfg["name"],
                    "hnsw_m": hnsw_cfg["M"],
                    "hnsw_efc": hnsw_cfg["efConstruction"],
                    "docker_name": docker_cfg["name"],
                    "docker_threads": env.get("THREADS", "16"),
                    "docker_num_wrk": env.get("NUM_WRK", "1"),
                })

    return jobs


def load_worker_script() -> str:
    """Load the worker startup script template."""
    script_path = Path(__file__).parent / "worker_startup.sh"
    return script_path.read_text()


def create_user_data(
    database: str,
    dataset: str,
    run_id: str,
    benchmark_type: str = "competitive",
    pull_latest: str = "",
    hnsw_m: str = "",
    hnsw_efc: str = "",
    hnsw_name: str = "",
    docker_config_name: str = "",
    docker_threads: str = "",
    docker_num_wrk: str = "",
) -> str:
    """Create user-data script for a worker instance."""
    template = load_worker_script()

    # Substitute placeholders
    script = template.replace("{{DATABASE}}", database)
    script = script.replace("{{DATASET}}", dataset)
    script = script.replace("{{S3_BUCKET}}", S3_BUCKET)
    script = script.replace("{{RUN_ID}}", run_id)
    script = script.replace("{{BENCHMARK_TYPE}}", benchmark_type)
    script = script.replace("{{PULL_LATEST}}", pull_latest)

    # Tuning-specific placeholders (worker derives tuning config from BENCHMARK_TYPE)
    script = script.replace("{{HNSW_M}}", hnsw_m)
    script = script.replace("{{HNSW_EFC}}", hnsw_efc)
    script = script.replace("{{HNSW_NAME}}", hnsw_name)
    script = script.replace("{{DOCKER_CONFIG_NAME}}", docker_config_name)
    script = script.replace("{{DOCKER_THREADS}}", docker_threads)
    script = script.replace("{{DOCKER_NUM_WRK}}", docker_num_wrk)

    # Base64 encode for EC2 user-data
    return base64.b64encode(script.encode()).decode()


def launch_worker(
    ec2_client,
    database: str,
    dataset: str,
    run_id: str,
    benchmark_type: str = "competitive",
    pull_latest: str = "",
    dry_run: bool = False,
    hnsw_m: str = "",
    hnsw_efc: str = "",
    hnsw_name: str = "",
    docker_config_name: str = "",
    docker_threads: str = "",
    docker_num_wrk: str = "",
) -> Optional[str]:
    """Launch a worker EC2 instance."""

    user_data = create_user_data(
        database, dataset, run_id, benchmark_type, pull_latest,
        hnsw_m=hnsw_m, hnsw_efc=hnsw_efc, hnsw_name=hnsw_name,
        docker_config_name=docker_config_name,
        docker_threads=docker_threads, docker_num_wrk=docker_num_wrk,
    )

    # Include tuning params in instance name for identifiability
    if hnsw_name and docker_config_name:
        instance_name = f"vectordb-worker-{database}-{dataset}-{hnsw_name}-{docker_config_name}-{run_id}"
    else:
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


def check_job_status(
    s3_client, run_id: str, job_name: str, benchmark_type: str = "competitive",
) -> str:
    """Check the status of a job from S3.

    Args:
        s3_client: boto3 S3 client
        run_id: Run identifier
        job_name: Job directory name (e.g., 'kdbai-sift' or 'kdbai-sift-M32_efC128-1wrk_16thr')
        benchmark_type: Benchmark type for S3 path
    """
    key = f"runs/{benchmark_type}/{run_id}/jobs/{job_name}/status.json"
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
    job_names: list[str],
    benchmark_type: str = "competitive",
    timeout_minutes: int = 180,
) -> dict:
    """Wait for all jobs to complete and return status summary.

    Args:
        s3_client: boto3 S3 client
        run_id: Run identifier
        job_names: List of job directory names (e.g., 'kdbai-sift' or 'kdbai-sift-M32_efC128-1wrk_16thr')
        benchmark_type: Benchmark type for S3 path
        timeout_minutes: Maximum wait time
    """

    print(f"\nWaiting for {len(job_names)} jobs to complete (timeout: {timeout_minutes}m)...")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    results = {job: "pending" for job in job_names}

    while time.time() - start_time < timeout_seconds:
        all_done = True

        for job_name in job_names:
            if results[job_name] in ("completed", "failed", "skipped"):
                continue

            status = check_job_status(s3_client, run_id, job_name, benchmark_type)
            results[job_name] = status

            if status == "pending":
                all_done = False

        if all_done:
            break

        # Print status update
        completed = sum(1 for s in results.values() if s == "completed")
        skipped = sum(1 for s in results.values() if s == "skipped")
        failed = sum(1 for s in results.values() if s == "failed")
        pending = sum(1 for s in results.values() if s == "pending")
        parts = [f"{completed} completed"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")
        parts.append(f"{pending} pending")
        print(f"  Status: {', '.join(parts)}")

        time.sleep(30)  # Check every 30 seconds

    return results


def generate_and_upload_report(
    s3_client,
    run_id: str,
    benchmark_type: str,
    num_completed: int,
) -> bool:
    """Download per-job results, merge, generate report, and upload to S3.

    Uses scripts/pull_run.py to download and merge per-job databases,
    then uploads the merged DB and HTML report back to S3.
    Reports are also copied to a flat reports/ prefix for easy browsing.
    """
    if num_completed == 0:
        print("\nNo completed jobs — skipping report generation.")
        return False

    print(f"\n{'=' * 60}")
    print("GENERATING REPORT")
    print(f"{'=' * 60}")

    # pull_run.py handles: S3 download → merge → report generation
    result = subprocess.run(
        [sys.executable, "scripts/pull_run.py", run_id,
         "--benchmark-type", benchmark_type,
         "--output-dir", "results"],
    )

    if result.returncode != 0:
        print("ERROR: Report generation failed")
        return False

    # File naming: vectordb-benchmark-{type}-{run_id}.ext
    base_name = f"vectordb-benchmark-{benchmark_type}-{run_id}"
    run_prefix = f"runs/{benchmark_type}/{run_id}"

    # Upload combined report and merged DB to S3
    # Each file goes to its run directory + the report HTML also to reports/
    uploads: list[tuple[str, str, str]] = [
        (f"results/{base_name}.db", f"{run_prefix}/{base_name}.db", "application/octet-stream"),
        (f"results/{base_name}.html", f"{run_prefix}/{base_name}.html", "text/html"),
        (f"results/{base_name}.html", f"reports/{base_name}.html", "text/html"),
    ]

    for local_path, s3_key, content_type in uploads:
        if Path(local_path).exists():
            print(f"  Uploading {local_path} → s3://{S3_BUCKET}/{s3_key}")
            s3_client.upload_file(
                local_path, S3_BUCKET, s3_key,
                ExtraArgs={"ContentType": content_type},
            )
        else:
            print(f"  WARNING: {local_path} not found, skipping upload")

    print(f"\nReports: s3://{S3_BUCKET}/{run_prefix}/")
    return True


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
        "--benchmark-type", "-t",
        default="competitive",
        help="Benchmark type for S3 organization (default: competitive)",
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
    benchmark_type = args.benchmark_type

    # Each benchmark type has explicit handling
    if benchmark_type == "competitive":
        is_tuning = False
        tuning_cfg = None
    elif benchmark_type == "kdbai-tuning":
        is_tuning = True
        tuning_cfg = load_tuning_config(benchmark_type)
        if tuning_cfg is None:
            print(f"Error: Tuning config not found: configs/tuning/{benchmark_type}.yaml")
            return 1
    else:
        print(f"Unknown benchmark type: {benchmark_type}")
        print("Known types: competitive, kdbai-tuning")
        return 1

    # Validate datasets
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}")
            return 1

    # Validate databases (skip for tuning — always kdbai)
    if not is_tuning:
        for db in databases:
            if db not in DATABASES:
                print(f"Unknown database: {db}")
                return 1

    print("=" * 60)
    print("VECTORDB BENCHMARK ORCHESTRATOR")
    print("=" * 60)
    print(f"Run ID:     {run_id}")
    print(f"Type:       {benchmark_type}")
    if is_tuning:
        print(f"Mode:       TUNING (configs/tuning/{benchmark_type}.yaml)")
    print(f"Databases:  {', '.join(databases)}")
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Pull latest: {args.pull_latest or 'none'}")
    print(f"Dry run:    {args.dry_run}")
    print("=" * 60)

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

    # Build job list — competitive or tuning mode
    if is_tuning:
        tuning_jobs = generate_tuning_jobs(tuning_cfg, datasets)
        # Job names for S3 tracking: kdbai-sift-M32_efC128-1wrk_16thr
        job_names = [
            f"{j['database']}-{j['dataset']}-{j['hnsw_name']}-{j['docker_name']}"
            for j in tuning_jobs
        ]
        print(f"\nTuning jobs: {len(tuning_jobs)} "
              f"({len(datasets)} datasets x "
              f"{len(tuning_cfg.get('method_params', {}).get('hnsw_configs', []))} hnsw_configs x "
              f"{len(tuning_cfg.get('docker_params', {}).get('configs', []))} docker_configs)")
    else:
        tuning_jobs = None
        job_names = [f"{db}-{ds}" for db in databases for ds in datasets]
        print(f"\nTotal jobs: {len(job_names)}")

    # Create run config in S3
    run_prefix = f"runs/{benchmark_type}/{run_id}"
    if not args.dry_run:
        run_config = {
            "run_id": run_id,
            "benchmark_type": benchmark_type,
            "databases": databases,
            "datasets": datasets,
            "pull_latest": args.pull_latest,
            "tuning_config": f"configs/tuning/{benchmark_type}.yaml" if is_tuning else None,
            "started_at": datetime.now().isoformat(),
            "jobs": (
                [{"job_name": name} for name in job_names]
            ),
        }
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"{run_prefix}/config.json",
            Body=json.dumps(run_config, indent=2),
            ContentType="application/json",
        )
        print(f"\nRun config saved to s3://{S3_BUCKET}/{run_prefix}/config.json")

    # Launch workers
    print("\nLaunching workers...")
    instance_ids = []

    if is_tuning:
        for job in tuning_jobs:
            instance_id = launch_worker(
                ec2_client,
                job["database"],
                job["dataset"],
                run_id,
                benchmark_type=benchmark_type,
                pull_latest=args.pull_latest,
                dry_run=args.dry_run,
                hnsw_m=str(job["hnsw_m"]),
                hnsw_efc=str(job["hnsw_efc"]),
                hnsw_name=job["hnsw_name"],
                docker_config_name=job["docker_name"],
                docker_threads=job["docker_threads"],
                docker_num_wrk=job["docker_num_wrk"],
            )
            if instance_id:
                instance_ids.append(instance_id)
    else:
        for db in databases:
            for ds in datasets:
                instance_id = launch_worker(
                    ec2_client,
                    db,
                    ds,
                    run_id,
                    benchmark_type=benchmark_type,
                    pull_latest=args.pull_latest,
                    dry_run=args.dry_run,
                )
                if instance_id:
                    instance_ids.append(instance_id)

    if args.dry_run:
        print("\n[DRY RUN] No instances launched.")
        for name in job_names:
            print(f"  Would launch: {name}")
        return 0

    print(f"\nLaunched {len(instance_ids)} workers")

    if args.no_wait:
        print("\n--no-wait specified, exiting without waiting.")
        print(f"Check results at: s3://{S3_BUCKET}/{run_prefix}/")
        return 0

    # Wait for completion
    results = wait_for_completion(s3_client, run_id, job_names, benchmark_type, args.timeout)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    completed = [name for name, s in results.items() if s == "completed"]
    skipped = [name for name, s in results.items() if s == "skipped"]
    failed = [name for name, s in results.items() if s == "failed"]
    pending = [name for name, s in results.items() if s == "pending"]

    print(f"Completed: {len(completed)}")
    if skipped:
        print(f"Skipped:   {len(skipped)} (no supported index types)")
    print(f"Failed:    {len(failed)}")
    print(f"Pending:   {len(pending)} (timed out)")

    if skipped:
        print("\nSkipped jobs:")
        for name in skipped:
            print(f"  - {name}")

    if failed:
        print("\nFailed jobs:")
        for name in failed:
            print(f"  - {name}")

    # Generate merged report and upload to S3
    if completed:
        generate_and_upload_report(s3_client, run_id, benchmark_type, len(completed))

    print(f"\nResults: s3://{S3_BUCKET}/{run_prefix}/")

    return 0 if not failed and not pending else 1


if __name__ == "__main__":
    exit(main())
