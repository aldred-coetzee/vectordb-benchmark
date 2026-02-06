#!/usr/bin/env python3
"""
Orchestrator for AWS-based vector database benchmarks.

Launches worker EC2 instances, monitors progress, and aggregates results.
"""

import argparse
import base64
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

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
    max_retries: int = 3,
) -> tuple[Optional[str], Optional[str]]:
    """Launch a worker EC2 instance with retry on transient errors.

    Returns (instance_id, error_message) — one is always None.
    """

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
        return None, None

    # EC2 error codes that are transient and worth retrying
    retryable_codes = {
        "InsufficientInstanceCapacity",
        "RequestLimitExceeded",
        "Server.InternalError",
        "Unavailable",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = ec2_client.run_instances(**launch_params)
            instance_id = response["Instances"][0]["InstanceId"]
            if attempt > 0:
                print(f"  Launched: {instance_name} ({instance_id}) (succeeded on attempt {attempt + 1})")
            else:
                print(f"  Launched: {instance_name} ({instance_id})")
            return instance_id, None
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            last_error = f"{error_code}: {e.response['Error']['Message']}"
            if error_code not in retryable_codes:
                break  # Non-retryable, fail immediately
            if attempt < max_retries - 1:
                delay = (5 * 3 ** attempt) + random.uniform(0, 2)
                print(f"  Retry {attempt + 1}/{max_retries} for {instance_name} in {delay:.0f}s: {last_error}")
                time.sleep(delay)
        except Exception as e:
            last_error = str(e)
            break  # Unknown error, don't retry

    print(f"  FAILED to launch {instance_name}: {last_error}")
    return None, last_error


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


def check_instance_health(
    ec2_client,
    job_instances: dict[str, str],
) -> dict[str, str]:
    """Check EC2 instance states for pending jobs.

    Args:
        ec2_client: boto3 EC2 client
        job_instances: {job_name: instance_id} for jobs to check

    Returns:
        {job_name: state} for instances that are NOT running/pending.
        Only returns entries for unhealthy instances (terminated, stopped, etc.).
    """
    if not job_instances:
        return {}

    instance_to_job = {iid: name for name, iid in job_instances.items()}
    instance_ids = list(job_instances.values())

    try:
        # describe_instances supports up to 1000 IDs per call
        unhealthy = {}
        for i in range(0, len(instance_ids), 100):
            batch = instance_ids[i:i + 100]
            response = ec2_client.describe_instances(InstanceIds=batch)
            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    iid = instance["InstanceId"]
                    state = instance["State"]["Name"]
                    if state in ("terminated", "stopped", "shutting-down"):
                        job_name = instance_to_job[iid]
                        unhealthy[job_name] = state
        return unhealthy
    except Exception as e:
        print(f"  WARNING: Instance health check failed: {e}")
        return {}


def wait_for_completion(
    ec2_client,
    s3_client,
    run_id: str,
    job_names: list[str],
    launched: dict[str, str],
    failed: dict[str, tuple[str, dict]],
    benchmark_type: str = "competitive",
    timeout_minutes: int = 180,
    relaunch_fn=None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Wait for all jobs to complete and return status summary.

    Periodically checks S3 for job completion, verifies instance health,
    and re-launches failed jobs if capacity becomes available.

    Args:
        ec2_client: boto3 EC2 client (for health checks and re-launches)
        s3_client: boto3 S3 client
        run_id: Run identifier
        job_names: List of job directory names (e.g., 'kdbai-sift' or 'kdbai-sift-M32_efC128-1wrk_16thr')
        launched: {job_name: instance_id} for successfully launched jobs
        failed: {job_name: (error_message, job_params)} for jobs that failed to launch
        benchmark_type: Benchmark type for S3 path
        timeout_minutes: Maximum wait time
        relaunch_fn: Callback to re-launch a failed job. Called as relaunch_fn(job_params)
            and should return (instance_id, error_message).
    """

    print(f"\nWaiting for {len(job_names)} jobs to complete (timeout: {timeout_minutes}m)...")

    if failed:
        print(f"  {len(failed)} jobs failed to launch — will retry within first 30 min")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    relaunch_window_seconds = 30 * 60  # Re-launch attempts within first 30 minutes

    results = {job: "pending" for job in job_names}
    # Mark jobs that failed to launch with a distinct status
    for job_name in failed:
        results[job_name] = "launch_failed"

    # Track re-launch attempts per job (max 5 — capacity frees up gradually)
    relaunch_attempts: dict[str, int] = {}
    max_relaunch_attempts = 5
    # Track why jobs failed for final reporting
    failure_reasons: dict[str, str] = {
        name: err for name, (err, _) in failed.items()
    }
    # Track how many jobs have completed so we know when capacity frees up
    prev_completed = 0

    poll_cycle = 0

    while time.time() - start_time < timeout_seconds:
        poll_cycle += 1

        # Check S3 for job completion (BEFORE re-launch, so we know if capacity freed up)
        all_done = True
        for job_name in job_names:
            if results[job_name] in ("completed", "failed", "skipped", "instance_crashed", "launch_failed"):
                continue

            status = check_job_status(s3_client, run_id, job_name, benchmark_type)
            results[job_name] = status

            if status == "pending":
                all_done = False

        # Don't exit yet if there are launch_failed jobs that could still be re-launched
        elapsed_check = time.time() - start_time
        retriable_failures = (
            elapsed_check < relaunch_window_seconds
            and relaunch_fn
            and any(
                results[name] == "launch_failed"
                and relaunch_attempts.get(name, 0) < max_relaunch_attempts
                for name in failed
            )
        )
        if all_done and not retriable_failures:
            break

        # Instance health check every 5 cycles (~2.5 min)
        if poll_cycle % 5 == 0:
            # Build map of pending jobs that have instances
            pending_instances = {
                name: launched[name]
                for name in job_names
                if results[name] == "pending" and name in launched
            }
            if pending_instances:
                unhealthy = check_instance_health(ec2_client, pending_instances)
                for job_name, state in unhealthy.items():
                    reason = f"instance {state} without uploading results"
                    print(f"  CRASHED: {job_name} — {reason}")
                    results[job_name] = "instance_crashed"
                    failure_reasons[job_name] = reason

        # Print status update
        completed = sum(1 for s in results.values() if s == "completed")
        skipped = sum(1 for s in results.values() if s == "skipped")
        s3_failed = sum(1 for s in results.values() if s == "failed")
        crashed = sum(1 for s in results.values() if s == "instance_crashed")
        launch_failed = sum(1 for s in results.values() if s == "launch_failed")
        pending = sum(1 for s in results.values() if s == "pending")
        parts = [f"{completed} completed"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if s3_failed:
            parts.append(f"{s3_failed} failed")
        if crashed:
            parts.append(f"{crashed} crashed")
        if launch_failed:
            parts.append(f"{launch_failed} launch failed")
        parts.append(f"{pending} pending")
        print(f"  Status: {', '.join(parts)}")

        # Re-launch failed jobs when capacity frees up (new completions detected)
        cur_completed = completed + skipped + s3_failed + crashed
        elapsed = time.time() - start_time
        has_failed_jobs = any(
            results[name] == "launch_failed" and relaunch_attempts.get(name, 0) < max_relaunch_attempts
            for name in failed
        )
        if (has_failed_jobs and elapsed < relaunch_window_seconds
                and relaunch_fn and cur_completed > prev_completed):
            for job_name in list(failed.keys()):
                if results[job_name] != "launch_failed":
                    continue  # Already re-launched successfully
                attempts = relaunch_attempts.get(job_name, 0)
                if attempts >= max_relaunch_attempts:
                    continue
                relaunch_attempts[job_name] = attempts + 1
                _, job_params = failed[job_name]
                print(f"  Re-launching {job_name} (attempt {attempts + 1}/{max_relaunch_attempts}, "
                      f"previous: {failure_reasons[job_name]})")
                instance_id, error = relaunch_fn(job_params)
                if instance_id:
                    launched[job_name] = instance_id
                    results[job_name] = "pending"
                    del failed[job_name]
                    failure_reasons.pop(job_name, None)
                else:
                    failure_reasons[job_name] = error or "unknown error"
        prev_completed = cur_completed

        time.sleep(30)  # Check every 30 seconds

    return results, failure_reasons


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

    # Launch workers — track successes and failures separately
    print("\nLaunching workers...")
    launched = {}   # job_name -> instance_id
    failed = {}     # job_name -> (error_message, job_params)

    if is_tuning:
        for i, job in enumerate(tuning_jobs):
            job_name = job_names[i]
            instance_id, error = launch_worker(
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
                launched[job_name] = instance_id
            elif error:  # Not a dry run
                # Store job params for potential re-launch
                failed[job_name] = (error, {
                    "database": job["database"],
                    "dataset": job["dataset"],
                    "run_id": run_id,
                    "benchmark_type": benchmark_type,
                    "pull_latest": args.pull_latest,
                    "hnsw_m": str(job["hnsw_m"]),
                    "hnsw_efc": str(job["hnsw_efc"]),
                    "hnsw_name": job["hnsw_name"],
                    "docker_config_name": job["docker_name"],
                    "docker_threads": job["docker_threads"],
                    "docker_num_wrk": job["docker_num_wrk"],
                })
    else:
        job_idx = 0
        for db in databases:
            for ds in datasets:
                job_name = job_names[job_idx]
                job_idx += 1
                instance_id, error = launch_worker(
                    ec2_client,
                    db,
                    ds,
                    run_id,
                    benchmark_type=benchmark_type,
                    pull_latest=args.pull_latest,
                    dry_run=args.dry_run,
                )
                if instance_id:
                    launched[job_name] = instance_id
                elif error:  # Not a dry run
                    failed[job_name] = (error, {
                        "database": db,
                        "dataset": ds,
                        "run_id": run_id,
                        "benchmark_type": benchmark_type,
                        "pull_latest": args.pull_latest,
                    })

    if args.dry_run:
        print("\n[DRY RUN] No instances launched.")
        for name in job_names:
            print(f"  Would launch: {name}")
        return 0

    print(f"\nLaunched {len(launched)}/{len(job_names)} workers")
    if failed:
        print(f"{len(failed)} jobs failed to launch:")
        for name, (err, _) in failed.items():
            print(f"  {name}: {err}")

    if args.no_wait:
        print("\n--no-wait specified, exiting without waiting.")
        print(f"Check results at: s3://{S3_BUCKET}/{run_prefix}/")
        return 0

    # Build re-launch callback
    def relaunch_fn(job_params: dict) -> tuple[Optional[str], Optional[str]]:
        """Re-launch a failed job using stored parameters."""
        return launch_worker(ec2_client, **job_params, dry_run=False)

    # Wait for completion with health checks and re-launch
    results, failure_reasons = wait_for_completion(
        ec2_client, s3_client, run_id, job_names,
        launched, failed,
        benchmark_type=benchmark_type,
        timeout_minutes=args.timeout,
        relaunch_fn=relaunch_fn,
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    completed = [name for name, s in results.items() if s == "completed"]
    skipped = [name for name, s in results.items() if s == "skipped"]
    s3_failed = [name for name, s in results.items() if s == "failed"]
    crashed = [name for name, s in results.items() if s == "instance_crashed"]
    launch_failed = [name for name, s in results.items() if s == "launch_failed"]
    pending = [name for name, s in results.items() if s == "pending"]

    print(f"Completed:        {len(completed)}")
    if skipped:
        print(f"Skipped:          {len(skipped)} (no supported index types)")
    if s3_failed:
        print(f"Failed:           {len(s3_failed)} (benchmark error)")
    if crashed:
        print(f"Instance crashed: {len(crashed)}")
    if launch_failed:
        print(f"Launch failed:    {len(launch_failed)} (never started)")
    if pending:
        print(f"Timed out:        {len(pending)}")

    if skipped:
        print("\nSkipped jobs:")
        for name in skipped:
            print(f"  - {name}")

    if s3_failed:
        print("\nFailed jobs (benchmark error):")
        for name in s3_failed:
            print(f"  - {name}")

    if crashed:
        print("\nCrashed instances:")
        for name in crashed:
            reason = failure_reasons.get(name, "unknown")
            print(f"  - {name}: {reason}")

    if launch_failed:
        print("\nFailed to launch:")
        for name in launch_failed:
            reason = failure_reasons.get(name, "unknown")
            print(f"  - {name}: {reason}")

    if pending:
        print("\nTimed out (still pending):")
        for name in pending:
            print(f"  - {name}")

    # Generate merged report and upload to S3
    if completed:
        generate_and_upload_report(s3_client, run_id, benchmark_type, len(completed))

    print(f"\nResults: s3://{S3_BUCKET}/{run_prefix}/")

    any_failures = s3_failed or crashed or launch_failed or pending
    return 0 if not any_failures else 1


if __name__ == "__main__":
    exit(main())
