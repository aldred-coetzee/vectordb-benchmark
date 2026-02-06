#!/usr/bin/env python3
"""
Pull benchmark results from S3 and generate a report.

Downloads per-job benchmark databases from an AWS orchestrator run,
merges them into a single database, and generates an HTML report.

Usage:
    python scripts/pull_run.py 2026-02-05-1816
    python scripts/pull_run.py 2026-02-05-1816 --no-report
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

S3_BUCKET = "vectordb-benchmark-590780615264"
AWS_REGION = "us-west-2"


def s3_cmd(args: list[str]) -> subprocess.CompletedProcess:
    """Run an AWS CLI S3 command. Tries vectordb profile, falls back to default."""
    base_cmd = ["aws", "s3"] + args + ["--region", AWS_REGION]
    for profile in ["vectordb", None]:
        cmd = base_cmd + (["--profile", profile] if profile else [])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result
    return result


def list_jobs(run_id: str) -> list[str]:
    """List all jobs for a run from S3."""
    result = s3_cmd(["ls", f"s3://{S3_BUCKET}/runs/{run_id}/jobs/"])
    if result.returncode != 0:
        print(f"Error listing jobs: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    jobs = []
    for line in result.stdout.strip().split("\n"):
        if "PRE" in line:
            job_name = line.strip().split()[-1].rstrip("/")
            jobs.append(job_name)
    return jobs


def get_job_status(run_id: str, job: str) -> dict:
    """Get the status of a job from S3."""
    result = s3_cmd(["cp", f"s3://{S3_BUCKET}/runs/{run_id}/jobs/{job}/status.json", "-"])
    if result.returncode != 0:
        return {"status": "unknown"}
    return json.loads(result.stdout)


def download_job_db(run_id: str, job: str, dest: str) -> bool:
    """Download benchmark.db for a job."""
    result = s3_cmd(["cp", f"s3://{S3_BUCKET}/runs/{run_id}/jobs/{job}/benchmark.db", dest])
    return result.returncode == 0


def merge_databases(job_dbs: dict[str, str], output_path: str, run_label: str) -> None:
    """
    Merge per-job benchmark databases into a single DB.

    All rows get the same run_label (the orchestrator run ID) for traceability.
    Internal run_ids are remapped to be unique across jobs.
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    # Copy first DB as base
    first_job = next(iter(job_dbs))
    shutil.copy2(job_dbs[first_job], output_path)
    print(f"  Base: {first_job}")

    merged = sqlite3.connect(output_path)

    # Set run_label for rows from first DB
    merged.execute("UPDATE runs SET run_label = ?", (run_label,))

    # Get next available run_id
    next_run_id = merged.execute("SELECT MAX(run_id) FROM runs").fetchone()[0] + 1

    # Merge remaining DBs
    for job, path in job_dbs.items():
        if job == first_job:
            continue

        src = sqlite3.connect(path)
        src.row_factory = sqlite3.Row

        for run in src.execute("SELECT * FROM runs").fetchall():
            old_run_id = run["run_id"]
            new_run_id = next_run_id
            next_run_id += 1

            # Insert run with new ID and run_label
            merged.execute("""
                INSERT INTO runs (run_id, timestamp, start_time, end_time, duration_seconds,
                    database, db_version, db_client_version, dataset, vector_count, dimensions,
                    cpus, memory_gb, config_json, benchmark_config_json, hostname, notes,
                    run_label, instance_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (new_run_id, run["timestamp"], run["start_time"], run["end_time"],
                  run["duration_seconds"], run["database"], run["db_version"],
                  run["db_client_version"] if "db_client_version" in run.keys() else None,
                  run["dataset"], run["vector_count"], run["dimensions"],
                  run["cpus"], run["memory_gb"], run["config_json"],
                  run["benchmark_config_json"], run["hostname"], run["notes"],
                  run_label, run["instance_type"]))

            # Copy child tables with remapped run_id
            for row in src.execute("SELECT * FROM ingest_results WHERE run_id=?", (old_run_id,)):
                merged.execute(
                    "INSERT INTO ingest_results (run_id, index_type, index_config_json, "
                    "total_time_s, throughput_vps, peak_memory_gb, final_memory_gb, batch_size) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (new_run_id,) + tuple(row)[2:]
                )

            for row in src.execute("SELECT * FROM search_results WHERE run_id=?", (old_run_id,)):
                merged.execute(
                    "INSERT INTO search_results (run_id, index_type, ef_search, qps, "
                    "p50_ms, p95_ms, p99_ms, recall_at_10, recall_at_100, "
                    "avg_cpu_pct, avg_memory_gb, num_queries) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (new_run_id,) + tuple(row)[2:]
                )

            for row in src.execute("SELECT * FROM resource_samples WHERE run_id=?", (old_run_id,)):
                merged.execute(
                    "INSERT INTO resource_samples (run_id, timestamp, phase, cpu_pct, memory_gb) "
                    "VALUES (?,?,?,?,?)",
                    (new_run_id,) + tuple(row)[2:]
                )

        src.close()
        print(f"  Merged: {job}")

    merged.commit()

    # Summary
    total_runs = merged.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    total_search = merged.execute("SELECT COUNT(*) FROM search_results").fetchone()[0]
    total_ingest = merged.execute("SELECT COUNT(*) FROM ingest_results").fetchone()[0]
    print(f"  Total: {total_runs} runs, {total_ingest} ingest, {total_search} search results")

    merged.close()


def main():
    parser = argparse.ArgumentParser(
        description="Pull benchmark results from S3 and generate report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/pull_run.py 2026-02-05-1816
    python scripts/pull_run.py 2026-02-05-1816 --no-report
        """,
    )

    parser.add_argument("run_id", help="Orchestrator run ID (e.g., 2026-02-05-1816)")
    parser.add_argument("--no-report", action="store_true",
                        help="Only download and merge, skip report generation")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results)")

    args = parser.parse_args()
    run_id = args.run_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. List jobs
    print(f"Pulling results for run {run_id}...")
    jobs = list_jobs(run_id)
    if not jobs:
        print(f"No jobs found for run {run_id}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(jobs)} jobs\n")

    # 2. Check status and download
    completed: dict[str, str] = {}
    skipped: list[str] = []
    failed: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for job in sorted(jobs):
            status = get_job_status(run_id, job)
            s = status.get("status", "unknown")

            if s == "completed":
                dest = os.path.join(tmpdir, f"{job}.db")
                if download_job_db(run_id, job, dest):
                    # Verify it has actual data
                    conn = sqlite3.connect(dest)
                    count = conn.execute("SELECT COUNT(*) FROM search_results").fetchone()[0]
                    conn.close()
                    if count > 0:
                        completed[job] = dest
                        print(f"  {job}: {count} search results")
                    else:
                        print(f"  {job}: completed but empty (no search results)")
                        failed.append(job)
                else:
                    print(f"  {job}: download failed")
                    failed.append(job)
            elif s == "skipped":
                reason = status.get("reason", "no supported index types")
                print(f"  {job}: skipped ({reason})")
                skipped.append(job)
            elif s == "failed":
                exit_code = status.get("exit_code", "?")
                print(f"  {job}: failed (exit code {exit_code})")
                failed.append(job)
            else:
                print(f"  {job}: {s}")
                failed.append(job)

        if not completed:
            print("\nNo completed jobs with results!", file=sys.stderr)
            sys.exit(1)

        # 3. Merge
        db_path = output_dir / f"benchmark-{run_id}.db"
        print(f"\nMerging {len(completed)} jobs into {db_path}...")
        merge_databases(completed, str(db_path), run_id)

    # 4. Summary
    print(f"\nRun {run_id}:")
    print(f"  Completed: {len(completed)}")
    if skipped:
        print(f"  Skipped:   {len(skipped)} ({', '.join(skipped)})")
    if failed:
        print(f"  Failed:    {len(failed)} ({', '.join(failed)})")

    # 5. Generate combined report (with charts)
    if not args.no_report:
        report_path = output_dir / f"report-{run_id}.html"
        print(f"\nGenerating report: {report_path}")

        cmd = [
            sys.executable, "generate_report.py",
            "--db-path", str(db_path),
            "--output", str(report_path),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("Error generating report", file=sys.stderr)
            sys.exit(1)

    print(f"\nDone!")
    print(f"  Database: {db_path}")
    if not args.no_report:
        print(f"  Report:   {output_dir}/report-{run_id}.html")


if __name__ == "__main__":
    main()
