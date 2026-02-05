#!/usr/bin/env python3
"""
Generate benchmark comparison reports.

Usage:
    # Default: one report per dataset in the DB
    python generate_report.py --db-path results/benchmark.db --output results/report.html

    # Single dataset
    python generate_report.py --dataset sift --output results/report-sift.html

    # From a run ID (generates per-dataset reports automatically)
    python generate_report.py --run-id 2026-02-05-1919

    # Specific databases
    python generate_report.py --databases faiss,kdbai,qdrant

    # Specific run IDs
    python generate_report.py --runs 1,2,3,4,5
"""

import argparse
import sys
from pathlib import Path

from benchmark.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate per-dataset HTML reports from a run
    python generate_report.py --run-id 2026-02-05-1919

    # Generate HTML report for a single dataset
    python generate_report.py --dataset sift --db-path results/benchmark.db -o results/report-sift.html

    # Filter specific databases
    python generate_report.py --databases faiss,kdbai,qdrant

    # Use specific run IDs
    python generate_report.py --runs 1,2,3
        """,
    )

    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout). "
             "When generating per-dataset reports, dataset name is inserted before extension.",
    )

    parser.add_argument(
        "--databases", "-d",
        help="Comma-separated list of databases to include",
    )

    parser.add_argument(
        "--runs", "-r",
        help="Comma-separated list of run IDs to include",
    )

    parser.add_argument(
        "--run-id",
        help="Orchestrator run ID (e.g., 2026-02-05-1816). "
             "Auto-resolves db-path to results/benchmark-{run_id}.db "
             "and generates per-dataset reports.",
    )

    parser.add_argument(
        "--dataset",
        help="Single dataset to report on (e.g., sift, gist). "
             "If omitted with multi-dataset DB, generates one report per dataset.",
    )

    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite database (default: results/benchmark.db)",
    )

    parser.add_argument(
        "--configs-dir",
        default="configs",
        help="Path to configs directory (default: configs)",
    )

    args = parser.parse_args()

    # Resolve paths from --run-id if provided
    if args.run_id:
        if args.db_path is None:
            args.db_path = f"results/benchmark-{args.run_id}.db"
        if args.output is None:
            args.format = "html"
            args.output = f"results/report-{args.run_id}.html"
    else:
        if args.db_path is None:
            args.db_path = "results/benchmark.db"

    # Auto-detect format from output file extension
    if args.output and args.format == "markdown":
        if args.output.endswith(".html"):
            args.format = "html"

    # Validate database file exists
    if not Path(args.db_path).exists():
        print(f"Error: Database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    # Parse filters
    databases = None
    if args.databases:
        databases = [d.strip() for d in args.databases.split(",")]

    run_ids = None
    if args.runs:
        try:
            run_ids = [int(r.strip()) for r in args.runs.split(",")]
        except ValueError:
            print("Error: --runs must be comma-separated integers", file=sys.stderr)
            sys.exit(1)

    # Generate report
    generator = ReportGenerator(db_path=args.db_path, configs_dir=args.configs_dir)

    try:
        if args.dataset:
            # Single dataset report
            report = generator.generate_report(
                format=args.format,
                databases=databases,
                run_ids=run_ids,
                dataset=args.dataset,
            )
            _write_report(report, args.output)
        else:
            # Check how many datasets exist
            available_datasets = generator.get_datasets()

            if len(available_datasets) <= 1:
                # Single dataset — generate one report
                report = generator.generate_report(
                    format=args.format,
                    databases=databases,
                    run_ids=run_ids,
                )
                _write_report(report, args.output)
            else:
                # Multiple datasets — generate one report per dataset
                ext = "html" if args.format == "html" else "md"
                for ds in available_datasets:
                    report = generator.generate_report(
                        format=args.format,
                        databases=databases,
                        run_ids=run_ids,
                        dataset=ds,
                    )
                    if args.output:
                        # Insert dataset name: report-RUN.html -> report-RUN-sift.html
                        base = Path(args.output)
                        ds_output = str(base.with_name(f"{base.stem}-{ds.lower()}.{ext}"))
                    else:
                        ds_output = None
                    _write_report(report, ds_output, label=ds)
    finally:
        generator.close()


def _write_report(report: str, output: str | None, label: str | None = None):
    """Write report to file or stdout."""
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report written to: {output}")
    else:
        if label:
            print(f"--- {label} ---")
        print(report)


if __name__ == "__main__":
    main()
