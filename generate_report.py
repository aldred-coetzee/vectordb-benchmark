#!/usr/bin/env python3
"""
Generate benchmark comparison reports.

Usage:
    # Default: latest run of each database, markdown output
    python generate_report.py

    # HTML output
    python generate_report.py --format html --output results/report.html

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
    # Generate markdown report for latest run of each database
    python generate_report.py

    # Generate HTML report
    python generate_report.py --format html -o results/report.html

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
        help="Output file path (default: stdout)",
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
        "--db-path",
        default="results/benchmark.db",
        help="Path to SQLite database (default: results/benchmark.db)",
    )

    parser.add_argument(
        "--configs-dir",
        default="configs",
        help="Path to configs directory (default: configs)",
    )

    args = parser.parse_args()

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
        report = generator.generate_report(
            format=args.format,
            databases=databases,
            run_ids=run_ids,
        )
    finally:
        generator.close()

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
