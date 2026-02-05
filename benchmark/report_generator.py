"""Report generator for benchmark results."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_yaml_config
from .db import BenchmarkDatabase


@dataclass
class IngestData:
    """Ingest benchmark data."""
    index_type: str
    throughput_vps: float
    total_time_s: float
    peak_memory_gb: Optional[float] = None
    final_memory_gb: Optional[float] = None


@dataclass
class SearchData:
    """Search benchmark data."""
    index_type: str
    ef_search: Optional[int]
    qps: float
    recall_at_10: Optional[float] = None
    recall_at_100: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None


@dataclass
class RunData:
    """Complete run data for a database."""
    run_id: int
    database: str
    db_version: Optional[str]
    dataset: str
    vector_count: int
    dimensions: int
    timestamp: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    cpus: Optional[float] = None
    memory_gb: Optional[float] = None
    hostname: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    ingest_results: List[IngestData] = field(default_factory=list)
    search_results: List[SearchData] = field(default_factory=list)


class ReportGenerator:
    """Generate benchmark comparison reports."""

    def __init__(self, db_path: str = "results/benchmark.db", configs_dir: str = "configs"):
        """Initialize the report generator."""
        self.db = BenchmarkDatabase(db_path)
        self.configs_dir = Path(configs_dir)

    def close(self):
        """Close database connection."""
        self.db.close()

    def get_latest_runs(self, databases: Optional[List[str]] = None) -> List[RunData]:
        """
        Get the most recent run for each database.

        Args:
            databases: Optional list of database names to filter

        Returns:
            List of RunData objects
        """
        conn = self.db._get_connection()
        cursor = conn.cursor()

        # Get latest run_id for each database
        query = """
            SELECT r.* FROM runs r
            INNER JOIN (
                SELECT database, MAX(run_id) as max_id
                FROM runs
                GROUP BY database
            ) latest ON r.run_id = latest.max_id
        """

        if databases:
            placeholders = ",".join("?" * len(databases))
            query += f" WHERE r.database IN ({placeholders})"
            cursor.execute(query, databases)
        else:
            cursor.execute(query)

        runs = []
        for row in cursor.fetchall():
            run_data = self._build_run_data(dict(row))
            runs.append(run_data)

        return runs

    def get_runs_by_ids(self, run_ids: List[int]) -> List[RunData]:
        """
        Get specific runs by their IDs.

        Args:
            run_ids: List of run IDs

        Returns:
            List of RunData objects
        """
        runs = []
        for run_id in run_ids:
            details = self.db.get_run_details(run_id)
            if details:
                run_data = self._build_run_data(details)
                runs.append(run_data)
        return runs

    def _build_run_data(self, row: Dict[str, Any]) -> RunData:
        """Build RunData from database row."""
        run_id = row["run_id"]
        database = row["database"]

        # Get ingest and search results
        conn = self.db._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM ingest_results WHERE run_id = ?", (run_id,))
        ingest_results = [
            IngestData(
                index_type=r["index_type"],
                throughput_vps=r["throughput_vps"],
                total_time_s=r["total_time_s"],
                peak_memory_gb=r["peak_memory_gb"],
                final_memory_gb=r["final_memory_gb"],
            )
            for r in cursor.fetchall()
        ]

        cursor.execute("SELECT * FROM search_results WHERE run_id = ? ORDER BY index_type, ef_search", (run_id,))
        search_results = [
            SearchData(
                index_type=r["index_type"],
                ef_search=r["ef_search"],
                qps=r["qps"],
                recall_at_10=r["recall_at_10"],
                recall_at_100=r["recall_at_100"],
                p50_ms=r["p50_ms"],
                p95_ms=r["p95_ms"],
                p99_ms=r["p99_ms"],
            )
            for r in cursor.fetchall()
        ]

        # Load metadata from config file
        metadata = {}
        config = {}
        # Normalize database name for config lookup (e.g. "KDB.AI" -> "kdbai")
        config_name = database.lower().replace(".", "")
        config_path = self.configs_dir / f"{config_name}.yaml"
        if config_path.exists():
            try:
                full_config = load_yaml_config(str(config_path))
                metadata = full_config.get("metadata", {})
                config = full_config
            except Exception:
                pass

        return RunData(
            run_id=run_id,
            database=database,
            db_version=row.get("db_version"),
            dataset=row.get("dataset", "unknown"),
            vector_count=row.get("vector_count", 0),
            dimensions=row.get("dimensions", 0),
            timestamp=row.get("timestamp", ""),
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            duration_seconds=row.get("duration_seconds"),
            cpus=row.get("cpus"),
            memory_gb=row.get("memory_gb"),
            hostname=row.get("hostname"),
            metadata=metadata,
            config=config,
            ingest_results=ingest_results,
            search_results=search_results,
        )

    def _get_best_search_result(self, run: RunData, min_recall: float = 0.97) -> Optional[SearchData]:
        """Get the search result with highest QPS at minimum recall threshold."""
        hnsw_results = [s for s in run.search_results if s.index_type.lower() == "hnsw"]
        valid_results = [s for s in hnsw_results if s.recall_at_10 and s.recall_at_10 >= min_recall]

        if valid_results:
            return max(valid_results, key=lambda s: s.qps)
        elif hnsw_results:
            # Return highest recall if none meet threshold
            return max(hnsw_results, key=lambda s: s.recall_at_10 or 0)
        return None

    def generate_findings(self, runs: List[RunData]) -> List[str]:
        """
        Generate evidence-based findings from results.

        Only generates findings that can be proven from the data.
        Uses hedging language and cites specific numbers.
        """
        findings = []

        if len(runs) < 2:
            return findings

        # Group by architecture
        embedded = [r for r in runs if r.metadata.get("architecture") == "embedded"]
        client_server = [r for r in runs if r.metadata.get("architecture") == "client-server"]

        # Find best performers for HNSW search
        def get_best_qps(run: RunData) -> float:
            best = self._get_best_search_result(run)
            return best.qps if best else 0

        def get_best_recall(run: RunData) -> float:
            best = self._get_best_search_result(run)
            return best.recall_at_10 if best and best.recall_at_10 else 0

        # Compare embedded vs client-server
        if embedded and client_server:
            best_embedded = max(embedded, key=get_best_qps)
            best_embedded_result = self._get_best_search_result(best_embedded)

            best_cs = max(client_server, key=get_best_qps)
            best_cs_result = self._get_best_search_result(best_cs)

            if best_embedded_result and best_cs_result and best_cs_result.qps > 0:
                ratio = best_embedded_result.qps / best_cs_result.qps
                findings.append(
                    f"**Embedded vs Client-Server**: {best_embedded.database} (embedded) achieved "
                    f"{best_embedded_result.qps:,.0f} QPS at {best_embedded_result.recall_at_10:.1%} recall, "
                    f"which is {ratio:.1f}x higher than {best_cs.database} (client-server) at "
                    f"{best_cs_result.qps:,.0f} QPS. This difference is consistent with the architectural "
                    f"overhead of network communication in client-server deployments."
                )

        # Find fastest and slowest ingest (HNSW)
        runs_with_hnsw_ingest = [
            (r, next((i for i in r.ingest_results if i.index_type.lower() == "hnsw"), None))
            for r in runs
        ]
        runs_with_hnsw_ingest = [(r, i) for r, i in runs_with_hnsw_ingest if i]

        if len(runs_with_hnsw_ingest) >= 2:
            fastest = max(runs_with_hnsw_ingest, key=lambda x: x[1].throughput_vps)
            slowest = min(runs_with_hnsw_ingest, key=lambda x: x[1].throughput_vps)

            if slowest[1].throughput_vps > 0:
                ratio = fastest[1].throughput_vps / slowest[1].throughput_vps
                findings.append(
                    f"**Ingest Performance**: {fastest[0].database} achieved the highest HNSW ingest "
                    f"throughput at {fastest[1].throughput_vps:,.0f} vectors/second, which is "
                    f"{ratio:.0f}x faster than {slowest[0].database} at {slowest[1].throughput_vps:,.0f} vectors/second."
                )

        # Check for anomalies - recall not improving with higher ef
        for run in runs:
            hnsw_results = sorted(
                [s for s in run.search_results if s.index_type.lower() == "hnsw" and s.ef_search],
                key=lambda s: s.ef_search
            )
            if len(hnsw_results) >= 3:
                recalls = [s.recall_at_10 for s in hnsw_results if s.recall_at_10]
                if recalls and len(set(recalls)) == 1:
                    findings.append(
                        f"**Anomaly**: {run.database} shows identical recall ({recalls[0]:.4f}) across all "
                        f"efSearch values tested. This may indicate the efSearch parameter is not being "
                        f"applied correctly. Manual investigation recommended."
                    )

        # Protocol comparison within client-server
        if len(client_server) >= 2:
            by_protocol = {}
            for r in client_server:
                protocol = r.metadata.get("protocol", "unknown")
                best = self._get_best_search_result(r)
                if best:
                    if protocol not in by_protocol or best.qps > by_protocol[protocol][1].qps:
                        by_protocol[protocol] = (r, best)

            if len(by_protocol) >= 2:
                sorted_protocols = sorted(by_protocol.items(), key=lambda x: x[1][1].qps, reverse=True)
                fastest_proto, (fastest_run, fastest_result) = sorted_protocols[0]
                slowest_proto, (slowest_run, slowest_result) = sorted_protocols[-1]

                if slowest_result.qps > 0:
                    ratio = fastest_result.qps / slowest_result.qps
                    if ratio > 1.5:  # Only report if difference is significant
                        findings.append(
                            f"**Protocol Efficiency**: Among client-server databases, {fastest_run.database} "
                            f"using {fastest_proto} achieved {fastest_result.qps:,.0f} QPS, "
                            f"{ratio:.1f}x higher than {slowest_run.database} using {slowest_proto} "
                            f"at {slowest_result.qps:,.0f} QPS."
                        )

        return findings

    def _load_benchmark_config(self) -> Dict[str, Any]:
        """Load shared benchmark configuration."""
        config_path = Path("benchmark.yaml")
        if config_path.exists():
            try:
                return load_yaml_config(str(config_path))
            except Exception:
                pass
        return {}

    def generate_markdown(self, runs: List[RunData]) -> str:
        """Generate markdown report."""
        lines = []
        bench_config = self._load_benchmark_config()

        # Header
        lines.append("# Vector Database Benchmark Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Configuration summary
        if runs:
            sample = runs[0]
            lines.append("## Benchmark Configuration")

            # Dataset description from config
            datasets = bench_config.get("datasets", {})
            ds_info = datasets.get(sample.dataset, {}) or datasets.get(sample.dataset.lower(), {})
            ds_desc = ds_info.get("description", "")
            lines.append(f"- **Dataset**: {sample.dataset} ({sample.vector_count:,} vectors, {sample.dimensions} dimensions)")
            if ds_desc:
                lines.append(f"  - {ds_desc}")
            lines.append(f"- **Search**: k=100, 10,000 queries (sequential, single-client)")
            if sample.hostname:
                lines.append(f"- **Host**: {sample.hostname}")
            lines.append("")

            # Index descriptions from config
            indexes = bench_config.get("indexes", {})
            if indexes:
                lines.append("### Index Types Tested")
                lines.append("")
                for idx_name, idx_config in indexes.items():
                    desc = idx_config.get("description", "") if isinstance(idx_config, dict) else ""
                    if desc and "not yet implemented" not in desc.lower():
                        lines.append(f"- **{idx_name.upper()}**: {desc}")
                lines.append("")

            # Metric glossary from config
            metrics = bench_config.get("metrics", {})
            if metrics:
                lines.append("### Metrics Glossary")
                lines.append("")
                lines.append("| Metric | Description |")
                lines.append("|--------|-------------|")
                for key, info in metrics.items():
                    if isinstance(info, dict):
                        name = info.get("name", key)
                        desc = info.get("description", "")
                        lines.append(f"| {name} | {desc} |")
                lines.append("")

        # Database Configuration Summary
        lines.append("## Database Configuration Summary")
        lines.append("")
        lines.append("| Database | Version | Architecture | Protocol | Persistence | License |")
        lines.append("|----------|---------|--------------|----------|-------------|---------|")
        for run in sorted(runs, key=lambda r: r.database):
            meta = run.metadata
            lines.append(
                f"| {run.database} | {run.db_version or 'N/A'} | "
                f"{meta.get('architecture', 'N/A')} | {meta.get('protocol', 'N/A')} | "
                f"{meta.get('persistence', 'N/A')} | {meta.get('license', 'N/A')} |"
            )
        lines.append("")

        # Database notes (if any)
        notes = [(run.database, run.metadata.get("notes")) for run in runs if run.metadata.get("notes")]
        if notes:
            lines.append("**Notes:**")
            for db_name, note in sorted(notes):
                lines.append(f"- **{db_name}**: {note}")
            lines.append("")

        # Ingest Performance Summary - sorted by HNSW throughput (descending)
        lines.append("## Ingest Performance Summary")
        lines.append("")
        lines.append("| Database | FLAT (vec/s) | FLAT Time | HNSW (vec/s) | HNSW Time | Peak Memory |")
        lines.append("|----------|--------------|-----------|--------------|-----------|-------------|")

        def get_hnsw_throughput(run):
            hnsw = next((i for i in run.ingest_results if i.index_type.lower() == "hnsw"), None)
            return hnsw.throughput_vps if hnsw else 0

        for run in sorted(runs, key=get_hnsw_throughput, reverse=True):
            flat = next((i for i in run.ingest_results if i.index_type.lower() == "flat"), None)
            hnsw = next((i for i in run.ingest_results if i.index_type.lower() == "hnsw"), None)

            flat_vps = f"{flat.throughput_vps:,.0f}" if flat else "-"
            flat_time = f"{flat.total_time_s:.1f}s" if flat else "-"
            hnsw_vps = f"{hnsw.throughput_vps:,.0f}" if hnsw else "-"
            hnsw_time = f"{hnsw.total_time_s:.1f}s" if hnsw else "-"
            peak_mem = f"{hnsw.peak_memory_gb:.2f} GB" if hnsw and hnsw.peak_memory_gb else "-"

            lines.append(f"| {run.database} | {flat_vps} | {flat_time} | {hnsw_vps} | {hnsw_time} | {peak_mem} |")
        lines.append("")

        # Search Performance Summary (Best Recall Config) - sorted by QPS (descending)
        lines.append("## Search Performance Summary")
        lines.append("*HNSW index at ~97%+ recall where available, sorted by QPS*")
        lines.append("")
        lines.append("| Database | Config | QPS | R@10 | R@100 | P50 (ms) | P99 (ms) |")
        lines.append("|----------|--------|-----|------|-------|----------|----------|")

        def get_best_qps(run):
            best = self._get_best_search_result(run)
            return best.qps if best else 0

        for run in sorted(runs, key=get_best_qps, reverse=True):
            best = self._get_best_search_result(run)
            if best:
                config = f"ef={best.ef_search}" if best.ef_search else "flat"
                r10 = f"{best.recall_at_10:.4f}" if best.recall_at_10 else "-"
                r100 = f"{best.recall_at_100:.4f}" if best.recall_at_100 else "-"
                p50 = f"{best.p50_ms:.2f}" if best.p50_ms else "-"
                p99 = f"{best.p99_ms:.2f}" if best.p99_ms else "-"
                lines.append(f"| {run.database} | {config} | {best.qps:,.0f} | {r10} | {r100} | {p50} | {p99} |")
            else:
                lines.append(f"| {run.database} | - | - | - | - | - | - |")
        lines.append("")

        # QPS vs Recall Tradeoff
        lines.append("## QPS vs Recall Tradeoff (HNSW)")
        lines.append("")

        # Collect all ef_search values
        all_ef_values = set()
        for run in runs:
            for s in run.search_results:
                if s.index_type.lower() == "hnsw" and s.ef_search:
                    all_ef_values.add(s.ef_search)
        all_ef_values = sorted(all_ef_values)

        if all_ef_values:
            header = "| Database |" + "".join(f" ef={ef} QPS | ef={ef} R@10 |" for ef in all_ef_values)
            lines.append(header)
            lines.append("|" + "---|" * (1 + len(all_ef_values) * 2))

            # Sort by best QPS (descending)
            for run in sorted(runs, key=get_best_qps, reverse=True):
                row = f"| {run.database} |"
                for ef in all_ef_values:
                    result = next(
                        (s for s in run.search_results if s.index_type.lower() == "hnsw" and s.ef_search == ef),
                        None
                    )
                    if result:
                        r10 = f"{result.recall_at_10:.3f}" if result.recall_at_10 else "-"
                        row += f" {result.qps:,.0f} | {r10} |"
                    else:
                        row += " - | - |"
                lines.append(row)
            lines.append("")

        # Key Findings
        findings = self.generate_findings(runs)
        if findings:
            lines.append("## Key Findings")
            lines.append("")
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. {finding}")
                lines.append("")

        # Detailed Results Per Database
        lines.append("## Detailed Results Per Database")
        lines.append("")

        for run in sorted(runs, key=lambda r: r.database):
            lines.append(f"### {run.database}")
            lines.append("")

            # Run metadata
            lines.append("**Run Information:**")
            lines.append(f"- Run ID: {run.run_id}")
            lines.append(f"- Timestamp: {run.timestamp}")
            if run.duration_seconds:
                lines.append(f"- Duration: {run.duration_seconds:.1f}s ({run.duration_seconds/60:.1f} min)")
            if run.cpus:
                lines.append(f"- CPU Limit: {run.cpus}")
            if run.memory_gb:
                lines.append(f"- Memory Limit: {run.memory_gb} GB")
            lines.append("")

            # Ingest results
            if run.ingest_results:
                lines.append("**Ingest Results:**")
                lines.append("| Index | Throughput | Time | Peak Memory | Final Memory |")
                lines.append("|-------|------------|------|-------------|--------------|")
                for ingest in run.ingest_results:
                    peak = f"{ingest.peak_memory_gb:.2f} GB" if ingest.peak_memory_gb else "-"
                    final = f"{ingest.final_memory_gb:.2f} GB" if ingest.final_memory_gb else "-"
                    lines.append(
                        f"| {ingest.index_type.upper()} | {ingest.throughput_vps:,.0f} vec/s | "
                        f"{ingest.total_time_s:.1f}s | {peak} | {final} |"
                    )
                lines.append("")

            # Search results
            if run.search_results:
                lines.append("**Search Results:**")
                lines.append("| Index | Config | QPS | R@10 | R@100 | P50 | P95 | P99 |")
                lines.append("|-------|--------|-----|------|-------|-----|-----|-----|")
                for search in run.search_results:
                    config = f"ef={search.ef_search}" if search.ef_search else "-"
                    r10 = f"{search.recall_at_10:.4f}" if search.recall_at_10 else "-"
                    r100 = f"{search.recall_at_100:.4f}" if search.recall_at_100 else "-"
                    p50 = f"{search.p50_ms:.2f}ms" if search.p50_ms else "-"
                    p95 = f"{search.p95_ms:.2f}ms" if search.p95_ms else "-"
                    p99 = f"{search.p99_ms:.2f}ms" if search.p99_ms else "-"
                    lines.append(
                        f"| {search.index_type.upper()} | {config} | {search.qps:,.0f} | "
                        f"{r10} | {r100} | {p50} | {p95} | {p99} |"
                    )
                lines.append("")

        # Appendix
        lines.append("## Appendix")
        lines.append("")
        lines.append("### Methodology")
        lines.append("- Each database was tested with the same dataset and query workload")
        lines.append("- HNSW indexes used M=16, efConstruction=64")
        lines.append("- Search tests used 10,000 queries with 100 warmup queries")
        lines.append("- Recall is calculated against brute-force ground truth")
        lines.append("")

        return "\n".join(lines)

    def generate_html(self, runs: List[RunData]) -> str:
        """Generate HTML report with embedded CSS."""
        markdown = self.generate_markdown(runs)

        # Convert markdown to HTML (simple conversion)
        html_content = self._markdown_to_html(markdown)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector Database Benchmark Report</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --bg-color: #ffffff;
            --text-color: #1f2937;
            --border-color: #e5e7eb;
            --code-bg: #f3f4f6;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg-color);
        }}

        h1 {{
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 0.5rem;
        }}

        h2 {{
            color: var(--text-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.3rem;
            margin-top: 2rem;
        }}

        h3 {{
            color: var(--text-color);
            margin-top: 1.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}

        th, td {{
            padding: 0.75rem;
            text-align: left;
            border: 1px solid var(--border-color);
        }}

        th {{
            background: var(--code-bg);
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background: #f9fafb;
        }}

        tr:hover {{
            background: #f3f4f6;
        }}

        ul, ol {{
            padding-left: 1.5rem;
        }}

        li {{
            margin: 0.5rem 0;
        }}

        code {{
            background: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }}

        strong {{
            color: var(--primary-color);
        }}

        em {{
            color: #6b7280;
        }}

        @media print {{
            body {{
                padding: 0;
                font-size: 12pt;
            }}

            h1 {{
                font-size: 18pt;
            }}

            h2 {{
                font-size: 14pt;
                page-break-after: avoid;
            }}

            table {{
                font-size: 10pt;
                page-break-inside: avoid;
            }}
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}

            table {{
                font-size: 0.8rem;
            }}

            th, td {{
                padding: 0.5rem;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        return html

    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        lines = markdown.split("\n")
        html_lines = []
        in_table = False
        in_list = False
        list_type = None

        for line in lines:
            # Headers
            if line.startswith("# "):
                html_lines.append(f"<h1>{self._escape_html(line[2:])}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{self._escape_html(line[3:])}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{self._escape_html(line[4:])}</h3>")

            # Tables
            elif line.startswith("|"):
                if not in_table:
                    html_lines.append("<table>")
                    in_table = True

                cells = [c.strip() for c in line.split("|")[1:-1]]

                if all(c.replace("-", "") == "" for c in cells):
                    continue  # Skip separator row

                is_header = not any("<tr>" in l for l in html_lines[-5:] if "<tr>" in l)
                tag = "th" if is_header else "td"

                row = "<tr>" + "".join(f"<{tag}>{self._format_inline(c)}</{tag}>" for c in cells) + "</tr>"
                html_lines.append(row)

            # List items
            elif line.startswith("- "):
                if not in_list or list_type != "ul":
                    if in_list:
                        html_lines.append(f"</{list_type}>")
                    html_lines.append("<ul>")
                    in_list = True
                    list_type = "ul"
                html_lines.append(f"<li>{self._format_inline(line[2:])}</li>")

            elif line.strip() and line[0].isdigit() and ". " in line:
                if not in_list or list_type != "ol":
                    if in_list:
                        html_lines.append(f"</{list_type}>")
                    html_lines.append("<ol>")
                    in_list = True
                    list_type = "ol"
                content = line.split(". ", 1)[1] if ". " in line else line
                html_lines.append(f"<li>{self._format_inline(content)}</li>")

            # Empty line
            elif line.strip() == "":
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                    list_type = None

            # Paragraph
            else:
                if line.startswith("*") and line.endswith("*"):
                    html_lines.append(f"<p><em>{self._format_inline(line[1:-1])}</em></p>")
                else:
                    html_lines.append(f"<p>{self._format_inline(line)}</p>")

        # Close any open tags
        if in_table:
            html_lines.append("</table>")
        if in_list:
            html_lines.append(f"</{list_type}>")

        return "\n".join(html_lines)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _format_inline(self, text: str) -> str:
        """Format inline markdown (bold, italic, code)."""
        import re

        text = self._escape_html(text)

        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

        # Italic
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

        # Code
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)

        return text

    def generate_report(
        self,
        format: str = "markdown",
        databases: Optional[List[str]] = None,
        run_ids: Optional[List[int]] = None,
    ) -> str:
        """
        Generate a benchmark report.

        Args:
            format: Output format ('markdown' or 'html')
            databases: Optional list of database names to include
            run_ids: Optional list of specific run IDs to include

        Returns:
            Report content as string
        """
        if run_ids:
            runs = self.get_runs_by_ids(run_ids)
        else:
            runs = self.get_latest_runs(databases)

        if not runs:
            return "No benchmark results found."

        if format == "html":
            return self.generate_html(runs)
        else:
            return self.generate_markdown(runs)
