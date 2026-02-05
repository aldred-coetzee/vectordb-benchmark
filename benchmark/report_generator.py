"""Report generator for benchmark results."""

import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    def get_latest_runs(
        self,
        databases: Optional[List[str]] = None,
        dataset: Optional[str] = None,
    ) -> List[RunData]:
        """
        Get the most recent run for each (database, dataset) pair.

        Args:
            databases: Optional list of database names to filter
            dataset: Optional dataset name to filter (case-insensitive)

        Returns:
            List of RunData objects
        """
        conn = self.db._get_connection()
        cursor = conn.cursor()

        # Get latest run_id for each (database, dataset) pair
        query = """
            SELECT r.* FROM runs r
            INNER JOIN (
                SELECT database, dataset, MAX(run_id) as max_id
                FROM runs
                GROUP BY database, dataset
            ) latest ON r.run_id = latest.max_id
        """

        conditions = []
        params: list = []
        if databases:
            placeholders = ",".join("?" * len(databases))
            conditions.append(f"r.database IN ({placeholders})")
            params.extend(databases)
        if dataset:
            conditions.append("UPPER(r.dataset) = UPPER(?)")
            params.append(dataset)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)

        runs = []
        for row in cursor.fetchall():
            run_data = self._build_run_data(dict(row))
            runs.append(run_data)

        return runs

    def get_datasets(self) -> List[str]:
        """Get distinct dataset names from the database."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT dataset FROM runs ORDER BY dataset")
        return [row[0] for row in cursor.fetchall()]

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

    def _get_best_search_result(self, run: RunData) -> Optional[SearchData]:
        """Get the HNSW search result with highest QPS (lowest efSearch)."""
        hnsw_results = [s for s in run.search_results if s.index_type.lower() == "hnsw"]
        if hnsw_results:
            return max(hnsw_results, key=lambda s: s.qps)
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

            # Datasets section — show metadata for datasets present in results
            result_datasets = set(r.dataset for r in runs)
            ds_entries = []
            for ds_key, ds_meta in datasets.items():
                if not isinstance(ds_meta, dict):
                    continue
                # Match by key or by uppercase name
                if ds_key in result_datasets or ds_key.upper() in result_datasets:
                    ds_entries.append(ds_meta)
            # Also try case-insensitive match
            if not ds_entries:
                for ds_key, ds_meta in datasets.items():
                    if not isinstance(ds_meta, dict):
                        continue
                    if ds_key.lower() in {d.lower() for d in result_datasets}:
                        ds_entries.append(ds_meta)

            if ds_entries:
                lines.append("### Datasets")
                lines.append("")
                lines.append("| Dataset | Vectors | Dims | Metric | Purpose |")
                lines.append("|---------|---------|------|--------|---------|")
                for ds_meta in ds_entries:
                    ds_name = ds_meta.get("name", "")
                    ds_vectors = f"{ds_meta.get('vectors', 0):,}" if ds_meta.get("vectors") else "-"
                    ds_dims = ds_meta.get("dimensions", "-")
                    ds_metric = ds_meta.get("metric", "L2")
                    # Truncate purpose to first sentence for table
                    purpose = ds_meta.get("purpose", "")
                    purpose_short = purpose.split(".")[0] + "." if purpose else "-"
                    lines.append(f"| {ds_name} | {ds_vectors} | {ds_dims} | {ds_metric} | {purpose_short} |")
                lines.append("")

                # Full descriptions
                for ds_meta in ds_entries:
                    ds_name = ds_meta.get("name", "")
                    ds_desc_full = ds_meta.get("description", "")
                    ds_purpose = ds_meta.get("purpose", "")
                    if ds_desc_full:
                        lines.append(f"**{ds_name}**: {ds_desc_full}")
                        if ds_purpose:
                            lines.append(f"*{ds_purpose}*")
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
        lines.append("*HNSW index at highest-QPS config (lowest efSearch), sorted by QPS. Recall shown for comparison.*")
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

        # Batch Search Performance
        has_batch = any(
            s.index_type.upper().endswith("_BATCH")
            for run in runs
            for s in run.search_results
        )
        if has_batch:
            lines.append("## Batch Search Performance")
            lines.append("*All queries sent in a single API call. Per-query latency percentiles are not available — the database processes queries together, potentially in parallel.*")
            lines.append("")

            # Collect all batch ef_search values
            batch_ef_values = set()
            for run in runs:
                for s in run.search_results:
                    if s.index_type.upper() == "HNSW_BATCH" and s.ef_search:
                        batch_ef_values.add(s.ef_search)
            batch_ef_values = sorted(batch_ef_values)

            if batch_ef_values:
                header = "| Database |" + "".join(f" ef={ef} QPS | ef={ef} R@10 |" for ef in batch_ef_values)
                lines.append(header)
                lines.append("|" + "---|" * (1 + len(batch_ef_values) * 2))

                # Only include runs that have batch results
                batch_runs = [r for r in runs if any(s.index_type.upper().endswith("_BATCH") for s in r.search_results)]
                for run in sorted(batch_runs, key=lambda r: max((s.qps for s in r.search_results if s.index_type.upper() == "HNSW_BATCH"), default=0), reverse=True):
                    row = f"| {run.database} |"
                    for ef in batch_ef_values:
                        result = next(
                            (s for s in run.search_results if s.index_type.upper() == "HNSW_BATCH" and s.ef_search == ef),
                            None
                        )
                        if result:
                            r10 = f"{result.recall_at_10:.3f}" if result.recall_at_10 else "-"
                            row += f" {result.qps:,.0f} | {r10} |"
                        else:
                            row += " - | - |"
                    lines.append(row)
                lines.append("")

            # Also show flat batch if present
            flat_batch_runs = [
                (run, next((s for s in run.search_results if s.index_type.upper() == "FLAT_BATCH"), None))
                for run in runs
            ]
            flat_batch_runs = [(r, s) for r, s in flat_batch_runs if s]
            if flat_batch_runs:
                lines.append("### Flat Index (Batch)")
                lines.append("")
                lines.append("| Database | QPS | R@10 | R@100 |")
                lines.append("|----------|-----|------|-------|")
                for run, s in sorted(flat_batch_runs, key=lambda x: x[1].qps, reverse=True):
                    r10 = f"{s.recall_at_10:.4f}" if s.recall_at_10 else "-"
                    r100 = f"{s.recall_at_100:.4f}" if s.recall_at_100 else "-"
                    lines.append(f"| {run.database} | {s.qps:,.0f} | {r10} | {r100} |")
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

            # Search results (sequential)
            sequential_results = [s for s in run.search_results if not s.index_type.upper().endswith("_BATCH")]
            if sequential_results:
                lines.append("**Search Results (Sequential):**")
                lines.append("| Index | Config | QPS | R@10 | R@100 | P50 | P95 | P99 |")
                lines.append("|-------|--------|-----|------|-------|-----|-----|-----|")
                for search in sequential_results:
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

            # Search results (batch)
            batch_results = [s for s in run.search_results if s.index_type.upper().endswith("_BATCH")]
            if batch_results:
                lines.append("**Search Results (Batch):**")
                lines.append("*All queries in single API call — no per-query latency available*")
                lines.append("")
                lines.append("| Index | Config | QPS | R@10 | R@100 |")
                lines.append("|-------|--------|-----|------|-------|")
                for search in batch_results:
                    config = f"ef={search.ef_search}" if search.ef_search else "-"
                    r10 = f"{search.recall_at_10:.4f}" if search.recall_at_10 else "-"
                    r100 = f"{search.recall_at_100:.4f}" if search.recall_at_100 else "-"
                    lines.append(
                        f"| {search.index_type.upper()} | {config} | {search.qps:,.0f} | "
                        f"{r10} | {r100} |"
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
        dataset: Optional[str] = None,
    ) -> str:
        """
        Generate a benchmark report.

        Args:
            format: Output format ('markdown' or 'html')
            databases: Optional list of database names to include
            run_ids: Optional list of specific run IDs to include
            dataset: Optional dataset name to filter (case-insensitive)

        Returns:
            Report content as string
        """
        if run_ids:
            runs = self.get_runs_by_ids(run_ids)
            if dataset:
                runs = [r for r in runs if r.dataset.upper() == dataset.upper()]
        else:
            runs = self.get_latest_runs(databases, dataset=dataset)

        if not runs:
            return "No benchmark results found."

        if format == "html":
            return self.generate_html(runs)
        else:
            return self.generate_markdown(runs)


# =============================================================================
# Color palette and chart utilities
# =============================================================================

# Accessible color palette — distinct hues for up to 9 databases
DB_COLORS = {
    "FAISS": "#636363",       # gray (embedded baseline)
    "Qdrant": "#e41a1c",      # red
    "Milvus": "#377eb8",      # blue
    "Chroma": "#4daf4a",      # green
    "Weaviate": "#984ea3",    # purple
    "Redis": "#ff7f00",       # orange
    "pgvector": "#a65628",    # brown
    "KDB.AI": "#f781bf",      # pink
    "LanceDB": "#999999",     # light gray (embedded)
}

DB_MARKERS = {
    "FAISS": "s",       # square
    "Qdrant": "o",
    "Milvus": "D",      # diamond
    "Chroma": "^",      # triangle up
    "Weaviate": "v",    # triangle down
    "Redis": "P",       # plus (filled)
    "pgvector": "X",    # x (filled)
    "KDB.AI": "*",      # star
    "LanceDB": "h",     # hexagon
}


def _get_color(db_name: str) -> str:
    """Get color for a database name."""
    return DB_COLORS.get(db_name, "#333333")


def _get_marker(db_name: str) -> str:
    """Get marker for a database name."""
    return DB_MARKERS.get(db_name, "o")


def _fig_to_svg(fig) -> str:
    """Convert a matplotlib figure to inline SVG string."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    svg = buf.getvalue()
    # Strip XML declaration and DOCTYPE for inline embedding
    lines = svg.split("\n")
    filtered = [l for l in lines if not l.startswith("<?xml") and not l.startswith("<!DOCTYPE")]
    return "\n".join(filtered)


class ComparisonReportGenerator:
    """Generate combined multi-dataset benchmark report with charts."""

    # Reference ef for executive summary (common operating point)
    REFERENCE_EF = 128

    # Databases excluded from ranking (not comparable to client-server disk-persisted)
    EXCLUDED_DBS = {"FAISS", "Redis"}

    def __init__(self, db_path: str = "results/benchmark.db", configs_dir: str = "configs"):
        self._rg = ReportGenerator(db_path=db_path, configs_dir=configs_dir)

    def close(self):
        self._rg.close()

    def generate(
        self,
        databases: Optional[List[str]] = None,
        run_ids: Optional[List[int]] = None,
    ) -> str:
        """Generate a combined HTML report across all datasets."""
        if run_ids:
            all_runs = self._rg.get_runs_by_ids(run_ids)
        else:
            all_runs = self._rg.get_latest_runs(databases)

        if not all_runs:
            return "<html><body><p>No benchmark results found.</p></body></html>"

        # Group runs by dataset
        by_dataset: Dict[str, List[RunData]] = {}
        for run in all_runs:
            ds = run.dataset.upper()
            by_dataset.setdefault(ds, []).append(run)

        datasets = sorted(by_dataset.keys())
        bench_config = self._rg._load_benchmark_config()
        datasets_meta = bench_config.get("datasets", {})

        # Build report sections
        sections: List[str] = []
        sections.append(self._render_header(datasets, all_runs))
        sections.append(self._render_executive_summary(by_dataset, datasets))

        # Charts
        chart_svgs = self._render_charts(by_dataset, datasets)
        sections.append(chart_svgs)

        # Key findings (cross-dataset)
        sections.append(self._render_findings(by_dataset, datasets))

        # Per-dataset details (collapsible)
        sections.append(self._render_per_dataset_details(by_dataset, datasets))

        # Configuration and methodology
        sections.append(self._render_config_section(all_runs, bench_config, datasets_meta))
        sections.append(self._render_methodology())

        body = "\n".join(sections)
        return self._wrap_html(body)

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------
    def _render_header(self, datasets: List[str], runs: List[RunData]) -> str:
        sample = runs[0]
        db_names = sorted(set(r.database for r in runs))
        ds_display = ", ".join(datasets)
        return f"""
        <h1>Vector Database Benchmark Report</h1>
        <p class="meta">
            Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &mdash;
            {len(db_names)} databases &times; {len(datasets)} datasets
            ({ds_display})
        </p>"""

    # -----------------------------------------------------------------
    # Executive Summary
    # -----------------------------------------------------------------
    def _render_executive_summary(
        self,
        by_dataset: Dict[str, List[RunData]],
        datasets: List[str],
    ) -> str:
        lines: List[str] = []
        lines.append("<h2>Executive Summary</h2>")
        lines.append(f'<p class="note">All search metrics at HNSW efSearch={self.REFERENCE_EF}. '
                      f'Sorted by average QPS across datasets (descending). '
                      f'<span class="rank-top-inline">Green</span> = top 2, '
                      f'<span class="rank-bottom-inline">red</span> = bottom 2. '
                      f'<span class="row-excluded-inline">Yellow rows</span> '
                      f'(FAISS, Redis) excluded from ranking.</p>')

        # Caveats callout
        lines.append('<div class="caveat-box">'
                     '<strong>Comparability notes:</strong>'
                     '<ul>'
                     '<li><strong>FAISS</strong> runs embedded (in-process) with zero network overhead '
                     '&mdash; not directly comparable to client-server databases. Shown as a baseline.</li>'
                     '<li><strong>Redis</strong> is purely in-memory (no disk persistence during operation) '
                     '&mdash; other client-server databases persist to disk. '
                     'Redis numbers reflect RAM-only performance.</li>'
                     '</ul></div>')

        # Collect all unique databases
        all_dbs = sorted(set(r.database for ds_runs in by_dataset.values() for r in ds_runs))

        # Build header row with sort indicator
        header = "<tr><th>Database</th><th>Notes</th>"
        for ds in datasets:
            header += f"<th>{ds} QPS</th><th>{ds} R@10</th>"
        header += "<th>Avg QPS ▼</th>"
        # Add ingest columns
        for ds in datasets:
            header += f"<th>{ds} Ingest (v/s)</th>"
        header += "</tr>"

        # Collect data per database
        db_rows: List[Tuple[str, str, Dict[str, Optional[SearchData]], Dict[str, Optional[IngestData]], float]] = []

        for db_name in all_dbs:
            # Find architecture from metadata
            sample_run = next((r for ds_runs in by_dataset.values() for r in ds_runs if r.database == db_name), None)
            arch = sample_run.metadata.get("architecture", "unknown") if sample_run else "unknown"

            search_by_ds: Dict[str, Optional[SearchData]] = {}
            ingest_by_ds: Dict[str, Optional[IngestData]] = {}
            qps_values = []

            for ds in datasets:
                ds_runs = by_dataset.get(ds, [])
                run = next((r for r in ds_runs if r.database == db_name), None)
                if run:
                    # Find HNSW result at reference ef
                    result = next(
                        (s for s in run.search_results
                         if s.index_type.lower() == "hnsw" and s.ef_search == self.REFERENCE_EF),
                        None
                    )
                    search_by_ds[ds] = result
                    if result:
                        qps_values.append(result.qps)

                    hnsw_ingest = next(
                        (i for i in run.ingest_results if i.index_type.lower() == "hnsw"), None
                    )
                    ingest_by_ds[ds] = hnsw_ingest
                else:
                    search_by_ds[ds] = None
                    ingest_by_ds[ds] = None

            avg_qps = sum(qps_values) / len(qps_values) if qps_values else 0
            db_rows.append((db_name, arch, search_by_ds, ingest_by_ds, avg_qps))

        # Sort by avg QPS descending
        db_rows.sort(key=lambda x: x[4], reverse=True)

        # Compute ranks per dataset for QPS (exclude FAISS/Redis)
        cs_qps_ranks: Dict[str, Dict[str, int]] = {}  # ds -> {db_name: rank}
        for ds in datasets:
            ranked_entries = [
                (row[0], row[2].get(ds))
                for row in db_rows
                if row[0] not in self.EXCLUDED_DBS and row[2].get(ds)
            ]
            ranked_entries.sort(key=lambda x: x[1].qps if x[1] else 0, reverse=True)
            cs_qps_ranks[ds] = {name: i + 1 for i, (name, _) in enumerate(ranked_entries)}

        # Build rows
        rows_html: List[str] = []
        for db_name, arch, search_by_ds, ingest_by_ds, avg_qps in db_rows:
            # Build annotation notes
            notes_parts: List[str] = []
            sample_run = next((r for ds_runs in by_dataset.values() for r in ds_runs if r.database == db_name), None)
            persistence = sample_run.metadata.get("persistence", "") if sample_run else ""
            if arch == "embedded":
                notes_parts.append("embedded")
            if persistence == "memory" and arch != "embedded":
                notes_parts.append("in-memory")
            notes_label = ", ".join(notes_parts)
            excluded = db_name in self.EXCLUDED_DBS
            row_class = ' class="row-excluded"' if excluded else ""
            row = f"<tr{row_class}><td><strong>{db_name}</strong></td><td>{notes_label}</td>"

            for ds in datasets:
                s = search_by_ds.get(ds)
                rank = cs_qps_ranks.get(ds, {}).get(db_name, 0)
                n_cs = len(cs_qps_ranks.get(ds, {}))

                if s:
                    rank_class = ""
                    if not excluded and n_cs >= 3:
                        if rank <= 2:
                            rank_class = " class=\"rank-top\""
                        elif rank >= n_cs - 1:
                            rank_class = " class=\"rank-bottom\""
                    row += f"<td{rank_class}>{s.qps:,.0f}</td>"
                    row += f"<td>{s.recall_at_10:.3f}</td>" if s.recall_at_10 else "<td>-</td>"
                else:
                    row += "<td>-</td><td>-</td>"

            row += f"<td><strong>{avg_qps:,.0f}</strong></td>"

            for ds in datasets:
                ig = ingest_by_ds.get(ds)
                row += f"<td>{ig.throughput_vps:,.0f}</td>" if ig else "<td>-</td>"

            row += "</tr>"
            rows_html.append(row)

        table = f"""<div class="table-wrap"><table>
        <thead>{header}</thead>
        <tbody>{"".join(rows_html)}</tbody>
        </table></div>"""

        lines.append(table)
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------
    def _render_charts(
        self,
        by_dataset: Dict[str, List[RunData]],
        datasets: List[str],
    ) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return "<p><em>Charts unavailable (matplotlib not installed).</em></p>"

        lines: List[str] = ["<h2>Performance Charts</h2>", '<div class="chart-grid">']

        # Recall vs QPS scatter (one per dataset)
        for ds in datasets:
            svg = self._chart_recall_vs_qps(by_dataset.get(ds, []), ds, plt)
            lines.append(f'<div class="chart-cell">{svg}</div>')

        lines.append("</div>")  # close chart-grid

        # Ingest throughput bar chart
        svg = self._chart_ingest_throughput(by_dataset, datasets, plt)
        lines.append(f'<div class="chart-full">{svg}</div>')

        # Search QPS bar chart at reference ef
        lines.append('<div class="chart-grid">')
        for ds in datasets:
            svg = self._chart_search_qps_bars(by_dataset.get(ds, []), ds, plt)
            lines.append(f'<div class="chart-cell">{svg}</div>')
        lines.append("</div>")

        return "\n".join(lines)

    def _chart_recall_vs_qps(self, runs: List[RunData], dataset: str, plt) -> str:
        """Scatter plot: Recall@10 vs QPS for all databases."""
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for run in sorted(runs, key=lambda r: r.database):
            hnsw = sorted(
                [s for s in run.search_results if s.index_type.lower() == "hnsw" and s.ef_search],
                key=lambda s: s.ef_search,
            )
            if not hnsw:
                continue

            recalls = [s.recall_at_10 for s in hnsw if s.recall_at_10]
            qps_vals = [s.qps for s in hnsw if s.recall_at_10]
            if not recalls:
                continue

            color = _get_color(run.database)
            marker = _get_marker(run.database)

            ax.plot(recalls, qps_vals, marker=marker, color=color, label=run.database,
                    markersize=8, linewidth=1.5, linestyle="-", zorder=3)

            # FLAT as separate point (triangle marker)
            flat = next(
                (s for s in run.search_results if s.index_type.lower() == "flat" and s.recall_at_10),
                None,
            )
            if flat and flat.recall_at_10:
                ax.scatter([flat.recall_at_10], [flat.qps], marker="x", color=color,
                          s=60, zorder=4, alpha=0.6)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"Recall vs QPS — {dataset}")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0.65, right=1.01)

        fig.tight_layout()
        return _fig_to_svg(fig)

    def _chart_ingest_throughput(
        self,
        by_dataset: Dict[str, List[RunData]],
        datasets: List[str],
        plt,
    ) -> str:
        """Horizontal grouped bar chart: ingest throughput (HNSW) per database."""
        import numpy as np

        # Collect (db_name, {ds: throughput})
        all_dbs = sorted(set(r.database for ds_runs in by_dataset.values() for r in ds_runs))

        data: Dict[str, Dict[str, float]] = {}
        for db_name in all_dbs:
            data[db_name] = {}
            for ds in datasets:
                run = next((r for r in by_dataset.get(ds, []) if r.database == db_name), None)
                if run:
                    hnsw = next((i for i in run.ingest_results if i.index_type.lower() == "hnsw"), None)
                    data[db_name][ds] = hnsw.throughput_vps if hnsw else 0
                else:
                    data[db_name][ds] = 0

        # Sort by max throughput across datasets (descending)
        sorted_dbs = sorted(all_dbs, key=lambda db: max(data[db].values()), reverse=True)

        fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_dbs) * 0.5 + 1)))

        y = np.arange(len(sorted_dbs))
        bar_height = 0.8 / len(datasets) if datasets else 0.8
        ds_colors = ["#2563eb", "#e41a1c", "#4daf4a", "#ff7f00"]  # per-dataset colors

        for i, ds in enumerate(datasets):
            offsets = y - 0.4 + bar_height * (i + 0.5)
            values = [data[db].get(ds, 0) for db in sorted_dbs]
            ax.barh(offsets, values, height=bar_height * 0.9,
                    label=ds, color=ds_colors[i % len(ds_colors)], alpha=0.85)
            # Add value labels
            for j, v in enumerate(values):
                if v > 0:
                    ax.text(v + max(values) * 0.01, offsets[j], f"{v:,.0f}",
                            va="center", fontsize=7, color="#333")

        ax.set_yticks(y)
        ax.set_yticklabels(sorted_dbs)
        ax.set_xlabel("Vectors/sec (HNSW ingest)")
        ax.set_title("Ingest Throughput by Database")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

        fig.tight_layout()
        return _fig_to_svg(fig)

    def _chart_search_qps_bars(self, runs: List[RunData], dataset: str, plt) -> str:
        """Horizontal bar chart: search QPS at reference efSearch."""
        import numpy as np

        db_qps: List[Tuple[str, float, Optional[float]]] = []
        for run in runs:
            result = next(
                (s for s in run.search_results
                 if s.index_type.lower() == "hnsw" and s.ef_search == self.REFERENCE_EF),
                None,
            )
            if result:
                db_qps.append((run.database, result.qps, result.recall_at_10))

        if not db_qps:
            return ""

        db_qps.sort(key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(6, max(2.5, len(db_qps) * 0.4 + 1)))

        names = [x[0] for x in db_qps]
        values = [x[1] for x in db_qps]
        recalls = [x[2] for x in db_qps]
        colors = [_get_color(n) for n in names]

        y = np.arange(len(names))
        ax.barh(y, values, color=colors, alpha=0.85)

        # Annotate with recall
        for i, (v, r) in enumerate(zip(values, recalls)):
            label = f"{v:,.0f}"
            if r is not None:
                label += f"  (R@10={r:.3f})"
            ax.text(v + max(values) * 0.01, i, label, va="center", fontsize=7)

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("QPS")
        ax.set_title(f"Search QPS at ef={self.REFERENCE_EF} — {dataset}")
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

        fig.tight_layout()
        return _fig_to_svg(fig)

    # -----------------------------------------------------------------
    # Key Findings (cross-dataset)
    # -----------------------------------------------------------------
    def _render_findings(
        self,
        by_dataset: Dict[str, List[RunData]],
        datasets: List[str],
    ) -> str:
        findings: List[str] = []

        # 1. Best client-server ingest per dataset
        for ds in datasets:
            runs = by_dataset.get(ds, [])
            cs_ingest = []
            for r in runs:
                if r.metadata.get("architecture") != "client-server":
                    continue
                hnsw = next((i for i in r.ingest_results if i.index_type.lower() == "hnsw"), None)
                if hnsw:
                    cs_ingest.append((r.database, hnsw.throughput_vps))

            if cs_ingest:
                best = max(cs_ingest, key=lambda x: x[1])
                findings.append(
                    f"<strong>{ds} Ingest</strong>: {best[0]} achieved the highest client-server "
                    f"HNSW ingest at {best[1]:,.0f} vectors/sec."
                )

        # 2. Embedded vs client-server gap
        for ds in datasets:
            runs = by_dataset.get(ds, [])
            embedded = [(r.database, s.qps) for r in runs for s in r.search_results
                        if r.metadata.get("architecture") == "embedded"
                        and s.index_type.lower() == "hnsw" and s.ef_search == self.REFERENCE_EF]
            cs = [(r.database, s.qps) for r in runs for s in r.search_results
                  if r.metadata.get("architecture") == "client-server"
                  and s.index_type.lower() == "hnsw" and s.ef_search == self.REFERENCE_EF]

            if embedded and cs:
                best_emb = max(embedded, key=lambda x: x[1])
                best_cs = max(cs, key=lambda x: x[1])
                if best_cs[1] > 0:
                    ratio = best_emb[1] / best_cs[1]
                    if ratio > 1.5:
                        findings.append(
                            f"<strong>{ds} Embedded vs Client-Server</strong>: "
                            f"{best_emb[0]} (embedded) at {best_emb[1]:,.0f} QPS is "
                            f"{ratio:.1f}x faster than the best client-server ({best_cs[0]} at "
                            f"{best_cs[1]:,.0f} QPS)."
                        )

        # 3. Batch search speedup
        for ds in datasets:
            runs = by_dataset.get(ds, [])
            speedups = []
            for r in runs:
                seq = next(
                    (s for s in r.search_results
                     if s.index_type.lower() == "hnsw" and s.ef_search == self.REFERENCE_EF),
                    None,
                )
                batch = next(
                    (s for s in r.search_results
                     if s.index_type.upper() == "HNSW_BATCH" and s.ef_search == self.REFERENCE_EF),
                    None,
                )
                if seq and batch and seq.qps > 0:
                    speedups.append((r.database, batch.qps / seq.qps))

            if speedups:
                best = max(speedups, key=lambda x: x[1])
                if best[1] > 2:
                    findings.append(
                        f"<strong>{ds} Batch Speedup</strong>: {best[0]} achieves "
                        f"{best[1]:.1f}x throughput improvement with batch search."
                    )

        # 4. Cross-dataset consistency
        if len(datasets) >= 2:
            db_ranks: Dict[str, List[int]] = {}
            for ds in datasets:
                runs = by_dataset.get(ds, [])
                cs_results = [
                    (r.database, next(
                        (s.qps for s in r.search_results
                         if s.index_type.lower() == "hnsw" and s.ef_search == self.REFERENCE_EF),
                        0,
                    ))
                    for r in runs if r.metadata.get("architecture") == "client-server"
                ]
                cs_results.sort(key=lambda x: x[1], reverse=True)
                for rank, (name, _) in enumerate(cs_results, 1):
                    db_ranks.setdefault(name, []).append(rank)

            # Find databases with consistent ranking
            for name, ranks in db_ranks.items():
                if len(ranks) >= 2 and max(ranks) - min(ranks) <= 1:
                    findings.append(
                        f"<strong>Consistency</strong>: {name} ranks #{min(ranks)} across "
                        f"all datasets tested — consistent performance across different "
                        f"vector dimensions and distributions."
                    )

        if not findings:
            return ""

        items = "".join(f"<li>{f}</li>" for f in findings)
        return f"""
        <h2>Key Findings</h2>
        <ol class="findings">{items}</ol>"""

    # -----------------------------------------------------------------
    # Per-dataset details (collapsible)
    # -----------------------------------------------------------------
    def _render_per_dataset_details(
        self,
        by_dataset: Dict[str, List[RunData]],
        datasets: List[str],
    ) -> str:
        lines: List[str] = ["<h2>Per-Dataset Details</h2>"]

        for i, ds in enumerate(datasets):
            runs = by_dataset.get(ds, [])
            if not runs:
                continue

            open_attr = " open" if i == 0 else ""
            lines.append(f'<details{open_attr}>')
            lines.append(f'<summary><strong>{ds}</strong> — {runs[0].vector_count:,} vectors, '
                         f'{runs[0].dimensions}D</summary>')

            # Search performance table (sequential)
            lines.append("<h3>Search Performance (Sequential)</h3>")
            lines.append(self._search_table(runs, batch=False))

            # QPS vs Recall tradeoff
            lines.append("<h3>QPS vs Recall Tradeoff</h3>")
            lines.append(self._qps_recall_table(runs))

            # Batch search (if available)
            has_batch = any(
                s.index_type.upper().endswith("_BATCH")
                for r in runs for s in r.search_results
            )
            if has_batch:
                lines.append("<h3>Batch Search</h3>")
                lines.append('<p class="note">All queries sent in a single API call. '
                             'Per-query latency unavailable.</p>')
                lines.append(self._search_table(runs, batch=True))

            lines.append("</details>")

        return "\n".join(lines)

    def _search_table(self, runs: List[RunData], batch: bool = False) -> str:
        """Render search results table for a single dataset, sorted by QPS descending."""
        rows_data: List[Tuple[str, str, str, float, str, str, str, str, str]] = []

        if batch:
            header = "<tr><th>Database</th><th>Index</th><th>Config</th><th>QPS ▼</th><th>R@10</th><th>R@100</th></tr>"
        else:
            header = "<tr><th>Database</th><th>Index</th><th>Config</th><th>QPS ▼</th><th>R@10</th><th>R@100</th><th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th></tr>"

        for run in runs:
            for s in run.search_results:
                if batch:
                    if not s.index_type.upper().endswith("_BATCH"):
                        continue
                else:
                    if s.index_type.upper().endswith("_BATCH"):
                        continue

                config = f"ef={s.ef_search}" if s.ef_search else "-"
                r10 = f"{s.recall_at_10:.4f}" if s.recall_at_10 else "-"
                r100 = f"{s.recall_at_100:.4f}" if s.recall_at_100 else "-"
                idx_display = s.index_type.upper().replace("_BATCH", "")
                p50 = f"{s.p50_ms:.2f}" if s.p50_ms else "-"
                p95 = f"{s.p95_ms:.2f}" if s.p95_ms else "-"
                p99 = f"{s.p99_ms:.2f}" if s.p99_ms else "-"

                rows_data.append((run.database, idx_display, config, s.qps, r10, r100, p50, p95, p99))

        # Sort by QPS descending
        rows_data.sort(key=lambda x: x[3], reverse=True)

        rows_html: List[str] = []
        for db, idx, config, qps, r10, r100, p50, p95, p99 in rows_data:
            row_class = ' class="row-excluded"' if db in self.EXCLUDED_DBS else ""
            if batch:
                rows_html.append(
                    f"<tr{row_class}><td>{db}</td><td>{idx}</td>"
                    f"<td>{config}</td><td>{qps:,.0f}</td>"
                    f"<td>{r10}</td><td>{r100}</td></tr>"
                )
            else:
                rows_html.append(
                    f"<tr{row_class}><td>{db}</td><td>{idx}</td>"
                    f"<td>{config}</td><td>{qps:,.0f}</td>"
                    f"<td>{r10}</td><td>{r100}</td>"
                    f"<td>{p50}</td><td>{p95}</td><td>{p99}</td></tr>"
                )

        return f'<div class="table-wrap"><table><thead>{header}</thead><tbody>{"".join(rows_html)}</tbody></table></div>'

    def _qps_recall_table(self, runs: List[RunData]) -> str:
        """Render QPS vs Recall tradeoff table (HNSW only), sorted by best QPS descending."""
        # Collect all ef values
        all_ef: set = set()
        for r in runs:
            for s in r.search_results:
                if s.index_type.lower() == "hnsw" and s.ef_search:
                    all_ef.add(s.ef_search)
        ef_values = sorted(all_ef)

        if not ef_values:
            return ""

        header = "<tr><th>Database</th>"
        for ef in ef_values:
            header += f"<th>ef={ef} QPS</th><th>ef={ef} R@10</th>"
        header += "</tr>"
        sort_note = '<p class="note">Sorted by best QPS across ef values (descending).</p>'

        # Sort runs by best HNSW QPS descending
        def _best_hnsw_qps(run: RunData) -> float:
            return max((s.qps for s in run.search_results if s.index_type.lower() == "hnsw"), default=0)

        rows: List[str] = []
        for run in sorted(runs, key=_best_hnsw_qps, reverse=True):
            row_class = ' class="row-excluded"' if run.database in self.EXCLUDED_DBS else ""
            row = f"<tr{row_class}><td>{run.database}</td>"
            for ef in ef_values:
                result = next(
                    (s for s in run.search_results
                     if s.index_type.lower() == "hnsw" and s.ef_search == ef),
                    None,
                )
                if result:
                    r10 = f"{result.recall_at_10:.3f}" if result.recall_at_10 else "-"
                    row += f"<td>{result.qps:,.0f}</td><td>{r10}</td>"
                else:
                    row += "<td>-</td><td>-</td>"
            row += "</tr>"
            rows.append(row)

        return f'{sort_note}<div class="table-wrap"><table><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table></div>'

    # -----------------------------------------------------------------
    # Configuration section
    # -----------------------------------------------------------------
    def _render_config_section(
        self,
        runs: List[RunData],
        bench_config: Dict[str, Any],
        datasets_meta: Dict[str, Any],
    ) -> str:
        lines: List[str] = ["<h2>Benchmark Configuration</h2>"]

        # Datasets table
        result_datasets = set(r.dataset.upper() for r in runs)
        ds_entries = []
        for ds_key, ds_meta in datasets_meta.items():
            if not isinstance(ds_meta, dict):
                continue
            if ds_key.upper() in result_datasets or ds_key in result_datasets:
                ds_entries.append(ds_meta)

        if ds_entries:
            lines.append("<h3>Datasets</h3>")
            rows = ""
            for ds_meta in ds_entries:
                name = ds_meta.get("name", "")
                vectors = f"{ds_meta.get('vectors', 0):,}" if ds_meta.get("vectors") else "-"
                dims = ds_meta.get("dimensions", "-")
                metric = ds_meta.get("metric", "L2")
                purpose = ds_meta.get("purpose", "")
                purpose_short = purpose.split(".")[0] + "." if purpose else "-"
                rows += f"<tr><td>{name}</td><td>{vectors}</td><td>{dims}</td><td>{metric}</td><td>{purpose_short}</td></tr>"

            lines.append(f'''<div class="table-wrap"><table>
            <thead><tr><th>Dataset</th><th>Vectors</th><th>Dims</th><th>Metric</th><th>Purpose</th></tr></thead>
            <tbody>{rows}</tbody></table></div>''')

        # Index types
        indexes = bench_config.get("indexes", {})
        if indexes:
            lines.append("<h3>Index Types Tested</h3><ul>")
            for idx_name, idx_config in indexes.items():
                desc = idx_config.get("description", "") if isinstance(idx_config, dict) else ""
                if desc and "not yet implemented" not in desc.lower():
                    lines.append(f"<li><strong>{idx_name.upper()}</strong>: {desc}</li>")
            lines.append("</ul>")

        # Metrics glossary
        metrics = bench_config.get("metrics", {})
        if metrics:
            lines.append("<h3>Metrics Glossary</h3>")
            rows = ""
            for key, info in metrics.items():
                if isinstance(info, dict):
                    rows += f"<tr><td><strong>{info.get('name', key)}</strong></td><td>{info.get('description', '')}</td></tr>"
            lines.append(f'<div class="table-wrap"><table><thead><tr><th>Metric</th><th>Description</th></tr></thead>'
                         f'<tbody>{rows}</tbody></table></div>')

        # Database Configuration Summary
        lines.append("<h3>Database Configuration</h3>")
        db_rows = ""
        for db_name in sorted(set(r.database for r in runs)):
            sample = next(r for r in runs if r.database == db_name)
            meta = sample.metadata
            persistence = meta.get("persistence", "N/A")
            persistence_html = (f'<strong class="caveat-highlight">{persistence}</strong>'
                                if persistence == "memory" else persistence)
            db_rows += (
                f"<tr><td>{sample.database}</td><td>{sample.db_version or 'N/A'}</td>"
                f"<td>{meta.get('architecture', 'N/A')}</td>"
                f"<td>{meta.get('protocol', 'N/A')}</td>"
                f"<td>{persistence_html}</td>"
                f"<td>{meta.get('license', 'N/A')}</td></tr>"
            )

        lines.append(f'''<div class="table-wrap"><table>
        <thead><tr><th>Database</th><th>Version</th><th>Architecture</th><th>Protocol</th><th>Persistence</th><th>License</th></tr></thead>
        <tbody>{db_rows}</tbody></table></div>''')

        # Database notes
        notes = [(r.database, r.metadata.get("notes")) for r in runs if r.metadata.get("notes")]
        seen_dbs = set()
        if notes:
            lines.append("<h3>Database Notes</h3><ul>")
            for db_name, note in sorted(notes):
                if db_name not in seen_dbs:
                    lines.append(f"<li><strong>{db_name}</strong>: {note}</li>")
                    seen_dbs.add(db_name)
            lines.append("</ul>")

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Methodology
    # -----------------------------------------------------------------
    def _render_methodology(self) -> str:
        return """
        <h2>Methodology</h2>
        <ul>
            <li>Each database was tested with the same dataset and query workload</li>
            <li>HNSW indexes used M=16, efConstruction=64 (consistent across all databases)</li>
            <li>Search tests used 10,000 queries with 100 warmup queries, executed sequentially (single-client, one-at-a-time)</li>
            <li>Recall is calculated against brute-force ground truth provided with each dataset</li>
            <li>Client-server databases run in Docker containers with consistent CPU/memory limits</li>
            <li>FAISS runs in-process (no network overhead) &mdash; shown as embedded baseline, not directly comparable to client-server databases</li>
            <li>Redis operates entirely in-memory (no disk persistence during benchmarks) &mdash; other client-server databases persist to disk, making Redis results not directly comparable</li>
        </ul>"""

    # -----------------------------------------------------------------
    # HTML wrapper
    # -----------------------------------------------------------------
    def _wrap_html(self, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector Database Benchmark Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --text: #1f2937;
            --border: #e5e7eb;
            --bg-alt: #f9fafb;
            --bg-code: #f3f4f6;
            --green: #16a34a;
            --red: #dc2626;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            background: #fff;
        }}

        h1 {{
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 0.5rem;
        }}

        h2 {{
            color: var(--text);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.3rem;
            margin-top: 2.5rem;
        }}

        h3 {{ margin-top: 1.5rem; }}

        p.meta {{ color: #6b7280; font-size: 0.95rem; }}
        p.note {{ color: #6b7280; font-size: 0.9rem; font-style: italic; }}

        /* Tables */
        .table-wrap {{ overflow-x: auto; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.875rem;
        }}

        th, td {{
            padding: 0.6rem 0.75rem;
            text-align: left;
            border: 1px solid var(--border);
            white-space: nowrap;
        }}

        th {{ background: var(--bg-code); font-weight: 600; }}
        tr:nth-child(even) {{ background: var(--bg-alt); }}
        tr:hover {{ background: #f3f4f6; }}

        /* Rank badges */
        .rank-top {{ background: #dcfce7 !important; }}
        .rank-bottom {{ background: #fee2e2 !important; }}
        .rank-top-inline {{ background: #dcfce7; padding: 1px 5px; border-radius: 3px; }}
        .rank-bottom-inline {{ background: #fee2e2; padding: 1px 5px; border-radius: 3px; }}
        .row-excluded-inline {{ background: #fef9c3; padding: 1px 5px; border-radius: 3px; }}

        /* Excluded rows (FAISS/Redis — not comparable) */
        .row-excluded td {{ background: #fef9c3 !important; }}

        /* Caveat callout box */
        .caveat-box {{
            background: #fffbeb;
            border: 1px solid #f59e0b;
            border-left: 4px solid #f59e0b;
            border-radius: 4px;
            padding: 0.75rem 1rem;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        .caveat-box ul {{
            margin: 0.25rem 0 0 0;
            padding-left: 1.25rem;
        }}
        .caveat-box li {{ margin: 0.25rem 0; }}
        .caveat-highlight {{ color: #d97706; }}

        /* Chart grid */
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin: 1rem 0;
        }}

        .chart-cell {{ text-align: center; }}
        .chart-cell svg, .chart-full svg {{
            max-width: 100%;
            height: auto;
        }}
        .chart-full {{ margin: 1.5rem 0; text-align: center; }}

        /* Collapsible details */
        details {{
            border: 1px solid var(--border);
            border-radius: 6px;
            margin: 0.75rem 0;
            padding: 0;
        }}

        summary {{
            cursor: pointer;
            padding: 0.75rem 1rem;
            background: var(--bg-alt);
            border-radius: 6px;
            user-select: none;
        }}

        summary:hover {{ background: #f3f4f6; }}

        details[open] summary {{
            border-bottom: 1px solid var(--border);
            border-radius: 6px 6px 0 0;
        }}

        details > :not(summary) {{
            padding: 0 1rem;
        }}

        /* Findings */
        ol.findings li {{
            margin: 0.75rem 0;
            line-height: 1.5;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
            table {{ font-size: 0.8rem; }}
            th, td {{ padding: 0.4rem; }}
        }}

        @media print {{
            body {{ padding: 0; font-size: 11pt; }}
            h1 {{ font-size: 16pt; }}
            h2 {{ font-size: 13pt; page-break-after: avoid; }}
            table {{ font-size: 9pt; page-break-inside: avoid; }}
            details {{ break-inside: avoid; }}
            details[open] {{ border: none; }}
            summary {{ background: none; }}
        }}
    </style>
</head>
<body>
{body}
</body>
</html>"""
