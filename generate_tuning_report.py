#!/usr/bin/env python3
"""
Generate KDB.AI tuning benchmark report.

Compares HNSW parameter configurations (M, efConstruction, efSearch, THREADS/NUM_WRK)
across datasets. Produces Pareto charts, config comparison tables, and recommendations.

Usage:
    # From a run ID (auto-resolves DB path)
    python generate_tuning_report.py --run-id 2026-02-08-1200

    # From a specific DB file
    python generate_tuning_report.py --db-path results/benchmark-kdbai-tuning-2026-02-08-1200.db

    # Custom output
    python generate_tuning_report.py --run-id 2026-02-08-1200 -o results/tuning-report.html
"""

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TuningResult:
    """A single search config result from a tuning run."""
    dataset: str
    index_type: str       # e.g., HNSW_M32_efC128_1wrk_16thr
    ef_search: Optional[int]
    qps: float
    recall_at_10: Optional[float]
    recall_at_100: Optional[float]
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]
    # Parsed from index_type
    M: Optional[int] = None
    efC: Optional[int] = None
    docker_config: Optional[str] = None
    index_only: bool = False


@dataclass
class IngestResult:
    """Ingest result from a tuning run."""
    dataset: str
    index_type: str
    throughput_vps: float
    total_time_s: float
    # Parsed from index_type
    M: Optional[int] = None
    efC: Optional[int] = None
    docker_config: Optional[str] = None


def parse_index_type(index_type: str) -> Dict[str, Any]:
    """Parse dynamic index_type name into components.

    Examples:
        HNSW_M32_efC128_1wrk_16thr -> {M: 32, efC: 128, docker: '1wrk_16thr', indexOnly: False}
        HNSW_M16_efC64_idxOnly     -> {M: 16, efC: 64, docker: None, indexOnly: True}
        HNSW_M48_efC200_2wrk_8thr_idxOnly -> {M: 48, efC: 200, docker: '2wrk_8thr', indexOnly: True}
    """
    result: Dict[str, Any] = {"M": None, "efC": None, "docker": None, "indexOnly": False}

    m_match = re.search(r'M(\d+)', index_type)
    if m_match:
        result["M"] = int(m_match.group(1))

    efc_match = re.search(r'efC(\d+)', index_type)
    if efc_match:
        result["efC"] = int(efc_match.group(1))

    result["indexOnly"] = "_idxOnly" in index_type

    # Docker config: pattern like NwrkMthr (e.g., 1wrk_16thr, 2wrk_8thr)
    docker_match = re.search(r'(\d+wrk_\d+thr)', index_type)
    if docker_match:
        result["docker"] = docker_match.group(1)

    return result


def load_tuning_data(db_path: str) -> Tuple[List[TuningResult], List[IngestResult]]:
    """Load tuning results from SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # Load search results
    cursor.execute("""
        SELECT r.dataset, s.index_type, s.ef_search, s.qps,
               s.recall_at_10, s.recall_at_100, s.p50_ms, s.p95_ms, s.p99_ms
        FROM search_results s
        JOIN runs r ON s.run_id = r.run_id
        WHERE UPPER(r.database) = 'KDB.AI'
        ORDER BY r.dataset, s.index_type, s.ef_search
    """)

    search_results = []
    for row in cursor.fetchall():
        parsed = parse_index_type(row["index_type"])
        search_results.append(TuningResult(
            dataset=row["dataset"],
            index_type=row["index_type"],
            ef_search=row["ef_search"],
            qps=row["qps"],
            recall_at_10=row["recall_at_10"],
            recall_at_100=row["recall_at_100"],
            p50_ms=row["p50_ms"],
            p95_ms=row["p95_ms"],
            p99_ms=row["p99_ms"],
            M=parsed["M"],
            efC=parsed["efC"],
            docker_config=parsed["docker"],
            index_only=parsed["indexOnly"],
        ))

    # Load ingest results
    cursor.execute("""
        SELECT r.dataset, i.index_type, i.throughput_vps, i.total_time_s
        FROM ingest_results i
        JOIN runs r ON i.run_id = r.run_id
        WHERE UPPER(r.database) = 'KDB.AI'
        ORDER BY r.dataset, i.index_type
    """)

    ingest_results = []
    for row in cursor.fetchall():
        parsed = parse_index_type(row["index_type"])
        ingest_results.append(IngestResult(
            dataset=row["dataset"],
            index_type=row["index_type"],
            throughput_vps=row["throughput_vps"],
            total_time_s=row["total_time_s"],
            M=parsed["M"],
            efC=parsed["efC"],
            docker_config=parsed["docker"],
        ))

    conn.close()
    return search_results, ingest_results


class TuningReportGenerator:
    """Generate tuning benchmark report."""

    REFERENCE_EF = 128

    def __init__(self, search_results: List[TuningResult], ingest_results: List[IngestResult]):
        self.search_results = search_results
        self.ingest_results = ingest_results

        # Group by dataset
        self.datasets = sorted(set(r.dataset for r in search_results))
        self.search_by_dataset: Dict[str, List[TuningResult]] = {}
        for r in search_results:
            self.search_by_dataset.setdefault(r.dataset, []).append(r)
        self.ingest_by_dataset: Dict[str, List[IngestResult]] = {}
        for r in ingest_results:
            self.ingest_by_dataset.setdefault(r.dataset, []).append(r)

    def generate(self) -> str:
        """Generate complete HTML report."""
        sections: List[str] = []
        sections.append(self._render_header())
        sections.append(self._render_summary_table())
        sections.append(self._render_pareto_charts())
        sections.append(self._render_ingest_comparison())
        sections.append(self._render_threading_impact())
        sections.append(self._render_index_only_impact())
        sections.append(self._render_cross_dataset_view())
        sections.append(self._render_recommendation())

        body = "\n".join(sections)
        return self._wrap_html(body)

    def _render_header(self) -> str:
        n_configs = len(set(
            (r.M, r.efC, r.docker_config) for r in self.search_results if not r.index_only
        ))
        return f"""
        <h1>KDB.AI HNSW Tuning Report</h1>
        <p class="meta">
            Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &mdash;
            {n_configs} configurations &times; {len(self.datasets)} datasets
            ({', '.join(self.datasets)})
        </p>"""

    def _render_summary_table(self) -> str:
        """All configs ranked by recall@10 at reference efSearch."""
        lines: List[str] = []
        lines.append("<h2>Summary: All Configurations at efSearch=128</h2>")
        lines.append('<p class="note">Sorted by average Recall@10 across datasets (descending). '
                      'Only non-indexOnly results shown.</p>')

        # Header
        header = "<tr><th>Config</th><th>M</th><th>efC</th><th>Docker</th>"
        for ds in self.datasets:
            header += f"<th>{ds} R@10</th><th>{ds} QPS</th>"
        header += "<th>Avg R@10</th><th>Avg QPS</th></tr>"

        # Collect data per unique config
        configs: Dict[str, Dict[str, TuningResult]] = {}
        for r in self.search_results:
            if r.index_only or r.ef_search != self.REFERENCE_EF:
                continue
            key = f"M{r.M}_efC{r.efC}_{r.docker_config or 'default'}"
            configs.setdefault(key, {})[r.dataset] = r

        # Sort by avg recall
        sorted_configs = sorted(
            configs.items(),
            key=lambda kv: _avg([r.recall_at_10 for r in kv[1].values() if r.recall_at_10]),
            reverse=True,
        )

        # Find best recall per dataset for highlighting
        best_recall = {}
        for ds in self.datasets:
            recalls = [r.recall_at_10 for r in self.search_results
                       if r.dataset == ds and r.ef_search == self.REFERENCE_EF
                       and not r.index_only and r.recall_at_10]
            best_recall[ds] = max(recalls) if recalls else 0

        rows = []
        for key, ds_results in sorted_configs:
            sample = next(iter(ds_results.values()))
            avg_r = _avg([r.recall_at_10 for r in ds_results.values() if r.recall_at_10])
            avg_qps = _avg([r.qps for r in ds_results.values()])

            row = f"<tr><td><strong>{key}</strong></td>"
            row += f"<td>{sample.M}</td><td>{sample.efC}</td><td>{sample.docker_config or 'default'}</td>"

            for ds in self.datasets:
                r = ds_results.get(ds)
                if r:
                    r10_class = ' class="rank-top"' if r.recall_at_10 and abs(r.recall_at_10 - best_recall.get(ds, 0)) < 0.001 else ''
                    row += f"<td{r10_class}>{r.recall_at_10:.4f}</td><td>{r.qps:,.0f}</td>"
                else:
                    row += "<td>-</td><td>-</td>"

            row += f"<td><strong>{avg_r:.4f}</strong></td><td><strong>{avg_qps:,.0f}</strong></td></tr>"
            rows.append(row)

        table = f"""
        <div class="table-wrap">
        <table>
            <thead>{header}</thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        </div>"""

        lines.append(table)
        return "\n".join(lines)

    def _render_pareto_charts(self) -> str:
        """Recall vs QPS scatter plots per dataset."""
        lines: List[str] = []
        lines.append("<h2>Recall vs QPS Pareto Curves</h2>")
        lines.append('<p class="note">Each point is a (M, efC, efSearch) combo. '
                      'Lines connect efSearch values for a given (M, efC). '
                      'Non-indexOnly results only.</p>')

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            lines.append("<p>matplotlib not available — charts skipped.</p>")
            return "\n".join(lines)

        chart_cells = []
        for ds in self.datasets:
            svg = self._chart_pareto(ds, plt)
            if svg:
                chart_cells.append(f'<div class="chart-cell">{svg}</div>')

        if chart_cells:
            lines.append(f'<div class="chart-grid">{"".join(chart_cells)}</div>')
        return "\n".join(lines)

    def _chart_pareto(self, dataset: str, plt) -> str:
        """Recall@10 vs QPS scatter for one dataset."""
        import io

        results = [r for r in self.search_by_dataset.get(dataset, [])
                   if not r.index_only and r.recall_at_10 and r.ef_search]

        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(6, 4))

        # Group by (M, efC, docker_config) — each group is a line
        groups: Dict[str, List[TuningResult]] = {}
        for r in results:
            key = f"M{r.M}_efC{r.efC}_{r.docker_config or 'default'}"
            groups.setdefault(key, []).append(r)

        colors = plt.cm.tab10.colors
        for i, (key, group) in enumerate(sorted(groups.items())):
            group.sort(key=lambda r: r.ef_search)
            recalls = [r.recall_at_10 for r in group]
            qps = [r.qps for r in group]
            color = colors[i % len(colors)]
            ax.plot(recalls, qps, marker="o", color=color, label=key,
                    markersize=6, linewidth=1.5, zorder=3)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(f"Recall vs QPS — {dataset}")
        ax.legend(fontsize=6, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue().decode("utf-8")

    def _render_ingest_comparison(self) -> str:
        """Bar chart comparing ingest throughput across M/efC configs."""
        lines: List[str] = []
        lines.append("<h2>Ingest Throughput</h2>")
        lines.append('<p class="note">Higher M and efConstruction build better graphs '
                      'but slow ingest. Quantifies the tradeoff.</p>')

        # Table
        header = "<tr><th>Config</th><th>M</th><th>efC</th>"
        for ds in self.datasets:
            header += f"<th>{ds} (vec/s)</th><th>{ds} Time (s)</th>"
        header += "</tr>"

        configs: Dict[str, Dict[str, IngestResult]] = {}
        for r in self.ingest_results:
            # Dedupe by (M, efC) — docker config doesn't affect ingest params
            key = f"M{r.M}_efC{r.efC}"
            configs.setdefault(key, {})[r.dataset] = r

        rows = []
        for key, ds_results in sorted(configs.items()):
            sample = next(iter(ds_results.values()))
            row = f"<tr><td><strong>{key}</strong></td><td>{sample.M}</td><td>{sample.efC}</td>"
            for ds in self.datasets:
                r = ds_results.get(ds)
                if r:
                    row += f"<td>{r.throughput_vps:,.0f}</td><td>{r.total_time_s:.1f}</td>"
                else:
                    row += "<td>-</td><td>-</td>"
            row += "</tr>"
            rows.append(row)

        table = f"""
        <div class="table-wrap">
        <table>
            <thead>{header}</thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        </div>"""

        lines.append(table)

        # Chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            svg = self._chart_ingest(configs, plt)
            if svg:
                lines.append(f'<div class="chart-full">{svg}</div>')
        except ImportError:
            pass

        return "\n".join(lines)

    def _chart_ingest(self, configs: Dict[str, Dict[str, IngestResult]], plt) -> str:
        """Grouped bar chart of ingest throughput."""
        import io
        import numpy as np

        config_names = sorted(configs.keys())
        if not config_names:
            return ""

        fig, ax = plt.subplots(figsize=(8, max(3, len(config_names) * 0.5 + 1)))
        y = np.arange(len(config_names))
        bar_height = 0.8 / len(self.datasets) if self.datasets else 0.8
        ds_colors = ["#2563eb", "#e41a1c", "#4daf4a", "#ff7f00"]

        for i, ds in enumerate(self.datasets):
            offsets = y - 0.4 + bar_height * (i + 0.5)
            values = [configs[c].get(ds, IngestResult("", "", 0, 0)).throughput_vps for c in config_names]
            ax.barh(offsets, values, height=bar_height * 0.9,
                    label=ds, color=ds_colors[i % len(ds_colors)], alpha=0.85)

        ax.set_yticks(y)
        ax.set_yticklabels(config_names)
        ax.set_xlabel("Vectors/sec")
        ax.set_title("Ingest Throughput by Configuration")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue().decode("utf-8")

    def _render_threading_impact(self) -> str:
        """THREADS/NUM_WRK impact on QPS and ingest."""
        lines: List[str] = []
        lines.append("<h2>Threading Configuration Impact</h2>")
        lines.append('<p class="note">Compares QPS across THREADS/NUM_WRK combos '
                      'for the same HNSW params. efSearch=128, non-indexOnly.</p>')

        # Group by (M, efC) then by docker_config
        hnsw_groups: Dict[str, Dict[str, Dict[str, TuningResult]]] = {}
        for r in self.search_results:
            if r.index_only or r.ef_search != self.REFERENCE_EF:
                continue
            hnsw_key = f"M{r.M}_efC{r.efC}"
            docker_key = r.docker_config or "default"
            hnsw_groups.setdefault(hnsw_key, {}).setdefault(docker_key, {})[r.dataset] = r

        if not hnsw_groups:
            lines.append("<p>No threading comparison data available.</p>")
            return "\n".join(lines)

        for hnsw_key in sorted(hnsw_groups.keys()):
            docker_configs = hnsw_groups[hnsw_key]
            if len(docker_configs) < 2:
                continue

            lines.append(f"<h3>{hnsw_key}</h3>")
            header = "<tr><th>Docker Config</th>"
            for ds in self.datasets:
                header += f"<th>{ds} QPS</th><th>{ds} R@10</th>"
            header += "</tr>"

            rows = []
            for docker_key in sorted(docker_configs.keys()):
                ds_results = docker_configs[docker_key]
                row = f"<tr><td><strong>{docker_key}</strong></td>"
                for ds in self.datasets:
                    r = ds_results.get(ds)
                    if r:
                        row += f"<td>{r.qps:,.0f}</td><td>{r.recall_at_10:.4f}</td>"
                    else:
                        row += "<td>-</td><td>-</td>"
                row += "</tr>"
                rows.append(row)

            lines.append(f"""
            <div class="table-wrap">
            <table>
                <thead>{header}</thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
            </div>""")

        return "\n".join(lines)

    def _render_index_only_impact(self) -> str:
        """Side-by-side QPS comparison with/without indexOnly."""
        lines: List[str] = []
        lines.append("<h2>indexOnly Impact</h2>")
        lines.append('<p class="note">Does returning only IDs (without vectors) improve QPS? '
                      'Compared at efSearch=128.</p>')

        # Find pairs of (config, dataset) with and without indexOnly
        pairs: Dict[str, Dict[str, Tuple[Optional[TuningResult], Optional[TuningResult]]]] = {}
        for r in self.search_results:
            if r.ef_search != self.REFERENCE_EF:
                continue
            config_key = f"M{r.M}_efC{r.efC}_{r.docker_config or 'default'}"
            pairs.setdefault(config_key, {}).setdefault(r.dataset, (None, None))
            current = pairs[config_key][r.dataset]
            if r.index_only:
                pairs[config_key][r.dataset] = (current[0], r)
            else:
                pairs[config_key][r.dataset] = (r, current[1])

        header = "<tr><th>Config</th>"
        for ds in self.datasets:
            header += f"<th>{ds} Normal QPS</th><th>{ds} idxOnly QPS</th><th>{ds} Speedup</th>"
        header += "</tr>"

        rows = []
        for config_key in sorted(pairs.keys()):
            ds_pairs = pairs[config_key]
            row = f"<tr><td><strong>{config_key}</strong></td>"
            for ds in self.datasets:
                normal, idx_only = ds_pairs.get(ds, (None, None))
                if normal and idx_only:
                    speedup = idx_only.qps / normal.qps if normal.qps > 0 else 0
                    row += f"<td>{normal.qps:,.0f}</td><td>{idx_only.qps:,.0f}</td>"
                    color = "green" if speedup > 1.05 else ("red" if speedup < 0.95 else "inherit")
                    row += f'<td style="color:{color}">{speedup:.2f}x</td>'
                elif normal:
                    row += f"<td>{normal.qps:,.0f}</td><td>-</td><td>-</td>"
                else:
                    row += "<td>-</td><td>-</td><td>-</td>"
            row += "</tr>"
            rows.append(row)

        lines.append(f"""
        <div class="table-wrap">
        <table>
            <thead>{header}</thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        </div>""")

        return "\n".join(lines)

    def _render_cross_dataset_view(self) -> str:
        """Do best configs on SIFT also work best on GLOVE/GIST?"""
        lines: List[str] = []
        lines.append("<h2>Cross-Dataset Consistency</h2>")
        lines.append('<p class="note">Heatmap showing Recall@10 rank per config across datasets. '
                      'efSearch=128, non-indexOnly. Lower rank = better.</p>')

        if len(self.datasets) < 2:
            lines.append("<p>Need at least 2 datasets for cross-dataset comparison.</p>")
            return "\n".join(lines)

        # Rank configs per dataset
        config_ranks: Dict[str, Dict[str, int]] = {}  # config -> {ds: rank}
        for ds in self.datasets:
            ds_configs = []
            for r in self.search_results:
                if r.dataset != ds or r.index_only or r.ef_search != self.REFERENCE_EF:
                    continue
                key = f"M{r.M}_efC{r.efC}_{r.docker_config or 'default'}"
                ds_configs.append((key, r.recall_at_10 or 0))

            ds_configs.sort(key=lambda x: x[1], reverse=True)
            for rank, (key, _) in enumerate(ds_configs, 1):
                config_ranks.setdefault(key, {})[ds] = rank

        header = "<tr><th>Config</th>"
        for ds in self.datasets:
            header += f"<th>{ds} Rank</th>"
        header += "<th>Avg Rank</th></tr>"

        rows = []
        sorted_configs = sorted(
            config_ranks.items(),
            key=lambda kv: _avg(list(kv[1].values())),
        )
        for key, ranks in sorted_configs:
            avg_rank = _avg(list(ranks.values()))
            row = f"<tr><td><strong>{key}</strong></td>"
            for ds in self.datasets:
                r = ranks.get(ds, "-")
                bg = ""
                if isinstance(r, int):
                    if r <= 3:
                        bg = ' class="rank-top"'
                    elif r >= len(config_ranks) - 2:
                        bg = ' class="rank-bottom"'
                row += f"<td{bg}>{r}</td>"
            row += f"<td><strong>{avg_rank:.1f}</strong></td></tr>"
            rows.append(row)

        lines.append(f"""
        <div class="table-wrap">
        <table>
            <thead>{header}</thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        </div>""")

        return "\n".join(lines)

    def _render_recommendation(self) -> str:
        """Recommend best config based on recall target."""
        lines: List[str] = []
        lines.append("<h2>Recommendation</h2>")

        # Find config with best average recall at efSearch=128
        config_stats: Dict[str, Dict[str, List[float]]] = {}
        for r in self.search_results:
            if r.index_only or r.ef_search != self.REFERENCE_EF:
                continue
            key = f"M{r.M}_efC{r.efC}_{r.docker_config or 'default'}"
            config_stats.setdefault(key, {"recalls": [], "qps": []})
            if r.recall_at_10:
                config_stats[key]["recalls"].append(r.recall_at_10)
            config_stats[key]["qps"].append(r.qps)

        if not config_stats:
            lines.append("<p>No data available for recommendation.</p>")
            return "\n".join(lines)

        # Sort by avg recall
        ranked = sorted(
            config_stats.items(),
            key=lambda kv: _avg(kv[1]["recalls"]),
            reverse=True,
        )

        best_key, best_stats = ranked[0]
        best_recall = _avg(best_stats["recalls"])
        best_qps = _avg(best_stats["qps"])

        # Baseline (M16_efC64)
        baseline_key = next((k for k, _ in ranked if "M16_efC64" in k), None)
        baseline_recall = _avg(config_stats[baseline_key]["recalls"]) if baseline_key else 0
        baseline_qps = _avg(config_stats[baseline_key]["qps"]) if baseline_key else 0

        lines.append(f"""
        <div class="caveat-box">
        <strong>Best configuration: {best_key}</strong>
        <ul>
            <li>Average Recall@10: {best_recall:.4f} (vs baseline {baseline_recall:.4f}, +{(best_recall - baseline_recall):.4f})</li>
            <li>Average QPS: {best_qps:,.0f} (vs baseline {baseline_qps:,.0f})</li>
        </ul>
        <p>This configuration should be used in the next competitive benchmark to see if it
        closes the recall gap with Qdrant.</p>
        </div>""")

        # Also show top 3
        lines.append("<h3>Top 3 Configurations</h3>")
        lines.append('<table><thead><tr><th>Rank</th><th>Config</th><th>Avg R@10</th><th>Avg QPS</th></tr></thead><tbody>')
        for i, (key, stats) in enumerate(ranked[:3], 1):
            r = _avg(stats["recalls"])
            q = _avg(stats["qps"])
            lines.append(f'<tr><td>{i}</td><td>{key}</td><td>{r:.4f}</td><td>{q:,.0f}</td></tr>')
        lines.append('</tbody></table>')

        return "\n".join(lines)

    def _wrap_html(self, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KDB.AI HNSW Tuning Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --text: #1f2937;
            --border: #e5e7eb;
            --bg-alt: #f9fafb;
            --bg-code: #f3f4f6;
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

        .rank-top {{ background: #dcfce7 !important; }}
        .rank-bottom {{ background: #fee2e2 !important; }}

        .caveat-box {{
            background: #dbeafe;
            border: 1px solid #3b82f6;
            border-left: 4px solid #3b82f6;
            border-radius: 4px;
            padding: 0.75rem 1rem;
            margin: 1rem 0;
            font-size: 0.95rem;
        }}
        .caveat-box ul {{
            margin: 0.25rem 0 0 0;
            padding-left: 1.25rem;
        }}

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
        }}
    </style>
</head>
<body>
{body}
</body>
</html>"""


def _avg(values: list) -> float:
    """Average of a list, 0 if empty."""
    return sum(values) / len(values) if values else 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate KDB.AI tuning benchmark report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        help="Run ID (auto-resolves to results/vectordb-benchmark-kdbai-tuning-{run_id}.db)",
    )
    parser.add_argument(
        "--db-path",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--benchmark-type",
        default="kdbai-tuning",
        help="Benchmark type for filename resolution (default: kdbai-tuning)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    # Resolve DB path
    if args.db_path:
        db_path = args.db_path
    elif args.run_id:
        db_path = f"results/vectordb-benchmark-{args.benchmark_type}-{args.run_id}.db"
    else:
        print("Error: either --run-id or --db-path required", file=sys.stderr)
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    output_path = args.output
    if not output_path and args.run_id:
        output_path = f"results/vectordb-benchmark-{args.benchmark_type}-{args.run_id}.html"

    # Load data
    search_results, ingest_results = load_tuning_data(db_path)
    if not search_results:
        print("No KDB.AI tuning results found in database.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(search_results)} search results, {len(ingest_results)} ingest results")

    # Generate report
    generator = TuningReportGenerator(search_results, ingest_results)
    report = generator.generate()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)
        print(f"Report written to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
