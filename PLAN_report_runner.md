# Report Runner Plan

## Overview
Create a report runner that generates comprehensive benchmark comparison reports from SQLite results and YAML configs.

## Command Interface
```bash
# Default: latest run of each database
python generate_report.py

# Specific run IDs
python generate_report.py --runs 1,2,3,4,5

# Filter by database
python generate_report.py --databases kdbai,qdrant,faiss

# Output format
python generate_report.py --format markdown|html|console

# Output file
python generate_report.py --output results/report.md
```

## Report Structure

### 1. Header
- Report title with generation timestamp
- Benchmark configuration (dataset, vector count, dimensions)
- Hardware info (hostname, CPU/memory limits)

### 2. Database Configuration Summary
Table showing config metadata for all databases tested:
| Database | Version | Architecture | Protocol | Persistence | License | URL |
|----------|---------|--------------|----------|-------------|---------|-----|
| FAISS | 1.7.4 | embedded | in-process | memory | MIT | github.com/... |
| KDB.AI | 1.8.2 | client-server | QIPC | disk | Commercial | kdb.ai |
| ... | | | | | | |

### 3. Ingest Performance Summary
| Database | FLAT (vec/s) | FLAT Time | HNSW (vec/s) | HNSW Time | Peak Memory |
|----------|--------------|-----------|--------------|-----------|-------------|
| FAISS | 1,446,381 | 0.7s | 66,126 | 15.1s | - |
| KDB.AI | 5,006 | 199.7s | 3,684 | 271.4s | 4.05 GB |

### 4. Search Performance Summary (Best Recall Config)
| Database | Index | QPS | R@10 | R@100 | Latency P50 | Latency P99 |
|----------|-------|-----|------|-------|-------------|-------------|
| FAISS | HNSW ef=256 | 1,685 | 0.994 | 0.974 | 0.5ms | 1.2ms |

### 5. QPS vs Recall Tradeoff (HNSW)
Table showing all efSearch values for each database:
| Database | ef=8 QPS | ef=8 R@10 | ef=16 QPS | ef=16 R@10 | ... |

### 6. Key Findings (Auto-Generated, Evidence-Based)
Auto-generated findings based ONLY on actual data comparisons:
- Compare embedded vs client-server (only if both types present)
- Identify fastest/slowest for ingest and search
- Note any anomalies (e.g., recall not improving with higher ef)

**Rules for auto-generation:**
- MUST cite specific numbers from results
- MUST NOT speculate or guess causes
- MUST use hedging language ("appears to", "data suggests")
- MUST flag unexpected results for manual review

Example:
> "FAISS achieved 2,983 QPS at 97.9% recall, which is 12.3x higher than
> KDB.AI's 242 QPS at the same recall level. This difference is consistent
> with FAISS being an embedded (in-process) database while KDB.AI operates
> as a client-server architecture."

### 7. Detailed Results Per Database
For each database, a dedicated section with:
- Run metadata (run_id, timestamp, duration)
- All ingest results (FLAT, HNSW)
- All search results with all efSearch variations
- Resource utilization (CPU%, memory)

### 8. Appendix
- Benchmark methodology notes
- Full benchmark.yaml configuration
- Hardware specifications

## Implementation

### New Files
```
benchmark/
├── report.py          # Existing - enhance or keep
└── report_generator.py  # New - main report logic

generate_report.py     # New - CLI entry point
templates/             # New - report templates
├── report.md.j2       # Markdown template
└── report.html.j2     # HTML template (optional)
```

### Key Classes/Functions

```python
# generate_report.py
class ReportGenerator:
    def __init__(self, db_path: str = "results/benchmark.db"):
        self.db = BenchmarkDatabase(db_path)

    def get_latest_runs(self, databases: List[str] = None) -> Dict[str, RunData]:
        """Get most recent run for each database."""

    def load_metadata(self, database: str) -> Dict:
        """Load config metadata from YAML."""

    def generate_summary_tables(self, runs: Dict) -> Dict:
        """Generate comparison tables."""

    def generate_report(self, format: str = "markdown") -> str:
        """Generate full report."""

# Data classes
@dataclass
class RunData:
    run_id: int
    database: str
    metadata: Dict
    config: Dict
    ingest_results: List[IngestResult]
    search_results: List[SearchResult]
    timing: Dict  # start, end, duration
```

### Database Queries Needed
```sql
-- Latest run per database
SELECT * FROM runs
WHERE run_id IN (
    SELECT MAX(run_id) FROM runs GROUP BY database
)

-- Ingest results for run
SELECT * FROM ingest_results WHERE run_id = ?

-- Search results for run
SELECT * FROM search_results WHERE run_id = ?
```

## Output Example (Markdown)

```markdown
# Vector Database Benchmark Report
Generated: 2026-02-02 14:30:00

## Configuration
- Dataset: SIFT-1M (1,000,000 vectors, 128 dimensions)
- Search: k=100, 10,000 queries
- Hardware: 4 CPUs, 8GB memory per container

## Database Overview
| Database | Architecture | Protocol | Persistence |
|----------|--------------|----------|-------------|
| FAISS | embedded | in-process | memory |
| KDB.AI | client-server | QIPC | disk |
...

## Ingest Performance
| Database | HNSW Throughput | Time | Peak Memory |
|----------|-----------------|------|-------------|
| FAISS | 66,126 vec/s | 15s | - |
...

## Search Performance (HNSW, ~98% Recall)
| Database | QPS | R@10 | Config |
|----------|-----|------|--------|
| FAISS | 2,983 | 0.979 | ef=128 |
...

## Key Findings
1. **Embedded vs Client-Server**: FAISS (embedded) achieves 12x higher
   QPS than KDB.AI (client-server) due to zero network overhead.
2. **Protocol efficiency**: KDB.AI's QIPC protocol shows better
   performance than HTTP-based alternatives.
...
```

## Decisions Made

1. **Output formats**: Markdown AND HTML from the start
2. **Charts**: Defer - visualization is separate concern
3. **Key Findings**: Auto-generate but MUST be evidence-based only
   - Cite specific numbers
   - No speculation on causes
   - Use hedging language
   - Flag anomalies for manual review
4. **Historical comparison**: Defer for now
5. **CSV Export**: Defer - share MD/HTML files instead

## Implementation Details

### HTML Output
- Use a simple, clean CSS theme (no external dependencies)
- Tables should be sortable (optional enhancement)
- Mobile-responsive layout
- Print-friendly styling

### Evidence-Based Findings Generator
```python
def generate_findings(results: Dict) -> List[str]:
    findings = []

    # Only generate findings we can prove from data
    embedded = [r for r in results if r.metadata.architecture == "embedded"]
    client_server = [r for r in results if r.metadata.architecture == "client-server"]

    if embedded and client_server:
        # Compare best embedded vs best client-server
        best_embedded = max(embedded, key=lambda r: r.best_qps)
        best_cs = max(client_server, key=lambda r: r.best_qps)
        ratio = best_embedded.best_qps / best_cs.best_qps

        findings.append(
            f"{best_embedded.name} (embedded) achieved {best_embedded.best_qps:,} QPS, "
            f"which is {ratio:.1f}x higher than {best_cs.name} (client-server) "
            f"at {best_cs.best_qps:,} QPS. This is consistent with the architectural "
            f"difference (in-process vs network communication)."
        )

    # Flag anomalies
    for r in results:
        if r.recall_not_improving_with_ef:
            findings.append(
                f"⚠️ ANOMALY: {r.name} recall did not improve with higher efSearch. "
                f"This may indicate a configuration issue. Manual review recommended."
            )

    return findings
```

## Estimated Effort
- Core report generator: ~250 lines
- CLI interface: ~50 lines
- Markdown template: ~100 lines
- HTML template + CSS: ~150 lines
- Findings generator: ~100 lines

Total: ~650 lines, ~3-4 hours implementation
