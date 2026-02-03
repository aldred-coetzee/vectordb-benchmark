# Vector Database Benchmark Tool

## Claude Code Settings

```
allowedTools: ["Bash(*)", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch"]
```

### Workflow
- **Before starting**: Always `git pull` to ensure local repo is up to date
- **After completing work**: Commit and push changes (don't wait to be asked)
- **Commit messages**: Concise, imperative mood (e.g., "Fix Redis result limit")

### Code Standards
- Python 3.12+, use type hints
- Follow existing patterns in `benchmark/clients/`
- New DB clients must implement `BaseVectorDBClient` interface
- Keep benchmark configs in `configs/*.yaml`
- Results go in `results/` — never commit large result files (.csv, .json with data)

### Public Release Consideration
This project may be shared publicly so others can verify benchmarks independently. Design with this in mind:
- Clear setup instructions and dependencies
- Reproducible results (pinned versions, documented configs)
- No hardcoded paths, credentials, or internal references
- Self-contained — users should be able to clone and run

### Testing
- Run `python run_benchmark.py --help` to verify CLI changes
- Test client changes with small dataset before full benchmark

## Project Overview

Benchmarking tool for comparing vector database performance on the SIFT-1M dataset (1M vectors, 128 dimensions). Measures ingest speed, query throughput (QPS), latency percentiles, and recall accuracy.

## Supported Databases (9 total)

| Database | Type | Client File |
|----------|------|-------------|
| FAISS | Embedded | `faiss_client.py` |
| LanceDB | Embedded | `lancedb_client.py` |
| ChromaDB | Client-server | `chroma_client.py` |
| Qdrant | Client-server | `qdrant_client.py` |
| Milvus | Client-server | `milvus_client.py` |
| Weaviate | Client-server | `weaviate_client.py` |
| pgvector | Client-server | `pgvector_client.py` |
| Redis Stack | Client-server | `redis_client.py` |
| KDB.AI | Client-server | `kdbai_client.py` |

## Project Structure

```
benchmark/
├── clients/           # Database client implementations (base.py defines interface)
├── runner.py          # Orchestrates single benchmark runs
├── report_generator.py # HTML/markdown report generation
├── data_loader.py     # Dataset loading (.fvecs/.ivecs for SIFT, HDF5 for ann-benchmarks)
├── metrics.py         # Recall calculation
├── docker_manager.py  # Container lifecycle management
├── docker_monitor.py  # Container resource monitoring
└── config.py          # Configuration parsing

configs/               # YAML configs per database (index params, efSearch sweeps)
results/               # Benchmark output (CSV, JSON, HTML reports)
datasets/              # SIFT-1M data files
```

## Key Commands

```bash
# Run single database benchmark
python run_benchmark.py --config configs/qdrant.yaml

# Run all databases sequentially
python run_all.py

# Generate comparison report from results
python generate_report.py
```

## Configuration Format (configs/*.yaml)

```yaml
database: qdrant
endpoint: localhost:6333
container: qdrant-bench
index_configs:
  - type: hnsw
    params: {M: 16, efConstruction: 64}
search_configs:
  - index_type: hnsw
    params: {efSearch: 64}
  - index_type: hnsw
    params: {efSearch: 128}
```

## Datasets

All datasets include pre-computed query vectors and ground truth nearest neighbors, enabling recall calculation without brute-force.

### Texmex Corpus (Primary) — ftp://ftp.irisa.fr/local/texmex/corpus/

The standard benchmark corpus with queries and ground truth included.
Each dataset contains: `*_base.fvecs` (vectors), `*_query.fvecs` (queries), `*_groundtruth.ivecs` (true neighbors).

| Dataset | Vectors | Dims | Queries | Metric | Size | Purpose |
|---------|---------|------|---------|--------|------|---------|
| **SIFT-1M** | 1M | 128 | 10K | L2 | ~500MB | Current baseline |
| **GIST-1M** | 1M | 960 | 1K | L2 | ~4GB | High-dimension stress |
| **SIFT-10M** | 10M | 128 | 10K | L2 | ~5GB | Scale stress |
| **GloVe-100** | 1.2M | 100 | 10K | Cosine | ~500MB | Cosine metric (NLP/embeddings) |

### Recommended Progression

```
SIFT-1M → GIST-1M → SIFT-10M → GloVe-100
   ↓         ↓          ↓          ↓
baseline   dims       scale      cosine
```

Four axes: baseline, dimension stress (960D), volume stress (10M), distance metric (cosine vs L2)

## Current Limitations

- **Sequential execution**: `run_all.py` runs databases one at a time
- **Single machine**: All benchmarks run on local Docker
- **Long runtime**: Full suite takes several hours (9 DBs x multiple configs)
- **Single dataset**: Only SIFT-1M currently implemented

---

## Next Phase: AWS Parallelization

### Goal
Parallelize benchmark runs across AWS infrastructure to reduce total runtime from hours to minutes.

### Architecture Principle: Separation of Concerns

```
┌─────────────────────────────────────────────────────────┐
│                    Benchmark Core                        │
│  (runner.py, clients/, metrics.py, data_loader.py)      │
│           Platform-agnostic, no infra dependencies       │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Local   │  │   AWS    │  │  Future  │
        │ (current)│  │Orchestr. │  │ (GCP/K8s)│
        └──────────┘  └──────────┘  └──────────┘
```

- **Local runs must continue to work** — AWS is additive, not a replacement
- **Orchestration is pluggable** — AWS-specific code isolated in its own module
- **Core benchmark code has no cloud dependencies** — keeps it testable and portable
- **Common interface** — orchestrators share same contract for invoking benchmarks

### Requirements
1. **Isolated environments**: Each database runs in its own EC2 instance or container
2. **Parallel execution**: Run multiple databases simultaneously
3. **Result aggregation**: Collect results from all workers into unified report
4. **Cost efficiency**: Use spot instances, auto-terminate on completion
5. **Reproducibility**: Infrastructure as code (Terraform/CDK/Pulumi)

### Design Considerations
- Worker orchestration: AWS Batch vs ECS vs direct EC2
- Data distribution: S3 for dataset, results collection
- Coordination: Step Functions vs simple SQS queue vs orchestrator script
- Instance sizing: Match current benchmark specs (8 CPU, 32GB RAM)

### Open Questions
- Should embedded DBs (FAISS, LanceDB) run differently than client-server?
- How to handle databases that need specific Docker images?
- Central orchestrator vs fully distributed workers?

### Future: Filtered & Hybrid Search (Out of Scope for Now)

Primary focus is pure ANN benchmarking. However, the AWS infrastructure design should accommodate future expansion to:

- **Filtered search**: Vector search with metadata predicates (e.g., `WHERE category = 'X'`)
  - Requires: datasets with metadata (yfcc-10M, synthetic)
  - Requires: filtered ground truth computation

- **Hybrid search**: BM25 + dense vector combination
  - Requires: text datasets (MS MARCO, BEIR)
  - Requires: sparse index support in clients

Design consideration: Keep dataset/query loading modular so filtered queries can be added later without restructuring.

## Data Loader Status

- **Texmex (.fvecs/.ivecs)**: Already supported — SIFT-1M, GIST-1M, SIFT-10M
- **ANN-Benchmarks (HDF5)**: Needed for GloVe-100 — download from `https://ann-benchmarks.com/glove-100-angular.hdf5`
