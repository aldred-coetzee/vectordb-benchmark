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

### AWS Architecture

**Job Model**: One worker = one (database, dataset) pair
- 9 DBs × 4 datasets = 36 jobs total
- Max 9 concurrent workers (one per DB)
- Jobs queued and dispatched by orchestrator

```
┌─────────────────┐
│  Your Machine   │
│  (local)        │
│                 │
│  run_aws.py     │──── Launch via Console (for now)
│  --pull-report  │◄─── Pull final report from S3
└─────────────────┘

         ┌────────────────────────────────────┐
         │         Orchestrator EC2           │
         │           (t3.micro)               │
         │                                    │
         │  - Manages job queue (36 jobs)     │
         │  - Launches up to 9 workers        │
         │  - Monitors heartbeats             │
         │  - As workers finish, launches     │
         │    next job from queue             │
         │  - Aggregates all results          │
         │  - Generates report → S3           │
         │  - Auto-terminates when done       │
         └──────────────┬─────────────────────┘
                        │ launches/monitors
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Worker 1 │   │ Worker 2 │   │ Worker 3 │   ... (up to 9)
   │ qdrant + │   │ milvus + │   │ redis +  │
   │ sift-1m  │   │ sift-1m  │   │ sift-1m  │
   │          │   │          │   │          │
   │ result → │   │ result → │   │ result → │
   │ S3, exit │   │ S3, exit │   │ S3, exit │
   └──────────┘   └──────────┘   └──────────┘
                        │
                        ▼
                  ┌──────────┐
                  │    S3    │
                  │ (bucket) │
                  └──────────┘
```

**Job Matrix**:
```
                 SIFT-1M  GIST-1M  SIFT-10M  GloVe-100
Fast DBs (8)       ✓        ✓         ✓         ✓       = 32 jobs
pgvector           ✓        ✓         ✗         ✓       =  3 jobs
                                                Total:   35 jobs
```

- pgvector excluded from SIFT-10M by default (HNSW build takes ~9 hours)
- Use `--include-slow` flag to override if needed
- Report notes exclusions with reason

**Job Queue Example**:
```
Wave 1: qdrant+sift1m, milvus+sift1m, redis+sift1m, ...  (9 parallel)
Wave 2: qdrant+gist1m, milvus+gist1m, ...               (as slots free)
Wave 3: qdrant+sift10m, ... (pgvector skipped)
Wave 4: qdrant+glove100, ...
```

### Instance Configuration

Only **2 configurations** needed:

| Role | Instance Type | Count | Purpose |
|------|---------------|-------|---------|
| Orchestrator | t3.micro | 1 | Coordinate jobs, aggregate results |
| Worker | m5.2xlarge (8 CPU, 32GB) | up to 9 | Run benchmarks |

### Worker AMI (Pre-baked)

Datasets baked into AMI for efficiency (avoids 360GB repeated downloads):

```
Worker AMI contains:
  /data/
    sift-1m/      (~500MB)
    gist-1m/      (~4GB)
    sift-10m/     (~5GB)
    glove-100/    (~500MB)
  /app/
    vectordb-benchmark/   (benchmark code)
    docker images         (pre-pulled)
```

- AMI size: ~12GB
- AMI storage cost: ~$1.20/month
- Workers launch ready to run (no download wait)

### S3 Structure

```
s3://kdbai-rnd-bucket/vectordb-benchmark/
  runs/
    2024-02-03-1430/
      config.json          # What to run (databases, datasets, timeouts)
      status.json          # Live status (updated by orchestrator)
      report.html          # Final report (when complete)
      jobs/
        qdrant-sift1m/
          status.json      # pending|running|completed|failed
          heartbeat.txt    # Last heartbeat timestamp
          result.json      # Benchmark results
          error.log        # If failed
        qdrant-gist1m/
          ...
        milvus-sift1m/
          ...
```

### CLI Interface

```bash
# Local (unchanged)
python run_benchmark.py --config configs/qdrant.yaml
python run_all.py --databases qdrant,milvus

# AWS
python run_aws.py                                    # Run all (35 jobs)
python run_aws.py --databases qdrant,milvus,redis   # Specific DBs
python run_aws.py --dataset gist-1m                  # Specific dataset
python run_aws.py --databases qdrant --dataset sift-10m  # Combined
python run_aws.py --include-slow                     # Include pgvector+SIFT-10M (~9hrs)
python run_aws.py --pull-report runs/2024-02-03-1430     # Download report
```

### Failure Handling (Simple v1)

| Failure | Detection | Response |
|---------|-----------|----------|
| Worker won't start | Orchestrator timeout | Log error, skip, continue |
| Benchmark crashes | Exit code ≠ 0 | Upload error.log, mark failed |
| DB container fails | Health check timeout | Upload error, mark failed |
| Worker hangs | No heartbeat for 15 min | Mark timeout |

- **No auto-retry** in v1 — report shows what succeeded vs failed
- Failed DBs show error message in report
- Re-run specific DBs manually if needed

### Cost Safety

| Component | Max Lifetime | Action |
|-----------|--------------|--------|
| Orchestrator | 4 hours | Self-terminates, kills stuck workers |
| Worker (per job) | 2 hours | Self-terminate after single benchmark |

**Realistic Time Estimates** (per job):

| DB Type | SIFT-1M | GIST-1M | SIFT-10M | GloVe-100 |
|---------|---------|---------|----------|-----------|
| Fast (FAISS, Qdrant, LanceDB) | 20 min | 30 min | 60 min | 25 min |
| Medium (Milvus, Weaviate, Redis, ChromaDB, KDB.AI) | 30 min | 45 min | 90 min | 35 min |
| Slow (pgvector) | 70 min | 80 min | **~9 hrs** | 80 min |

**Time Estimate** (full run: 35 jobs, pgvector skips SIFT-10M):
- Longest job: ~90 min (medium DB + SIFT-10M)
- 4 waves with 9 parallel workers
- Total wall clock: ~3-4 hours

**Cost Estimate** (full run):
- Orchestrator: 4 hrs × $0.0104/hr = ~$0.04
- Workers: 35 jobs × avg 0.75 hrs × $0.384/hr = ~$10
- **Total: ~$10 per full benchmark run**

### Current Constraints

- **No CLI EC2 create**: Org policy blocks programmatic launch
- **Console launch**: Manually start orchestrator via AWS Console for now
- **Future**: Request programmatic access to automate fully

### Open Questions
- Should embedded DBs (FAISS, LanceDB) run differently than client-server?
- Spot instances vs on-demand? (spot cheaper but can be interrupted)

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
