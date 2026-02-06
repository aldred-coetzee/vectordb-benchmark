# Vector Database Benchmark Tool

## Primary Objective

**This benchmark is maintained by KX, the company behind KDB.AI.** The primary purpose is to understand where KDB.AI stands relative to competitors and identify areas for improvement. When analyzing results, always surface KDB.AI-specific insights: performance gaps, recall differences, configuration opportunities, and actionable recommendations. KDB.AI findings take priority in any investigation or report review.

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
- When discovering noteworthy database quirks (payload limits, index behavior, API oddities), update the `metadata.notes` field in the relevant `configs/*.yaml` — these surface in generated reports

### Public Release Consideration
This project may be shared publicly so others can verify benchmarks independently. Design with this in mind:
- Clear setup instructions and dependencies
- Reproducible results (pinned versions, documented configs)
- No hardcoded paths, credentials, or internal references
- Self-contained — users should be able to clone and run

### Testing
- **Dev datasets**: Use `--dataset sift-dev` for fast iteration (~12s per DB vs 30+ min)
- Generate dev data: `python scripts/generate_dev_datasets.py --datasets sift`
- Run `python run_benchmark.py --help` to verify CLI changes
- Test client changes on sift-dev before full benchmark

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
data/                  # Datasets (sift/, gist/, sift-dev/) and DB runtime storage
scripts/               # Utility scripts (generate_dev_datasets.py, pull_run.py)
aws/                   # AWS orchestration (orchestrator.py, worker_startup.sh)
```

## Key Commands

```bash
# Run single database benchmark
python run_benchmark.py --config configs/qdrant.yaml

# Fast dev test (~12 seconds)
python run_benchmark.py --config configs/qdrant.yaml --dataset sift-dev

# Run all databases sequentially
python run_all.py

# Pull AWS run results and generate report
python scripts/pull_run.py 2026-02-05-1816
python scripts/pull_run.py 2026-02-05-1816 --benchmark-type kdbai-tuning

# Generate report from existing merged DB
python generate_report.py --run-id 2026-02-05-1816

# Generate dev datasets from full datasets
python scripts/generate_dev_datasets.py --datasets sift

# Download HDF5 datasets (GloVe, DBpedia-OpenAI)
python scripts/download_datasets.py --datasets glove-100,dbpedia-openai
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

### Texmex Corpus — ftp://ftp.irisa.fr/local/texmex/corpus/

Format: `.fvecs`/`.ivecs` — `*_base.fvecs` (vectors), `*_query.fvecs` (queries), `*_groundtruth.ivecs` (true neighbors).

| Dataset | Vectors | Dims | Queries | Metric | Size | Purpose |
|---------|---------|------|---------|--------|------|---------|
| **SIFT-1M** | 1M | 128 | 10K | L2 | ~500MB | Baseline |
| **GIST-1M** | 1M | 960 | 1K | L2 | ~4GB | Dimension stress |
| **SIFT-10M** | 10M | 128 | 10K | L2 | ~5GB | Scale stress (not yet supported) |

### ANN-Benchmarks (HDF5)

Format: `.hdf5` — keys: `train` (base vectors), `test` (queries), `neighbors` (ground truth).

| Dataset | Vectors | Dims | Queries | Metric | Size | Purpose |
|---------|---------|------|---------|--------|------|---------|
| **GloVe-100** | 1.18M | 100 | 10K | Cosine | ~463MB | Cosine metric |
| **DBpedia-OpenAI-1M** | 990K | 1536 | 10K | Cosine | ~6GB | Production-representative (ada-002) |

Download: `python scripts/download_datasets.py --datasets glove-100,dbpedia-openai`

### Dev Datasets (for development testing)

Small subsets for fast iteration — NOT for benchmarking. Generated from full datasets.

| Dataset | Vectors | Dims | Queries | Source |
|---------|---------|------|---------|--------|
| **sift-dev** | 10K | 128 | 100 | SIFT-1M |
| **gist-dev** | 10K | 960 | 100 | GIST-1M |

Generate with: `python scripts/generate_dev_datasets.py`

Dev datasets use recomputed exact ground truth (FAISS brute-force) and the same `.fvecs/.ivecs` format, so the same code paths are exercised.

### Recommended Progression

```
SIFT-1M → GIST-1M → GloVe-100 → DBpedia-OpenAI-1M
   ↓         ↓          ↓              ↓
baseline   dims      cosine     production (1536D, cosine)
```

Four axes: baseline (128D L2), dimension stress (960D L2), cosine metric (100D), production workload (1536D cosine)

### Distance Metric Support

All 9 database clients support both L2 and cosine distance. The metric is configured per-dataset in `benchmark.yaml` and propagated automatically to each client:

| Client | L2 | Cosine |
|--------|-----|--------|
| FAISS | `IndexFlatL2` | `IndexFlatIP` + normalize |
| Qdrant | `Distance.EUCLID` | `Distance.COSINE` |
| Milvus | `metric_type: "L2"` | `metric_type: "COSINE"` |
| ChromaDB | `hnsw:space: "l2"` | `hnsw:space: "cosine"` |
| Weaviate | `VectorDistances.L2_SQUARED` | `VectorDistances.COSINE` |
| pgvector | `<->` operator | `<=>` operator |
| Redis | `DISTANCE_METRIC: "L2"` | `DISTANCE_METRIC: "COSINE"` |
| KDB.AI | `metric: "L2"` | `metric: "CS"` |
| LanceDB | `metric="L2"` | `metric="cosine"` |

## Current Limitations

- **Sequential execution**: `run_all.py` runs databases one at a time
- **Single machine**: All benchmarks run on local Docker
- **Long runtime**: Full suite takes several hours (9 DBs x multiple configs)

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

**Job Queue** (ordered smallest → largest for early results):
```
Wave 1: All 9 DBs × SIFT-1M    (~30 min) → First comparison ready
Wave 2: All 9 DBs × GloVe-100  (~35 min) → Cosine metric results
Wave 3: All 9 DBs × GIST-1M    (~80 min) → High-dimension results
Wave 4: 8 DBs × SIFT-10M       (~90 min) → Scale results (pgvector skipped)
```

Early results arrive in ~30 min; full run completes in ~3-4 hours.

### Instance Configuration

Only **2 configurations** needed:

| Role | Instance Type | Count | Purpose |
|------|---------------|-------|---------|
| Orchestrator | t3.micro | 1 | Coordinate jobs, aggregate results |
| Worker | m5.4xlarge (16 CPU, 64GB) | up to 9 | Run benchmarks |

### AMIs (Separate Concerns)

Two separate AMIs — orchestrator is lightweight, worker is heavy with datasets and Docker images.

#### Orchestrator AMI

`ami-09ed5dd071675cfef` (`vectordb-benchmark-orchestrator-v1`)

```
Orchestrator AMI contains:
  /app/
    vectordb-benchmark/     - benchmark code (git pull at startup for latest)
  Python 3.12.12 + boto3 1.42.42
  No Docker, no datasets
```

- Base: Amazon Linux 2023 (`ami-0fcee47b1475c1af3`)
- AMI size: ~3GB
- AMI storage cost: ~$0.15/month
- Purpose: Run `orchestrator.py` only — launch/monitor workers, aggregate results
- Stable — rarely needs rebuilding

#### Worker AMI (Pre-baked)

`ami-0be95bd332a011cc2` (`vectordb-benchmark-worker-v2`)

```
Worker AMI (v2) contains:
  /data/
    sift/         (~500MB)  - SIFT-1M dataset
    gist/         (~4GB)    - GIST-1M dataset
    glove-100/    (~463MB)  - GloVe-100 (HDF5, cosine, 100D)
    dbpedia-openai/ (~5.8GB) - DBpedia-OpenAI-1M (HDF5, cosine, 1536D)
  /app/
    vectordb-benchmark/     - benchmark code (git pull at startup for latest)
  Python packages: h5py 3.15.1
  Docker images (pre-pulled):
    - qdrant/qdrant:latest
    - milvusdb/milvus:latest
    - semitechnologies/weaviate:latest
    - chromadb/chroma:latest
    - redis/redis-stack:latest
    - pgvector/pgvector:pg16
    - portal.dl.kx.com/kdbai-db:latest
```

- AMI size: ~25GB
- AMI storage cost: ~$1.25/month
- Workers launch ready to run (no download wait)

**Not yet included**:
- SIFT-10M (different format - .bvecs)

**Docker Image Updates**:
- Default: Use images baked in AMI (fast)
- Optional: `--pull-latest` or `--pull-latest=qdrant,kdbai` to pull fresh images at startup
- KDB.AI pulls require credentials from S3 (see below)

### S3 Structure

Runs are organized by benchmark type (`competitive`, `kdbai-tuning`, etc.). Reports use self-describing filenames so they make sense when downloaded: `vectordb-benchmark-{type}-{run_id}.html/.db`.

```
s3://vectordb-benchmark-590780615264/
  config/                                                    # Shared credentials
    kc.lic                 # KDB.AI license (encrypted, fetched at startup)
    docker-config.json     # Docker registry credentials for KDB.AI (encrypted)

  reports/                                                   # All reports, flat
    vectordb-benchmark-competitive-2026-02-06-1430.html
    vectordb-benchmark-kdbai-tuning-2026-02-08-1200.html

  runs/                                                      # Raw data, by type
    competitive/
      2026-02-06-1430/
        config.json          # What to run (databases, datasets, timeouts)
        vectordb-benchmark-competitive-2026-02-06-1430.db
        vectordb-benchmark-competitive-2026-02-06-1430.html
        orchestrator.log
        jobs/
          qdrant-sift/
            status.json      # pending|running|completed|failed
            benchmark.db     # Per-job results
            worker.log       # Worker stdout/stderr

    kdbai-tuning/
      2026-02-08-1200/
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

### Cost & Time Summary

**Instances:**

| Role | Type | Specs | Count | When |
|------|------|-------|-------|------|
| Orchestrator | t3.micro | 2 vCPU, 1GB RAM | 1 | During run |
| Worker | m5.4xlarge | 16 vCPU, 64GB RAM | up to 9 | During run |

**Auto-Terminate Timeouts:**

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

**Full Run** (35 jobs, pgvector skips SIFT-10M):
- First results (SIFT-1M): ~30 min
- Total wall clock: ~3-4 hours

**Running Costs** (per full benchmark run):

| Component | Calculation | Cost |
|-----------|-------------|------|
| Orchestrator | 4 hrs × $0.0104/hr | $0.04 |
| Workers | 35 jobs × 0.75 hrs × $0.768/hr | ~$20 |
| S3 transfer | Results only | ~$0.01 |
| **Total per run** | | **~$20** |

**Standby Costs** (monthly, no runs):

| Component | Calculation | Cost |
|-----------|-------------|------|
| Worker AMI (EBS snapshot) | 12GB × $0.05/GB/mo | $0.60 |
| Orchestrator AMI | 3GB × $0.05/GB/mo | $0.15 |
| S3 results storage | ~100MB | ~$0.01 |
| **Total standby** | | **~$0.80/month** |

**No instances running when idle** — all auto-terminate after completion.

### AWS Setup Progress

**AWS Profile**: `vectordb` (SSO)
**Region**: `us-west-2` (Oregon)

**Tagging Convention** (all resources):
| Key | Value |
|-----|-------|
| `Owner` | `acoetzee` |
| `Project` | `vectordb-benchmark` |

**VPC & Networking** (created):
| Resource | Name | ID | Details |
|----------|------|-----|---------|
| VPC | vectordb-benchmark | vpc-04b7bfbbe10306a9d | 10.3.0.0/16 |
| Subnet | vectordb-benchmark-public | subnet-08871c182a622a9e9 | 10.3.1.0/24, us-west-2a |
| Internet Gateway | vectordb-benchmark-igw | igw-0134919a8af2d1869 | Attached to VPC |
| Route Table | vectordb-benchmark-rt | (default for VPC) | 0.0.0.0/0 → IGW |
| Security Group | vectordb-benchmark-sg | sg-0fe6723dd1004558d | SSH (22) restricted to admin IP |

**Other Resources** (created):
- [x] Key pair: `vectordb-benchmark` (stored at `~/.ssh/vectordb-benchmark.pem`)
- [x] S3 bucket: `vectordb-benchmark-590780615264` (us-west-2)
- [x] IAM role: `vectordb-benchmark-role` (EC2 trust, 3 policies attached):
  - `AmazonS3FullAccess` — S3 read/write for results and config
  - `vectordb-benchmark-passrole` — pass IAM role to worker instances
  - `vectordb-benchmark-ec2` — RunInstances, DescribeInstances, DescribeTags, CreateTags
- [x] S3 config files: `config/kc.lic` (KDB.AI license), `config/docker-config.json` (KDB.AI registry creds)

**Completed**:
- [x] Test EC2 launch (verified SSH access works)
- [x] Worker AMI v2: `ami-0be95bd332a011cc2` (`vectordb-benchmark-worker-v2`) — adds h5py, GloVe-100, DBpedia-OpenAI
- [x] Orchestrator AMI v1: `ami-09ed5dd071675cfef` (`vectordb-benchmark-orchestrator-v1`)
- [x] Refactored `SIFTDataset` → `TexmexDataset` (auto-detects dataset name from directory)
- [x] Standardized dataset paths to `data/` (relative paths, symlinked to `/data/` on AWS)
- [x] Created `aws/worker_startup.sh` with auto-termination, S3 result upload, credential fetching
- [x] Created `aws/orchestrator.py` for launching and monitoring worker instances
- [x] Created `aws/orchestrator_startup.sh` with tag-based config, self-tagging
- [x] Launch Template v13 (`vectordb-benchmark-full`) with orchestrator AMI, BenchmarkType + PullLatest tags
- [x] IAM policies: PassRole + EC2 permissions
- [x] Worker auto-termination (trap on exit + scheduled shutdown)
- [x] Orchestrator self-tags with run ID
- [x] Test full orchestrator flow (qdrant/sift minimal test — passed)
- [x] Fix all benchmark code bugs (see bugs fixed below)
- [x] All 9 databases validated on sift-dev locally
- [x] Dev dataset tooling (`scripts/generate_dev_datasets.py`)

**Still TODO**:
- [ ] Run clean full benchmark suite (all 9 DBs × sift + gist)
- [ ] Generate comparison report from results
- [ ] Add SIFT-10M support (`.bvecs` format - needs `read_bvecs()` in data_loader.py)
- [x] Add GloVe-100 + DBpedia-OpenAI support (HDF5 format — h5py + datasets baked into Worker AMI v2)
- [ ] (Optional) Streamlit UI if team usage increases

### Bugs Fixed (from AWS test runs 2026-02-05)

| # | Bug | Affected | Root Cause | Fix |
|---|-----|----------|-----------|-----|
| 1 | Hardcoded `sift_*` filenames | All gist jobs | `run_benchmark.py` checked for `sift_base.fvecs` regardless of dataset | Dynamic `{dataset_name}_*.fvecs` |
| 2 | KDB.AI license env var lost | KDB.AI | `sudo -u` doesn't preserve env vars | `sudo -E -u` in `worker_startup.sh` |
| 3 | Connection failed after container health check | Redis, pgvector, KDB.AI | Health check passes (port open) but app not ready | 8 retries over 45s with exponential backoff |
| 4 | pgvector `vector type not found` | pgvector | `register_vector()` called before `CREATE EXTENSION` | Swapped order in `pgvector_client.py` |
| 5 | Qdrant 193MB payload on GIST | Qdrant + GIST | 50K × 960-dim = 193MB, exceeds 32MB HTTP limit | Auto-calculate batch size based on dimensions |
| 6 | Redis `BusyLoadingError` on startup | Redis | Volume mount loaded old 1GB RDB dump | Removed volume mount (benchmarks start fresh) |
| 7 | Milvus 192MB gRPC payload on GIST | Milvus + GIST | Entire vector array sent in single gRPC call, exceeds 64MB limit | Auto-calculate batch size based on dimensions in `milvus_client.py` |
| 8 | KDB.AI large payload on GIST | KDB.AI + GIST | Same pattern as Milvus/Qdrant | Auto-calculate batch size based on dimensions in `kdbai_client.py` |
| 9 | KDB.AI metadata missing in report | KDB.AI | `"KDB.AI".lower()` → `"kdb.ai"` → looks for `kdb.ai.yaml` not `kdbai.yaml` | Strip dots in config lookup in `report_generator.py` |
| 10 | Qdrant GIST batch size still too large | Qdrant + GIST | Batch sizing used `dims * 4` (binary) but Qdrant sends JSON where floats are ~12 bytes | Changed to `dims * 12 + 200` in `qdrant_client.py` |
| 11 | KDB.AI insert 82x slower than necessary | KDB.AI | `batch_vectors.tolist()` converted numpy to Python lists (millions of float objects) | Use `list(batch_vectors.astype(np.float32))` to pass numpy arrays directly |
| 12 | Qdrant GIST insert still too large (49MB) | Qdrant + GIST | JSON float estimate of 12 bytes/float too low — GIST floats have long decimals (~20 bytes) | Changed to `dims * 20 + 200` in `qdrant_client.py` |
| 13 | Qdrant batch search timeout on SIFT | Qdrant + SIFT | 10,000 FLAT queries sent in single HTTP call, server-side brute-force too slow for default timeout | Cap batch at 500 queries + increase client timeout to 300s |
| 14 | Batch search error kills entire benchmark | All batch DBs | Batch search exception propagates, losing sequential results already collected | Wrap batch search in try/except in `runner.py` |
| 15 | EC2 launch failures silent and unrecoverable | Orchestrator | `launch_worker()` returned `None` on failure, caller skipped job, waited 180m | Retry with exponential backoff (3 attempts), re-launch in wait loop, instance health checks |
| 16 | vCPU limit exceeded launching 45 workers | Tuning run | 45 × 16 vCPU = 720, exceeds account limit | 4 GIST jobs never launched. **TODO**: queue jobs with concurrency limit instead of launching all at once |
| 17 | Docker stop timeout loses results | KDB.AI tuning | `container.stop(timeout=10)` times out → exit code 1 → benchmark.db not uploaded | Worker now uploads results before cleanup; `set +e` around benchmark cmd; cleanup trap won't overwrite status.json |
| 18 | Tuning report averages over different dataset counts | Report generator | `_avg()` averages whatever values exist — configs missing lowest-recall dataset get inflated scores | Impute recall from same-HNSW different-docker; complete-data configs rank above incomplete; "Data" column shows actual/imputed |

### Config Improvements

- **efSearch sweep**: Trimmed from [8,16,32,64,128,256] to [32,64,128,256] — lower values have unusable recall
- **Dev datasets**: 10K vectors / 100 queries for ~12s smoke tests (`--dataset sift-dev`)
- **KDB.AI THREADS**: Set to 16 (was 4) to match available CPU cores per [docs](https://code.kx.com/kdbai/latest/reference/multithreading.html)
- **KDB.AI indexes**: Using `qFlat`/`qHnsw` with `mmapLevel=0` (fully in-memory). Default mmapLevel=1 memory-maps vectors to disk, causing 5-14% recall gap vs competitors. mmapLevel=0 keeps everything in RAM while retaining multi-threaded insert support (THREADS env var) that in-memory `hnsw` lacks. Batch QPS improved 11x on sift-dev.
- **Worker instances**: Upgraded from m5.2xlarge (8 CPU, 32GB) to m5.4xlarge (16 CPU, 64GB) — matches local benchmark config (cpus: 16, memory: 64g)
- **Database notes**: All 9 configs have `metadata.notes` documenting benchmark-relevant quirks (payload limits, index types, protocol caveats)
- **Batch search benchmark**: Added alongside sequential search. 5/9 databases support native batch search APIs (FAISS, Qdrant, Milvus, ChromaDB, KDB.AI). Sends all queries in one API call to measure throughput. P50/P95/P99 latency not available for batch (all queries processed together). Results stored with `HNSW_BATCH`/`FLAT_BATCH` index type suffix.
- **Batch ≠ parallel**: Batch reduces network round trips but server-side parallelism varies. FAISS parallelizes via OpenMP across queries. Qdrant parallelizes since v1.14 (chunked across shards). Milvus parallelizes via Knowhere thread pool. KDB.AI only parallelizes for partitioned tables — single-table batch is sequential server-side. ChromaDB behavior undocumented.
- **Qdrant batch API**: Use `query_batch_points()` with `QueryRequest`, NOT `search_batch()`/`SearchRequest` (deprecated/removed).
- **FAISS threading**: Single-query search is always single-threaded. Only batch search benefits from `omp_set_num_threads()`. Never use concurrent client threads — causes harmful OpenMP thread nesting.
- **LanceDB excluded**: Does not support pure HNSW or FLAT — only IVF-based indexes (IVF_PQ, IVF_HNSW_SQ, IVF_HNSW_PQ). Previous results used IVF_HNSW_SQ disguised as HNSW, producing unfairly low QPS (65 vs FAISS 2,403). Now excluded via `supported_indexes = set()`. Will be re-added when IVF-PQ benchmarks are implemented.
- **Redis persistence**: Corrected from `disk` to `memory`. Redis operates entirely in-memory — RDB snapshots are background-only and don't affect benchmark performance.
- **Skipped job status**: Databases with no supported indexes (LanceDB) now get `status: skipped` instead of `failed`. Worker handles missing results dir gracefully, orchestrator shows skipped separately.
- **pull_run.py**: Single command to pull AWS run results from S3, merge per-job DBs, and generate a named report. Usage: `python scripts/pull_run.py 2026-02-05-1816 [--benchmark-type competitive]`. Output: `results/vectordb-benchmark-{type}-{run_id}.db` + `.html`. All merged runs share a `run_label` column for traceability.
- **Batch search resilience**: Batch search failures are now non-fatal — sequential results are preserved even if batch search times out or errors (bugs #12-14).
- **Auto report in S3**: Orchestrator now generates a merged report after all jobs complete and uploads to `s3://{bucket}/runs/{type}/{run_id}/` plus a copy of the HTML to `s3://{bucket}/reports/` for easy browsing. Startup script installs pyyaml only (result dataclasses extracted to `benchmark/results.py` to break the `db.py` → `runner.py` → numpy/docker import chain).
- **Docker launch commands in report**: The Benchmark Configuration section now includes a "Docker Launch Commands" subsection showing the exact `docker run` command for each client-server database. Built from config YAML `container` sections. Secrets/shell variables are redacted. Embedded databases (FAISS, LanceDB) are skipped.
- **Test environment in report**: Benchmark Configuration starts with a "Test Environment" two-column table (Orchestrator vs Worker) showing role, instance type, vCPUs, memory, count, OS, region. Worker specs (vCPUs, memory, instance type) are dynamic from run data; orchestrator info from `benchmark.yaml` (not in results DB). Auto-detects AWS via `.compute.internal` hostnames; falls back to local hostname display. Repo link for reproducibility. Instance type stored in `runs.instance_type` column via EC2 IMDSv2 (1s timeout, `None` on non-EC2).
- **Qdrant gRPC**: Switched from REST/JSON to gRPC (`prefer_grpc=True`). All operations (upsert, search, batch search) now use binary protobuf (~4 bytes/float instead of ~20 bytes JSON). Eliminates GIST payload size issues and makes benchmark fairer vs Milvus (also gRPC). Batch sizing updated to 50MB limit (gRPC max is 64MB).
- **Chroma efSearch fix**: `collection.modify(metadata={"hnsw:search_ef": ef})` does NOT work in ChromaDB 1.x — only updates metadata dict, not the actual hnswlib index. Correct API: `collection.modify(configuration={"hnsw": {"ef_search": ef}})`. Previous runs had ef stuck at 64 (creation default), producing identical recall at ef=128 and ef=256.
- **KDB.AI recall fix**: Set `mmapLevel=0` on `qHnsw` to force fully in-memory operation. Default mmapLevel=1 memory-maps vectors to disk, degrading recall. Keeps `qHnsw`/`qFlat` (not `hnsw`/`flat`) to retain multi-threaded insert via THREADS env var. Also fixed `get_version()` to return server version (1.8.2) instead of client library version (1.8.0).
- **Qdrant version fix**: Made `get_version()` more robust with HTTP fallback when gRPC-preferred client can't access REST service API. Config `version` changed from `"latest"` to null (auto-detected). Now correctly reports 1.16.3.
- **Client + server versions in report**: Database Configuration table now shows both "Server Version" and "Client Version" columns. All 9 clients implement `get_client_version()` returning their Python SDK version via `importlib.metadata`. Stored in new `db_client_version` column in SQLite `runs` table (backward-compatible migration).
- **Client SDK auto-upgrade on pull-latest**: When `PULL_LATEST` triggers a Docker image pull, the worker now also runs `pip install --upgrade` for the corresponding Python client SDK. Only upgrades the client for the database this worker is running (not all clients). Ensures client/server version compatibility when benchmarking new releases.
- **Orchestrator launch retry**: `launch_worker()` retries transient EC2 errors (`InsufficientInstanceCapacity`, `RequestLimitExceeded`, `Server.InternalError`) up to 3 times with exponential backoff (5s, 15s, 45s + jitter). Non-retryable errors fail immediately. Returns `(instance_id, error)` tuple for tracking.
- **Orchestrator re-launch and health checks**: `wait_for_completion()` re-launches failed jobs within the first 30 minutes (max 2 attempts per job, in case capacity frees up). Every ~2.5 min, calls `describe_instances()` to detect crashed workers (terminated/stopped without uploading `status.json`). Results summary now distinguishes: completed, failed (benchmark error), instance crashed, launch failed (never started), timed out.
- **AWS resource cleanup**: No orphan volumes/ENIs/EIPs from terminated instances (`InstanceInitiatedShutdownBehavior=terminate` + `DeleteOnTermination` on root volumes works correctly). Worker-v1 AMI (`ami-0f9bf04496aedd923`, 50GB, $2.50/mo) can be deregistered — superseded by worker-v2.

**Run 2026-02-06-1750** (KDB.AI tuning: 15 configs × 3 datasets = 45 jobs):
- 37/45 jobs completed. 4 never launched (vCPU limit), 3 docker stop timeout (data exists but not uploaded), 1 worker never reported.
- Report: `results/kdbai-tuning-2026-02-06-1750.html`, DB: `results/vectordb-benchmark-kdbai-tuning-2026-02-06-1750.db`
- **Conclusions invalid** — report averages over different dataset counts. M32_efC128 ranked #1 but only because GloVe-100 (lowest recall) was missing.
- **Preliminary recommendation**: M=32, efC=128, 2wrk_8thr (best balance of recall 0.919, ingest ~4,600 v/s, QPS 172). Runner-up: M=32, efC=200 (+0.5% recall, -15% ingest).
- M=48 data incomplete (all GIST jobs failed) — cannot recommend with confidence yet.
- Needs re-run after fixing bugs #16-18 for clean data.

**Run 2026-02-05-1816** (6 DBs × sift + gist, excludes pgvector + LanceDB):
- 12/16 jobs completed with data. Qdrant empty (bugs #12-14, fixed). LanceDB skipped (expected).
- Report: `results/report-2026-02-05-1816.html`, DB: `results/benchmark-2026-02-05-1816.db`
- KDB.AI ingest 9-11x faster than previous run (numpy fix). HNSW ingest: 7,993 vec/s (fastest client-server).
- Pull with: `python scripts/pull_run.py 2026-02-05-1816`

**First Worker Test Results** (2026-02-04):
- Qdrant on SIFT-1M: Completed successfully
- Results uploaded to S3: `s3://vectordb-benchmark-590780615264/runs/2026-02-04-2117/jobs/qdrant-sift/`
- Instance auto-terminated
- HNSW efSearch=128: 381 QPS, 2.4ms p50 latency, 99.3% recall

**Implementation Notes**:
- Orchestrator AMI has boto3 pre-installed; Worker AMI uses AWS CLI only (no boto3 needed)
- `orchestrator.py` Session handling: tries `profile_name="vectordb"` (local SSO), falls back to default (EC2 IAM role)
- AL2023 requires IMDSv2 tokens for instance metadata (plain curl returns empty)
- User-data does `git pull` then runs local script to avoid GitHub CDN caching issues
- Orchestrator self-tags with `vectordb-orchestrator-{run-id}` after generating run ID
- Workers tagged with Name, Owner, Project, RunId, Database, Dataset at launch
- Dimension-aware batch sizing: Qdrant (64MB gRPC), Milvus (64MB gRPC), KDB.AI (qIPC) each auto-calculate insert batch size based on vector dimensions
- Connection retry: 8 attempts with exponential backoff capped at 10s (total ~45s window)
- KDB.AI performance: `THREADS` env var controls parallelism for qHNSW insert/search. `NUM_WRK` (worker processes) defaults to 1, correct for single-table benchmarks. Rule: `workers × threads <= cores`.

### Triggering Benchmarks

**Two launch templates — one per benchmark type:**

| Template | Purpose | Launches |
|----------|---------|----------|
| `vectordb-benchmark-full` | Competitive benchmark (all DBs) | 7 DBs × 4 datasets = 28 jobs |
| `vectordb-benchmark-kdbai-tuning` | KDB.AI HNSW tuning | 3 datasets × 5 hnsw × 3 docker = 45 jobs |

Both use the same orchestrator AMI and startup script. Fixed values (BenchmarkType, Databases) are baked into each template's UserData. Variable values (Datasets, PullLatest) are instance tags editable at launch.

#### Template 1: Competitive (`vectordb-benchmark-full`)

Launch Template v13 — `lt-...` (existing)

**Fixed in UserData**: benchmark-type=competitive (default)
**Variable tags** (editable at launch):
| Tag | Default | Description |
|-----|---------|-------------|
| `Databases` | `faiss,qdrant,milvus,redis,chroma,weaviate,kdbai` | Remove entries to run a subset |
| `Datasets` | `sift,gist,glove-100,dbpedia-openai` | Remove entries to run a subset |
| `PullLatest` | `all` | `all`, comma-separated list, or delete to skip |

#### Template 2: KDB.AI Tuning (`vectordb-benchmark-kdbai-tuning`)

Launch Template v1 — `lt-09cf10d368856c5ea`

**Fixed in UserData**: `--benchmark-type kdbai-tuning --databases kdbai`
**Variable tags** (editable at launch):
| Tag | Default | Description |
|-----|---------|-------------|
| `Datasets` | `sift,glove-100,gist` | Remove entries for quick test (no dbpedia-openai — OOMs) |
| `PullLatest` | `kdbai` | Pull fresh KDB.AI image, or delete to skip |

#### How to Launch

1. EC2 Console → Launch Templates → pick template → Launch
2. Edit variable tags if needed (optional)
3. Orchestrator launches workers, monitors, generates report
4. All instances auto-terminate on completion
5. Results in `s3://vectordb-benchmark-590780615264/runs/{type}/{run-id}/`

#### CLI (Custom Runs)

```bash
# Ensure AWS SSO is active
aws sso login --profile vectordb

# KDB.AI release comparison (pull fresh image)
python aws/orchestrator.py --databases kdbai,qdrant,milvus --pull-latest kdbai --datasets sift

# KDB.AI tuning via CLI
python aws/orchestrator.py --databases kdbai --datasets sift,glove-100,gist \
    --benchmark-type kdbai-tuning

# Quick single-DB test
python aws/orchestrator.py --databases qdrant --datasets sift

# Full competitive run from CLI (same as launch template)
python aws/orchestrator.py

# Full run, don't wait for completion
python aws/orchestrator.py --no-wait
```

### Next Steps (Priority Order)

1. ~~**Build Orchestrator**~~ ✓
2. ~~**Verify Worker End-to-End**~~ ✓
3. ~~**Create Launch Templates**~~ ✓ — `vectordb-benchmark-full` (competitive, v13) + `vectordb-benchmark-kdbai-tuning` (tuning, v1)
4. ~~**Add IAM Permissions**~~ ✓ — PassRole + EC2 policies
5. ~~**Build Orchestrator AMI**~~ ✓ — `ami-09ed5dd071675cfef`
6. ~~**Update Launch Template**~~ ✓ — Orchestrator AMI + tags + git-based user-data
7. ~~**Fix Worker Name Tags**~~ ✓
8. ~~**Test Full Orchestrator Flow**~~ ✓ — Minimal test passed
9. ~~**Fix Benchmark Code Bugs**~~ ✓ — All 9 bugs fixed, all 9 DBs pass on sift-dev
10. **Run Clean Full Benchmark** — 7 DBs × 4 datasets (28 jobs) via Launch Template v13
11. **Generate Comparison Report** — From S3 results
12. **Later Enhancements** — SIFT-10M (.bvecs), `run_aws.py` CLI, Web UI, per-job cost tracking (capture full instance uptime boot-to-shutdown, not just benchmark runtime — enables accurate cost-per-run reporting from results DB)

### Open Questions
- Should embedded DBs (FAISS, LanceDB) run differently than client-server?
- Spot instances vs on-demand? (spot cheaper but can be interrupted)
- ~~Orphan cleanup: orchestrator should tag workers and terminate on exit~~ — Verified clean: `InstanceInitiatedShutdownBehavior=terminate` + `DeleteOnTermination` handles this. No orphan resources found.

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

- **Texmex (.fvecs/.ivecs)**: Supported — SIFT-1M, GIST-1M
- **Texmex (.bvecs)**: NOT YET SUPPORTED — SIFT-10M uses byte vectors, needs `read_bvecs()` function
- **ANN-Benchmarks (HDF5)**: Supported — GloVe-100, DBpedia-OpenAI-1M (`AnnBenchmarkDataset` class, h5py baked into Worker AMI v2)
- **Cosine distance**: Supported — all 9 clients handle metric from `benchmark.yaml` dataset config

### Dataset Paths

Datasets use relative `data/` paths (configured in `benchmark.yaml`):
- `data/sift/` — SIFT-1M (sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs)
- `data/gist/` — GIST-1M (gist_base.fvecs, gist_query.fvecs, gist_groundtruth.ivecs)
- `data/glove-100/glove-100-angular.hdf5` — GloVe-100 (HDF5, cosine)
- `data/dbpedia-openai/dbpedia-openai-1000k-angular.hdf5` — DBpedia-OpenAI-1M (HDF5, cosine)
- `data/sift-dev/` — Dev subset (10K vectors, 100 queries, recomputed ground truth)
- `data/gist-dev/` — Dev subset (10K vectors, 100 queries, recomputed ground truth)

**Local**: Datasets stored in project `data/` directory. Download HDF5 datasets: `python scripts/download_datasets.py`
**AWS**: Worker startup script creates symlink `data -> /data` (AMI has datasets at `/data/`)
