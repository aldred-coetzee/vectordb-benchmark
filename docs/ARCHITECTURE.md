# Architecture

## Overview

This tool benchmarks 9 vector databases on standardized datasets, measuring ingest speed, query throughput, latency, and recall accuracy. It runs locally via Docker or at scale on AWS.

```
run_benchmark.py          CLI entry point (single DB)
run_all.py                CLI entry point (all DBs sequentially)
aws/orchestrator.py       AWS entry point (parallel workers)

benchmark/
├── runner.py             Orchestrates ingest → search → metrics pipeline
├── clients/              Database-specific implementations
│   ├── base.py           Abstract interface (BaseVectorDBClient)
│   ├── faiss_client.py   Embedded — no container
│   ├── qdrant_client.py  gRPC, prefer_grpc=True
│   ├── milvus_client.py  gRPC via pymilvus
│   ├── chroma_client.py  HTTP REST
│   ├── weaviate_client.py  gRPC
│   ├── pgvector_client.py  PostgreSQL wire protocol
│   ├── redis_client.py   Redis protocol (RESP)
│   ├── kdbai_client.py   qIPC binary protocol
│   └── lancedb_client.py Embedded (excluded — IVF only)
├── data_loader.py        Dataset loading (.fvecs, .ivecs, HDF5)
├── metrics.py            Recall@k calculation
├── db.py                 SQLite result storage
├── results.py            Result dataclasses (no heavy imports)
├── config.py             YAML config parsing
├── docker_manager.py     Container lifecycle (start/stop/health)
├── docker_monitor.py     Container resource monitoring (CPU/mem)
└── report_generator.py   HTML report from SQLite data

configs/                  Per-database YAML configs
configs/tuning/           KDB.AI tuning sweep configs
scripts/                  Utilities (pull_run, generate_dev_datasets, download_datasets)
aws/                      Orchestrator, worker startup, orchestrator startup
results/                  Output (.db, .html, .csv)
data/                     Datasets (symlinked to /data on AWS)
```

## Two Benchmark Types

The tool supports two distinct benchmark flows, each with its own purpose, job matrix, and report format:

### Competitive Benchmark

Compares all databases against each other on the same HNSW parameters (M=16, efConstruction=64). Used to rank databases on recall, QPS, latency, and ingest speed.

- **Scope**: 7+ databases x 4 datasets = 28+ jobs
- **Parameters**: Fixed M and efConstruction (from `benchmark.yaml`), sweep efSearch
- **Runner method**: `run_full_benchmark()` — tests both FLAT and HNSW indexes, includes batch search
- **Report**: `ComparisonReportGenerator` — cross-database tables, per-dataset breakdowns
- **AWS template**: `vectordb-benchmark-full`

### KDB.AI Tuning Benchmark

Sweeps HNSW build parameters (M, efConstruction) and Docker threading configs (NUM_WRK, THREADS) for KDB.AI only. Used to find optimal configuration for competitive runs.

- **Scope**: 1 database x 3 datasets x 5 HNSW configs x 3 docker configs = 45 jobs
- **Parameters**: Multiple M/efConstruction combinations (from `configs/tuning/kdbai-tuning.yaml`)
- **Runner method**: `run_tuning_benchmark()` — HNSW only, no FLAT, no batch search
- **Report**: `generate_tuning_report.py` — parameter impact tables, config rankings
- **AWS template**: `vectordb-benchmark-kdbai-tuning`

Both flows share the same core infrastructure: `run_benchmark.py` entry point, `BenchmarkRunner`, database clients, Docker management, and SQLite storage. The `--benchmark-type` flag and `--tuning-config` argument control which flow executes.

## Data Flow

### 1. Entry Point (`run_benchmark.py`)

```
CLI args → load config YAML → get_client(database_name) → create BenchmarkRunner
         → DockerManager.start_container() → wait_for_ready()
         → client.connect() → runner.run_full_benchmark()
         → BenchmarkDatabase.save() → DockerManager.stop_container()
```

Key function: `get_client()` (~line 770) maps database names to client classes. Each client is instantiated with database-specific parameters.

### 2. Runner (`benchmark/runner.py`)

`BenchmarkRunner` is the core orchestrator. It holds a client, dataset, and monitor.

**`run_full_benchmark()`** (~line 390):
```
For each index type [flat, hnsw]:
  1. run_ingest_benchmark() → create table, insert vectors, measure throughput
  2. For each efSearch value [128, 256, 512]:
     - _warm_cache() → 1000 untimed queries (pre-sweep)
     - run_search_benchmark() → timed queries, collect latencies + IDs
  3. For each efSearch (if has_batch_search):
     - run_batch_search_benchmark() → all queries in one API call
  4. drop_table()
```

**`run_tuning_benchmark()`** (~line 562):
```
For each HNSW config (M, efConstruction):
  1. Ingest with this config
  2. For each efSearch × indexOnly combination:
     - run_search_benchmark()
  3. Drop table, repeat with next config
```

**`run_search_benchmark()`** (~line 167):
```
1. Scaled warmup: max(config_warmup, num_queries/10) queries
2. Query padding: if < 5000 queries, pad with shuffled repeats
3. Time all queries: record per-query latency + retrieved IDs
4. Compute: QPS, latency percentiles (p50/p95/p99)
5. Compute: Recall@10 and Recall@100 vs ground truth
```

### 3. Client Interface (`benchmark/clients/base.py`)

All 9 clients implement `BaseVectorDBClient`:

```python
class BaseVectorDBClient(ABC):
    name: str                          # Display name (e.g., "KDB.AI")
    get_version() -> str               # Server version
    get_client_version() -> str        # Python SDK version
    connect(endpoint, **kwargs)        # Connect to database
    disconnect()                       # Clean disconnect
    create_table(name, dimension, IndexConfig)  # Create with index
    drop_table(name)                   # Drop if exists
    insert(name, ids, vectors)         # Batch insert (numpy arrays)
    search(name, query, k, SearchConfig) -> SearchResult  # Single query
    has_batch_search: bool             # Whether batch API exists
    batch_search(name, queries, k, SearchConfig) -> List[SearchResult]
    get_stats(name) -> Dict            # Row count, etc.
```

Each client handles its own:
- Wire protocol and serialization
- Batch size calculation based on dimensions and payload limits
- Metric mapping (L2/cosine → database-specific enum)
- Connection retry with exponential backoff

### 4. Result Storage (`benchmark/db.py`)

SQLite database with three tables:

```sql
runs:            run_id, database, dataset, dimensions, config_json, hostname, instance_type, ...
ingest_results:  run_id, index_type, throughput_vps, total_time_s, peak_memory_gb
search_results:  run_id, index_type, ef_search, qps, recall_at_10, recall_at_100,
                 p50_ms, p95_ms, p99_ms, num_queries
```

Each benchmark run produces one `.db` file. AWS runs produce per-job DBs that `scripts/pull_run.py` merges by remapping run_ids.

### 5. Report Generation (`benchmark/report_generator.py`)

`ReportGenerator` reads SQLite, loads config YAMLs for metadata, renders HTML:
- Per-dataset comparison tables (recall, QPS, latency, ingest)
- Recall vs QPS scatter plots
- Database configuration details
- Docker launch commands
- Test environment specs (auto-detects AWS via hostname)

`ComparisonReportGenerator` wraps `ReportGenerator` (composition, not inheritance) for multi-dataset reports with cross-dataset summary tables.

### 6. Docker Management (`benchmark/docker_manager.py`)

`DockerManager` handles the container lifecycle:
```
start_container() → pull image (optional) → create container → start
                  → apply_env_overrides() (THREADS, NUM_WRK from env)
wait_for_ready()  → health check URL with configurable timeout/interval
                  → connection retry: 8 attempts, exponential backoff, 45s total
stop_container()  → stop with timeout → remove container
```

`DockerMonitor` (`docker_monitor.py`) samples container stats at configurable intervals during benchmarks, reporting CPU%, memory usage, and peak values.

### 7. AWS Architecture (`aws/`)

```
Orchestrator (t3.small)              Workers (m5.4xlarge × N)
┌─────────────────────┐              ┌───────────────────┐
│ orchestrator.py     │──launches──→ │ worker_startup.sh │
│ - Build job matrix  │              │ - git pull        │
│ - Launch workers    │              │ - Start container │
│ - Monitor via S3    │              │ - Run benchmark   │
│ - Merge results     │              │ - Upload to S3    │
│ - Generate report   │              │ - Auto-terminate  │
└─────────────────────┘              └───────────────────┘
         │                                    │
         └──────── S3 Bucket ─────────────────┘
```

**Job model**: One EC2 instance per (database, dataset) pair. Each worker runs a single benchmark, uploads results to S3, and self-terminates.

**Orchestrator flow** (`aws/orchestrator.py`):
1. Parse tags from EC2 instance metadata (Databases, Datasets, PullLatest)
2. Build job matrix, exclude known failures (Milvus + GIST/DBpedia)
3. Launch workers with retry (3 attempts, exponential backoff)
4. Poll S3 for `status.json` every ~2.5 min, check instance health
5. Re-launch failed jobs within first 30 min
6. Merge all per-job DBs, generate report, upload to S3
7. Self-terminate

**Worker flow** (`aws/worker_startup.sh`):
1. Set trap for cleanup on exit (upload log, set status, shutdown)
2. Fetch credentials from S3 (KDB.AI license, Docker registry)
3. `git pull origin main` for latest code
4. Optionally pull fresh Docker images + upgrade client SDK
5. Run `python run_benchmark.py` with appropriate flags
6. Upload results to S3 regardless of exit code
7. Auto-terminate via `shutdown -h now`

## Key Design Decisions

- **Composition over inheritance**: `ComparisonReportGenerator` wraps `ReportGenerator` as `self._rg`
- **Numpy arrays everywhere**: Clients receive numpy arrays directly (not Python lists) for performance
- **Dimension-aware batching**: Each client calculates insert batch size based on `dims × bytes_per_element` to stay within protocol limits
- **Non-fatal batch search**: Batch search failures are caught and don't lose sequential results
- **Results before cleanup**: Worker uploads results to S3 before stopping containers (docker stop can timeout)
- **`results.py` decoupled from `runner.py`**: Dataclasses in separate module to avoid importing numpy/docker on orchestrator
