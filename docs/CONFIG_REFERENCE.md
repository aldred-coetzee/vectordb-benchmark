# Configuration Reference

Configuration is split across two layers:

1. **Competitive benchmarks** use `benchmark.yaml` (global settings) + `configs/<db>.yaml` (per-database). HNSW parameters are fixed — all databases use the same M/efConstruction for fair comparison.

2. **KDB.AI tuning benchmarks** additionally use `configs/tuning/kdbai-tuning.yaml`, which defines the HNSW parameter sweep (multiple M/efConstruction combinations) and Docker threading configs to test.

## Global Benchmark Config (`benchmark.yaml`)

Controls datasets, index parameters, search settings, and exclusions shared across all databases. Used by both benchmark types.

### Datasets

```yaml
datasets:
  sift:
    path: data/sift                    # Relative to project root
    vectors: 1000000
    dimensions: 128
    metric: L2                         # L2 or cosine
    format: fvecs                      # fvecs or hdf5
```

| Dataset | Vectors | Dims | Queries | Metric | Format |
|---------|---------|------|---------|--------|--------|
| `sift` | 1M | 128 | 10K | L2 | fvecs |
| `gist` | 1M | 960 | 1K | L2 | fvecs |
| `glove-100` | 1.18M | 100 | 10K | cosine | hdf5 |
| `dbpedia-openai` | 990K | 1536 | 10K | cosine | hdf5 |
| `sift-dev` | 10K | 128 | 100 | L2 | fvecs |
| `gist-dev` | 10K | 960 | 100 | L2 | fvecs |

### Index Parameters

```yaml
indexes:
  hnsw:
    M: 16                              # Graph connectivity (edges per node)
    efConstruction: 64                 # Build-time beam width
    efSearch: [128, 256, 512]          # Search-time beam widths to sweep
```

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `M` | 16 | 4-64 | Higher = better recall, more memory, slower ingest |
| `efConstruction` | 64 | 16-500 | Higher = better graph quality, slower ingest |
| `efSearch` | [128,256,512] | Must be >= k | Higher = better recall, lower QPS |

**Recommended**: M=16, efC=64 for competitive benchmarks. For KDB.AI tuning, M=32 with efC=128 provides the best recall/ingest tradeoff (see [FINDINGS.md](FINDINGS.md)).

### Search Settings

```yaml
search:
  k: 10                               # Number of nearest neighbors
  num_queries: 10000                   # Max queries to run
  warmup: 100                          # Minimum warmup queries (scaled up automatically)
  batch_size: 50000                    # Max queries per batch search call
```

### Exclusions and Caveats

```yaml
exclusions:
  - database: milvus
    dataset: gist
    reason: "Container crashed (OOM) during search"

caveats:
  - database: kdbai
    dataset: all
    note: "qHnsw with mmapLevel=0 for dims <= 960"
```

Exclusions prevent jobs from launching. Caveats are informational, shown in reports.

## Database Config (`configs/<database>.yaml`)

Each database has a YAML config controlling container setup, connection, and metadata.

### Structure

```yaml
database:
  name: qdrant                         # Used by get_client() for dispatch
  version: "1.16.3"                    # null for auto-detect
  endpoint: localhost:6333

container:
  image: qdrant/qdrant:latest
  cpus: 16
  memory: 64g
  ports: ["6333:6333", "6334:6334"]
  env:                                 # Environment variables
    QDRANT__SERVICE__GRPC_PORT: "6334"
  volumes:                             # Host:container mount paths
    - ./data/qdrant:/qdrant/storage
  setup:
    directories:                       # Created before container start
      - path: ./data/qdrant
        mode: "0777"

health_check:
  url: http://localhost:6333/healthz
  timeout: 60                          # Seconds to wait for ready
  interval: 2                          # Seconds between checks

metadata:                              # Shown in reports
  full_name: "Qdrant"
  architecture: client-server
  protocol: gRPC
  persistence: disk
  license: Apache-2.0
  url: https://qdrant.tech
  notes: "..."                         # Benchmark-relevant quirks

params:                                # Database-specific parameters
  timeout: 30
```

### Database-Specific Parameters

#### KDB.AI (`configs/kdbai.yaml`)

```yaml
container:
  env:
    KDB_LICENSE_B64: "${KDB_LICENSE_B64}"  # Base64-encoded license
    NUM_WRK: "2"                       # Worker processes
    THREADS: "8"                       # Threads per worker
    VDB_DIR: "/tmp/kx/data/vdb"
```

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `NUM_WRK` | 2 | Worker processes. `workers × threads <= cores` |
| `THREADS` | 8 | Threads per worker for qHnsw insert/search |
| `mmapLevel` | 0 (auto) | Set in `kdbai_client.py`, not config. 0=in-memory, 1=mmap |

**Important**: `mmapLevel` is set programmatically in `kdbai_client.py:create_table()`:
- dims <= 960: `mmapLevel=0` (fully in-memory, best recall)
- dims > 960: `mmapLevel=1` (memory-mapped, avoids OOM)
- Only applies to `qHnsw`, NOT `qFlat`

#### KDB.AI FAISS Variant (`configs/kdbai-faiss.yaml`)

Separate config using FAISS-based `flat`/`hnsw` indexes instead of q-language `qFlat`/`qHnsw`. Runs on different ports to allow side-by-side comparison.

```yaml
container:
  ports: ["8092:8082", "8091:8081"]    # Different host ports
  env:
    NUM_WRK: "1"                       # FAISS hnsw is single-threaded
    THREADS: "16"
  volumes:
    - ./data/kdbai-faiss:/tmp/kx/data  # Separate data directory
```

#### Qdrant (`configs/qdrant.yaml`)

Uses gRPC (`prefer_grpc=True`). Batch sizing: `dims * 4 + 100` bytes per vector, 50MB limit (gRPC max 64MB).

#### Milvus (`configs/milvus.yaml`)

gRPC protocol. Batch sizing: `dims * 4` bytes per vector. Excluded from GIST and DBpedia-OpenAI (OOM/disk issues).

#### Redis (`configs/redis.yaml`)

No volume mounts — benchmarks start fresh each run. Previous volume mounts caused `BusyLoadingError` from stale RDB dumps.

## Tuning Config (`configs/tuning/kdbai-tuning.yaml`)

Controls KDB.AI-specific HNSW parameter sweeps.

```yaml
method_params:
  hnsw_configs:
    - name: M16_efC64
      M: 16
      efConstruction: 64
    - name: M16_efC200
      M: 16
      efConstruction: 200
    - name: M32_efC128
      M: 32
      efConstruction: 128
    - name: M32_efC200
      M: 32
      efConstruction: 200
    - name: M48_efC200
      M: 48
      efConstruction: 200

  efSearch_values: [128, 256, 512]

  search_options:
    indexOnly: [false, true]           # Test with and without post-filtering

docker_params:
  configs:
    - name: 1wrk_16thr
      env:
        NUM_WRK: "1"
        THREADS: "16"
    - name: 2wrk_8thr
      env:
        NUM_WRK: "2"
        THREADS: "8"
    - name: 4wrk_4thr
      env:
        NUM_WRK: "4"
        THREADS: "4"
```

`method_params` are swept within a single container run. `docker_params` require a container restart (different threading config), so on AWS each docker config gets its own worker instance.

Total jobs: 3 datasets x 5 HNSW configs x 3 docker configs = 45

## CLI Arguments

### `run_benchmark.py`

```
Required:
  --config PATH            Database config YAML (e.g., configs/qdrant.yaml)

Dataset:
  --dataset NAME           Dataset from benchmark.yaml (default: sift)

HNSW overrides:
  --hnsw-m N               Override M parameter
  --hnsw-efc N             Override efConstruction
  --ef-search N [N ...]    Override efSearch values

Tuning:
  --tuning-config PATH     Tuning config YAML
  --docker-config-name STR Docker config name for index_type suffix

Docker:
  --no-docker              Skip Docker management (external container)
  --pull-latest            Pull fresh Docker image before starting
  --cold                   Restart container between efSearch values

Output:
  --output DIR             Output directory (default: results)
```

### `aws/orchestrator.py`

```
  --databases LIST         Comma-separated (default: all 8)
  --datasets LIST          Comma-separated (default: sift,gist,glove-100,dbpedia-openai)
  --benchmark-type TYPE    competitive or kdbai-tuning
  --pull-latest LIST       Docker images to pull fresh
  --no-wait                Launch workers and exit without monitoring
```

### `scripts/pull_run.py`

```
  RUN_ID                   Run ID (e.g., 2026-02-07-0010)
  --benchmark-type TYPE    competitive or kdbai-tuning (default: competitive)
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — code structure and data flow
- [METHODOLOGY.md](METHODOLOGY.md) — what we measure, how, known caveats
- [FINDINGS.md](FINDINGS.md) — benchmark results and recommended configs
- [RUNBOOK.md](RUNBOOK.md) — step-by-step instructions to run benchmarks
