# Methodology

## What We Measure

### Ingest Throughput (vectors/sec)

Wall-clock time to insert all vectors into a table with index. For HNSW, each insert updates the graph, so throughput decreases as the dataset grows. Measured end-to-end including batch serialization overhead.

**Implementation**: `runner.py:run_ingest_benchmark()` (~line 95). Vectors inserted in batches sized per-client based on dimension and protocol overhead.

### Queries Per Second (QPS)

Sequential single-query throughput. One query at a time, wait for response, measure total wall-clock time for all queries. `QPS = num_queries / total_time`.

Not concurrent load — measures single-client latency-bound throughput.

**Implementation**: `runner.py:run_search_benchmark()` (~line 167). Outer timer wraps entire query loop.

### Latency Percentiles (P50, P95, P99)

Per-query latency measured with `time.perf_counter()` inside each client's `search()` method. Captures round-trip time including serialization, network, and deserialization.

**Implementation**: Each client wraps its search call with perf_counter. `runner.py` collects all latencies and computes percentiles via `metrics.py:calculate_latency_percentiles()`.

### Recall@k

Fraction of true k nearest neighbors returned by the search. Computed against pre-computed ground truth shipped with each dataset.

```
Recall@10 = |returned ∩ true_top_10| / 10
```

**Implementation**: `metrics.py:calculate_recall()`. Compares retrieved IDs against ground truth for each query, averages across all queries.

### Batch QPS

All queries sent in a single API call. Measures throughput under batch workloads. Available for 5/9 databases: FAISS, Qdrant, Milvus, ChromaDB, KDB.AI. Per-query latency percentiles are not available for batch (queries processed together).

**Implementation**: `runner.py:run_batch_search_benchmark()` (~line 310). Single timer wraps one `client.batch_search()` call.

## How We Measure

### Test Environment and Run Isolation

All benchmarks run on identical AWS `m5.4xlarge` instances (16 vCPU, 64 GB RAM). Docker containers are configured with 16 CPUs and 64 GB memory.

**Isolation model**: Each benchmark job runs on a dedicated EC2 instance. The database container and benchmark process are the only workloads on the machine — no resource contention between databases or between jobs. This ensures one database's memory pressure, CPU usage, or background tasks cannot affect another's measurements.

**Parallelization**: All jobs launch simultaneously (subject to account vCPU limits). For competitive runs, this means up to 28 instances running in parallel (7 databases x 4 datasets). For tuning runs, up to 9 instances (3 datasets x 3 docker configs). Workers are fully independent — no cross-worker communication. The orchestrator monitors progress by polling S3 status files.

**Local runs** do not have this isolation. `run_all.py` runs databases sequentially on the same machine, so system state (OS page cache, memory fragmentation) may carry across databases. For comparable measurements, use AWS or restart between databases.

### Two Benchmark Flows

There are two separate benchmark flows with different purposes:

1. **Competitive** (`run_full_benchmark()`): Compares all databases on fixed HNSW parameters (M=16, efC=64). Tests both FLAT and HNSW indexes, includes batch search. One job per (database, dataset) pair.

2. **KDB.AI Tuning** (`run_tuning_benchmark()`): Sweeps HNSW build parameters (M, efConstruction) and Docker threading configs for KDB.AI only. HNSW only — no FLAT, no batch search. One job per (dataset, HNSW config, docker config) triple.

Both flows share `run_search_benchmark()` for the actual query measurement. The key difference is what varies: competitive holds HNSW params constant and varies the database; tuning holds the database constant and varies HNSW params.

### Competitive Benchmark Sequence

For each database × dataset combination (`runner.py:run_full_benchmark()`, ~line 390):

```
1. Start Docker container, wait for health check
2. FLAT index:
   a. Create table with flat index
   b. Insert all vectors → measure ingest throughput
   c. Run 10,000 queries → measure QPS, latency, recall (baseline)
   d. Run batch search (if supported)
3. HNSW index:
   a. Create table with HNSW index (M=16, efConstruction=64)
   b. Insert all vectors → measure ingest throughput
   c. Cache warming: 1,000 untimed queries
   d. For efSearch in [128, 256, 512]:
      - Per-ef warmup: max(100, num_queries/10) untimed queries
      - Run all queries → measure QPS, latency, recall
   e. Batch search for each efSearch (if supported)
4. Drop tables, stop container
```

### Tuning Benchmark Sequence

For each (HNSW config, docker config) pair on a single dataset (`runner.py:run_tuning_benchmark()`, ~line 562):

```
1. Start Docker container with docker config (NUM_WRK, THREADS)
2. For each HNSW config (M, efConstruction):
   a. Create table with HNSW index using this M/efConstruction
   b. Insert all vectors → measure ingest throughput
   c. Cache warming: 1,000 untimed queries
   d. For each efSearch in [128, 256, 512]:
      - For each indexOnly in [false, true]:
        - Per-ef warmup → timed queries → QPS, latency, recall
   e. Drop table
3. Stop container
```

The tuning flow also tests `indexOnly` (a KDB.AI search option that skips post-filtering) at each efSearch point, doubling the search configurations measured.

### Warmup Strategy

Two levels of warmup to reduce cache effects:

1. **Pre-sweep warming** (`runner.py:_warm_cache()`, ~line 147): 1,000 untimed queries before the first efSearch value. Brings HNSW graph into OS page cache so the first efSearch point isn't penalized by cold-cache overhead.

2. **Per-efSearch warming** (`runner.py`, ~line 202): `max(config_warmup, num_queries/10)` queries before each timed measurement. For SIFT (10K queries): 1,000 warmup. For GIST (1K queries): 100 warmup.

### Query Padding for Small Datasets

GIST has only 1,000 queries — too few for stable QPS measurement (~5 seconds at typical throughput). Queries are padded to 5,000 by repeating and shuffling with a fixed seed (`rng(42)`). Recall is computed on original queries only; QPS includes all padded queries.

**Implementation**: `runner.py`, ~line 208-221.

### Cold Restart Mode (`--cold`)

Optional flag that restarts the database container between each efSearch value, providing true cache-isolated measurements at the cost of much longer runtime. Not used by default.

**Implementation**: `runner.py:_restart_container()` (~line 156).

## Known Caveats

### QPS Variance (±20-40%)

QPS measurements show significant run-to-run variance, particularly for KDB.AI. Root causes:

1. **Server-side stalls**: Some queries experience 10-100x higher latency (visible in P50 vs total-time discrepancy). Likely caused by server-side GC, buffer management, or background compaction. Not addressable from the client side.

2. **Cache effects between efSearch sweeps**: Despite pre-sweep warming, different efSearch values access different graph neighborhoods. ef=128 touches fewer nodes than ef=512, so subsequent sweeps may benefit from cached graph edges.

3. **Small query set for GIST**: Only 1,000 native queries (padded to 5,000). Addressed with padding but variance remains higher than SIFT/GloVe (10K queries).

**Impact**: QPS rankings between databases are reliable (order-of-magnitude differences). QPS differences under 2x between databases should not be considered significant.

### KDB.AI mmapLevel

KDB.AI `qHnsw` supports `mmapLevel` controlling whether vectors are memory-mapped from disk (1) or fully in-memory (0).

- **mmapLevel=0** (current default for dims <= 960): Best recall, highest memory usage
- **mmapLevel=1** (used for dims > 960): Reduces memory but degrades recall by 1-2% on SIFT, negligible on GIST/GloVe

The `mmapLevel=0` fix improved SIFT recall from 0.970 to 0.982 at ef=256 (tuning run comparison). GIST and GloVe showed minimal change.

**Note**: `qFlat` does NOT support mmapLevel (causes "invalid arguments" error). Only `qHnsw` accepts it. See `kdbai_client.py:create_table()` (~line 143-153).

### ChromaDB efSearch Freeze

ChromaDB's `collection.modify(configuration={"hnsw": {"ef_search": N}})` API does not reliably change the underlying hnswlib search parameter. In the latest competitive run (2026-02-07), recall was identical at ef=128, 256, and 512 across all 4 datasets — confirming efSearch is frozen at the creation-time default (64).

The legacy API `collection.modify(metadata={"hnsw:search_ef": N})` only updates metadata, not the actual index. This is a known ChromaDB limitation.

**Impact**: ChromaDB results represent ef=64 performance regardless of the configured sweep values.

### KDB.AI qHnsw vs FAISS hnsw

KDB.AI offers two HNSW implementations:
- **qHnsw** (q-language): supports mmapLevel, multi-threaded insert via THREADS env var, higher QPS
- **hnsw** (FAISS-based): always in-memory, single-threaded insert, often higher recall on cosine

We benchmark both as "KDB.AI" (qHnsw) and "KDB.AI (FAISS)" (hnsw). On GloVe-100 (cosine), the FAISS variant shows 8 percentage points higher recall (0.852 vs 0.771 at ef=256), suggesting qHnsw may handle cosine distance differently.

### Batch Search Semantics

"Batch" means all queries in one API call — not parallel execution. Server-side parallelism varies:
- **FAISS**: Parallelizes via OpenMP across queries
- **Qdrant**: Parallelizes across shards (since v1.14)
- **Milvus**: Parallelizes via Knowhere thread pool
- **KDB.AI**: Sequential for single-table; only partitioned tables parallelize
- **ChromaDB**: Undocumented

### Recall Is Deterministic, QPS Is Not

For identical HNSW parameters (M, efConstruction, efSearch), recall is deterministic — same index structure produces same results. Verified: tuning runs with different docker configs (1wrk_16thr, 2wrk_8thr, 4wrk_4thr) produce identical recall.

QPS varies with threading config, system load, and cache state. Always compare recall across configs; treat QPS as indicative.

### Excluded Databases/Datasets

| Database | Dataset | Reason |
|----------|---------|--------|
| Milvus | GIST | OOM during search (30.6GB peak memory) |
| Milvus | DBpedia-OpenAI | Container disk space exhaustion during search |
| LanceDB | All | Only supports IVF-based indexes, no pure HNSW/FLAT |
| KDB.AI (FAISS) | DBpedia-OpenAI | OOM (1536D fully in-memory) |
