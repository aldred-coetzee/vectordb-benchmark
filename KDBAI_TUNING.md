# KDB.AI HNSW Parameter Tuning

## Purpose

Find optimal HNSW parameters for KDB.AI across all benchmark datasets. The competitive benchmark uses identical params for all databases (M=16, efConstruction=64). KDB.AI has strong ingest but lower recall and QPS — tuning should close the gap.

## Competitive Baselines (efSearch=128, client-server DBs only)

| Dataset | Metric | Best Recall@10 | KDB.AI Recall | Best QPS | KDB.AI QPS | Best Ingest | KDB.AI Ingest |
|---------|--------|----------------|---------------|----------|------------|-------------|---------------|
| SIFT | L2 128D | 0.9962 (Qdrant) | 0.9456 | 354 (Qdrant) | 243 | **7,277 (KDB.AI)** | 7,277 |
| GloVe-100 | Cos 100D | 0.9036 (Qdrant) | 0.7023 | 318 (Qdrant) | 209 | 8,130 (Qdrant) | 6,774 |
| GIST | L2 960D | 0.9344 (Qdrant) | 0.7905 | 203 (Redis) | 130 | **2,243 (KDB.AI)** | 2,243 |

### Targets

- **Recall@10**: Within 2% of Qdrant at equivalent efSearch
- **QPS**: Not below 50% of current (e.g., SIFT >= 120 at ef=128)
- **Ingest**: Quantify degradation from higher M/efC — acceptable if recall target met

## Parameter Inventory

| Parameter | Type | Current | Sweep | Impact |
|-----------|------|---------|-------|--------|
| M | method | 16 | 16, 32, 48 | recall up, ingest down, memory up |
| efConstruction | method | 64 | 64, 128, 200 | recall up, ingest down |
| efSearch | method | [128, 256] | [128, 256, 512] | recall up, QPS down |
| indexOnly | method | false | true/false | QPS up? (less response payload) |
| mmapLevel | fixed | 0 | 0 | Already validated — 1 degrades recall 5-14% |
| THREADS | docker | 16 | 16, 8, 4 | ingest throughput, search (partitioned only?) |
| NUM_WRK | docker | 1 | 1, 2, 4 | concurrent capacity, memory per worker |
| Partitioning | future | none | — | Could enable multi-threaded search. Separate investigation. |

## How to Run

### Local (development testing)

```bash
python run_benchmark.py --config configs/kdbai.yaml --dataset sift-dev \
    --tuning-config configs/tuning/kdbai-tuning.yaml
```

### AWS (full benchmark)

```bash
# Via Launch Template (recommended):
#   EC2 Console → Launch Templates → vectordb-benchmark-kdbai-tuning → Launch
#   Edit Datasets/PullLatest tags if needed, then launch.

# Via CLI:
python aws/orchestrator.py --databases kdbai --datasets sift,glove-100,gist \
    --benchmark-type kdbai-tuning
```

### Job Structure

Each unique combination of (dataset x hnsw_config x docker_config) = one worker instance.

- 3 datasets x 5 hnsw_configs x 3 docker_configs = 45 workers
- Wall clock: ~20 min (longest: GIST + M48/efC200)
- Cost: 45 x ~20 min x $0.768/hr ~ $11

## Tuning Config

See `configs/tuning/kdbai-tuning.yaml` — the single source of truth for what gets swept.

## Results

Results stored in SQLite with dynamically named index_type, e.g., `HNSW_M32_efC128_wrk1_thr16`.

Report generated with:
```bash
python generate_tuning_report.py --run-id <run-id> --benchmark-type kdbai-tuning
```

## Results Log

| Run ID | Datasets | Notes |
|--------|----------|-------|
| (pending) | | |
