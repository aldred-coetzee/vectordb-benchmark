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

| Run ID | Datasets | Jobs | Status | Notes |
|--------|----------|------|--------|-------|
| 2026-02-06-1750 | SIFT, GloVe-100, GIST | 37/45 | Partial | 4 vCPU limit, 3 docker timeout, 1 unreported. Report averages invalid (bug #18). |

### Run 2026-02-06-1750 — Detailed Findings

**Report**: `results/kdbai-tuning-2026-02-06-1750.html`
**DB**: `results/vectordb-benchmark-kdbai-tuning-2026-02-06-1750.db`

#### Missing Jobs (8 of 45)

| Category | Count | Jobs | Root Cause |
|----------|-------|------|------------|
| vCPU limit exceeded | 4 | All GIST: M32_efC200-4wrk, M48_efC200-{1wrk,2wrk,4wrk} | 45 × 16 vCPU = 720, hit account limit |
| Docker stop timeout | 3 | SIFT M48_efC200-{2wrk,4wrk}, GloVe M32_efC128-1wrk | Results exist locally but exit code 1 → not uploaded |
| Worker never reported | 1 | GloVe M16_efC64-1wrk | Orchestrator killed at 4hr deadline |

#### Report Averaging Bug

The summary table ranks configs by average Recall@10, but `_avg()` in `generate_tuning_report.py` averages over however many datasets have data. Configs missing the lowest-recall dataset (GloVe-100) get inflated scores:

| Config | GIST | GloVe-100 | SIFT | Reported Avg | True Avg* | Datasets |
|--------|------|-----------|------|-------------|-----------|----------|
| M32_efC128_1wrk_16thr | 0.922 | **missing** | 0.988 | **0.955** | **0.919** | 2 of 3 |
| M48_efC200_1wrk_16thr | **missing** | 0.896 | 0.981 | **0.938** | **?** | 2 of 3 |
| M32_efC200_1wrk_16thr | 0.927 | 0.857 | 0.988 | **0.924** | **0.924** | 3 of 3 |

*\*Imputed from same-HNSW different-docker variants (threading doesn't affect recall).*

#### Preliminary Results (at efSearch=128)

**HNSW params determine recall; threading determines QPS/ingest:**

| HNSW Config | Avg Recall@10 | Avg Ingest (v/s) | Avg QPS (2wrk_8thr) | Data Completeness |
|-------------|--------------|-------------------|---------------------|-------------------|
| M=16, efC=64 | 0.813 | ~5,500 | 164 | 8/9 jobs |
| M=16, efC=200 | 0.847 | ~3,700 | 193 | 9/9 jobs |
| M=32, efC=128 | 0.919 | ~4,600 | 172 | 7/9 jobs |
| M=32, efC=200 | 0.924 | ~3,900 | 166 | 7/9 jobs |
| M=48, efC=200 | 0.938* | ~3,300 | 174 | 4/9 jobs (all GIST missing) |

#### Threading Impact (no effect on recall, affects QPS only)

| Docker Config | Avg QPS (M32 configs) | Notes |
|---------------|----------------------|-------|
| 1wrk_16thr | 145 | Simplest setup |
| **2wrk_8thr** | **172** | **Best QPS (+19%)** |
| 4wrk_4thr | 155 | Middle ground |

#### Preliminary Recommendation

**Best balanced config: M=32, efConstruction=128, 2wrk_8thr**

| Metric | Baseline (M16/efC64) | Recommended (M32/efC128) | Improvement |
|--------|---------------------|--------------------------|-------------|
| Recall@10 | 0.813 | 0.919 | +13% |
| QPS | 164 | 172 | +5% |
| Ingest (v/s) | ~5,500 | ~4,600 | -16% |

Runner-up: **M=32, efC=200** — +0.5% recall for -15% ingest. Worth considering if recall is the overriding priority.

M=48 shows highest recall (0.938) but all GIST data is missing — cannot recommend until re-run provides complete data.

#### vs Competitive Baselines

| Dataset | Target (Qdrant) | Baseline (M16/efC64) | Gap | Tuned (M32/efC128) | Remaining Gap |
|---------|----------------|---------------------|-----|--------------------|----|
| SIFT | 0.996 | 0.946 | -5.0% | 0.988 | **-0.8%** |
| GloVe-100 | 0.904 | 0.702 | -20.2% | 0.848 | **-5.6%** |
| GIST | 0.934 | 0.791 | -14.4% | 0.922 | **-1.3%** |

Tuning closes most of the recall gap. GloVe-100 still has a 5.6% deficit — may need M=48 or different approach for cosine metric.

#### Bugs to Fix Before Re-Run

| # | Bug | Fix Needed |
|---|-----|-----------|
| 16 | vCPU limit launching 45 workers | Add concurrency limit to orchestrator (max ~40 concurrent) |
| 17 | Docker stop timeout loses results | Upload benchmark.db before `container.stop()` |
| 18 | Report averages over different dataset counts | Fix `_avg()` in `generate_tuning_report.py` — impute recall from same-HNSW variants or only average complete configs |
