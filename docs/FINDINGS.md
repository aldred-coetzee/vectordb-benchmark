# Findings

All benchmarks ran on AWS m5.4xlarge (16 vCPU, 64 GB), Docker containers with 16 CPUs / 64 GB.

This document covers results from two separate benchmark flows:

1. **Competitive benchmark** (run 2026-02-07-0010): 8 databases compared head-to-head on fixed HNSW parameters (M=16, efC=64) across 4 datasets. Answers: "How does KDB.AI compare to competitors?"

2. **KDB.AI tuning benchmark** (run 2026-02-07-0915): 45 KDB.AI configurations (5 HNSW param sets x 3 docker threading configs x 3 datasets). Answers: "What KDB.AI HNSW parameters maximize recall and throughput?"

## Competitive Results (M=16, efC=64)

### Overall Rankings (HNSW ef=256, averaged across available datasets)

**Recall@10** (higher is better):

| # | Database | Avg Recall | Datasets |
|---|----------|-----------|----------|
| 1 | Qdrant | 0.975 | 4/4 |
| 2 | Milvus | 0.938 | 2/4 |
| 3 | FAISS | 0.920 | 4/4 |
| 4 | Redis | 0.916 | 4/4 |
| 5 | Weaviate | 0.913 | 4/4 |
| 6 | KDB.AI | 0.893 | 4/4 |
| 7 | Chroma | 0.858 | 4/4 |

**HNSW Ingest Throughput** (vectors/sec, higher is better):

| # | Database | Avg Ingest |
|---|----------|-----------|
| 1 | FAISS | 29,959 |
| 2 | KDB.AI | 4,938 |
| 3 | Milvus | 4,639 |
| 4 | Qdrant | 4,122 |
| 5 | Chroma | 2,820 |
| 6 | Redis | 2,805 |
| 7 | Weaviate | 2,010 |

**Note**: Chroma recall is frozen due to an efSearch bug (see [METHODOLOGY.md](METHODOLOGY.md#chromadb-efsearch-freeze)). Actual ef=256 recall would be higher. Milvus averages are over 2 datasets only (excluded from GIST and DBpedia-OpenAI).

### Per-Dataset Recall@10 (HNSW ef=256)

| Database | SIFT | GIST | GloVe-100 | DBpedia |
|----------|------|------|-----------|---------|
| Qdrant | 0.999 | 0.966 | 0.943 | 0.992 |
| FAISS | 0.993 | 0.862 | 0.849 | 0.975 |
| KDB.AI (FAISS) | 0.993 | 0.862 | 0.852 | -- |
| Milvus | -- | -- | 0.932 | -- |
| Redis | 0.972 | 0.867 | 0.831 | 0.993 |
| Weaviate | 0.973 | 0.854 | 0.821 | 1.000 |
| KDB.AI | 0.982 | 0.867 | 0.771 | 0.954 |
| Chroma | 0.969 | 0.747 | 0.767 | 0.951 |

### Per-Dataset QPS (HNSW ef=256)

| Database | SIFT | GIST | GloVe-100 | DBpedia |
|----------|------|------|-----------|---------|
| FAISS | 1,434 | 513 | 1,386 | 452 |
| Chroma | 426 | 282 | 427 | 255 |
| Qdrant | 330 | 168 | 295 | 155 |
| KDB.AI | 261 | 148 | 200 | 114 |
| Milvus | 431 | -- | 235 | -- |
| Redis | 254 | 138 | 220 | 78 |
| Weaviate | 214 | 170 | 147 | 170 |
| KDB.AI (FAISS) | 100 | 68 | 92 | -- |

## KDB.AI-Specific Findings

### 1. Recall Gap on Cosine Datasets

KDB.AI (qHnsw) ranks 6th/7th on recall overall. The gap is largest on cosine-metric datasets:

- **GloVe-100 (cosine)**: KDB.AI recall=0.771 vs Qdrant 0.943 (gap: 0.172)
- **SIFT (L2)**: KDB.AI recall=0.982 vs Qdrant 0.999 (gap: 0.017)

The FAISS-based `hnsw` variant achieves recall=0.852 on GloVe-100 (8 points higher than qHnsw), suggesting `qHnsw`'s cosine distance implementation may be less optimal.

### 2. Ingest Speed Is a Strength

KDB.AI ranks 2nd in HNSW ingest throughput (4,938 v/s avg), behind only FAISS (embedded, 29,959 v/s). Among client-server databases, KDB.AI is fastest for ingest.

### 3. mmapLevel Impact

Switching from `mmapLevel=1` (memory-mapped) to `mmapLevel=0` (fully in-memory) for `qHnsw`:

| Dataset | Old Recall (mmap=1) | New Recall (mmap=0) | Delta |
|---------|--------------------|--------------------|-------|
| SIFT | 0.970 | 0.982 | +0.012 |
| GIST | 0.868 | 0.867 | -0.001 |
| GloVe-100 | 0.772 | 0.771 | -0.001 |

SIFT benefited significantly (+1.2%). GIST and GloVe showed no meaningful change, suggesting the recall gap on these datasets is not caused by mmapLevel.

### 4. qHnsw vs FAISS hnsw (within KDB.AI)

| Metric | qHnsw (KDB.AI) | FAISS hnsw (KDB.AI FAISS) | Winner |
|--------|----------------|--------------------------|--------|
| SIFT recall | 0.982 | 0.993 | FAISS (+0.011) |
| GloVe recall | 0.771 | 0.852 | FAISS (+0.081) |
| SIFT QPS | 261 | 100 | qHnsw (2.6x) |
| GloVe QPS | 200 | 92 | qHnsw (2.2x) |
| SIFT ingest | 8,006 v/s | 11,367 v/s | FAISS (1.4x) |

qHnsw is 2-3x faster on QPS (multi-threaded). FAISS has higher recall, especially on cosine. FAISS ingest is faster despite single-threaded insert (simpler index structure).

### 5. KDB.AI vs Qdrant (Closest Competitor)

At default competitive settings (M=16, efC=64), KDB.AI trails Qdrant on recall across all datasets. With tuning (M=48, efC=200), the gap closes:

| Dataset | KDB.AI (M48,efC200) | Qdrant (M16,efC64) | Gap |
|---------|--------------------|--------------------|-----|
| SIFT | 0.999 | 0.999 | 0.000 |
| GIST | 0.983 | 0.966 | +0.017 |
| GloVe-100 | 0.935 | 0.943 | -0.008 |

With aggressive tuning, KDB.AI matches Qdrant on SIFT, beats it on GIST, and trails slightly on GloVe (cosine). However, this comparison is not apples-to-apples — Qdrant at M=48/efC=200 would likely also improve.

## Tuning Run Results

### Impact of M on Recall (efC=200, ef=256)

| Dataset | M=16 | M=32 | M=48 | M16->M32 | M32->M48 |
|---------|------|------|------|----------|----------|
| SIFT | 0.988 | 0.997 | 0.999 | +0.009 | +0.002 |
| GIST | 0.896 | 0.967 | 0.983 | +0.070 | +0.017 |
| GloVe-100 | 0.832 | 0.904 | 0.935 | +0.073 | +0.030 |

M=16->M=32 is the biggest lever. M=32->M=48 shows diminishing returns. GIST and GloVe benefit most from higher M (high-dimensional / cosine datasets).

### Impact of efConstruction on Recall (ef=256)

| Dataset | M=16,efC=64 | M=16,efC=200 | Delta |
|---------|-------------|-------------|-------|
| SIFT | 0.982 | 0.988 | +0.006 |
| GIST | 0.867 | 0.896 | +0.029 |
| GloVe-100 | 0.771 | 0.832 | +0.061 |

| Dataset | M=32,efC=128 | M=32,efC=200 | Delta |
|---------|-------------|-------------|-------|
| SIFT | 0.996 | 0.997 | +0.000 |
| GIST | 0.963 | 0.967 | +0.003 |
| GloVe-100 | 0.897 | 0.904 | +0.008 |

efConstruction has large impact at M=16 (especially on cosine: +6.1% on GloVe) but diminishing returns at M=32.

### Impact of M on Ingest (efC=200, avg across docker configs)

| M | SIFT (v/s) | GIST (v/s) | GloVe-100 (v/s) |
|---|-----------|-----------|----------------|
| 16 | 6,011 | 1,655 | 5,579 |
| 32 | 4,174 | 1,212 | 3,739 |
| 48 | 3,284 | 978 | 2,900 |

Higher M reduces ingest throughput ~30% per step. M=48 is 45% slower than M=16.

### Docker Threading Impact

Threading config (NUM_WRK x THREADS) does NOT affect recall (confirmed: identical recall across all docker configs for the same HNSW params). It affects QPS and ingest:

- **2wrk_8thr**: Best overall QPS
- **1wrk_16thr**: Competitive QPS, simplest config
- **4wrk_4thr**: No advantage over 2wrk_8thr

### Recommended KDB.AI Configuration

For competitive benchmarks (balance of recall and ingest speed):

```
M=32, efConstruction=128, NUM_WRK=2, THREADS=8
```

This achieves:
- SIFT recall: 0.996 (vs 0.982 at default M=16/efC=64)
- GIST recall: 0.963 (vs 0.867)
- GloVe recall: 0.897 (vs 0.771)
- Ingest: ~4,200 v/s on SIFT (vs ~8,000 at M=16)

For maximum recall (at the cost of 45% slower ingest):

```
M=48, efConstruction=200, NUM_WRK=2, THREADS=8
```

## QPS Measurement Quality

The QPS noise reduction fixes (cache warming, scaled warmup, query padding) reduced non-monotonic QPS violations from 35% to 18% of configs. Remaining violations are caused by server-side stalls in KDB.AI (observable as QPS << 1000/P50, e.g., QPS=53 when P50=3.6ms). This is a KDB.AI server behavior, not a benchmarking methodology issue.

## Open Issues

1. **Cosine recall gap**: qHnsw recall on cosine datasets lags FAISS hnsw by 8 points (GloVe). Root cause unclear — may be in qHnsw's distance computation or graph construction for angular distance.
2. **QPS stalls**: Intermittent server-side stalls cause 5-10x QPS drops on random efSearch points. Not reproducible — appears to be GC or internal buffer management.
3. **Chroma efSearch**: `collection.modify(configuration=...)` doesn't change hnswlib ef. All Chroma results are at ef=64.
