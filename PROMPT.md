Create a vector database benchmark tool in Python.
Goal: Benchmark vector database performance measuring ingest speed, query speed, and recall accuracy. Starting with KDB.AI using Flat and HNSW indexes on the SIFT-1M dataset. The tool will be extended to support other vector databases (Milvus, Qdrant, Weaviate, pgvector) in the future.
Project Structure:
vectordb-benchmark/
├── benchmark/
│   ├── init.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── docker_monitor.py
│   ├── runner.py
│   ├── report.py
│   └── clients/
│       ├── init.py
│       ├── base.py
│       └── kdbai_client.py
├── datasets/
│   └── download_sift.py
├── results/
├── run_benchmark.py
└── requirements.txt
Dependencies: kdbai-client, numpy, pandas, matplotlib, docker
Architecture:

benchmark/clients/base.py: Abstract base class defining the interface all database clients must implement. Methods should include: connect(), disconnect(), create_table(), drop_table(), insert(), search(), get_stats()
benchmark/clients/kdbai_client.py: KDB.AI implementation of the base class
benchmark/runner.py: Orchestrates benchmarks using any client that implements the base class
benchmark/data_loader.py: Dataset loading, independent of database
benchmark/metrics.py: Recall calculation, independent of database
benchmark/report.py: Results formatting and export, independent of database
benchmark/docker_monitor.py: Container resource monitoring, independent of database

Dataset: SIFT-1M

Download from: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
Files after extraction:

sift_base.fvecs: 1,000,000 base vectors (128 dimensions)
sift_query.fvecs: 10,000 query vectors
sift_groundtruth.ivecs: pre-computed true nearest neighbors (10000 x 100)


File format: each vector is prefixed with 4-byte int32 dimension, then dim*4 bytes of float32 (fvecs) or int32 (ivecs) data

KDB.AI Specifics (for kdbai_client.py):

Connect via: kdbai.Session(endpoint='http://localhost:8082')
Database: session.database('default')
Schema for vectors: [{'name': 'id', 'type': 'int64'}, {'name': 'vectors', 'type': 'float32s'}]
Index builds automatically during insert (no separate build step)
Flat index config: {'name': 'flat_index', 'type': 'flat', 'column': 'vectors', 'params': {'dims': 128, 'metric': 'L2'}}
HNSW index config: {'name': 'hnsw_index', 'type': 'hnsw', 'column': 'vectors', 'params': {'dims': 128, 'M': 16, 'efConstruction': 64, 'metric': 'L2'}}
Search with params: table.search(vectors={'hnsw_index': [query.tolist()]}, n=100, index_params={'hnsw_index': {'efSearch': 64}})
Insert using pandas DataFrame: table.insert(df)

Indexes to Benchmark:

Flat - baseline with 100% recall (ground truth)
HNSW - sweep efSearch values: [8, 16, 32, 64, 128, 256]

Metrics to Capture:
Ingest (per index):

Total time (seconds)
Vectors per second
Peak memory during ingest (GB)
Final memory after ingest (GB)

Search (per configuration):

QPS (queries per second)
Latency P50, P95, P99 (milliseconds)
Recall@10 (compare to ground truth)
Recall@100 (compare to ground truth)
CPU utilization (%)
Memory usage (GB)

System context:

Docker CPU limit
Docker memory limit
Dataset name, vector count, dimensions

Docker Monitoring:

Use the docker Python package to get container stats
Container name passed as parameter
Memory: stats['memory_stats']['usage']
CPU: calculate from cpu_stats deltas

Console Output Format:
================================================================================
VECTOR DATABASE BENCHMARK RESULTS
Database:       KDB.AI
Timestamp:      2025-01-30 14:32:15
Dataset:        SIFT-1M (1,000,000 vectors, 128 dimensions)
Queries:        10,000
Docker CPU:     8 cores
Docker Memory:  32 GB
INGEST (index built during insert)
Index      Config                 Vectors      Time(s)    Vec/sec    Peak Mem   Final Mem
Flat       dims=128               1,000,000    45.2       22,124     2.1 GB     1.8 GB
HNSW       M=16,efC=64            1,000,000    312.5      3,200      4.8 GB     4.2 GB
SEARCH
Index      Config         QPS      P50(ms)  P95(ms)  P99(ms)  R@10    R@100   CPU%   Mem
Flat       -              52       18.2     21.5     24.8     1.0000  1.0000  95%    1.8GB
HNSW       efSearch=8     48521    0.18     0.31     0.52     0.7823  0.6512  78%    4.2GB
HNSW       efSearch=16    34200    0.24     0.42     0.74     0.8891  0.7834  82%    4.2GB
HNSW       efSearch=32    22150    0.38     0.72     1.12     0.9456  0.8923  85%    4.2GB
HNSW       efSearch=64    12840    0.65     1.24     1.85     0.9812  0.9534  88%    4.2GB
HNSW       efSearch=128   6520     1.21     2.45     3.42     0.9934  0.9823  91%    4.2GB
HNSW       efSearch=256   3180     2.58     4.82     6.21     0.9978  0.9945  93%    4.2GB
================================================================================
Files to Generate:

results/benchmark_ingest.csv
results/benchmark_search.csv
results/recall_vs_qps.png (plot with Recall@10 on x-axis, QPS on y-axis, log scale)

CLI Usage:
python run_benchmark.py --database kdbai --dataset datasets/sift --container kdbai-bench --cpus 8 --memory 32
Notes:

Include warmup queries before timing (100 queries)
Insert in batches (50,000 vectors per batch)
Drop tables before creating to ensure clean state
Handle errors gracefully (connection issues, missing dataset, etc.)
