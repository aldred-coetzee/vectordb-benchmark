# Runbook

## Prerequisites

- Python 3.12+
- Docker (for client-server databases)
- AWS CLI + `vectordb` SSO profile (for AWS runs)
- ~15 GB disk for datasets

## Local Setup

### 1. Clone and Install

```bash
git clone https://github.com/aldred-coetzee/vectordb-benchmark.git
cd vectordb-benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Texmex datasets (SIFT, GIST) — manual download
# SIFT: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz → extract to data/sift/
# GIST: ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz → extract to data/gist/

# HDF5 datasets (GloVe, DBpedia-OpenAI)
python scripts/download_datasets.py --datasets glove-100,dbpedia-openai

# Dev datasets (for fast testing)
python scripts/generate_dev_datasets.py --datasets sift
```

### 3. Verify Setup

```bash
# Quick smoke test (~12 seconds)
python run_benchmark.py --config configs/faiss.yaml --dataset sift-dev
```

## Running Benchmarks Locally

### Single Database

```bash
# Full SIFT benchmark
python run_benchmark.py --config configs/qdrant.yaml --dataset sift

# Dev test (fast iteration)
python run_benchmark.py --config configs/qdrant.yaml --dataset sift-dev

# Specific dataset
python run_benchmark.py --config configs/kdbai.yaml --dataset glove-100

# Skip Docker (if container already running)
python run_benchmark.py --config configs/qdrant.yaml --no-docker
```

### All Databases

```bash
python run_all.py                                    # All DBs, SIFT
python run_all.py --databases qdrant,milvus,kdbai    # Subset
python run_all.py --dataset gist                     # Different dataset
```

### KDB.AI with License

KDB.AI requires a commercial license. Set the environment variable before running:

```bash
export KDB_LICENSE_B64=$(base64 -w 0 /path/to/kc.lic)
python run_benchmark.py --config configs/kdbai.yaml --dataset sift
```

### Custom HNSW Parameters

```bash
# Override M and efConstruction
python run_benchmark.py --config configs/kdbai.yaml --hnsw-m 32 --hnsw-efc 128

# Custom efSearch sweep
python run_benchmark.py --config configs/kdbai.yaml --ef-search 64 128 256 512

# Cold-cache mode (restarts container between efSearch values)
python run_benchmark.py --config configs/kdbai.yaml --cold
```

## Running Benchmarks on AWS

### Prerequisites

```bash
# Ensure AWS SSO is active
aws sso login --profile vectordb
```

### Option 1: Launch Template (Console)

1. Go to EC2 Console > Launch Templates
2. Select template:
   - `vectordb-benchmark-full` for competitive benchmark
   - `vectordb-benchmark-kdbai-tuning` for KDB.AI tuning
3. Click "Launch instance from template"
4. Optionally edit tags:
   - `Databases`: comma-separated list (remove entries to run subset)
   - `Datasets`: comma-separated list
   - `PullLatest`: `all`, specific DBs, or remove to skip
5. Launch — orchestrator handles everything automatically

### Option 2: CLI

```bash
# Full competitive run
python aws/orchestrator.py

# Specific databases/datasets
python aws/orchestrator.py --databases qdrant,kdbai --datasets sift,gist

# KDB.AI tuning
python aws/orchestrator.py --databases kdbai --datasets sift,glove-100,gist \
    --benchmark-type kdbai-tuning

# Pull fresh Docker images
python aws/orchestrator.py --databases kdbai --pull-latest kdbai

# Fire and forget (don't wait for completion)
python aws/orchestrator.py --no-wait
```

### Monitoring a Run

Workers auto-terminate on completion. To check progress:

```bash
# List running instances
aws ec2 describe-instances --profile vectordb --region us-west-2 \
    --filters "Name=tag:Project,Values=vectordb-benchmark" "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Name`].Value|[0],State.Name,LaunchTime]' \
    --output table

# Check job status in S3
aws s3 ls s3://vectordb-benchmark-590780615264/runs/competitive/<RUN_ID>/jobs/ \
    --profile vectordb --region us-west-2

# Check a specific job
aws s3 cp s3://vectordb-benchmark-590780615264/runs/competitive/<RUN_ID>/jobs/qdrant-sift/status.json - \
    --profile vectordb --region us-west-2
```

### Pulling Results

```bash
# Pull competitive results (merges per-job DBs, generates report)
python scripts/pull_run.py <RUN_ID>

# Pull tuning results
python scripts/pull_run.py <RUN_ID> --benchmark-type kdbai-tuning
```

Output:
- `results/vectordb-benchmark-<type>-<RUN_ID>.db` — merged SQLite database
- `results/vectordb-benchmark-<type>-<RUN_ID>.html` — HTML report

### Costs

| Component | Cost per Run |
|-----------|-------------|
| Workers (28 jobs x ~45 min) | ~$16 |
| Orchestrator (3-4 hrs) | ~$0.04 |
| S3 | ~$0.01 |
| **Total** | **~$16-20** |

Standby cost: ~$0.80/month (AMI snapshots).

## Generating Reports

```bash
# From a pulled run
python generate_report.py --run-id 2026-02-07-0010

# From a specific database file
python generate_report.py --db results/vectordb-benchmark-competitive-2026-02-07-0010.db
```

## Troubleshooting

### Container won't start

```bash
# Check if port is in use
docker ps
# Check container logs
docker logs <container_name>
# Remove stale container
docker rm -f <container_name>
```

### KDB.AI "index not supported"

The index type spelling must be exact: `qFlat`, `qHnsw`, `flat`, `hnsw`. `HNSW` (uppercase) is NOT valid.

### KDB.AI "invalid arguments: mmapLevel"

`qFlat` does not support mmapLevel. Only `qHnsw` accepts it. Check `kdbai_client.py:create_table()`.

### Milvus OOM on GIST/DBpedia

Expected. Milvus with HNSW on 960D/1536D vectors exceeds 64 GB. These combinations are excluded in `benchmark.yaml`.

### Redis BusyLoadingError

Remove volume mounts from Redis config. Stale RDB dumps cause this on startup.

### pgvector "vector type not found"

`CREATE EXTENSION vector` must run before `register_vector()`. Check `pgvector_client.py`.

### AWS: Workers terminated but no results

Check worker logs in S3:
```bash
aws s3 cp s3://vectordb-benchmark-590780615264/runs/<type>/<RUN_ID>/jobs/<job>/worker.log - \
    --profile vectordb --region us-west-2
```

Exit code 143 = SIGTERM (safety timeout after 2 hours). Increase `MAX_RUNTIME_MINUTES` in `worker_startup.sh` for slow databases.

### AWS: vCPU limit exceeded

Account has a vCPU limit (default ~64 for on-demand). Running 45 tuning jobs x 16 vCPU = 720 vCPU exceeds this. Request a limit increase or run fewer concurrent jobs.

## Adding a New Database

1. Create client in `benchmark/clients/<name>_client.py` implementing `BaseVectorDBClient`
2. Create config in `configs/<name>.yaml`
3. Register in `run_benchmark.py:get_client()` (~line 770)
4. Add to `aws/orchestrator.py:DATABASES` list
5. Add pip install + docker pull cases to `aws/worker_startup.sh`
6. Test locally: `python run_benchmark.py --config configs/<name>.yaml --dataset sift-dev`
7. Add any exclusions/caveats to `benchmark.yaml`
