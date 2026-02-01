#!/bin/bash

# Vector Database Setup Script
# This script pulls and starts Docker containers for the vector databases
# used in the benchmark suite.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Container names (used by benchmark's docker_monitor.py)
QDRANT_CONTAINER="qdrant-benchmark"
PGVECTOR_CONTAINER="pgvector-benchmark"
KDBAI_CONTAINER="kdbai-benchmark"
WEAVIATE_CONTAINER="weaviate-benchmark"
MILVUS_CONTAINER="milvus-benchmark"

# Default resource limits (can be overridden via environment variables)
MEMORY_LIMIT="${MEMORY_LIMIT:-8g}"
CPU_LIMIT="${CPU_LIMIT:-4}"

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if Docker is installed and running
check_docker() {
    print_header "Checking Docker"

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi

    print_success "Docker is installed and running"
}

# Wait for a container to be healthy/ready
wait_for_container() {
    local container_name=$1
    local check_command=$2
    local max_attempts=${3:-30}
    local attempt=1

    echo -n "Waiting for $container_name to be ready"
    while [ $attempt -le $max_attempts ]; do
        if eval "$check_command" &> /dev/null; then
            echo ""
            print_success "$container_name is ready"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo ""
    print_error "$container_name failed to become ready after $max_attempts attempts"
    return 1
}

# Stop and remove existing container if it exists
cleanup_container() {
    local container_name=$1
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        print_info "Removing existing container: $container_name"
        docker stop "$container_name" &> /dev/null || true
        docker rm "$container_name" &> /dev/null || true
    fi
}

# Setup Qdrant
setup_qdrant() {
    print_header "Setting up Qdrant"

    cleanup_container "$QDRANT_CONTAINER"

    print_info "Pulling Qdrant image..."
    docker pull qdrant/qdrant:latest

    print_info "Starting Qdrant container..."
    docker run -d \
        --name "$QDRANT_CONTAINER" \
        -p 6333:6333 \
        -p 6334:6334 \
        --memory="$MEMORY_LIMIT" \
        --cpus="$CPU_LIMIT" \
        -v qdrant_storage:/qdrant/storage \
        qdrant/qdrant:latest

    # Wait for Qdrant to be ready
    wait_for_container "$QDRANT_CONTAINER" "curl -s http://localhost:6333/healthz"

    print_success "Qdrant is running on http://localhost:6333"
}

# Setup pgvector (PostgreSQL with vector extension)
setup_pgvector() {
    print_header "Setting up pgvector (PostgreSQL)"

    cleanup_container "$PGVECTOR_CONTAINER"

    print_info "Pulling pgvector image..."
    docker pull pgvector/pgvector:pg16

    print_info "Starting pgvector container..."
    docker run -d \
        --name "$PGVECTOR_CONTAINER" \
        -p 5432:5432 \
        --memory="$MEMORY_LIMIT" \
        --cpus="$CPU_LIMIT" \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_DB=postgres \
        -v pgvector_data:/var/lib/postgresql/data \
        pgvector/pgvector:pg16

    # Wait for PostgreSQL to be ready
    wait_for_container "$PGVECTOR_CONTAINER" \
        "docker exec $PGVECTOR_CONTAINER pg_isready -U postgres"

    # Enable the vector extension
    print_info "Enabling pgvector extension..."
    sleep 2  # Give a moment for PostgreSQL to fully initialize
    docker exec "$PGVECTOR_CONTAINER" psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" || true

    print_success "pgvector is running on localhost:5432"
    print_info "  Username: postgres"
    print_info "  Password: postgres"
    print_info "  Database: postgres"
}

# Setup KDB.AI
setup_kdbai() {
    print_header "Setting up KDB.AI"

    # KDB.AI Docker image from KX registry
    KDBAI_IMAGE="portal.dl.kx.com/kdbai-db"

    # Check for license - required for KDB.AI
    if [ -z "$KDB_LICENSE_B64" ]; then
        if [ -n "$KDBAI_LICENSE_FILE" ] && [ -f "$KDBAI_LICENSE_FILE" ]; then
            print_info "Reading license from $KDBAI_LICENSE_FILE"
            KDB_LICENSE_B64=$(base64 -w 0 "$KDBAI_LICENSE_FILE")
            export KDB_LICENSE_B64
        else
            print_warning "KDB.AI requires a license from KX."
            echo ""
            print_info "To set up KDB.AI:"
            echo ""
            echo "  1. Sign up for KDB.AI Server at https://kdb.ai/"
            echo "  2. You'll receive a welcome email with:"
            echo "     - Docker registry credentials (Bearer Token)"
            echo "     - License file (kc.lic)"
            echo ""
            echo "  3. Login to the KX Docker registry:"
            echo "     docker login portal.dl.kx.com -u <your-email> -p <bearer-token>"
            echo ""
            echo "  4. Run this script with your license:"
            echo "     export KDB_LICENSE_B64=\$(base64 -w 0 /path/to/kc.lic)"
            echo "     $0 kdbai"
            echo ""
            echo "  Or set the license file path:"
            echo "     KDBAI_LICENSE_FILE=/path/to/kc.lic $0 kdbai"
            echo ""
            print_info "Alternatively, use KDB.AI Cloud (no Docker needed):"
            echo "     https://cloud.kdb.ai/"
            return 0
        fi
    fi

    # Check if logged into KX registry
    if ! docker pull "$KDBAI_IMAGE" --quiet 2>/dev/null; then
        print_warning "Cannot pull KDB.AI image. You may need to login first:"
        echo ""
        echo "  docker login portal.dl.kx.com -u <your-email> -p <bearer-token>"
        echo ""
        print_info "The bearer token is in your KDB.AI welcome email."
        return 1
    fi

    cleanup_container "$KDBAI_CONTAINER"

    # Calculate CPU limit for --cpuset-cpus (KDB.AI Standard Edition limited to 24 cores)
    local cpu_count=${CPU_LIMIT:-4}
    if [ "$cpu_count" -gt 24 ]; then
        print_warning "KDB.AI Standard Edition limited to 24 cores, capping CPU limit"
        cpu_count=24
    fi
    local cpuset="0-$((cpu_count - 1))"

    print_info "Starting KDB.AI container..."
    docker run -d \
        --name "$KDBAI_CONTAINER" \
        -p 8081:8081 \
        -p 8082:8082 \
        --memory="$MEMORY_LIMIT" \
        --cpuset-cpus="$cpuset" \
        -e KDB_LICENSE_B64="$KDB_LICENSE_B64" \
        -e VDB_DIR="/tmp/kx/data/vdb" \
        -e THREADS="$cpu_count" \
        -v kdbai_data:/tmp/kx/data \
        "$KDBAI_IMAGE"

    # Wait for KDB.AI to be ready
    wait_for_container "$KDBAI_CONTAINER" "curl -s http://localhost:8082/api/v1/system/state" 60

    print_success "KDB.AI is running on http://localhost:8082"
}

# Setup Weaviate
setup_weaviate() {
    print_header "Setting up Weaviate"

    cleanup_container "$WEAVIATE_CONTAINER"

    print_info "Pulling Weaviate image..."
    docker pull semitechnologies/weaviate:latest

    print_info "Starting Weaviate container..."
    docker run -d \
        --name "$WEAVIATE_CONTAINER" \
        -p 8080:8080 \
        -p 50051:50051 \
        --memory="$MEMORY_LIMIT" \
        --cpus="$CPU_LIMIT" \
        -e QUERY_DEFAULTS_LIMIT=25 \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
        -e DEFAULT_VECTORIZER_MODULE='none' \
        -e CLUSTER_HOSTNAME='node1' \
        -v weaviate_data:/var/lib/weaviate \
        semitechnologies/weaviate:latest

    # Wait for Weaviate to be ready
    wait_for_container "$WEAVIATE_CONTAINER" "curl -s http://localhost:8080/v1/.well-known/ready"

    print_success "Weaviate is running on http://localhost:8080"
}

# Setup Milvus
setup_milvus() {
    print_header "Setting up Milvus"

    cleanup_container "$MILVUS_CONTAINER"

    print_info "Pulling Milvus image..."
    docker pull milvusdb/milvus:latest

    print_info "Starting Milvus container..."
    docker run -d \
        --name "$MILVUS_CONTAINER" \
        -p 19530:19530 \
        -p 9091:9091 \
        --memory="$MEMORY_LIMIT" \
        --cpus="$CPU_LIMIT" \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e COMMON_STORAGETYPE=local \
        -v milvus_data:/var/lib/milvus \
        milvusdb/milvus:latest \
        milvus run standalone

    # Wait for Milvus to be ready
    wait_for_container "$MILVUS_CONTAINER" "curl -s http://localhost:9091/healthz" 60

    print_success "Milvus is running on localhost:19530"
}

# Show status of all containers
show_status() {
    print_header "Container Status"

    echo "Database Containers:"
    echo "-------------------"

    for container in "$QDRANT_CONTAINER" "$PGVECTOR_CONTAINER" "$KDBAI_CONTAINER" "$WEAVIATE_CONTAINER" "$MILVUS_CONTAINER"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null)
            print_success "$container: $status"
        elif docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
            status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null)
            print_warning "$container: $status"
        else
            print_info "$container: not created"
        fi
    done
}

# Stop all benchmark containers
stop_all() {
    print_header "Stopping All Containers"

    for container in "$QDRANT_CONTAINER" "$PGVECTOR_CONTAINER" "$KDBAI_CONTAINER" "$WEAVIATE_CONTAINER" "$MILVUS_CONTAINER"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            print_info "Stopping $container..."
            docker stop "$container" &> /dev/null
            print_success "Stopped $container"
        fi
    done
}

# Remove all benchmark containers and volumes
cleanup_all() {
    print_header "Cleaning Up All Containers and Volumes"

    for container in "$QDRANT_CONTAINER" "$PGVECTOR_CONTAINER" "$KDBAI_CONTAINER" "$WEAVIATE_CONTAINER" "$MILVUS_CONTAINER"; do
        cleanup_container "$container"
    done

    print_info "Removing volumes..."
    docker volume rm qdrant_storage pgvector_data kdbai_data weaviate_data milvus_data 2>/dev/null || true

    print_success "Cleanup complete"
}

# Print usage examples
print_usage_examples() {
    print_header "Usage Examples"

    echo "After setup, you can run benchmarks like this:"
    echo ""
    echo "# Benchmark Qdrant:"
    echo "python run_benchmark.py --database qdrant --dataset datasets/sift \\"
    echo "    --container $QDRANT_CONTAINER --endpoint http://localhost:6333"
    echo ""
    echo "# Benchmark pgvector:"
    echo "python run_benchmark.py --database pgvector --dataset datasets/sift \\"
    echo "    --container $PGVECTOR_CONTAINER --endpoint localhost:5432"
    echo ""
    echo "# Benchmark KDB.AI:"
    echo "python run_benchmark.py --database kdbai --dataset datasets/sift \\"
    echo "    --container $KDBAI_CONTAINER --endpoint http://localhost:8082"
    echo ""
    echo "# Benchmark Weaviate:"
    echo "python run_benchmark.py --database weaviate --dataset datasets/sift \\"
    echo "    --container $WEAVIATE_CONTAINER --endpoint http://localhost:8080"
    echo ""
    echo "# Benchmark Milvus:"
    echo "python run_benchmark.py --database milvus --dataset datasets/sift \\"
    echo "    --container $MILVUS_CONTAINER --endpoint localhost:19530"
    echo ""
    echo "# Benchmark FAISS (no Docker needed):"
    echo "python run_benchmark.py --database faiss --dataset datasets/sift"
    echo ""
    echo "# Don't forget to download the dataset first:"
    echo "python datasets/download_sift.py"
}

# Main function
main() {
    local command=${1:-"all"}

    case $command in
        all)
            check_docker
            setup_qdrant
            setup_pgvector
            setup_weaviate
            setup_milvus
            setup_kdbai
            show_status
            print_usage_examples
            ;;
        qdrant)
            check_docker
            setup_qdrant
            ;;
        pgvector)
            check_docker
            setup_pgvector
            ;;
        weaviate)
            check_docker
            setup_weaviate
            ;;
        milvus)
            check_docker
            setup_milvus
            ;;
        kdbai)
            check_docker
            setup_kdbai
            ;;
        status)
            show_status
            ;;
        stop)
            stop_all
            ;;
        cleanup)
            cleanup_all
            ;;
        help|--help|-h)
            echo "Vector Database Setup Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  all       Setup all databases (default)"
            echo "  qdrant    Setup only Qdrant"
            echo "  pgvector  Setup only pgvector (PostgreSQL)"
            echo "  weaviate  Setup only Weaviate"
            echo "  milvus    Setup only Milvus"
            echo "  kdbai     Setup only KDB.AI (requires license)"
            echo "  status    Show status of all containers"
            echo "  stop      Stop all benchmark containers"
            echo "  cleanup   Remove all containers and volumes"
            echo "  help      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MEMORY_LIMIT       Memory limit for containers (default: 8g)"
            echo "  CPU_LIMIT          CPU limit for containers (default: 4)"
            echo "  KDB_LICENSE_B64    Base64-encoded KDB.AI license (for kdbai)"
            echo "  KDBAI_LICENSE_FILE Path to KDB.AI license file (alternative to KDB_LICENSE_B64)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Setup all databases"
            echo "  $0 qdrant                       # Setup only Qdrant"
            echo "  $0 weaviate                     # Setup only Weaviate"
            echo "  $0 milvus                       # Setup only Milvus"
            echo "  MEMORY_LIMIT=16g $0             # Setup with 16GB memory limit"
            echo ""
            echo "KDB.AI Setup (requires license from https://kdb.ai/):"
            echo "  docker login portal.dl.kx.com -u <email> -p <bearer-token>"
            echo "  export KDB_LICENSE_B64=\$(base64 -w 0 /path/to/kc.lic)"
            echo "  $0 kdbai"
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
