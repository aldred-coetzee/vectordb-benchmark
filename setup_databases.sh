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

    print_warning "KDB.AI typically requires a license or cloud subscription."
    print_info "For local development, you can use the KDB.AI Cloud free tier:"
    print_info "  1. Sign up at https://kdb.ai/"
    print_info "  2. Get your endpoint URL and API key"
    print_info "  3. Use --endpoint flag with the benchmark script"
    echo ""

    # Check if there's a local KDB.AI image available
    if docker images --format '{{.Repository}}' | grep -q "kdbai"; then
        print_info "Found local KDB.AI image. Attempting to start..."

        cleanup_container "$KDBAI_CONTAINER"

        # Try to start - this may fail without proper licensing
        docker run -d \
            --name "$KDBAI_CONTAINER" \
            -p 8082:8082 \
            --memory="$MEMORY_LIMIT" \
            --cpus="$CPU_LIMIT" \
            kdbai/kdbai-server:latest 2>/dev/null || {
            print_warning "Could not start KDB.AI container (may require license)"
        }
    else
        print_info "No local KDB.AI Docker image found."
        print_info "To use KDB.AI, either:"
        print_info "  - Use KDB.AI Cloud (recommended): https://kdb.ai/"
        print_info "  - Contact KX for enterprise Docker image"
    fi
}

# Show status of all containers
show_status() {
    print_header "Container Status"

    echo "Database Containers:"
    echo "-------------------"

    for container in "$QDRANT_CONTAINER" "$PGVECTOR_CONTAINER" "$KDBAI_CONTAINER"; do
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

    for container in "$QDRANT_CONTAINER" "$PGVECTOR_CONTAINER" "$KDBAI_CONTAINER"; do
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

    for container in "$QDRANT_CONTAINER" "$PGVECTOR_CONTAINER" "$KDBAI_CONTAINER"; do
        cleanup_container "$container"
    done

    print_info "Removing volumes..."
    docker volume rm qdrant_storage pgvector_data 2>/dev/null || true

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
            echo "  kdbai     Setup only KDB.AI"
            echo "  status    Show status of all containers"
            echo "  stop      Stop all benchmark containers"
            echo "  cleanup   Remove all containers and volumes"
            echo "  help      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MEMORY_LIMIT  Memory limit for containers (default: 8g)"
            echo "  CPU_LIMIT     CPU limit for containers (default: 4)"
            echo ""
            echo "Examples:"
            echo "  $0                    # Setup all databases"
            echo "  $0 qdrant             # Setup only Qdrant"
            echo "  MEMORY_LIMIT=16g $0   # Setup with 16GB memory limit"
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
