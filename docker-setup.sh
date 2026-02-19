#!/bin/bash

# Skintel Project - Docker Setup and Connection Script
# This script helps you set up and connect to the Skintel Docker environment

echo "=============================================="
echo "  Skintel - Docker Environment Setup"
echo "=============================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

print_status "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "Docker Compose is installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed logs models/checkpoints
print_status "Directories created"

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Please create one from .env.example"
else
    print_status ".env file found"
fi

# Main menu
echo ""
echo "What would you like to do?"
echo "1) Start all services (MinIO + MongoDB + FastAPI)"
echo "2) Stop all services"
echo "3) View service logs"
echo "4) Restart services"
echo "5) Check service status"
echo "6) Clean up (remove containers and volumes)"
echo "7) Enter backend container shell"
echo "8) Initialize Module 1 (Data Bedrock)"
echo "9) Exit"
echo ""
read -p "Enter your choice [1-9]: " choice

case $choice in
    1)
        echo ""
        print_status "Starting all Skintel services..."
        docker-compose up -d
        echo ""
        print_status "Services started successfully!"
        echo ""
        echo "Access points:"
        echo "  - FastAPI Backend: http://localhost:8000"
        echo "  - API Documentation: http://localhost:8000/docs"
        echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
        echo "  - MongoDB: mongodb://localhost:27017 (admin/skintel123)"
        echo ""
        print_status "Waiting for services to be healthy..."
        sleep 10
        docker-compose ps
        ;;
    2)
        echo ""
        print_status "Stopping all services..."
        docker-compose down
        print_status "Services stopped"
        ;;
    3)
        echo ""
        echo "Available services: backend, minio, mongodb"
        read -p "Enter service name (or 'all' for all services): " service
        if [ "$service" == "all" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f $service
        fi
        ;;
    4)
        echo ""
        print_status "Restarting all services..."
        docker-compose restart
        print_status "Services restarted"
        docker-compose ps
        ;;
    5)
        echo ""
        print_status "Service Status:"
        docker-compose ps
        echo ""
        print_status "Docker Stats:"
        docker stats --no-stream $(docker-compose ps -q)
        ;;
    6)
        echo ""
        print_warning "This will remove all containers and volumes!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" == "yes" ]; then
            docker-compose down -v
            print_status "Cleanup complete"
        else
            print_warning "Cleanup cancelled"
        fi
        ;;
    7)
        echo ""
        print_status "Entering backend container..."
        docker-compose exec backend /bin/bash
        ;;
    8)
        echo ""
        print_status "Initializing Module 1 (Data Bedrock)..."
        docker-compose exec backend python -m modules.module1_bedrock.initialize_bedrock
        ;;
    9)
        echo ""
        print_status "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_status "Done!"