#!/bin/bash

# Task Master AI - Local Infrastructure Setup Script
# Sets up complete local LLM infrastructure for Task Master AI migration

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/setup.log"
CONFIG_DIR="$PROJECT_ROOT/.taskmaster/config"
DATA_DIR="$PROJECT_ROOT/.taskmaster/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
    log "INFO: $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING: $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR: $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
    log "HEADER: $1"
}

# System requirements check
check_system_requirements() {
    print_header "=== Checking System Requirements ==="
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Operating System: macOS"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
        print_status "Total RAM: ${TOTAL_RAM}GB"
        if [ "$TOTAL_RAM" -lt 8 ]; then
            print_warning "Recommended minimum: 8GB RAM. Performance may be limited."
        fi
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS memory check
        TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        print_status "Total RAM: ${TOTAL_RAM}GB"
        if [ "$TOTAL_RAM" -lt 8 ]; then
            print_warning "Recommended minimum: 8GB RAM. Performance may be limited."
        fi
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    print_status "Available disk space: ${AVAILABLE_SPACE}GB"
    if [ "${AVAILABLE_SPACE%.*}" -lt 20 ]; then
        print_warning "Recommended minimum: 20GB free space for models and cache."
    fi
    
    # Check for GPU (NVIDIA)
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_status "GPU detected: $GPU_INFO"
    else
        print_warning "No NVIDIA GPU detected. Using CPU inference (slower)."
    fi
}

# Create directory structure
create_directories() {
    print_header "=== Creating Directory Structure ==="
    
    # Create necessary directories
    directories=(
        "$CONFIG_DIR"
        "$DATA_DIR/models"
        "$DATA_DIR/cache"
        "$DATA_DIR/logs"
        "$DATA_DIR/embeddings"
        "$PROJECT_ROOT/.taskmaster/migration/logs"
        "$PROJECT_ROOT/.taskmaster/migration/configs"
        "$PROJECT_ROOT/.taskmaster/migration/scripts"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory exists: $dir"
        fi
    done
}

# Install system dependencies
install_system_dependencies() {
    print_header "=== Installing System Dependencies ==="
    
    if [[ "$OS" == "linux" ]]; then
        # Linux dependencies
        if command -v apt-get >/dev/null 2>&1; then
            # Ubuntu/Debian
            print_status "Installing dependencies via apt..."
            sudo apt-get update
            sudo apt-get install -y curl wget git python3 python3-pip docker.io docker-compose redis-server
            
            # Start services
            sudo systemctl enable docker
            sudo systemctl start docker
            sudo systemctl enable redis-server
            sudo systemctl start redis-server
            
        elif command -v yum >/dev/null 2>&1; then
            # CentOS/RHEL
            print_status "Installing dependencies via yum..."
            sudo yum install -y curl wget git python3 python3-pip docker docker-compose redis
            
            # Start services
            sudo systemctl enable docker
            sudo systemctl start docker
            sudo systemctl enable redis
            sudo systemctl start redis
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        # macOS dependencies
        if ! command -v brew >/dev/null 2>&1; then
            print_status "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        print_status "Installing dependencies via Homebrew..."
        brew install curl wget git python3 docker docker-compose redis
        
        # Start services
        brew services start redis
    fi
    
    # Add user to docker group (Linux only)
    if [[ "$OS" == "linux" ]]; then
        sudo usermod -aG docker "$USER"
        print_warning "Please log out and back in for docker group changes to take effect."
    fi
}

# Install Python dependencies
install_python_dependencies() {
    print_header "=== Installing Python Dependencies ==="
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python packages..."
    cat > "$PROJECT_ROOT/requirements-local-llm.txt" << EOF
# Core dependencies
httpx>=0.24.0
aiofiles>=23.0.0
asyncio-mqtt>=0.11.0
redis>=4.5.0
psutil>=5.9.0
numpy>=1.24.0

# LLM and ML dependencies
sentence-transformers>=2.2.0
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
bitsandbytes>=0.39.0

# Vector database
qdrant-client>=1.3.0

# Monitoring and observability
prometheus-client>=0.16.0
opentelemetry-api>=1.18.0
opentelemetry-sdk>=1.18.0

# Development and testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
EOF

    pip install -r "$PROJECT_ROOT/requirements-local-llm.txt"
    
    print_status "Python dependencies installed successfully"
}

# Install Ollama
install_ollama() {
    print_header "=== Installing Ollama ==="
    
    if command -v ollama >/dev/null 2>&1; then
        print_status "Ollama already installed: $(ollama --version)"
        return
    fi
    
    print_status "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Verify installation
    if command -v ollama >/dev/null 2>&1; then
        print_status "Ollama installed successfully: $(ollama --version)"
    else
        print_error "Ollama installation failed"
        exit 1
    fi
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    if [[ "$OS" == "linux" ]]; then
        # Linux service
        sudo systemctl enable ollama
        sudo systemctl start ollama
    elif [[ "$OS" == "macos" ]]; then
        # macOS - start in background
        nohup ollama serve > "$DATA_DIR/logs/ollama.log" 2>&1 &
        echo $! > "$DATA_DIR/ollama.pid"
        print_status "Ollama started in background (PID: $(cat $DATA_DIR/ollama.pid))"
    fi
    
    # Wait for service to start
    sleep 5
    
    # Verify service is running
    if curl -s http://localhost:11434/api/tags >/dev/null; then
        print_status "Ollama service is running on port 11434"
    else
        print_error "Failed to start Ollama service"
        exit 1
    fi
}

# Download and setup models
setup_models() {
    print_header "=== Setting Up Local Models ==="
    
    models=(
        "llama3.1:8b-instruct-q4_0"    # Lightweight model
        "mistral:7b-instruct"          # General purpose
        "codellama:13b-instruct"       # Code generation
    )
    
    # Large models (optional - check available space)
    if [ "${AVAILABLE_SPACE%.*}" -gt 50 ]; then
        models+=(
            "llama3.1:70b-instruct-q4_0"   # High-capability model
            "qwen:32b-instruct"            # Research model
        )
        print_status "Sufficient space detected. Including large models."
    else
        print_warning "Limited space. Skipping large models (70B+)."
    fi
    
    for model in "${models[@]}"; do
        print_status "Downloading model: $model"
        if ollama pull "$model"; then
            print_status "Successfully downloaded: $model"
        else
            print_warning "Failed to download: $model (continuing with other models)"
        fi
    done
    
    # List installed models
    print_status "Installed models:"
    ollama list
}

# Setup LocalAI (alternative inference engine)
setup_localai() {
    print_header "=== Setting Up LocalAI ==="
    
    # Create LocalAI configuration
    cat > "$CONFIG_DIR/localai-docker-compose.yml" << EOF
version: '3.8'

services:
  localai:
    image: localai/localai:latest
    container_name: taskmaster-localai
    ports:
      - "8080:8080"
    environment:
      - DEBUG=true
      - MODELS_PATH=/models
      - CONTEXT_SIZE=4096
    volumes:
      - $DATA_DIR/models:/models
      - $DATA_DIR/cache:/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/models"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
    
    # Start LocalAI
    print_status "Starting LocalAI container..."
    cd "$CONFIG_DIR"
    docker-compose -f localai-docker-compose.yml up -d
    
    # Wait for service to be ready
    print_status "Waiting for LocalAI to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8080/v1/models >/dev/null; then
            print_status "LocalAI is ready on port 8080"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "LocalAI service did not start within 60 seconds"
    fi
}

# Setup Qdrant vector database
setup_qdrant() {
    print_header "=== Setting Up Qdrant Vector Database ==="
    
    # Create Qdrant configuration
    cat > "$CONFIG_DIR/qdrant-docker-compose.yml" << EOF
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: taskmaster-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - $DATA_DIR/qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
    
    # Start Qdrant
    print_status "Starting Qdrant container..."
    cd "$CONFIG_DIR"
    docker-compose -f qdrant-docker-compose.yml up -d
    
    # Wait for service to be ready
    print_status "Waiting for Qdrant to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:6333/health >/dev/null; then
            print_status "Qdrant is ready on port 6333"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Qdrant service did not start within 60 seconds"
    fi
}

# Setup monitoring and observability
setup_monitoring() {
    print_header "=== Setting Up Monitoring ==="
    
    # Create monitoring stack configuration
    cat > "$CONFIG_DIR/monitoring-docker-compose.yml" << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: taskmaster-prometheus
    ports:
      - "9090:9090"
    volumes:
      - $CONFIG_DIR/prometheus.yml:/etc/prometheus/prometheus.yml
      - $DATA_DIR/prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: taskmaster-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=taskmaster123
    volumes:
      - $DATA_DIR/grafana_data:/var/lib/grafana
    restart: unless-stopped
EOF
    
    # Create Prometheus configuration
    cat > "$CONFIG_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'taskmaster-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'

  - job_name: 'ollama'
    static_configs:
      - targets: ['host.docker.internal:11434']
    metrics_path: '/metrics'

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'
EOF
    
    # Start monitoring stack
    print_status "Starting monitoring stack..."
    cd "$CONFIG_DIR"
    docker-compose -f monitoring-docker-compose.yml up -d
    
    print_status "Monitoring services:"
    print_status "  - Prometheus: http://localhost:9090"
    print_status "  - Grafana: http://localhost:3000 (admin/taskmaster123)"
}

# Create configuration files
create_configuration_files() {
    print_header "=== Creating Configuration Files ==="
    
    # Create main configuration
    cat > "$CONFIG_DIR/local-llm-config.json" << EOF
{
  "providers": {
    "ollama": {
      "endpoint": "http://localhost:11434",
      "models": {
        "lightweight": "mistral:7b-instruct",
        "standard": "codellama:13b-instruct",
        "heavy": "llama3.1:70b-instruct-q4_0"
      },
      "enabled": true
    },
    "localai": {
      "endpoint": "http://localhost:8080",
      "models": {
        "general": "gpt-3.5-turbo"
      },
      "enabled": true
    }
  },
  "vector_db": {
    "qdrant": {
      "endpoint": "http://localhost:6333",
      "collection_name": "taskmaster_knowledge",
      "enabled": true
    }
  },
  "cache": {
    "redis": {
      "endpoint": "redis://localhost:6379",
      "ttl": 3600,
      "enabled": true
    }
  },
  "monitoring": {
    "prometheus": {
      "endpoint": "http://localhost:9090",
      "enabled": true
    },
    "grafana": {
      "endpoint": "http://localhost:3000",
      "enabled": true
    }
  },
  "resource_limits": {
    "max_concurrent_requests": 10,
    "cpu_threshold": 80,
    "memory_threshold": 85,
    "gpu_memory_threshold": 90
  }
}
EOF
    
    # Create Task Master integration configuration
    cat > "$CONFIG_DIR/taskmaster-local-llm.json" << EOF
{
  "local_llm": {
    "enabled": true,
    "fallback_to_external": false,
    "model_routing": {
      "simple_tasks": "ollama:mistral:7b-instruct",
      "moderate_tasks": "ollama:codellama:13b-instruct", 
      "complex_tasks": "ollama:llama3.1:70b-instruct-q4_0",
      "code_generation": "ollama:codellama:13b-instruct",
      "research": "ollama:llama3.1:70b-instruct-q4_0",
      "analysis": "ollama:llama3.1:70b-instruct-q4_0"
    },
    "performance_thresholds": {
      "max_response_time": 30,
      "min_success_rate": 0.9,
      "quality_threshold": 0.8
    }
  },
  "migration": {
    "phase": "testing",
    "external_api_enabled": false,
    "local_only": true,
    "validation_mode": false
  }
}
EOF
    
    print_status "Configuration files created successfully"
}

# Create startup scripts
create_startup_scripts() {
    print_header "=== Creating Startup Scripts ==="
    
    # Create main startup script
    cat > "$SCRIPT_DIR/start-local-llm-stack.sh" << 'EOF'
#!/bin/bash

# Task Master AI - Local LLM Stack Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/.taskmaster/config"
DATA_DIR="$PROJECT_ROOT/.taskmaster/data"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Start services
print_status "Starting Task Master AI Local LLM Stack..."

# Start Redis
if ! pgrep redis-server > /dev/null; then
    print_status "Starting Redis..."
    redis-server --daemonize yes
fi

# Start Ollama
if ! pgrep ollama > /dev/null; then
    print_status "Starting Ollama..."
    nohup ollama serve > "$DATA_DIR/logs/ollama.log" 2>&1 &
    sleep 5
fi

# Start Docker containers
print_status "Starting Docker containers..."
cd "$CONFIG_DIR"

if [ -f "localai-docker-compose.yml" ]; then
    docker-compose -f localai-docker-compose.yml up -d
fi

if [ -f "qdrant-docker-compose.yml" ]; then
    docker-compose -f qdrant-docker-compose.yml up -d
fi

if [ -f "monitoring-docker-compose.yml" ]; then
    docker-compose -f monitoring-docker-compose.yml up -d
fi

# Health checks
print_status "Performing health checks..."

services=(
    "http://localhost:11434/api/tags:Ollama"
    "http://localhost:8080/v1/models:LocalAI"
    "http://localhost:6333/health:Qdrant"
    "http://localhost:6379:Redis"
    "http://localhost:9090:Prometheus"
    "http://localhost:3000:Grafana"
)

for service in "${services[@]}"; do
    IFS=':' read -r url name <<< "$service"
    
    if curl -s "$url" > /dev/null 2>&1; then
        print_status "$name is running"
    else
        print_warning "$name is not responding"
    fi
done

print_status "Local LLM stack startup complete!"
print_status "Services available:"
print_status "  - Ollama API: http://localhost:11434"
print_status "  - LocalAI API: http://localhost:8080"
print_status "  - Qdrant DB: http://localhost:6333"
print_status "  - Prometheus: http://localhost:9090"
print_status "  - Grafana: http://localhost:3000"
EOF

    chmod +x "$SCRIPT_DIR/start-local-llm-stack.sh"

    # Create shutdown script
    cat > "$SCRIPT_DIR/stop-local-llm-stack.sh" << 'EOF'
#!/bin/bash

# Task Master AI - Local LLM Stack Shutdown Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/.taskmaster/config"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_status "Stopping Task Master AI Local LLM Stack..."

# Stop Docker containers
cd "$CONFIG_DIR"

if [ -f "monitoring-docker-compose.yml" ]; then
    print_status "Stopping monitoring stack..."
    docker-compose -f monitoring-docker-compose.yml down
fi

if [ -f "qdrant-docker-compose.yml" ]; then
    print_status "Stopping Qdrant..."
    docker-compose -f qdrant-docker-compose.yml down
fi

if [ -f "localai-docker-compose.yml" ]; then
    print_status "Stopping LocalAI..."
    docker-compose -f localai-docker-compose.yml down
fi

# Stop Ollama
if pgrep ollama > /dev/null; then
    print_status "Stopping Ollama..."
    pkill ollama
fi

# Stop Redis
if pgrep redis-server > /dev/null; then
    print_status "Stopping Redis..."
    pkill redis-server
fi

print_status "Local LLM stack shutdown complete!"
EOF

    chmod +x "$SCRIPT_DIR/stop-local-llm-stack.sh"

    # Create health check script
    cat > "$SCRIPT_DIR/health-check.sh" << 'EOF'
#!/bin/bash

# Task Master AI - Health Check Script

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_healthy() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

echo "Task Master AI - Local LLM Stack Health Check"
echo "=============================================="

# Check services
services=(
    "http://localhost:11434/api/tags:Ollama API"
    "http://localhost:8080/v1/models:LocalAI API"
    "http://localhost:6333/health:Qdrant Vector DB"
    "http://localhost:9090:Prometheus"
    "http://localhost:3000:Grafana"
)

all_healthy=true

for service in "${services[@]}"; do
    IFS=':' read -r url name <<< "$service"
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        print_healthy "$name"
    else
        print_error "$name (not responding)"
        all_healthy=false
    fi
done

# Check Redis separately
if redis-cli ping | grep -q PONG; then
    print_healthy "Redis Cache"
else
    print_error "Redis Cache (not responding)"
    all_healthy=false
fi

echo ""
if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}All services are healthy!${NC}"
    exit 0
else
    echo -e "${RED}Some services have issues. Check the logs.${NC}"
    exit 1
fi
EOF

    chmod +x "$SCRIPT_DIR/health-check.sh"
    
    print_status "Startup scripts created successfully"
}

# Create validation and testing scripts
create_validation_scripts() {
    print_header "=== Creating Validation Scripts ==="
    
    # Create model validation script
    cat > "$SCRIPT_DIR/validate-models.py" << 'EOF'
#!/usr/bin/env python3
"""
Model Validation Script for Task Master AI Local LLM Migration
Tests all models for basic functionality and performance
"""

import asyncio
import json
import time
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_abstraction import TaskMasterLLMInterface, ModelCapability, TaskComplexity

async def test_model_basic_functionality(interface: TaskMasterLLMInterface) -> Dict[str, Any]:
    """Test basic model functionality"""
    results = {}
    
    test_cases = [
        {
            "name": "simple_chat",
            "method": "research",
            "args": ["What is 2+2?", "Simple math question"],
            "expected_keywords": ["4", "four", "equals"]
        },
        {
            "name": "research_task",
            "method": "research", 
            "args": ["Benefits of local LLM deployment", "AI research context"],
            "expected_keywords": ["privacy", "control", "cost", "latency"]
        },
        {
            "name": "planning_task",
            "method": "plan",
            "args": ["Set up monitoring system", "DevOps context"],
            "expected_keywords": ["metrics", "alerts", "dashboard", "logs"]
        },
        {
            "name": "code_generation",
            "method": "generate_code",
            "args": ["Hello world function", "python"],
            "expected_keywords": ["def", "print", "hello", "world"]
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing {test_case['name']}...")
        
        try:
            start_time = time.time()
            method = getattr(interface, test_case['method'])
            result = await method(*test_case['args'])
            response_time = time.time() - start_time
            
            # Check if response contains expected keywords
            result_lower = result.lower()
            keywords_found = sum(1 for keyword in test_case['expected_keywords'] 
                               if keyword.lower() in result_lower)
            keyword_score = keywords_found / len(test_case['expected_keywords'])
            
            results[test_case['name']] = {
                'success': True,
                'response_time': response_time,
                'response_length': len(result),
                'keyword_score': keyword_score,
                'response_preview': result[:200] + '...' if len(result) > 200 else result
            }
            
            print(f"  ✓ Success (time: {response_time:.2f}s, score: {keyword_score:.2f})")
            
        except Exception as e:
            results[test_case['name']] = {
                'success': False,
                'error': str(e),
                'response_time': 0,
                'response_length': 0,
                'keyword_score': 0
            }
            print(f"  ✗ Failed: {e}")
    
    return results

async def test_model_performance(interface: TaskMasterLLMInterface) -> Dict[str, Any]:
    """Test model performance under load"""
    print("Testing performance under concurrent load...")
    
    # Create multiple concurrent requests
    tasks = []
    test_query = "Explain the benefits of asynchronous programming"
    
    start_time = time.time()
    for i in range(5):  # 5 concurrent requests
        task = interface.research(f"{test_query} (request {i+1})", "Performance test")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    successful_results = [r for r in results if isinstance(r, str)]
    failed_results = [r for r in results if not isinstance(r, str)]
    
    return {
        'concurrent_requests': len(tasks),
        'successful_requests': len(successful_results),
        'failed_requests': len(failed_results),
        'total_time': total_time,
        'average_time_per_request': total_time / len(tasks),
        'success_rate': len(successful_results) / len(tasks)
    }

async def main():
    """Main validation function"""
    print("🧪 Task Master AI - Model Validation")
    print("=" * 50)
    
    # Initialize interface
    interface = TaskMasterLLMInterface()
    
    try:
        await interface.initialize()
        
        # Get health status
        print("Checking system health...")
        health_status = await interface.get_health_status()
        
        healthy_providers = sum(1 for status in health_status.values() 
                              if status['status'] == 'healthy')
        total_providers = len(health_status)
        
        print(f"Provider health: {healthy_providers}/{total_providers} healthy")
        
        if healthy_providers == 0:
            print("❌ No healthy providers available. Aborting validation.")
            return
        
        # Test basic functionality
        basic_results = await test_model_basic_functionality(interface)
        
        # Test performance
        performance_results = await test_model_performance(interface)
        
        # Compile validation report
        validation_report = {
            'timestamp': time.time(),
            'system_health': health_status,
            'basic_functionality': basic_results,
            'performance_test': performance_results,
            'overall_score': calculate_overall_score(basic_results, performance_results)
        }
        
        # Save validation report
        report_path = '../config/validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        print(f"\n📊 Validation Report Saved: {report_path}")
        print(f"Overall Score: {validation_report['overall_score']:.2f}/10")
        
        # Print summary
        print_validation_summary(validation_report)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await interface.cleanup()

def calculate_overall_score(basic_results: Dict, performance_results: Dict) -> float:
    """Calculate overall validation score"""
    # Basic functionality score (0-7 points)
    basic_score = 0
    for test_name, result in basic_results.items():
        if result['success']:
            basic_score += result['keyword_score']  # 0-1 points per test
    
    # Performance score (0-3 points)
    perf_score = min(3.0, performance_results['success_rate'] * 3)
    
    return basic_score + perf_score

def print_validation_summary(report: Dict):
    """Print validation summary"""
    print("\n📋 Validation Summary")
    print("-" * 30)
    
    basic_tests = report['basic_functionality']
    successful_basic = sum(1 for r in basic_tests.values() if r['success'])
    print(f"Basic functionality: {successful_basic}/{len(basic_tests)} tests passed")
    
    perf_test = report['performance_test']
    print(f"Performance test: {perf_test['success_rate']:.2%} success rate")
    print(f"Average response time: {perf_test['average_time_per_request']:.2f}s")
    
    health = report['system_health']
    healthy_count = sum(1 for h in health.values() if h['status'] == 'healthy')
    print(f"System health: {healthy_count}/{len(health)} providers healthy")
    
    score = report['overall_score']
    if score >= 8:
        print("✅ Validation: EXCELLENT")
    elif score >= 6:
        print("✅ Validation: GOOD")
    elif score >= 4:
        print("⚠️  Validation: ACCEPTABLE")
    else:
        print("❌ Validation: NEEDS IMPROVEMENT")

if __name__ == "__main__":
    asyncio.run(main())
EOF

    chmod +x "$SCRIPT_DIR/validate-models.py"
    
    print_status "Validation scripts created successfully"
}

# Main setup function
main() {
    print_header "🚀 Task Master AI - Local LLM Infrastructure Setup"
    print_header "=================================================="
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Setup started at $(date)" > "$LOG_FILE"
    
    # Run setup steps
    check_system_requirements
    create_directories
    install_system_dependencies
    install_python_dependencies
    install_ollama
    setup_models
    setup_localai
    setup_qdrant
    setup_monitoring
    create_configuration_files
    create_startup_scripts
    create_validation_scripts
    
    print_header "🎉 Setup Complete!"
    print_status "Local LLM infrastructure has been successfully set up."
    print_status ""
    print_status "Next steps:"
    print_status "1. Run health check: $SCRIPT_DIR/health-check.sh"
    print_status "2. Validate models: $SCRIPT_DIR/validate-models.py"
    print_status "3. Start the stack: $SCRIPT_DIR/start-local-llm-stack.sh"
    print_status ""
    print_status "Services will be available at:"
    print_status "  - Ollama API: http://localhost:11434"
    print_status "  - LocalAI API: http://localhost:8080"
    print_status "  - Qdrant Vector DB: http://localhost:6333"
    print_status "  - Prometheus: http://localhost:9090"
    print_status "  - Grafana: http://localhost:3000"
    print_status ""
    print_status "Configuration files:"
    print_status "  - $CONFIG_DIR/local-llm-config.json"
    print_status "  - $CONFIG_DIR/taskmaster-local-llm.json"
    print_status ""
    print_status "Log file: $LOG_FILE"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --minimal)
            MINIMAL_INSTALL=true
            shift
            ;;
        --help)
            echo "Task Master AI - Local LLM Infrastructure Setup"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-models    Skip model downloads (faster setup)"
            echo "  --minimal        Minimal installation (no monitoring)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main setup
main