# SyftBox NSAI SDK

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/syft-nsai-sdk.svg)](https://pypi.org/project/syft-nsai-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-available-green.svg)](https://docs.syftbox.openmined.org/nsai-sdk)

A Python SDK for discovering and using AI models across the SyftBox network. Access chat and search models from various providers with automatic payment handling, health monitoring, and rich filtering capabilities.

## 🚀 Quick Start

### Prerequisites

1. **SyftBox Installation** (Required)
   ```bash
   # Install SyftBox
   curl -LsSf https://install.syftbox.openmined.org | sh
   
   # Setup SyftBox
   syftbox setup
   ```

2. **Install SDK**
   ```bash
   pip install syft-nsai-sdk
   uv pip install -e . --force-reinstall
   ```

### Basic Usage

```python
from syft_nsai_sdk import SyftBoxClient
import asyncio

async def main():
    # Initialize client
    client = SyftBoxClient()
    
    # Discover available models
    models = client.list_models()
    print(models)
    
    # Quick chat with automatic model selection
    response = await client.chat("Hello! How are you?")
    print(response.message.content)
    
    # Search documents
    results = await client.search("Python tutorials", limit=5)
    for result in results.results:
        print(f"- {result.content[:100]}...")

# Run the example
asyncio.run(main())
```

## 📋 Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Model Discovery](#model-discovery)
- [Chat Services](#chat-services)
- [Search Services](#search-services)
- [Payment & Accounting](#payment--accounting)
- [Health Monitoring](#health-monitoring)
- [CLI Usage](#cli-usage)
- [Environment Usage](#environment-usage)
- [API Reference](#api-reference)
- [Examples](#examples)

## 📦 Installation

### Standard Installation
```bash
pip install syft-nsai-sdk
```

### Development Installation
```bash
# Clone repository
git clone https://github.com/OpenMined/syft-nsai-sdk.git
cd syft-nsai-sdk

# Install in development mode
pip install -e ".[dev]"
```

### Requirements
- Python 3.8+
- SyftBox installed and configured
- Internet connection for model discovery

## 🔍 Model Discovery

### List All Models
```python
from syft_nsai_sdk import SyftBoxClient

client = SyftBoxClient()

# List all available models
print(client.list_models())

# List only chat models
print(client.list_models(service_type="chat"))

# List with health checks (slower but more accurate)
print(client.list_models(health_check="always"))
```

### Find Specific Models
```python
# Find by name
model = client.find_model("public-tinnyllama")

# Find by owner
models = client.find_models_by_owner("irina@openmined.org")

# Find by tags
models = client.find_models_by_tags(["opensource", "gpt"])
```

### Advanced Filtering
```python
from syft_nsai_sdk import FilterBuilder, ServiceType

# Build complex filters
filter_builder = FilterBuilder()
models = filter_builder \
    .by_service_type(ServiceType.CHAT) \
    .by_max_cost(0.10) \
    .by_tags(["opensource"], match_all=False) \
    .free_only() \
    .build() \
    .filter_models(client.discover_models())

print(f"Found {len(models)} matching models")
```

## 💬 Chat Services

### Simple Chat
```python
# Quick chat with automatic model selection
response = await client.chat("Explain quantum computing")
print(response.message.content)

# Chat with specific model
response = await client.chat(
    "Write a poem about AI",
    model_name="public-tinnyllama"
)
```

### Advanced Chat Options
```python
# Chat with generation options
response = await client.chat(
    "Tell me a story",
    max_cost=0.50,              # Maximum willing to pay
    max_tokens=200,             # Limit response length
    temperature=0.7,            # Control creativity
    preference="premium"        # Model selection preference
)
```

### Conversation Management
```python
from syft_nsai_sdk import ConversationManager

# Create conversation manager
conversation = client.create_conversation("public-tinnyllama")

# Set system message
conversation.set_system_message("You are a helpful AI assistant.")

# Multi-turn conversation
response1 = await conversation.send_message("What is machine learning?")
response2 = await conversation.send_message("Can you give me an example?")

# Get conversation summary
summary = conversation.get_conversation_summary()
print(f"Total messages: {summary['total_messages']}")
```

## 🔎 Search Services

### Simple Search
```python
# Search with automatic model selection
results = await client.search("Python machine learning tutorials")

for result in results.results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content[:100]}...")
    if result.metadata:
        print(f"Source: {result.metadata.get('filename', 'Unknown')}")
```

### Advanced Search Options
```python
# Search with options
results = await client.search(
    "artificial intelligence",
    model_name="knowledge-search",
    limit=10,                   # Number of results
    similarity_threshold=0.7,   # Minimum similarity score
    max_cost=0.25              # Cost limit
)
```

### Batch Search
```python
from syft_nsai_sdk import BatchSearchService

# Get search service
search_service = client.get_search_service("knowledge-search")
batch_service = BatchSearchService(search_service)

# Search multiple queries
queries = ["Python tutorials", "Machine learning", "Data science"]
responses = await batch_service.search_multiple_queries(queries)

for query, response in zip(queries, responses):
    print(f"Results for '{query}': {len(response.results)} found")
```

## 💰 Payment & Accounting

### Check Accounting Status
```python
# Check if accounting is configured
if client.is_accounting_configured():
    account_info = await client.get_account_info()
    print(f"Balance: ${account_info['balance']}")
else:
    print("Accounting not configured - limited to free models")

# Show detailed status
print(client.show_accounting_status())
```

### Manual Accounting Setup
```python
# Set up accounting credentials
await client.setup_accounting(
    email="user@example.com",
    password="your_password"
)

# Using environment variables (recommended for production)
# Set SYFTBOX_ACCOUNTING_EMAIL and SYFTBOX_ACCOUNTING_PASSWORD
# Client will auto-detect these
```

### Cost Management
```python
# Use only free models
free_models = client.discover_models(free_only=True)

# Set cost limits
response = await client.chat(
    "Expensive query",
    max_cost=1.00,      # Won't spend more than $1.00
    auto_pay=True       # Automatically handle payment
)

# Check model pricing before use
model = client.find_model("premium-gpt4")
if model:
    chat_service = model.get_service_info("chat")
    print(f"Cost: ${chat_service.pricing}/{chat_service.charge_type}")
```

## 🏥 Health Monitoring

### Check Model Health
```python
# Check single model
health = await client.check_model_health("public-tinnyllama")
print(f"Health status: {health}")

# Check all models
health_status = await client.check_all_models_health()
for model_name, status in health_status.items():
    print(f"{model_name}: {status}")
```

### Continuous Monitoring
```python
from syft_nsai_sdk import HealthMonitor

# Start health monitoring
monitor = client.start_health_monitoring(
    models=["public-tinnyllama", "premium-gpt4"],
    check_interval=30.0  # Check every 30 seconds
)

# Add callback for status changes
def on_health_change(model_name, old_status, new_status):
    print(f"{model_name} health changed: {old_status} → {new_status}")

monitor.add_callback(on_health_change)

# Get health summary
summary = monitor.get_health_summary()
print(f"Healthy models: {summary['healthy']}/{summary['total_models']}")
```

## 🖥️ CLI Usage

The SDK includes a command-line interface for quick operations:

### Installation & Setup
```bash
# Install with CLI support
pip install syft-nsai-sdk

# Check installation
syftbox-sdk --help
```

### Model Discovery
```bash
# List all models
syftbox-sdk list-models

# List only chat models
syftbox-sdk list-models --service chat

# List models with health checks
syftbox-sdk list-models --health-check always

# Filter by owner
syftbox-sdk list-models --owner "irina@openmined.org"

# Show detailed model info
syftbox-sdk model-info "public-tinnyllama"
```

### Chat Operations
```bash
# Quick chat
syftbox-sdk chat "Hello, how are you?"

# Chat with specific model
syftbox-sdk chat "Write a poem" --model "public-tinnyllama"

# Chat with cost limit
syftbox-sdk chat "Expensive query" --max-cost 0.50
```

### Search Operations
```bash
# Quick search
syftbox-sdk search "Python tutorials"

# Search with specific model
syftbox-sdk search "Machine learning" --model "knowledge-search"

# Search with options
syftbox-sdk search "AI research" --limit 10 --max-cost 0.25
```

### Health Monitoring
```bash
# Check health of all models
syftbox-sdk health-check --all

# Check specific model
syftbox-sdk health-check --model "public-tinnyllama"

# Health summary
syftbox-sdk health-summary
```

### Accounting Management
```bash
# Show accounting status
syftbox-sdk account-status

# Setup accounting (interactive)
syftbox-sdk setup-accounting

# Check balance
syftbox-sdk account-balance
```

## 🌍 Environment Usage

### Development Environment

#### Python REPL
```python
# Quick setup for development
from syft_nsai_sdk import *
import asyncio

client = SyftBoxClient()
models = client.list_models()

# Quick async helper
def run_async(coro):
    return asyncio.run(coro)

# Usage
response = run_async(client.chat("Hello"))
print(response.message.content)
```

#### IPython/Jupyter Enhanced
```python
# Enable async in Jupyter
%load_ext asyncio

from syft_nsai_sdk import SyftBoxClient
from IPython.display import display, HTML, Markdown

client = SyftBoxClient()

# Rich model display
models = client.discover_models(health_check="always")
display(HTML(client.list_models()))

# Interactive chat
message = "Explain machine learning"
response = await client.chat(message, max_cost=0.10)
display(Markdown(response.message.content))
```

### Production Environment

#### Docker Setup
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install SyftBox
RUN curl -LsSf https://install.syftbox.openmined.org | sh
RUN syftbox setup --non-interactive

# Install SDK
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Set environment variables for accounting
ENV SYFTBOX_ACCOUNTING_EMAIL=${ACCOUNTING_EMAIL}
ENV SYFTBOX_ACCOUNTING_PASSWORD=${ACCOUNTING_PASSWORD}
ENV SYFTBOX_ACCOUNTING_URL=${ACCOUNTING_URL}

CMD ["python", "app.py"]
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  syftbox-app:
    build: .
    environment:
      - SYFTBOX_ACCOUNTING_EMAIL=${ACCOUNTING_EMAIL}
      - SYFTBOX_ACCOUNTING_PASSWORD=${ACCOUNTING_PASSWORD}
      - SYFTBOX_ACCOUNTING_URL=${ACCOUNTING_URL}
    volumes:
      - syftbox_data:/root/.syftbox
    networks:
      - syftbox-network

volumes:
  syftbox_data:

networks:
  syftbox-network:
```

#### Kubernetes Deployment
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: syftbox-nsai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: syftbox-nsai
  template:
    metadata:
      labels:
        app: syftbox-nsai
    spec:
      containers:
      - name: nsai
        image: syftbox-nsai:latest
        env:
        - name: SYFTBOX_ACCOUNTING_EMAIL
          valueFrom:
            secretKeyRef:
              name: syftbox-creds
              key: email
        - name: SYFTBOX_ACCOUNTING_PASSWORD
          valueFrom:
            secretKeyRef:
              name: syftbox-creds
              key: password
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Secret
metadata:
  name: syftbox-creds
type: Opaque
data:
  email: <base64-encoded-email>
  password: <base64-encoded-password>
```

### CI/CD Integration

#### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Test SyftBox NSAI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install SyftBox
      run: |
        curl -LsSf https://install.syftbox.openmined.org | sh
        syftbox setup --non-interactive
    
    - name: Install dependencies
      run: |
        pip install syft-nsai-sdk[dev]
    
    - name: Run tests (free models only)
      run: |
        pytest tests/ -m "not paid"
      env:
        PYTHONPATH: .
    
    - name: Run integration tests
      if: github.ref == 'refs/heads/main'
      run: |
        pytest tests/ -m "integration"
      env:
        SYFTBOX_ACCOUNTING_EMAIL: ${{ secrets.ACCOUNTING_EMAIL }}
        SYFTBOX_ACCOUNTING_PASSWORD: ${{ secrets.ACCOUNTING_PASSWORD }}
```

### AWS Lambda
```python
# lambda_function.py
import json
import asyncio
from syft_nsai_sdk import SyftBoxClient

# Initialize client outside handler for reuse
client = SyftBoxClient()

def lambda_handler(event, context):
    """AWS Lambda handler for SyftBox chat API."""
    
    async def process_request():
        message = event.get('message', 'Hello')
        max_cost = event.get('max_cost', 0.10)
        
        try:
            response = await client.chat(message, max_cost=max_cost)
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'response': response.message.content,
                    'cost': response.cost,
                    'model': response.model
                })
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }
    
    # Run async function
    return asyncio.run(process_request())
```

### Environment Variables Reference

```bash
# Required for SyftBox
export SYFTBOX_CONFIG_PATH="/path/to/.syftbox/config.json"

# Accounting configuration
export SYFTBOX_ACCOUNTING_EMAIL="user@example.com"
export SYFTBOX_ACCOUNTING_PASSWORD="your_password"
export SYFTBOX_ACCOUNTING_URL="service_url"

# SDK configuration
export SYFTBOX_CACHE_SERVER_URL="https://syftbox.net"
export SYFTBOX_AUTO_HEALTH_CHECK_THRESHOLD="10"
export SYFTBOX_DEFAULT_MAX_COST="1.0"

# Development settings
export SYFTBOX_DEBUG="true"
export SYFTBOX_LOG_LEVEL="DEBUG"
```

## 🧪 Examples

See the [examples](./examples) directory for complete examples:

- **[basic_usage.py](./examples/basic_usage.py)** - Simple chat and search
- **[advanced_filtering.py](./examples/advanced_filtering.py)** - Complex model discovery
- **[batch_operations.py](./examples/batch_operations.py)** - Multiple model interactions
- **[health_monitoring.py](./examples/health_monitoring.py)** - Continuous health monitoring
- **[payment_handling.py](./examples/payment_handling.py)** - Working with paid models
- **[jupyter_notebook.ipynb](./examples/jupyter_notebook.ipynb)** - Interactive Jupyter example

## 📚 API Reference

### Main Classes
- **[SyftBoxClient](./docs/api/client.md)** - Main client for model discovery and usage
- **[ChatService](./docs/api/chat.md)** - Chat service client
- **[SearchService](./docs/api/search.md)** - Search service client
- **[HealthMonitor](./docs/api/health.md)** - Health monitoring service

### Types & Enums
- **[ModelInfo](./docs/api/types.md#modelinfo)** - Model information structure
- **[ServiceType](./docs/api/types.md#servicetype)** - Available service types
- **[HealthStatus](./docs/api/types.md#healthstatus)** - Health status values

### Utilities
- **[Filtering](./docs/api/filtering.md)** - Model filtering utilities
- **[Formatting](./docs/api/formatting.md)** - Display formatting functions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.syftbox.openmined.org/nsai-sdk](https://docs.syftbox.openmined.org/nsai-sdk)
- **Issues**: [GitHub Issues](https://github.com/OpenMined/syft-nsai-sdk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OpenMined/syft-nsai-sdk/discussions)
- **Discord**: [OpenMined Discord](https://discord.gg/openmined)

## 🔗 Related Projects

- **[SyftBox](https://github.com/OpenMined/SyftBox)** - The core SyftBox platform
- **[FastSyftBox](https://github.com/OpenMined/FastSyftBox)** - Framework for model providers
- **[Accounting SDK](https://github.com/OpenMined/accounting-sdk)** - Payment processing service