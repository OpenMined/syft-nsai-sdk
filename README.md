# Syft NSAI SDK

## Directory
```bash
.
├── LICENSE
├── main.py
├── pyproject.toml              # Updated with tabulate + aiohttp
├── README.md
├── run.sh
├── uv.lock
└── syft_nsai_sdk/
    ├── __init__.py            # Public API exports
    ├── types.py              # ServiceInfo, ModelMetadata dataclasses
    ├── config.py             # SyftBoxConfig class
    ├── models.py             # ModelObject with RPC logic
    └── sdk.py                # Main SyftBoxSDK class
```

## Setup
```bash
chmod +x run.sh
./run.sh
```

## Usage Across Platforms:
### 1. Start the Server:
```bash
./run.sh
# Server runs on http://localhost:8001
```

### 2. API Usage:
```bash
# Check SDK status
curl http://localhost:8001/sdk/status

# List models with filters
curl "http://localhost:8001/models?tag=finance"

# Chat with a model
curl -X POST http://localhost:8001/models/chat \
  -H "Content-Type: application/json" \
  -d '{"model_name": "alice-model", "prompt": "Hello!"}'
```

### 3. Python REPL:
```python
import syft_nsai_sdk as sb
models = sb.find_models(tags=["finance"])
response = models[0].chat("Hello!")
```

### 4. Jupyter Notebook:
```python
import syft_nsai_sdk as sb
models = sb.get_models()
sb.display_models(models)  # Pretty table
```

### 5. CLI:
```bash
uv run python main.py show-status
uv run python main.py demo
```


## Test Model Interactions:
### 1. Test Chat with a Model:
```bash
# Test with your own model
curl -X POST http://localhost:8001/models/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "callis",
    "prompt": "Hello! Can you help me test this API?",
    "temperature": 0.7
  }'

# Or test with the public tinyllama model
curl -X POST http://localhost:8001/models/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "public-tinnyllama",
    "prompt": "What can you help me with?",
    "owner": "irina@openmined.org",
    "temperature": 0.8
  }'
```
### 2. Test Search with a Model:
```bash
# Test search with a news model
curl -X POST http://localhost:8001/models/search \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "the-city",
    "query": "latest news",
    "owner": "speters@thecity.nyc"
  }'

# Or search OpenMined info
curl -X POST http://localhost:8001/models/search \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openmined-about",
    "query": "what is OpenMined",
    "owner": "irina@openmined.org"
  }'
```

### 3. Test Filtering:
```bash
# Find chat models
curl "http://localhost:8001/models?tag=chat" | jq

# Find models by specific owner
curl "http://localhost:8001/models?owner=irina@openmined.org" | jq

# Find news-related models
curl "http://localhost:8001/models?tag=news" | jq
```

### 4. Test CLI:
```bash
# List all models
syft-sdk list-models

# Filter by tag
syft-sdk list-models --tag chat

# Chat with a model
syft-sdk chat "callis" "Hello from CLI!" --owner callis@openmined.org

# Search with a model
syft-sdk search "the-city" "latest updates" --owner speters@thecity.nyc

# Run interactive demo
syft-sdk demo
```

### 5. Test in Jupyter:
```bash
uv run jupyter notebook
```