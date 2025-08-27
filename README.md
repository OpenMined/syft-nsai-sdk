# SyftBox NSAI SDK

Python SDK for discovering and using SyftBox AI models with built-in payment handling and RAG coordination.

## Quick Start

```python
import asyncio
from syft_nsai_sdk import SyftBoxClient

async def main():
    async with SyftBoxClient() as client:
        # Setup accounting for paid models
        await client.setup_accounting(
            email="your@email.com",
            password="your_password",
            service_url="https://accounting.syftbox.net"
        )
        
        # Chat with a model
        response = await client.chat(
            model_name="claude-sonnet-3.5",
            owner="aggregator@openmined.org",
            prompt="Hello! What is syftbox?",
            temperature=0.7,
            max_tokens=200
        )
        print(response.message.content)

asyncio.run(main())
```

## Client Methods

### Model Discovery

```python
# Discover available models
# Discover available models
models = client.discover_models(
    service_type="chat",  # "chat" or "search"
    owner="user@domain.com",
    tags=["opensource"],
    max_cost=0.10,
    health_check="auto"  # "auto", "always", "never"
)

# Find specific model
model = client.find_model("model-name", owner="user@domain.com")

# Find best model by criteria
best_chat = client.find_best_chat_model(
    preference="balanced",  # "cheapest", "balanced", "premium", "fastest"
    max_cost=0.50,
    tags=["helpful"]
)
```

### Chat Services

```python
# Direct chat
response = await client.chat(
    model_name="gpt-assistant",
    owner="ai-team@company.com",
    prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=500
)

# Auto-select best chat model
response = await client.chat_with_best(
    prompt="What's the weather like?",
    max_cost=0.25,
    tags=["helpful"],
    temperature=0.5
)
```

### Search Services

```python
# Direct search
results = await client.search(
    model_name="legal-database",
    owner="legal@company.com", 
    query="employment contracts",
    limit=10,
    similarity_threshold=0.8
)

# Auto-select best search model
results = await client.search_with_best(
    query="company policies remote work",
    max_cost=0.15,
    tags=["internal"],
    limit=5
)

# Search multiple models
results = await client.search_multiple_models(
    model_names=["docs", "wiki", "policies"],
    query="vacation policy",
    limit_per_model=3,
    total_limit=10,
    remove_duplicates=True
)
```

### RAG Coordination

```python
# Preview RAG workflow costs
preview = client.preview_rag_workflow(
    search_models=["legal-docs", "hr-policies"],
    chat_model="gpt-assistant"
)
print(preview)

# Chat only (no search context)
response = await client.chat_with_search_context(
    search_models=[],  # No search models = chat only
    chat_model="claude-assistant",
    prompt="What is machine learning?"
)

# RAG workflow (search + chat)
response = await client.chat_with_search_context(
    search_models=["legal-docs", "hr-policies", "wiki"],
    chat_model="claude-assistant", 
    prompt="What's our remote work policy?",
    max_search_results=5,
    temperature=0.7
)

# Simple search-then-chat
response = await client.search_then_chat(
    search_model="company-docs",
    chat_model="helpful-assistant", 
    prompt="How do I submit expenses?"
)
```

### Model Information

```python
# List available models
print(client.list_models(service_type="chat", format="table"))

# Get model details
details = client.show_model_details("model-name", owner="user@domain.com")

# Show usage examples
examples = client.show_model_usage("gpt-model", owner="ai-team@company.com")

# Get statistics
stats = client.get_statistics()
print(f"Total models: {stats['total_models']}")
```

### Health Monitoring

```python
# Check single model health
status = await client.check_model_health("model-name", timeout=5.0)

# Check all models
health_status = await client.check_all_models_health(service_type="chat")

# Start continuous monitoring
monitor = client.start_health_monitoring(
    models=["model1", "model2"],
    check_interval=30.0
)
```

### Account Management

```python
# Setup accounting
await client.setup_accounting("email", "password", "service_url")

# Check account info
info = await client.get_account_info()
print(f"Balance: ${info['balance']}")

# Show accounting status
print(client.show_accounting_status())

# Cost estimation
estimate = client.get_rag_cost_estimate(
    search_models=["docs1", "docs2"], 
    chat_model="premium-chat"
)
print(f"Total cost: ${estimate['total_cost']}")
```

## Response Objects

### ChatResponse
```python
response.message.content    # String: The AI's response
response.cost              # Float: Cost of the request  
response.usage.total_tokens # Int: Tokens used
response.model             # String: Model name used
response.provider_info     # Dict: Additional provider details
```

### SearchResponse  
```python
response.results           # List[DocumentResult]: Search results
response.cost              # Float: Cost of the request
response.query             # String: Original query
response.provider_info     # Dict: Search metadata

# Individual result
result = response.results[0]
result.content             # String: Document content
result.score              # Float: Similarity score (0-1)
result.metadata           # Dict: File info, etc.
```

## Error Handling

```python
from syft_nsai_sdk.core.exceptions import (
    ModelNotFoundError,
    ServiceNotSupportedError,
    PaymentError,
    AuthenticationError,
    ValidationError
)

try:
    response = await client.chat(model_name="invalid-model", prompt="test")
except ModelNotFoundError:
    print("Model not found")
except PaymentError:
    print("Payment issue - check accounting setup")
except ValidationError as e:
    print(f"Invalid parameters: {e}")
```

### Context Management
For maintaining context between messages, use conversation managers instead of multiple `client.chat()` calls:

```python
# ❌ No context between calls
response1 = await client.chat(model_name="model", prompt="What is AI?")
response2 = await client.chat(model_name="model", prompt="Give examples")  # No context

# ✅ Context maintained automatically  
conversation = client.create_conversation("model", owner="owner@email.com")
response1 = await conversation.send_message("What is AI?")
response2 = await conversation.send_message("Give examples")  # Remembers previous

# Instead of multiple client.chat() calls
conversation = client.create_conversation("claude-sonnet-3.5")
conversation.set_system_message("You are a helpful assistant.")

# Each message remembers previous context
response1 = await conversation.send_message("What is SyftBox?")
response2 = await conversation.send_message("How does it work?")  # Remembers previous
response3 = await conversation.send_message("Give me an example")   # Full context
```

### RAG vs Chat-Only
The `chat_with_search_context()` method supports both patterns:

```python
# Chat only (like frontend with no data sources)
response = await client.chat_with_search_context(
    search_models=[],  # Empty = chat only
    chat_model="assistant",
    prompt="What is Python?"
)

# RAG workflow (like frontend with data sources selected)
response = await client.chat_with_search_context(
    search_models=["docs", "wiki"],  # Search + chat
    chat_model="assistant",
    prompt="What's our policy?"
)
```

## Configuration

```python
# Custom configuration
client = SyftBoxClient(
    user_email="your@email.com",
    cache_server_url="https://custom.syftbox.net",
    auto_setup_accounting=True,
    auto_health_check_threshold=5
)

# Environment variables
# SYFTBOX_ACCOUNTING_EMAIL
# SYFTBOX_ACCOUNTING_PASSWORD  
# SYFTBOX_ACCOUNTING_URL
```