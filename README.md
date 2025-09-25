# SyftBox Hub NSAI SDK

Python SDK for discovering and using SyftBox AI services with built-in payment handling and RAG coordination.

## Installation

```bash
# Basic installation
pip install syft-hub

# With accounting features (for paid services)
pip install syft-hub[accounting]

# With all optional features
pip install syft-hub[all]
```

## Quick Start

```python
import asyncio
from syft_hub import Client

async def main():
    async with Client() as client:
        # Setup accounting for paid services
        await client.setup_accounting(
            email,
            password,
            # service_url,
        )
        
        # Chat with a service
        response = await client.chat(
            service_name="claude-sonnet-3.5",
            datasite="aggregator@openmined.org",
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
# Discover available services
# Discover available services
services = client.discover_services(
    service_type="chat",  # "chat" or "search"
    datasite="user@domain.com",
    tags=["opensource"],
    max_cost=0.10,
    health_check="auto"  # "auto", "always", "never"
)

# Find specific service
service = client.find_service("service-name", datasite="user@domain.com")

# Find best service by criteria
best_chat = client.find_best_chat_service(
    preference="balanced",  # "cheapest", "balanced", "premium", "fastest"
    max_cost=0.50,
    tags=["helpful"]
)
```

### Chat Services

```python
# Direct chat
response = await client.chat(
    service_name="gpt-assistant",
    datasite="ai-team@company.com",
    prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=500
)

# Auto-select best chat service
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
    service_name="legal-database",
    datasite="legal@company.com", 
    query="employment contracts",
    limit=10,
    similarity_threshold=0.8
)

# Auto-select best search service
results = await client.search_with_best(
    query="company policies remote work",
    max_cost=0.15,
    tags=["internal"],
    limit=5
)

# Search multiple services
results = await client.search_multiple_services(
    service_names=["docs", "wiki", "policies"],
    query="vacation policy",
    limit_per_service=3,
    total_limit=10,
    remove_duplicates=True
)
```

### RAG Coordination

```python
# Preview RAG workflow costs
preview = client.preview_rag_workflow(
    search_services=["legal-docs", "hr-policies"],
    chat_service="gpt-assistant"
)
print(preview)

# Chat only (no search context)
response = await client.chat_with_search_context(
    search_services=[],  # No search services = chat only
    chat_service="claude-assistant",
    prompt="What is machine learning?"
)

# RAG workflow (search + chat)
response = await client.chat_with_search_context(
    search_services=["legal-docs", "hr-policies", "wiki"],
    chat_service="claude-assistant", 
    prompt="What's our remote work policy?",
    max_search_results=5,
    temperature=0.7
)

# Simple search-then-chat
response = await client.search_then_chat(
    search_service="company-docs",
    chat_service="helpful-assistant", 
    prompt="How do I submit expenses?"
)
```

### RAG vs Chat-Only
The `chat_with_search_context()` method supports both patterns:

```python
# Chat only (like frontend with no data sources)
response = await client.chat_with_search_context(
    search_services=[],  # Empty = chat only
    chat_service="assistant",
    prompt="What is Python?"
)

# RAG workflow (like frontend with data sources selected)
response = await client.chat_with_search_context(
    search_services=["docs", "wiki"],  # Search + chat
    chat_service="assistant",
    prompt="What's our policy?"
)
```

### Model Information

```python
# List available services
print(client.format_services(service_type="chat", format="table"))

# Get service details
details = client.show_service_details("service-name", datasite="user@domain.com")

# Show usage examples
examples = client.show_service_usage("gpt-service", datasite="ai-team@company.com")

# Get statistics
stats = client.get_statistics()
print(f"Total services: {stats['total_services']}")
```

### Health Monitoring

```python
# Check single service health
status = await client.check_service_health("service-name", timeout=5.0)

# Check all services
health_status = await client.check_all_services_health(service_type="chat")

# Start continuous monitoring
monitor = client.start_health_monitoring(
    services=["service1", "service2"],
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
    search_services=["docs1", "docs2"], 
    chat_service="premium-chat"
)
print(f"Total cost: ${estimate['total_cost']}")
```

## Response Objects

### ChatResponse
```python
response.message.content    # String: The AI's response
response.cost              # Float: Cost of the request  
response.usage.total_tokens # Int: Tokens used
response.service             # String: Model name used
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
from syft_hub.core.exceptions import (
    ModelNotFoundError,
    ServiceNotSupportedError,
    PaymentError,
    AuthenticationError,
    ValidationError
)

try:
    response = await client.chat(service_name="invalid-service", prompt="test")
except ModelNotFoundError:
    print("Model not found")
except PaymentError:
    print("Payment issue - check accounting setup")
except ValidationError as e:
    print(f"Invalid parameters: {e}")
```

### Conversation Management

**For maintaining context between messages, use conversation managers:**

```python
# Create conversation manager
conversation = client.create_conversation(
    service_name="claude-sonnet-3.5",
    datasite="aggregator@openmined.org"
)

# Optional: Set system message
conversation.set_system_message("You are a helpful assistant.")

# Configure context retention (default: keeps last 2 exchanges)
conversation.set_max_exchanges(3)  # Keep last 3 exchanges

# Each message remembers previous context
response1 = await conversation.send_message("What is SyftBox?")

# This now prints just the content
print(f"Response:\n{response1}")

# Full object details still available via repr() or explicit access
print(repr(response1))  # Shows full ChatResponse details
print(response1.cost)   # Still works for specific attributes

response2 = await conversation.send_message("How does it work?")  # Remembers previous
response3 = await conversation.send_message("Give me an example")   # Full context

# Get conversation summary
summary = conversation.get_conversation_summary()
print(f"Total messages: {summary['total_messages']}")

# Clear history when needed
conversation.clear_history()
```

## Configuration

```python
# Environment variables
# SYFTBOX_ACCOUNTING_EMAIL
# SYFTBOX_ACCOUNTING_PASSWORD  
# SYFTBOX_ACCOUNTING_URL
# Custom configuration
client = Client(
    user_email="your@email.com",
    cache_server_url="https://custom.syftbox.net",
    auto_setup_accounting=True,
    auto_health_check_threshold=5
)
```