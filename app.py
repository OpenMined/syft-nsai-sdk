import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import syft_nsai_sdk as sb

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

def get_project_info():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    project = data["project"]
    return {
        "title": project.get("description", project["name"]),
        "version": project["version"]
    }

project_info = get_project_info()
app = FastAPI(title=project_info["title"], version=project_info["version"])

# Pydantic models for API
class ChatRequest(BaseModel):
    model_name: str
    prompt: str
    owner: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 100

class SearchRequest(BaseModel):
    model_name: str
    query: str
    owner: Optional[str] = None

class ModelResponse(BaseModel):
    name: str
    owner: str
    tags: List[str]
    services: List[str]
    summary: str
    status: str

# Main endpoints
@app.get("/")
async def root():
    return {"message": "SyftBox SDK API", "version": project_info["version"]}

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

# SDK Integration endpoints
@app.get("/sdk/status")
async def sdk_status():
    """Get SDK status information."""
    try:
        status_info = sb.status()
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SDK error: {str(e)}")

@app.get("/models", response_model=List[ModelResponse])
async def list_models(
    owner: Optional[str] = None,
    tag: Optional[str] = None,
    name: Optional[str] = None
):
    """List available models with optional filters."""
    try:
        owners = [owner] if owner else None
        tags = [tag] if tag else None
        
        models = sb.find_models(name=name, tags=tags, owners=owners)
        
        return [
            ModelResponse(
                name=model.name,
                owner=model.owner,
                tags=model.tags,
                services=model.services,
                summary=model.summary,
                status=model.status
            )
            for model in models
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding models: {str(e)}")

@app.get("/models/{model_name}")
async def get_model_info(model_name: str, owner: Optional[str] = None):
    """Get information about a specific model."""
    try:
        model = sb.get_model(model_name, owner=owner)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return ModelResponse(
            name=model.name,
            owner=model.owner,
            tags=model.tags,
            services=model.services,
            summary=model.summary,
            status=model.status
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model: {str(e)}")

@app.post("/models/chat")
async def chat_with_model(request: ChatRequest):
    """Chat with a specific model."""
    try:
        model = sb.get_model(request.model_name, owner=request.owner)
        if not model:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{request.model_name}' not found"
            )
        
        if "chat" not in model.services:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' does not have chat service enabled. Available: {model.services}"
            )
        
        response = model.chat(
            request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "model_name": model.name,
            "model_owner": model.owner,
            "prompt": request.prompt,
            "response": response
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/models/search")
async def search_with_model(request: SearchRequest):
    """Search using a specific model."""
    try:
        model = sb.get_model(request.model_name, owner=request.owner)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        
        if "search" not in model.services:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' does not have search service enabled. Available: {model.services}"
            )
        
        results = model.search(request.query)
        
        return {
            "model_name": model.name,
            "model_owner": model.owner,
            "query": request.query,
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")