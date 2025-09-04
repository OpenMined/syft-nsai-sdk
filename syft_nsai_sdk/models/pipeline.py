

from typing import List

from ..core.types import DocumentResult
from ..models.responses import ChatResponse


class PipelineResult:
    """Result from pipeline execution"""
    def __init__(self, response: ChatResponse, search_results: List[DocumentResult], cost: float):
        self.response = response
        self.search_results = search_results
        self.cost = cost
        self.content = response.message.content
    
    def __str__(self):
        return self.content