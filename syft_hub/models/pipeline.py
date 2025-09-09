

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
    
    def __repr__(self):
        """Display pipeline result with metadata in a clean format"""
        lines = [
            "Pipeline Result [Complete]",
            "",
        ]
        
        # Show response content (truncated if too long)
        content_preview = self.content[:200] + "..." if len(self.content) > 200 else self.content
        lines.append(f"Response:        {content_preview}")
        lines.append("")
        
        # Show search results count
        if self.search_results:
            lines.append(f"Search Results:  {len(self.search_results)} documents found")
            # Show first few sources
            sources = []
            for result in self.search_results[:3]:
                if result.metadata and 'filename' in result.metadata:
                    sources.append(result.metadata['filename'])
            if sources:
                lines.append(f"Sources:         {', '.join(sources)}{' ...' if len(self.search_results) > 3 else ''}")
        else:
            lines.append(f"Search Results:  No documents found")
        
        lines.append("")
        
        # Show cost
        lines.append(f"Total Cost:      ${self.cost:.4f}")
        
        # Show model used if available
        if hasattr(self.response, 'model'):
            lines.append(f"Model:           {self.response.model}")
        
        # Show token usage if available
        if hasattr(self.response, 'usage') and self.response.usage:
            lines.append(f"Tokens Used:     {self.response.usage.total_tokens} total")
        
        return "\n".join(lines)