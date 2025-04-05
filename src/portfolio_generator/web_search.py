import asyncio
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI


class PerplexitySearch:
    """Class to handle web searches using the Perplexity API."""
    
    def __init__(self, api_key: str):
        """Initialize with Perplexity API key."""
        self.api_key = api_key.strip('"\'')
        # Using OpenAI client with Perplexity base URL
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")
        
    async def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Search the web using Perplexity API for the given queries.
        
        Args:
            queries: List of search queries to execute
            
        Returns:
            List of search result objects
        """
        tasks = [self._search_single_query(query) for query in queries]
        return await asyncio.gather(*tasks)
    
    async def _search_single_query(self, query: str) -> Dict[str, Any]:
        """Execute a search for a single query using OpenAI client with Perplexity."""
        try:
            # Create messages for the search query
            messages = [
                {
                    "role": "system",
                    "content": "You are a search assistant that processes search queries and returns factual information about current events and data."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            # Use the OpenAI client with sonar-pro model (which has web search capability)
            response = self.client.chat.completions.create(
                model="sonar-pro",
                messages=messages
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            
            # Format the response in a way that's compatible with our existing code
            return {
                "query": query,
                "results": [
                    {
                        "title": "Perplexity Search Result",
                        "url": "https://perplexity.ai/search",
                        "content": response_content,
                        "raw_content": response_content
                    }
                ]
            }
            
        except Exception as e:
            error_msg = f"Exception searching '{query}': {str(e)}"
            print(error_msg)
            return {
                "query": query, 
                "results": [],
                "error": "exception",
                "message": error_msg
            }


def format_search_results(search_results: List[Dict], max_chars_per_source: int = 4000) -> str:
    """
    Format search results into a string for the model.
    
    Args:
        search_results: List of search result objects
        max_chars_per_source: Maximum characters to include per source
        
    Returns:
        Formatted string with search results
    """
    if not search_results:
        return "No search results found."
        
    # Collect all results
    sources_list = []
    for response in search_results:
        sources_list.extend(response.get('results', []))
    
    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}
    
    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"URL: {source['url']}\n"
        formatted_text += f"Most relevant content: {source['content']}\n"
        
        # Add raw content if available
        raw_content = source.get('raw_content', '')
        if raw_content and len(raw_content) > max_chars_per_source:
            raw_content = raw_content[:max_chars_per_source] + "... [truncated]"
        if raw_content:
            formatted_text += f"Full content:\n{raw_content}\n"
            
        formatted_text += f"{'='*80}\n\n"
        
    return formatted_text
