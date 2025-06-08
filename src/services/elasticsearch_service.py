from typing import Dict, Any, List
from elasticsearch import Elasticsearch
from ..config.settings import settings


class ElasticsearchService:
    """Service class for Elasticsearch operations"""
    
    def __init__(self):
        self.client = Elasticsearch(
            hosts=[settings.elasticsearch_url],
            basic_auth=settings.elasticsearch_auth
        )
        self.target_indexes = settings.target_indexes
    
    async def search(self, query: Dict[str, Any], indexes: List[str] = None, size: int = 20) -> Dict[str, Any]:
        """Execute search query across specified indexes"""
        if indexes is None:
            indexes = self.target_indexes
        
        try:
            response = self.client.search(
                index=",".join(indexes),
                body=query,
                size=size
            )
            return response
        except Exception as e:
            raise Exception(f"Elasticsearch search error: {str(e)}")
    
    async def check_index_health(self) -> List[str]:
        """Check which indexes are available and healthy"""
        available_indexes = []
        for index in self.target_indexes:
            try:
                if self.client.indices.exists(index=index):
                    available_indexes.append(index)
            except Exception:
                continue
        return available_indexes
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for all target indexes"""
        stats = {}
        for index in self.target_indexes:
            try:
                if self.client.indices.exists(index=index):
                    index_stats = self.client.indices.stats(index=index)
                    stats[index] = {
                        "doc_count": index_stats["indices"][index]["total"]["docs"]["count"],
                        "size": index_stats["indices"][index]["total"]["store"]["size_in_bytes"]
                    }
            except Exception as e:
                stats[index] = {"error": str(e)}
        return stats 