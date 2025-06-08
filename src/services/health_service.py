from datetime import datetime
from typing import Dict, Any
from ..models.schemas import HealthResponse
from ..config.settings import settings
from .elasticsearch_service import ElasticsearchService


class HealthService:
    """Service class for health checks and system status"""
    
    def __init__(self, es_service: ElasticsearchService):
        self.es_service = es_service
    
    async def get_health_status(self) -> HealthResponse:
        """Get comprehensive health status of the application"""
        try:
            # Check available indexes
            available_indexes = await self.es_service.check_index_health()
            
            # Check if cursor rules are loaded
            rules_loaded = bool(settings.cursor_rules.get("rules"))
            
            return HealthResponse(
                status="OK",
                timestamp=datetime.now().isoformat(),
                rules_loaded=rules_loaded,
                indexes_available=available_indexes
            )
        except Exception as e:
            return HealthResponse(
                status=f"ERROR: {str(e)}",
                timestamp=datetime.now().isoformat(),
                rules_loaded=False,
                indexes_available=[]
            )
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status including index statistics"""
        try:
            available_indexes = await self.es_service.check_index_health()
            index_stats = await self.es_service.get_index_stats()
            
            return {
                "status": "OK",
                "timestamp": datetime.now().isoformat(),
                "elasticsearch": {
                    "url": settings.elasticsearch_url,
                    "available_indexes": available_indexes,
                    "target_indexes": settings.target_indexes,
                    "index_stats": index_stats
                },
                "configuration": {
                    "rules_loaded": bool(settings.cursor_rules.get("rules")),
                    "total_rules": len(settings.cursor_rules.get("rules", [])),
                    "default_search_size": settings.default_search_size,
                    "max_search_size": settings.max_search_size,
                    "min_autocomplete_chars": settings.min_autocomplete_chars,
                    "max_autocomplete_results": settings.max_autocomplete_results
                },
                "api": {
                    "title": settings.api_title,
                    "version": settings.api_version
                }
            }
        except Exception as e:
            return {
                "status": f"ERROR: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error_details": str(e)
            } 