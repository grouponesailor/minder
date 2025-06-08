from .elasticsearch_service import ElasticsearchService
from .search_service import SearchService
from .autocomplete_service import AutoCompleteService
from .health_service import HealthService


class ServiceContainer:
    """Dependency injection container for managing service instances"""
    
    def __init__(self):
        # Initialize services
        self._elasticsearch_service = ElasticsearchService()
        self._search_service = SearchService(self._elasticsearch_service)
        self._autocomplete_service = AutoCompleteService(self._elasticsearch_service)
        self._health_service = HealthService(self._elasticsearch_service)
    
    @property
    def elasticsearch_service(self) -> ElasticsearchService:
        return self._elasticsearch_service
    
    @property
    def search_service(self) -> SearchService:
        return self._search_service
    
    @property
    def autocomplete_service(self) -> AutoCompleteService:
        return self._autocomplete_service
    
    @property
    def health_service(self) -> HealthService:
        return self._health_service


# Global container instance
container = ServiceContainer() 