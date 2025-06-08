from typing import List, Set
from ..models.schemas import AutoCompleteResponse
from .elasticsearch_service import ElasticsearchService


class AutoCompleteService:
    """Service class for autocomplete functionality"""
    
    def __init__(self, es_service: ElasticsearchService):
        self.es_service = es_service
    
    def build_autocomplete_query(self, query: str) -> dict:
        """Build Elasticsearch query for autocomplete with case-insensitive prefix matching"""
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Exact same query structure as the working backup
        autocomplete_query = {
            "bool": {
                "should": [
                    # Person full names with highest priority (case-insensitive)
                    {
                        "prefix": {
                            "full_name.keyword": {
                                "value": query_lower,
                                "case_insensitive": True,
                                "boost": 10.0
                            }
                        }
                    },
                    # Fallback to analyzed field for case-insensitive matching
                    {
                        "prefix": {
                            "full_name": {
                                "value": query_lower,
                                "boost": 9.0
                            }
                        }
                    },
                    # Person first/last names (case-insensitive)
                    {
                        "prefix": {
                            "first_name.keyword": {
                                "value": query_lower,
                                "case_insensitive": True,
                                "boost": 8.0
                            }
                        }
                    },
                    {
                        "prefix": {
                            "first_name": {
                                "value": query_lower,
                                "boost": 7.0
                            }
                        }
                    },
                    {
                        "prefix": {
                            "last_name.keyword": {
                                "value": query_lower,
                                "case_insensitive": True,
                                "boost": 8.0
                            }
                        }
                    },
                    {
                        "prefix": {
                            "last_name": {
                                "value": query_lower,
                                "boost": 7.0
                            }
                        }
                    },
                    # Organization/System names (case-insensitive)
                    {
                        "prefix": {
                            "name.keyword": {
                                "value": query_lower,
                                "case_insensitive": True,
                                "boost": 6.0
                            }
                        }
                    },
                    {
                        "prefix": {
                            "name": {
                                "value": query_lower,
                                "boost": 5.0
                            }
                        }
                    },
                    # Location names (case-insensitive)
                    {
                        "prefix": {
                            "location.city.keyword": {
                                "value": query_lower,
                                "case_insensitive": True,
                                "boost": 4.0
                            }
                        }
                    },
                    {
                        "prefix": {
                            "location.state.keyword": {
                                "value": query_lower,
                                "case_insensitive": True,
                                "boost": 3.0
                            }
                        }
                    },
                    {
                        "prefix": {
                            "location.country.keyword": {
                                "value": query_lower,
                                "case_insensitive": True,
                                "boost": 3.0
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
        
        return {
            "query": autocomplete_query,
            "_source": {
                "includes": [
                    "first_name", "last_name", "full_name", "name", 
                    "location.city", "location.state", "location.country"
                ]
            }
        }
    
    def extract_suggestions(self, response: dict, query: str) -> List[str]:
        """Extract unique suggestions from Elasticsearch response based on prefix matching"""
        suggestions = []
        seen_suggestions = set()  # To avoid duplicates
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            
            # Extract potential suggestion strings
            candidates = []
            
            # Priority 1: Full names (most relevant for "john doe" style results)
            if 'full_name' in source and source['full_name']:
                if source['full_name'].lower().startswith(query.lower()):
                    candidates.append(source['full_name'])
            
            # Priority 2: Combine first + last name for persons
            if 'first_name' in source and 'last_name' in source:
                if source['first_name'] and source['last_name']:
                    full_name_combo = f"{source['first_name']} {source['last_name']}"
                    if full_name_combo.lower().startswith(query.lower()):
                        candidates.append(full_name_combo)
            
            # Priority 3: Individual first/last names
            if 'first_name' in source and source['first_name']:
                if source['first_name'].lower().startswith(query.lower()):
                    candidates.append(source['first_name'])
            
            if 'last_name' in source and source['last_name']:
                if source['last_name'].lower().startswith(query.lower()):
                    candidates.append(source['last_name'])
            
            # Priority 4: Organization/System names
            if 'name' in source and source['name']:
                if source['name'].lower().startswith(query.lower()):
                    candidates.append(source['name'])
            
            # Priority 5: Location names
            if 'location' in source and isinstance(source['location'], dict):
                for location_field in ['city', 'state', 'country']:
                    if location_field in source['location'] and source['location'][location_field]:
                        if source['location'][location_field].lower().startswith(query.lower()):
                            candidates.append(source['location'][location_field])
            
            # Add unique candidates to suggestions
            for candidate in candidates:
                candidate_lower = candidate.lower()
                if candidate_lower not in seen_suggestions:
                    seen_suggestions.add(candidate_lower)
                    suggestions.append(candidate)
        
        # Convert to sorted list (case-insensitive sort)
        return sorted(suggestions, key=str.lower)
    
    async def get_suggestions(self, query: str, size: int = 10) -> AutoCompleteResponse:
        """Get autocomplete suggestions for the given query (prefix matching)"""
        if len(query.strip()) < 2:
            return AutoCompleteResponse(query=query, suggestions=[], total=0)
        
        # Build search body
        search_body = self.build_autocomplete_query(query)
        search_body["size"] = size * 3  # Get more to filter duplicates
        
        # Use the elasticsearch client directly like the working backup
        try:
            response = self.es_service.client.search(
                index=",".join(self.es_service.target_indexes),
                body=search_body
            )
            
            # Extract unique suggestions
            all_suggestions = self.extract_suggestions(response, query)
            
            # Limit to requested size
            suggestions = all_suggestions[:size]
            
            return AutoCompleteResponse(
                query=query,
                suggestions=suggestions,
                total=len(suggestions)
            )
        except Exception as e:
            # Return error details for debugging
            return AutoCompleteResponse(
                query=query,
                suggestions=[f"Error: {str(e)}"],
                total=0
            ) 