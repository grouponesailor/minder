import re
from typing import Dict, Any, List, Optional
from ..config.settings import settings
from ..models.schemas import SearchResult, SearchResponse, AdvancedSearchFilters
from .elasticsearch_service import ElasticsearchService


class SearchService:
    """Service class for search operations and query building"""
    
    def __init__(self, es_service: ElasticsearchService):
        self.es_service = es_service
        self.cursor_rules = settings.cursor_rules
    
    def determine_search_type(self, query: str) -> Dict[str, Any]:
        """Determine search type based on cursor rules"""
        query_lower = query.lower()
        
        for rule in self.cursor_rules.get("rules", []):
            # Check keywords
            if "keywords" in rule:
                for keyword in rule["keywords"]:
                    if keyword.lower() in query_lower:
                        return rule
            
            # Check patterns (basic pattern matching)
            if "patterns" in rule:
                for pattern in rule["patterns"]:
                    # Convert pattern to regex (replace * with .*)
                    pattern_regex = pattern.replace("*", ".*")
                    if re.search(pattern_regex, query, re.IGNORECASE):
                        return rule
        
        return {"type": "universal", "fields": ["all"]}
    
    def build_multi_index_query(self, query: str, search_rule: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive search with nested person support and better highlighting"""
        
        # Comprehensive query that includes nested person searches
        return {
            "query": {
                "bool": {
                    "should": [
                        # Person-only search (highest priority)
                        {
                            "bool": {
                                "must": [
                                    {"terms": {"_index": ["person_dataset"]}},
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": [
                                                "first_name^10",
                                                "last_name^10", 
                                                "full_name^15",
                                                "first_name.keyword^12",
                                                "last_name.keyword^12",
                                                "full_name.keyword^20"
                                            ],
                                            "type": "best_fields",
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                ],
                                "boost": 10.0
                            }
                        },
                        
                        # Organization search with nested persons (medium-high priority)
                        {
                            "bool": {
                                "must": [
                                    {"terms": {"_index": ["organization_units"]}},
                                    {
                                        "bool": {
                                            "should": [
                                                # Direct organization name match
                                                {
                                                    "multi_match": {
                                                        "query": query,
                                                        "fields": [
                                                            "name^6",
                                                            "name.keyword^8",
                                                            "description^2",
                                                            "type^2"
                                                        ],
                                                        "type": "best_fields",
                                                        "fuzziness": "AUTO"
                                                    }
                                                },
                                                # Nested persons search within organization
                                                {
                                                    "nested": {
                                                        "path": "persons",
                                                        "query": {
                                                            "multi_match": {
                                                                "query": query,
                                                                "fields": [
                                                                    "persons.name^8",
                                                                    "persons.role^3",
                                                                    "persons.email^3"
                                                                ],
                                                                "type": "best_fields",
                                                                "fuzziness": "AUTO"
                                                            }
                                                        },
                                                        "inner_hits": {
                                                            "name": "matching_persons",
                                                            "size": 3,
                                                            "highlight": {
                                                                "fields": {
                                                                    "persons.name": {},
                                                                    "persons.role": {},
                                                                    "persons.email": {}
                                                                }
                                                            }
                                                        }
                                                    }
                                                },
                                                # Manager search
                                                {
                                                    "multi_match": {
                                                        "query": query,
                                                        "fields": [
                                                            "manager.name^6",
                                                            "manager.role^3"
                                                        ],
                                                        "type": "best_fields",
                                                        "fuzziness": "AUTO"
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ],
                                "boost": 7.0
                            }
                        },
                        
                        # System search with nested owners (medium priority)
                        {
                            "bool": {
                                "must": [
                                    {"terms": {"_index": ["systems_index"]}},
                                    {
                                        "bool": {
                                            "should": [
                                                # Direct system name match
                                                {
                                                    "multi_match": {
                                                        "query": query,
                                                        "fields": [
                                                            "name^6",
                                                            "name.keyword^8",
                                                            "description^3",
                                                            "tags^4"
                                                        ],
                                                        "type": "best_fields",
                                                        "fuzziness": "AUTO"
                                                    }
                                                },
                                                # Nested owners search within system
                                                {
                                                    "nested": {
                                                        "path": "owners",
                                                        "query": {
                                                            "multi_match": {
                                                                "query": query,
                                                                "fields": [
                                                                    "owners.name^8",
                                                                    "owners.role^3",
                                                                    "owners.email^3"
                                                                ],
                                                                "type": "best_fields",
                                                                "fuzziness": "AUTO"
                                                            }
                                                        },
                                                        "inner_hits": {
                                                            "name": "matching_owners",
                                                            "size": 3,
                                                            "highlight": {
                                                                "fields": {
                                                                    "owners.name": {},
                                                                    "owners.role": {},
                                                                    "owners.email": {}
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ],
                                "boost": 6.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "highlight": {
                "fields": {
                    "first_name": {},
                    "last_name": {},
                    "full_name": {},
                    "name": {},
                    "description": {},
                    "tags": {},
                    "manager.name": {},
                    "manager.role": {}
                },
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }
        }
    
    def build_advanced_query(self, query: str, filters: AdvancedSearchFilters) -> Dict[str, Any]:
        """Build advanced search query with filters"""
        base_query = self.build_multi_index_query(query, {"type": "universal"})
        
        # Add filters to the query
        filter_conditions = []
        
        if filters.location:
            location_filter = {
                "bool": {
                    "should": [
                        {"wildcard": {"location.city": f"*{filters.location}*"}},
                        {"wildcard": {"location.state": f"*{filters.location}*"}},
                        {"wildcard": {"location.country": f"*{filters.location}*"}},
                        {"wildcard": {"address.city": f"*{filters.location}*"}},
                        {"wildcard": {"address.state": f"*{filters.location}*"}}
                    ]
                }
            }
            filter_conditions.append(location_filter)
        
        if filters.person_role:
            # Simplified role filter without nested query for now
            role_filter = {
                "bool": {
                    "should": [
                        {"wildcard": {"role": f"*{filters.person_role}*"}},
                        {"wildcard": {"position": f"*{filters.person_role}*"}}
                    ]
                }
            }
            filter_conditions.append(role_filter)
        
        if filters.org_type:
            filter_conditions.append({"term": {"type": filters.org_type}})
        
        if filters.org_level:
            filter_conditions.append({"term": {"level": filters.org_level}})
        
        if filters.system_status:
            filter_conditions.append({"term": {"status": filters.system_status}})
        
        # Add filters to query
        if filter_conditions:
            if "filter" not in base_query["query"]["bool"]:
                base_query["query"]["bool"]["filter"] = []
            base_query["query"]["bool"]["filter"].extend(filter_conditions)
        
        return base_query
    
    def get_result_name(self, source: Dict[str, Any], index: str) -> str:
        """Extract appropriate name from source document based on index type"""
        if "full_name" in source and source["full_name"]:
            return source["full_name"]
        elif "first_name" in source and "last_name" in source:
            return f"{source['first_name']} {source['last_name']}"
        elif "name" in source:
            return source["name"]
        else:
            return f"Document from {index}"
    
    def determine_index_type(self, index: str) -> str:
        """Determine the type of index for categorization"""
        if "person" in index:
            return "person"
        elif "organization" in index:
            return "organization"
        elif "system" in index:
            return "system"
        else:
            return "other"
    
    def extract_nested_matches(self, hit: Dict[str, Any]) -> List[str]:
        """Extract information about nested matches (persons, owners)"""
        nested_info = []
        
        # Check for nested person matches in organizations
        if "inner_hits" in hit and "matching_persons" in hit["inner_hits"]:
            for person_hit in hit["inner_hits"]["matching_persons"]["hits"]["hits"]:
                person_source = person_hit["_source"]
                person_name = person_source.get("name", "Unknown Person")
                person_role = person_source.get("role", "")
                if person_role:
                    nested_info.append(f"Person: {person_name} ({person_role})")
                else:
                    nested_info.append(f"Person: {person_name}")
        
        # Check for nested owner matches in systems
        if "inner_hits" in hit and "matching_owners" in hit["inner_hits"]:
            for owner_hit in hit["inner_hits"]["matching_owners"]["hits"]["hits"]:
                owner_source = owner_hit["_source"]
                owner_name = owner_source.get("name", "Unknown Owner")
                owner_role = owner_source.get("role", "")
                if owner_role:
                    nested_info.append(f"Owner: {owner_name} ({owner_role})")
                else:
                    nested_info.append(f"Owner: {owner_name}")
        
        return nested_info
    
    async def universal_search(self, query: str, size: int = 20, index_filter: Optional[str] = None) -> SearchResponse:
        """Perform universal search across all indexes"""
        search_rule = self.determine_search_type(query)
        es_query = self.build_multi_index_query(query, search_rule)
        
        # Determine which indexes to search
        indexes_to_search = self.es_service.target_indexes
        if index_filter and index_filter in self.es_service.target_indexes:
            indexes_to_search = [index_filter]
        
        # Execute search
        response = await self.es_service.search(es_query, indexes_to_search, size)
        
        # Process results
        results = []
        index_breakdown = {index: 0 for index in self.es_service.target_indexes}
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            index = hit["_index"]
            
            # Count results by index
            if index in index_breakdown:
                index_breakdown[index] += 1
            
            # Determine fields matched
            fields_matched = list(hit.get("highlight", {}).keys())
            if not fields_matched:
                fields_matched = ["name", "description"] if "name" in source else ["unknown"]
            
            # Extract nested match information
            nested_matches = self.extract_nested_matches(hit)
            if nested_matches:
                fields_matched.extend(nested_matches)
            
            # Check for exact matches
            exact_match = False
            if index == "person_dataset":
                query_lower = query.lower()
                exact_match = any([
                    source.get("first_name", "").lower() == query_lower,
                    source.get("last_name", "").lower() == query_lower,
                    source.get("full_name", "").lower() == query_lower
                ])
            elif index in ["organization_units", "systems_index"]:
                exact_match = source.get("name", "").lower() == query.lower()
            
            result = SearchResult(
                name=self.get_result_name(source, index),
                score=hit["_score"],
                search_type=search_rule.get("type", "universal"),
                fields_matched=fields_matched,
                exact_match=exact_match,
                source=source,
                highlights=hit.get("highlight", {}),
                index_type=self.determine_index_type(index)
            )
            results.append(result)
        
        return SearchResponse(
            total=response["hits"]["total"]["value"],
            search_type=search_rule.get("type", "universal"),
            query=query,
            results=results,
            index_breakdown=index_breakdown
        )
    
    async def advanced_search(self, query: str, filters: AdvancedSearchFilters, size: int = 20) -> SearchResponse:
        """Perform advanced search with filters"""
        es_query = self.build_advanced_query(query, filters)
        response = await self.es_service.search(es_query, None, size)
        
        # Process results (similar to universal_search)
        results = []
        index_breakdown = {index: 0 for index in self.es_service.target_indexes}
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            index = hit["_index"]
            
            if index in index_breakdown:
                index_breakdown[index] += 1
            
            fields_matched = list(hit.get("highlight", {}).keys())
            if not fields_matched:
                fields_matched = ["name", "description"] if "name" in source else ["unknown"]
            
            # Extract nested match information
            nested_matches = self.extract_nested_matches(hit)
            if nested_matches:
                fields_matched.extend(nested_matches)
            
            result = SearchResult(
                name=self.get_result_name(source, index),
                score=hit["_score"],
                search_type="advanced",
                fields_matched=fields_matched,
                exact_match=hit["_score"] > 5.0,
                source=source,
                highlights=hit.get("highlight", {}),
                index_type=self.determine_index_type(index)
            )
            results.append(result)
        
        return SearchResponse(
            total=len(results),
            search_type="advanced",
            query=query,
            results=results,
            index_breakdown=index_breakdown
        ) 