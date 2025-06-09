import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Universal Elasticsearch Search API",
    description="API for searching across persons, organization units, and systems with intelligent universal search",
    version="3.0.0"
)

# Load cursor rules
cursor_rules = {}
try:
    rules_path = Path(__file__).parent.parent / "cursor_rules.json"
    with open(rules_path, 'r', encoding='utf-8') as f:
        cursor_rules = json.load(f)
except Exception as e:
    print(f"Error loading cursor rules: {e}")
    cursor_rules = {"rules": []}

# Create Elasticsearch client
es_client = Elasticsearch(
    hosts=[os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
    basic_auth=(
        os.getenv("ELASTICSEARCH_USERNAME"),
        os.getenv("ELASTICSEARCH_PASSWORD")
    ) if os.getenv("ELASTICSEARCH_USERNAME") else None
)

# Define target indexes
TARGET_INDEXES = ["person_dataset", "organization_units", "systems_index"]

# Pydantic models
class SearchResult(BaseModel):
    name: str
    score: float
    search_type: str
    fields_matched: List[str]
    exact_match: bool
    source: Dict[str, Any]
    highlights: Dict[str, Any] = {}
    index_type: str

class SearchResponse(BaseModel):
    total: int
    search_type: str
    query: str
    results: List[SearchResult]
    index_breakdown: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    rules_loaded: bool
    indexes_available: List[str]

class AutoCompleteResponse(BaseModel):
    query: str
    suggestions: List[str]
    total: int

def determine_search_type(query: str) -> Dict[str, Any]:
    """Determine search type based on cursor rules"""
    query_lower = query.lower()
    
    for rule in cursor_rules.get("rules", []):
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

def build_multi_index_query(query: str, search_rule: Dict[str, Any]) -> Dict[str, Any]:
    """Build case-insensitive Elasticsearch query with exact string matching - no fuzzy matching"""
    
    # Normalize query for case-insensitive matching
    query_lower = query.lower()
    
    # Case-insensitive query focusing on exact substring matching
    universal_query = {
        "bool": {
            "should": [
                # Case-insensitive exact keyword matches using regexp
                {
                    "regexp": {
                        "first_name.keyword": {
                            "value": f"(?i){re.escape(query)}",
                            "boost": 10.0
                        }
                    }
                },
                {
                    "regexp": {
                        "last_name.keyword": {
                            "value": f"(?i){re.escape(query)}",
                            "boost": 10.0
                        }
                    }
                },
                {
                    "regexp": {
                        "full_name.keyword": {
                            "value": f"(?i){re.escape(query)}",
                            "boost": 12.0
                        }
                    }
                },
                {
                    "regexp": {
                        "name.keyword": {
                            "value": f"(?i){re.escape(query)}",
                            "boost": 10.0
                        }
                    }
                },
                
                # Case-insensitive wildcard searches using regexp
                {
                    "regexp": {
                        "first_name.keyword": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 4.0
                        }
                    }
                },
                {
                    "regexp": {
                        "last_name.keyword": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 4.0
                        }
                    }
                },
                {
                    "regexp": {
                        "full_name.keyword": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 5.0
                        }
                    }
                },
                {
                    "regexp": {
                        "name.keyword": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 4.0
                        }
                    }
                },
                
                # Text field matches (case-insensitive by default due to analyzers)
                {
                    "match": {
                        "first_name": {
                            "query": query,
                            "boost": 6.0
                        }
                    }
                },
                {
                    "match": {
                        "last_name": {
                            "query": query,
                            "boost": 6.0
                        }
                    }
                },
                {
                    "match": {
                        "full_name": {
                            "query": query,
                            "boost": 7.0
                        }
                    }
                },
                {
                    "match": {
                        "name": {
                            "query": query,
                            "boost": 6.0
                        }
                    }
                },
                {
                    "match": {
                        "description": {
                            "query": query,
                            "boost": 3.0
                        }
                    }
                },
                
                # Additional case-insensitive keyword field searches
                {
                    "regexp": {
                        "email": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 3.0
                        }
                    }
                },
                {
                    "regexp": {
                        "work_phone": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 2.5
                        }
                    }
                },
                {
                    "regexp": {
                        "mobile_phone": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 2.5
                        }
                    }
                },
                {
                    "regexp": {
                        "location.city": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 3.0
                        }
                    }
                },
                {
                    "regexp": {
                        "location.country": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 2.0
                        }
                    }
                },
                {
                    "regexp": {
                        "interests": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 2.0
                        }
                    }
                },
                {
                    "regexp": {
                        "gender": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 1.5
                        }
                    }
                },
                {
                    "regexp": {
                        "type": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 3.0
                        }
                    }
                },
                {
                    "regexp": {
                        "status": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 2.0
                        }
                    }
                },
                {
                    "regexp": {
                        "tags": {
                            "value": f"(?i).*{re.escape(query)}.*",
                            "boost": 3.0
                        }
                    }
                }
            ],
            "minimum_should_match": 1
        }
    }
    
    return universal_query

def get_result_name(source: Dict[str, Any], index: str) -> str:
    """Extract appropriate name based on document type and index"""
    if index == "person_dataset":
        return source.get("full_name", f"{source.get('first_name', '')} {source.get('last_name', '')}").strip()
    elif index == "organization_units":
        return source.get("name", "Unknown Organization")
    elif index == "systems_index":
        return source.get("name", "Unknown System")
    else:
        return source.get("name", source.get("full_name", "Unknown"))

def determine_index_type(index: str) -> str:
    """Determine the type of result based on index"""
    if index == "person_dataset":
        return "person"
    elif index == "organization_units":
        return "organization"
    elif index == "systems_index":
        return "system"
    else:
        return "unknown"

@app.get("/api/search", response_model=SearchResponse)
@app.get("/api/search/universal", response_model=SearchResponse)
async def universal_multi_index_search(
    q: str = Query(..., description="Search query - searches across persons, organizations, and systems"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100),
    index_filter: Optional[str] = Query(None, description="Filter by specific index: person_dataset, organization_units, or systems_index")
):
    """
    Universal search across all indexes including:
    - Persons: Names, contact info, location, professional experience, cars, interests
    - Organization Units: Names, types, hierarchy, persons, locations, managers
    - Systems: Names, descriptions, links, owners, tags, status
    """
    try:
        # Determine search type based on cursor rules
        search_rule = determine_search_type(q)
        
        # Build the comprehensive Elasticsearch query
        query = build_multi_index_query(q, search_rule)
        
        # Determine which indexes to search
        indexes_to_search = TARGET_INDEXES
        if index_filter and index_filter in TARGET_INDEXES:
            indexes_to_search = [index_filter]
        
        # Execute search across multiple indexes
        response = es_client.search(
            index=",".join(indexes_to_search),
            body={
                "size": size,
                "query": query,
                "highlight": {
                    "fields": {
                        "first_name": {},
                        "last_name": {},
                        "full_name": {},
                        "name": {},
                        "description": {},
                        "location.city": {},
                        "location.country": {},
                        "location.state": {},
                        "address.city": {},
                        "address.street": {},
                        "professional_experience.company": {},
                        "professional_experience.position": {},
                        "professional_experience.skills": {},
                        "professional_experience.description": {},
                        "persons.name": {},
                        "persons.role": {},
                        "owners.name": {},
                        "owners.role": {},
                        "cars.make": {},
                        "cars.model": {},
                        "cars.color": {},
                        "interests": {},
                        "work_phone": {},
                        "mobile_phone": {},
                        "type": {},
                        "tags": {},
                        "hierarchy_names": {},
                        "manager.name": {},
                        "link": {}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "_source": True,
                "sort": [
                    {"_score": {"order": "desc"}}
                ]
            }
        )
        
        # Process results and track index breakdown
        hits = []
        index_breakdown = {index: 0 for index in TARGET_INDEXES}
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            index = hit["_index"]
            index_breakdown[index] = index_breakdown.get(index, 0) + 1
            
            result_name = get_result_name(source, index)
            
            # Determine which fields matched based on highlights
            fields_matched = []
            highlights = hit.get("highlight", {})
            if highlights:
                fields_matched = list(highlights.keys())
            else:
                fields_matched = ["various_fields"]
            
            # Check for exact matches based on document type
            exact_match = False
            if index == "person_dataset":
                exact_match = any([
                    source.get("first_name", "").lower() == q.lower(),
                    source.get("last_name", "").lower() == q.lower(),
                    result_name.lower() == q.lower()
                ])
            elif index in ["organization_units", "systems_index"]:
                exact_match = source.get("name", "").lower() == q.lower()
            
            hits.append(SearchResult(
                name=result_name,
                score=hit["_score"],
                search_type=search_rule["type"],
                fields_matched=fields_matched,
                exact_match=exact_match,
                source=source,
                highlights=highlights,
                index_type=determine_index_type(index)
            ))
        
        return SearchResponse(
            total=response["hits"]["total"]["value"],
            search_type=search_rule["type"],
            query=q,
            results=hits,
            index_breakdown=index_breakdown
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing multi-index search: {str(e)}"
        )

# Keep the legacy endpoint for backward compatibility
@app.get("/api/search/name", response_model=SearchResponse)
async def search_name(
    name: str = Query(..., description="The name to search for")
):
    """Legacy name search endpoint - redirects to universal search"""
    return await universal_multi_index_search(q=name, size=20)

@app.get("/api/search/persons")
async def search_persons_only(
    q: str = Query(..., description="Search query for persons only"),
    size: int = Query(20, description="Number of results", ge=1, le=100)
):
    """Search only in person_dataset index"""
    return await universal_multi_index_search(q=q, size=size, index_filter="person_dataset")

@app.get("/api/search/organizations")
async def search_organizations_only(
    q: str = Query(..., description="Search query for organizations only"),
    size: int = Query(20, description="Number of results", ge=1, le=100)
):
    """Search only in organization_units index"""
    return await universal_multi_index_search(q=q, size=size, index_filter="organization_units")

@app.get("/api/search/systems")
async def search_systems_only(
    q: str = Query(..., description="Search query for systems only"),
    size: int = Query(20, description="Number of results", ge=1, le=100)
):
    """Search only in systems_index"""
    return await universal_multi_index_search(q=q, size=size, index_filter="systems_index")

@app.get("/api/search/advanced")
async def advanced_multi_index_search(
    q: str = Query(..., description="Search query"),
    location: Optional[str] = Query(None, description="Filter by location (city, state, or country)"),
    person_role: Optional[str] = Query(None, description="Filter by person role"),
    org_type: Optional[str] = Query(None, description="Filter by organization type"),
    org_level: Optional[int] = Query(None, description="Filter by organization level (1-6)"),
    system_status: Optional[str] = Query(None, description="Filter by system status"),
    size: int = Query(20, description="Number of results", ge=1, le=100)
):
    """
    Advanced search with additional filters across all indexes
    """
    try:
        # Start with universal query
        base_query = build_multi_index_query(q, {"type": "advanced"})
        
        # Add filters if provided
        filters = []
        
        if location:
            filters.append({
                "bool": {
                    "should": [
                        {"match": {"location.city": location}},
                        {"match": {"location.state": location}},
                        {"match": {"location.country": location}},
                        {"match": {"address.city": location}},
                        {"match": {"address.state": location}}
                    ]
                }
            })
        
        if person_role:
            filters.append({
                "bool": {
                    "should": [
                        {
                            "nested": {
                                "path": "professional_experience",
                                "query": {"match": {"professional_experience.position": person_role}}
                            }
                        },
                        {
                            "nested": {
                                "path": "persons",
                                "query": {"match": {"persons.role": person_role}}
                            }
                        }
                    ]
                }
            })
        
        if org_type:
            filters.append({"match": {"type": org_type}})
        
        if org_level:
            filters.append({"term": {"level": org_level}})
        
        if system_status:
            filters.append({"match": {"status": system_status}})
        
        # Combine base query with filters
        if filters:
            final_query = {
                "bool": {
                    "must": [base_query],
                    "filter": filters
                }
            }
        else:
            final_query = base_query
        
        response = es_client.search(
            index=",".join(TARGET_INDEXES),
            body={
                "size": size,
                "query": final_query,
                "highlight": {
                    "fields": {"*": {}}
                }
            }
        )
        
        results = []
        index_breakdown = {index: 0 for index in TARGET_INDEXES}
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            index = hit["_index"]
            index_breakdown[index] = index_breakdown.get(index, 0) + 1
            
            results.append({
                "name": get_result_name(source, index),
                "score": hit["_score"],
                "index_type": determine_index_type(index),
                "source": source,
                "highlights": hit.get("highlight", {})
            })
        
        return {
            "total": response["hits"]["total"]["value"],
            "query": q,
            "filters": {
                "location": location,
                "person_role": person_role,
                "org_type": org_type,
                "org_level": org_level,
                "system_status": system_status
            },
            "index_breakdown": index_breakdown,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing advanced search: {str(e)}")

@app.get("/api/search/rules")
async def get_search_rules():
    """Get available search rules from cursor_rules.json"""
    return cursor_rules

@app.get("/api/auto_complete", response_model=AutoCompleteResponse)
async def auto_complete(
    q: str = Query(..., description="Partial search query (minimum 2 characters)", min_length=2),
    size: int = Query(10, description="Number of suggestions to return", ge=1, le=10)
):
    """
    Get autocomplete suggestions for search queries
    Returns only matching strings (e.g., 'jo' -> 'john doe')
    Requires minimum 2 characters and returns top 10 results
    """
    if len(q.strip()) < 2:
        return AutoCompleteResponse(query=q, suggestions=[], total=0)
    
    # Convert query to lowercase for case-insensitive matching
    q_lower = q.lower()
    
    # Build autocomplete query using prefix matching for clean string results
    autocomplete_query = {
        "bool": {
            "should": [
                # Person full names with highest priority (case-insensitive)
                {
                    "prefix": {
                        "full_name.keyword": {
                            "value": q_lower,
                            "case_insensitive": True,
                            "boost": 10.0
                        }
                    }
                },
                # Fallback to analyzed field for case-insensitive matching
                {
                    "prefix": {
                        "full_name": {
                            "value": q_lower,
                            "boost": 9.0
                        }
                    }
                },
                # Person first/last names (case-insensitive)
                {
                    "prefix": {
                        "first_name.keyword": {
                            "value": q_lower,
                            "case_insensitive": True,
                            "boost": 8.0
                        }
                    }
                },
                {
                    "prefix": {
                        "first_name": {
                            "value": q_lower,
                            "boost": 7.0
                        }
                    }
                },
                {
                    "prefix": {
                        "last_name.keyword": {
                            "value": q_lower,
                            "case_insensitive": True,
                            "boost": 8.0
                        }
                    }
                },
                {
                    "prefix": {
                        "last_name": {
                            "value": q_lower,
                            "boost": 7.0
                        }
                    }
                },
                # Organization/System names (case-insensitive)
                {
                    "prefix": {
                        "name.keyword": {
                            "value": q_lower,
                            "case_insensitive": True,
                            "boost": 6.0
                        }
                    }
                },
                {
                    "prefix": {
                        "name": {
                            "value": q_lower,
                            "boost": 5.0
                        }
                    }
                },
                # Location names (case-insensitive)
                {
                    "prefix": {
                        "location.city.keyword": {
                            "value": q_lower,
                            "case_insensitive": True,
                            "boost": 4.0
                        }
                    }
                },
                {
                    "prefix": {
                        "location.state.keyword": {
                            "value": q_lower,
                            "case_insensitive": True,
                            "boost": 3.0
                        }
                    }
                },
                {
                    "prefix": {
                        "location.country.keyword": {
                            "value": q_lower,
                            "case_insensitive": True,
                            "boost": 3.0
                        }
                    }
                }
            ],
            "minimum_should_match": 1
        }
    }
    
    # Search configuration - get more results to ensure we have enough unique strings
    search_body = {
        "query": autocomplete_query,
        "size": size * 3,  # Get more to filter duplicates
        "_source": {
            "includes": [
                "first_name", "last_name", "full_name", "name", 
                "location.city", "location.state", "location.country"
            ]
        }
    }
    
    try:
        # Search across all indexes
        response = es_client.search(
            index=",".join(TARGET_INDEXES),
            body=search_body
        )
        
        suggestions = []
        seen_suggestions = set()  # To avoid duplicates
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Extract potential suggestion strings
            candidates = []
            
            # Priority 1: Full names (most relevant for "john doe" style results)
            if 'full_name' in source and source['full_name']:
                if source['full_name'].lower().startswith(q.lower()):
                    candidates.append(source['full_name'])
            
            # Priority 2: Combine first + last name for persons
            if 'first_name' in source and 'last_name' in source:
                if source['first_name'] and source['last_name']:
                    full_name_combo = f"{source['first_name']} {source['last_name']}"
                    if full_name_combo.lower().startswith(q.lower()):
                        candidates.append(full_name_combo)
            
            # Priority 3: Individual first/last names
            if 'first_name' in source and source['first_name']:
                if source['first_name'].lower().startswith(q.lower()):
                    candidates.append(source['first_name'])
            
            if 'last_name' in source and source['last_name']:
                if source['last_name'].lower().startswith(q.lower()):
                    candidates.append(source['last_name'])
            
            # Priority 4: Organization/System names
            if 'name' in source and source['name']:
                if source['name'].lower().startswith(q.lower()):
                    candidates.append(source['name'])
            
            # Priority 5: Location names
            if 'location' in source and isinstance(source['location'], dict):
                for location_field in ['city', 'state', 'country']:
                    if location_field in source['location'] and source['location'][location_field]:
                        if source['location'][location_field].lower().startswith(q.lower()):
                            candidates.append(source['location'][location_field])
            
            # Add unique candidates to suggestions
            for candidate in candidates:
                candidate_lower = candidate.lower()
                if candidate_lower not in seen_suggestions:
                    seen_suggestions.add(candidate_lower)
                    suggestions.append(candidate)
                    
                    # Stop when we have enough suggestions
                    if len(suggestions) >= size:
                        break
            
            if len(suggestions) >= size:
                break
        
        # Limit to requested size
        suggestions = suggestions[:size]
        
        return AutoCompleteResponse(
            query=q,
            suggestions=suggestions,
            total=len(suggestions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto complete error: {str(e)}")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    
    # Check which indexes are available
    available_indexes = []
    for index in TARGET_INDEXES:
        try:
            es_client.indices.get(index=index)
            available_indexes.append(index)
        except:
            pass
    
    return HealthResponse(
        status="OK",
        timestamp=datetime.now().isoformat(),
        rules_loaded=bool(cursor_rules.get("rules")),
        indexes_available=available_indexes
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Universal Elasticsearch Search API - Multi-Index Search",
        "version": "3.0.0",
        "description": "Search across persons, organization units, and systems with a single query",
        "indexes": TARGET_INDEXES,
        "endpoints": {
            "universal_search": "/api/search?q=<query>",
            "persons_only": "/api/search/persons?q=<query>",
            "organizations_only": "/api/search/organizations?q=<query>",
            "systems_only": "/api/search/systems?q=<query>",
            "advanced_search": "/api/search/advanced?q=<query>&location=<location>&org_type=<type>",
            "auto_complete": "/api/auto_complete?q=<partial_query>",
            "legacy_name_search": "/api/search/name?name=<name>",
            "health": "/api/health",
            "docs": "/docs"
        },
        "examples": {
            "search_by_name": "/api/search?q=John",
            "search_by_skill": "/api/search?q=React",
            "search_by_location": "/api/search?q=Tokyo",
            "search_by_company": "/api/search?q=Technology",
            "search_by_system": "/api/search?q=CRM",
            "auto_complete_names": "/api/auto_complete?q=jo",
            "auto_complete_locations": "/api/auto_complete?q=Ne",
            "auto_complete_organizations": "/api/auto_complete?q=Te",
            "search_persons_only": "/api/search/persons?q=engineer",
            "search_orgs_only": "/api/search/organizations?q=division",
            "search_systems_only": "/api/search/systems?q=management",
            "advanced_search": "/api/search/advanced?q=developer&location=USA&org_type=team"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Universal Search Server on port {port}")
    print(f"Loaded {len(cursor_rules.get('rules', []))} search rules")
    print(f"Target indexes: {', '.join(TARGET_INDEXES)}")
    uvicorn.run(app, host="0.0.0.0", port=port) 