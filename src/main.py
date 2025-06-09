import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import openai
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure OpenAI
openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
) if os.getenv("OPENAI_API_KEY") else None

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

class NaturalLanguageResponse(BaseModel):
    original_query: str
    interpreted_query: str
    search_strategy: str
    answer: str
    results: List[SearchResult]
    total: int
    confidence: float

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

def build_semantic_search_query(query: str, search_rule: Dict[str, Any]) -> Dict[str, Any]:
    """Build semantic search query with fuzzy matching, stemming, and intelligent relevance"""
    
    # Semantic search query with multiple matching strategies
    semantic_query = {
        "bool": {
            "should": [
                # High priority: Exact phrase matches
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "first_name.keyword^10", "last_name.keyword^10", "full_name.keyword^12",
                            "name.keyword^10"
                        ],
                        "type": "phrase",
                        "boost": 8.0
                    }
                },
                
                # Medium-high priority: Best fields semantic matching
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "first_name^6", "last_name^6", "full_name^7", "name^6",
                            "description^4"
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "prefix_length": 1,
                        "max_expansions": 50,
                        "boost": 6.0
                    }
                },
                
                # Medium priority: Cross-fields semantic matching
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "first_name^4", "last_name^4", "full_name^5", "name^4",
                            "description^3", "email^2"
                        ],
                        "type": "cross_fields",
                        "operator": "and",
                        "boost": 4.0
                    }
                },
                
                # Phrase matching with slop for flexible word order
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "first_name^3", "last_name^3", "full_name^4", "name^3",
                            "description^2"
                        ],
                        "type": "phrase",
                        "slop": 2,
                        "boost": 3.0
                    }
                },
                
                # Fuzzy matching for typos and variations
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "first_name^2", "last_name^2", "full_name^3", "name^2",
                            "description^1.5", "email^1", "work_phone^1", "mobile_phone^1"
                        ],
                        "type": "most_fields",
                        "fuzziness": "AUTO",
                        "prefix_length": 0,
                        "max_expansions": 100,
                        "boost": 2.0
                    }
                },
                
                # Wildcard matching for partial terms
                {
                    "query_string": {
                        "query": f"*{query}*",
                        "fields": [
                            "first_name.keyword^2", "last_name.keyword^2", "full_name.keyword^3",
                            "name.keyword^2", "email^1.5", "location.city^2", "location.country^1"
                        ],
                        "boost": 1.5
                    }
                },
                
                # Boolean query for compound searches
                {
                    "simple_query_string": {
                        "query": query,
                        "fields": [
                            "first_name^3", "last_name^3", "full_name^4", "name^3",
                            "description^2", "interests^2", "gender^1"
                        ],
                        "default_operator": "or",
                        "analyze_wildcard": True,
                        "boost": 1.0
                    }
                }
            ],
            "minimum_should_match": 1
        }
    }
    
    # Add conditional nested semantic searches for complex fields
    # These are wrapped in try/ignore blocks since not all indexes have these fields
    nested_semantic_queries = []
    
    # Professional experience semantic search (only for person_dataset)
    nested_semantic_queries.append({
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": "professional_experience",
                        "query": {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "professional_experience.company^4",
                                    "professional_experience.position^4",
                                    "professional_experience.description^3",
                                    "professional_experience.skills^5"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": 3.0
                            }
                        },
                        "ignore_unmapped": True
                    }
                }
            ],
            "minimum_should_match": 0
        }
    })
    
    # Organization persons semantic search (only for organization_units)
    nested_semantic_queries.append({
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": "persons",
                        "query": {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "persons.name^4",
                                    "persons.role^3",
                                    "persons.email^2"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": 2.5
                            }
                        },
                        "ignore_unmapped": True
                    }
                }
            ],
            "minimum_should_match": 0
        }
    })
    
    # System owners semantic search (only for systems_index)
    nested_semantic_queries.append({
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": "owners",
                        "query": {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "owners.name^4",
                                    "owners.email^2",
                                    "owners.role^3"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": 2.5
                            }
                        },
                        "ignore_unmapped": True
                    }
                }
            ],
            "minimum_should_match": 0
        }
    })
    
    # Cars semantic search (only for person_dataset)
    nested_semantic_queries.append({
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": "cars",
                        "query": {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "cars.make^3",
                                    "cars.model^3",
                                    "cars.color^2",
                                    "cars.license_plate^1"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": 1.5
                            }
                        },
                        "ignore_unmapped": True
                    }
                }
            ],
            "minimum_should_match": 0
        }
    })
    
    # Add nested queries to the main query with lower priority
    semantic_query["bool"]["should"].extend(nested_semantic_queries)
    
    return semantic_query

def parse_natural_language_query(query: str) -> Dict[str, Any]:
    """Parse natural language query using LLM and convert to search parameters"""
    
    if not openai_client:
        # Fallback parsing without LLM
        return {
            "search_terms": [query],
            "search_type": "universal",
            "fields": ["name", "first_name", "last_name", "full_name"],
            "answer_format": "direct",
            "confidence": 0.3
        }
    
    try:
        # System prompt to help the LLM understand our data structure
        system_prompt = """
You are a search query parser for an Elasticsearch database containing:

1. PERSON_DATASET: People with fields like first_name, last_name, full_name, email, work_phone, mobile_phone, location (city, state, country), professional_experience (company, position, skills), cars (make, model, color), interests, gender
2. ORGANIZATION_UNITS: Departments/teams with fields like name, type, level, location, persons (employees with name, role, email), description
3. SYSTEMS_INDEX: IT systems with fields like name, description, status, owners (name, email, role), type

Convert natural language queries to structured search parameters.

Return JSON with:
- search_terms: array of key terms to search for
- search_type: "person", "organization", "system", or "universal"
- fields: specific fields to focus on
- filters: any specific filters (location, role, status, etc.)
- answer_format: "direct" (simple answer), "list" (multiple results), or "detailed" (full information)
- confidence: 0.0-1.0 how confident you are in the interpretation

Examples:
"who is the manager of sales department?" -> {"search_terms": ["sales", "manager"], "search_type": "organization", "fields": ["name", "persons.name", "persons.role"], "filters": {"role": "manager"}, "answer_format": "direct", "confidence": 0.9}
"find people in Tokyo" -> {"search_terms": ["Tokyo"], "search_type": "person", "fields": ["location.city"], "filters": {"location": "Tokyo"}, "answer_format": "list", "confidence": 0.95}
"""

        user_prompt = f"Parse this query: '{query}'"
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        # Parse the LLM response
        llm_response = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Remove any markdown formatting
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0]
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1].split("```")[0]
            
            parsed_query = json.loads(llm_response)
            return parsed_query
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "search_terms": [query],
                "search_type": "universal",
                "fields": ["name", "first_name", "last_name", "full_name"],
                "answer_format": "direct",
                "confidence": 0.4
            }
            
    except Exception as e:
        print(f"Error in LLM parsing: {e}")
        # Fallback parsing
        return {
            "search_terms": [query],
            "search_type": "universal", 
            "fields": ["name", "first_name", "last_name", "full_name"],
            "answer_format": "direct",
            "confidence": 0.3
        }

def build_natural_language_query(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    """Build Elasticsearch query from parsed natural language query"""
    
    search_terms = parsed_query.get("search_terms", [])
    search_type = parsed_query.get("search_type", "universal")
    fields = parsed_query.get("fields", ["name"])
    filters = parsed_query.get("filters", {})
    
    # Build main search query
    should_clauses = []
    
    for term in search_terms:
        # Add matches for each field
        for field in fields:
            # Exact matches
            should_clauses.append({
                "match": {
                    field: {
                        "query": term,
                        "boost": 5.0
                    }
                }
            })
            
            # Keyword matches for exact terms
            if field.endswith(".keyword") or field in ["name", "first_name", "last_name", "full_name"]:
                keyword_field = f"{field}.keyword" if not field.endswith(".keyword") else field
                should_clauses.append({
                    "regexp": {
                        keyword_field: {
                            "value": f"(?i).*{re.escape(term)}.*",
                            "boost": 3.0
                        }
                    }
                })
        
        # Add nested searches for organization persons
        should_clauses.append({
            "nested": {
                "path": "persons",
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"persons.name": {"query": term, "boost": 4.0}}},
                            {"match": {"persons.role": {"query": term, "boost": 3.0}}}
                        ]
                    }
                },
                "ignore_unmapped": True
            }
        })
        
        # Add nested searches for system owners
        should_clauses.append({
            "nested": {
                "path": "owners",
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"owners.name": {"query": term, "boost": 4.0}}},
                            {"match": {"owners.role": {"query": term, "boost": 3.0}}}
                        ]
                    }
                },
                "ignore_unmapped": True
            }
        })

    # Build the main query
    main_query = {
        "bool": {
            "should": should_clauses,
            "minimum_should_match": 1
        }
    }
    
    # Add filters if any
    if filters:
        filter_clauses = []
        
        for filter_field, filter_value in filters.items():
            if filter_field == "role":
                # Search in nested role fields
                filter_clauses.append({
                    "bool": {
                        "should": [
                            {
                                "nested": {
                                    "path": "persons",
                                    "query": {"match": {"persons.role": filter_value}},
                                    "ignore_unmapped": True
                                }
                            },
                            {
                                "nested": {
                                    "path": "owners", 
                                    "query": {"match": {"owners.role": filter_value}},
                                    "ignore_unmapped": True
                                }
                            }
                        ]
                    }
                })
            elif filter_field == "location":
                # Search in location fields
                filter_clauses.append({
                    "bool": {
                        "should": [
                            {"match": {"location.city": filter_value}},
                            {"match": {"location.state": filter_value}},
                            {"match": {"location.country": filter_value}}
                        ]
                    }
                })
            else:
                # Generic field filter
                filter_clauses.append({
                    "match": {filter_field: filter_value}
                })
        
        if filter_clauses:
            main_query = {
                "bool": {
                    "must": [main_query],
                    "filter": filter_clauses
                }
            }
    
    return main_query

def format_natural_language_answer(query: str, parsed_query: Dict[str, Any], results: List[SearchResult]) -> str:
    """Generate a natural language answer based on the query and results"""
    
    if not results:
        return f"I couldn't find any information matching '{query}'. Please try rephrasing your question or being more specific."
    
    answer_format = parsed_query.get("answer_format", "direct")
    search_type = parsed_query.get("search_type", "universal")
    
    if answer_format == "direct" and len(results) == 1:
        result = results[0]
        name = result.name
        
        if "manager" in query.lower() or "head" in query.lower():
            return f"The manager/head is {name}."
        elif "who is" in query.lower():
            return f"That would be {name}."
        else:
            return f"I found: {name}."
    
    elif answer_format == "direct" and len(results) > 1:
        if "manager" in query.lower():
            names = [r.name for r in results[:3]]
            if len(names) == 1:
                return f"The manager is {names[0]}."
            elif len(names) == 2:
                return f"The managers are {names[0]} and {names[1]}."
            else:
                return f"The managers include {', '.join(names[:-1])}, and {names[-1]}."
        else:
            top_result = results[0]
            return f"The most relevant result is {top_result.name}. There are {len(results)} total matches."
    
    elif answer_format == "list":
        names = [r.name for r in results[:5]]
        if len(results) == 1:
            return f"I found 1 person: {names[0]}."
        elif len(results) <= 5:
            return f"I found {len(results)} people: {', '.join(names)}."
        else:
            return f"I found {len(results)} people. The top 5 are: {', '.join(names)}."
    
    else:
        # Default detailed response
        if len(results) == 1:
            result = results[0]
            details = []
            source = result.source
            
            if search_type == "person":
                if source.get("location"):
                    loc = source["location"]
                    location_str = f"{loc.get('city', '')}, {loc.get('state', '')}, {loc.get('country', '')}"
                    details.append(f"Location: {location_str.strip(', ')}")
                if source.get("email"):
                    details.append(f"Email: {source['email']}")
            
            elif search_type == "organization":
                if source.get("type"):
                    details.append(f"Type: {source['type']}")
                if source.get("level"):
                    details.append(f"Level: {source['level']}")
            
            elif search_type == "system":
                if source.get("status"):
                    details.append(f"Status: {source['status']}")
                if source.get("type"):
                    details.append(f"Type: {source['type']}")
            
            detail_str = ". ".join(details)
            return f"I found {result.name}. {detail_str}." if detail_str else f"I found {result.name}."
        
        else:
            return f"I found {len(results)} results. The top match is {results[0].name}."

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

# Semantic search endpoint with fuzzy matching and intelligent relevance
@app.get("/api/search/semantic", response_model=SearchResponse)
async def semantic_multi_index_search(
    q: str = Query(..., description="Semantic search query - supports fuzzy matching, typos, and meaning-based search"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100),
    index_filter: Optional[str] = Query(None, description="Filter by specific index: person_dataset, organization_units, or systems_index"),
    fuzziness: str = Query("AUTO", description="Fuzziness level: 0, 1, 2, AUTO"),
    min_score: float = Query(0.1, description="Minimum relevance score (0.0-1.0)", ge=0.0, le=1.0)
):
    """
    Semantic search with intelligent matching including:
    - Fuzzy matching for typos (john -> johm, jonh)
    - Partial word matching (john -> johnson)
    - Cross-field relevance scoring
    - Phrase matching with word order flexibility
    - Stemming and linguistic analysis
    - Nested field semantic search
    """
    try:
        # Determine search type based on cursor rules
        search_rule = determine_search_type(q)
        
        # Build the semantic Elasticsearch query
        query = build_semantic_search_query(q, search_rule)
        
        # Determine which indexes to search
        indexes_to_search = TARGET_INDEXES
        if index_filter and index_filter in TARGET_INDEXES:
            indexes_to_search = [index_filter]
        
        # Execute semantic search across multiple indexes
        response = es_client.search(
            index=indexes_to_search,
            body={
                "query": query,
                "size": size,
                "min_score": min_score,
                "_source": True,
                "highlight": {
                    "fields": {
                        "first_name": {},
                        "last_name": {},
                        "full_name": {},
                        "name": {},
                        "description": {},
                        "email": {},
                        "professional_experience.company": {},
                        "professional_experience.position": {},
                        "professional_experience.skills": {},
                        "persons.name": {},
                        "persons.role": {},
                        "owners.name": {},
                        "cars.make": {},
                        "cars.model": {}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "sort": [
                    {"_score": {"order": "desc"}}
                ]
            }
        )
        
        # Process results
        results = []
        index_breakdown = {"person_dataset": 0, "organization_units": 0, "systems_index": 0}
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            index = hit['_index']
            score = hit['_score']
            highlights = hit.get('highlight', {})
            
            # Count by index
            if index in index_breakdown:
                index_breakdown[index] += 1
            
            # Extract fields that matched for semantic analysis
            fields_matched = []
            if highlights:
                fields_matched = list(highlights.keys())
            else:
                fields_matched = ["semantic_match"]
            
            # Determine if this is an exact match (high score threshold for semantic)
            exact_match = score > 8.0
            
            result = SearchResult(
                name=get_result_name(source, index),
                score=score,
                search_type="semantic",
                fields_matched=fields_matched,
                exact_match=exact_match,
                source=source,
                highlights=highlights,
                index_type=determine_index_type(index)
            )
            results.append(result)
        
        return SearchResponse(
            total=response['hits']['total']['value'],
            search_type="semantic",
            query=q,
            results=results,
            index_breakdown=index_breakdown
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")

# Natural language search endpoint with LLM integration
@app.get("/api/search/natural", response_model=NaturalLanguageResponse)
async def natural_language_search(
    query: str = Query(..., description="Natural language query (e.g., 'who is the manager of sales department?')"),
    size: int = Query(20, description="Maximum number of results to return", ge=1, le=100),
    include_details: bool = Query(True, description="Include detailed information in the response")
):
    """
    Natural language search using LLM integration.
    
    Examples:
    - "Who is the manager of the sales department?"
    - "Find people working in Tokyo"
    - "What systems are owned by John?"
    - "Show me all developers in the engineering team"
    - "Who works at Microsoft?"
    """
    try:
        # Parse the natural language query using LLM
        parsed_query = parse_natural_language_query(query)
        
        # Extract search parameters
        search_terms = parsed_query.get("search_terms", [query])
        search_type = parsed_query.get("search_type", "universal")
        confidence = parsed_query.get("confidence", 0.5)
        
        # Build Elasticsearch query
        es_query = build_natural_language_query(parsed_query)
        
        # Determine which indexes to search based on search type
        if search_type == "person":
            indexes_to_search = ["person_dataset"]
        elif search_type == "organization":
            indexes_to_search = ["organization_units"]
        elif search_type == "system":
            indexes_to_search = ["systems_index"]
        else:
            indexes_to_search = TARGET_INDEXES
        
        # Execute the search
        response = es_client.search(
            index=indexes_to_search,
            body={
                "query": es_query,
                "size": size,
                "_source": True,
                "highlight": {
                    "fields": {
                        "first_name": {},
                        "last_name": {},
                        "full_name": {},
                        "name": {},
                        "description": {},
                        "persons.name": {},
                        "persons.role": {},
                        "owners.name": {},
                        "owners.role": {}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                },
                "sort": [
                    {"_score": {"order": "desc"}}
                ]
            }
        )
        
        # Process results
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            index = hit['_index']
            score = hit['_score']
            highlights = hit.get('highlight', {})
            
            # Extract fields that matched
            fields_matched = list(highlights.keys()) if highlights else ["natural_language_match"]
            
            result = SearchResult(
                name=get_result_name(source, index),
                score=score,
                search_type="natural_language",
                fields_matched=fields_matched,
                exact_match=score > 10.0,  # High score threshold for exact matches
                source=source if include_details else {},
                highlights=highlights,
                index_type=determine_index_type(index)
            )
            results.append(result)
        
        # Generate natural language answer
        answer = format_natural_language_answer(query, parsed_query, results)
        
        # Create interpreted query string
        interpreted_query = " + ".join(search_terms)
        if parsed_query.get("filters"):
            filter_parts = [f"{k}:{v}" for k, v in parsed_query["filters"].items()]
            interpreted_query += f" (filters: {', '.join(filter_parts)})"
        
        return NaturalLanguageResponse(
            original_query=query,
            interpreted_query=interpreted_query,
            search_strategy=f"{search_type} search with {len(search_terms)} terms",
            answer=answer,
            results=results,
            total=response['hits']['total']['value'],
            confidence=confidence
        )
        
    except Exception as e:
        # Fallback to universal search if natural language processing fails
        try:
            fallback_response = await universal_multi_index_search(q=query, size=size)
            answer = f"I searched for '{query}' and found {fallback_response.total} results."
            if fallback_response.results:
                answer += f" The top result is {fallback_response.results[0].name}."
            
            return NaturalLanguageResponse(
                original_query=query,
                interpreted_query=query,
                search_strategy="fallback universal search",
                answer=answer,
                results=fallback_response.results,
                total=fallback_response.total,
                confidence=0.3
            )
        except:
            raise HTTPException(status_code=500, detail=f"Error in natural language search: {str(e)}")

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
            "semantic_search": "/api/search/semantic?q=<query>&fuzziness=AUTO&min_score=0.1",
            "natural_language_search": "/api/search/natural?query=<natural_language_query>",
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
            "semantic_search_typos": "/api/search/semantic?q=jhon&fuzziness=AUTO",
            "semantic_search_partial": "/api/search/semantic?q=johnso&fuzziness=2", 
            "semantic_search_skills": "/api/search/semantic?q=developer&min_score=0.5",
            "natural_language_manager": "/api/search/natural?query=who%20is%20the%20manager%20of%20sales%20department",
            "natural_language_location": "/api/search/natural?query=find%20people%20working%20in%20Tokyo",
            "natural_language_systems": "/api/search/natural?query=what%20systems%20are%20owned%20by%20John",
            "natural_language_role": "/api/search/natural?query=show%20me%20all%20developers%20in%20engineering",
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