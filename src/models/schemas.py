from typing import Optional, List, Dict, Any
from pydantic import BaseModel


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


class AdvancedSearchFilters(BaseModel):
    location: Optional[str] = None
    person_role: Optional[str] = None
    org_type: Optional[str] = None
    org_level: Optional[int] = None
    system_status: Optional[str] = None 