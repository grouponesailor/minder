from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from ..models.schemas import SearchResponse, AdvancedSearchFilters
from ..services.container import container
from ..config.settings import settings

router = APIRouter()


@router.get("/search", response_model=SearchResponse)
@router.get("/search/universal", response_model=SearchResponse)
async def universal_search(
    q: str = Query(..., description="Search query"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100),
    index_filter: Optional[str] = Query(None, description="Filter by specific index")
):
    """Universal search across all indexes with restrictive entity separation"""
    try:
        return await container.search_service.universal_search(q, size, index_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/search/name", response_model=SearchResponse)
async def search_name(
    name: str = Query(..., description="Name to search for")
):
    """Legacy name search - redirects to persons search"""
    try:
        return await container.search_service.universal_search(name, 20, "person_dataset")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Name search error: {str(e)}")


@router.get("/search/persons", response_model=SearchResponse)
async def search_persons_only(
    q: str = Query(..., description="Search query for persons only"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100)
):
    """Search only in person_dataset index - no organization cross-matches"""
    try:
        return await container.search_service.universal_search(q, size, "person_dataset")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Person search error: {str(e)}")


@router.get("/search/organizations", response_model=SearchResponse)
async def search_organizations_only(
    q: str = Query(..., description="Search query for organizations only"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100)
):
    """Search only in organization_units index - no person cross-matches"""
    try:
        return await container.search_service.universal_search(q, size, "organization_units")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Organization search error: {str(e)}")


@router.get("/search/systems", response_model=SearchResponse)
async def search_systems_only(
    q: str = Query(..., description="Search query for systems only"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100)
):
    """Search only in systems_index - no cross-matches with other entities"""
    try:
        return await container.search_service.universal_search(q, size, "systems_index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System search error: {str(e)}")


@router.get("/search/advanced", response_model=SearchResponse)
async def advanced_search(
    q: str = Query(..., description="Search query"),
    location: Optional[str] = Query(None, description="Filter by location"),
    person_role: Optional[str] = Query(None, description="Filter by person role"),
    org_type: Optional[str] = Query(None, description="Filter by organization type"),
    org_level: Optional[int] = Query(None, description="Filter by organization level"),
    system_status: Optional[str] = Query(None, description="Filter by system status"),
    size: int = Query(20, description="Number of results to return", ge=1, le=100)
):
    """Advanced search with filters"""
    try:
        filters = AdvancedSearchFilters(
            location=location,
            person_role=person_role,
            org_type=org_type,
            org_level=org_level,
            system_status=system_status
        )
        return await container.search_service.advanced_search(q, filters, size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced search error: {str(e)}")


@router.get("/search/rules")
async def get_search_rules():
    """Get the current search rules configuration"""
    return {"cursor_rules": settings.cursor_rules} 