from fastapi import APIRouter, HTTPException, Query
from ..models.schemas import AutoCompleteResponse
from ..services.container import container
from ..config.settings import settings

router = APIRouter()


@router.get("/auto_complete", response_model=AutoCompleteResponse)
async def auto_complete(
    q: str = Query(..., description="Partial search query (minimum 2 characters)", min_length=settings.min_autocomplete_chars),
    size: int = Query(settings.max_autocomplete_results, description="Number of suggestions to return", ge=1, le=settings.max_autocomplete_results)
):
    """
    Get autocomplete suggestions based on partial query input.
    
    Features:
    - Case-insensitive matching
    - Minimum 2 characters required
    - Maximum 10 suggestions returned
    - Searches across persons, organizations, and systems
    - Returns simple strings (not full objects)
    """
    try:
        return await container.autocomplete_service.get_suggestions(q, size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autocomplete error: {str(e)}") 