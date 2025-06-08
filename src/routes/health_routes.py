from fastapi import APIRouter, HTTPException
from ..models.schemas import HealthResponse
from ..services.container import container

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify system status.
    
    Returns:
    - Status of the application
    - Timestamp
    - Whether cursor rules are loaded
    - List of available Elasticsearch indexes
    """
    try:
        return await container.health_service.get_health_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with comprehensive system information.
    
    Returns detailed information about:
    - Elasticsearch status and index statistics
    - Configuration settings
    - API information
    """
    try:
        return await container.health_service.get_detailed_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed health check error: {str(e)}")