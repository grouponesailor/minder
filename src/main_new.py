from fastapi import FastAPI
from .config.settings import settings
from .routes import search_routes, autocomplete_routes, health_routes
from .services.container import container

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Include routers with API prefix
app.include_router(search_routes.router, prefix="/api", tags=["search"])
app.include_router(autocomplete_routes.router, prefix="/api", tags=["autocomplete"])
app.include_router(health_routes.router, prefix="/api", tags=["health"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Universal Elasticsearch Search API",
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health",
        "endpoints": {
            "search": "/api/search",
            "autocomplete": "/api/auto_complete",
            "health": "/api/health"
        },
        "indexes": settings.target_indexes,
        "target_indexes": ", ".join(settings.target_indexes)
    }


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print(f"🚀 Starting {settings.api_title} v{settings.api_version}")
    print(f"🔍 Elasticsearch URL: {settings.elasticsearch_url}")
    print(f"📊 Target indexes: {', '.join(settings.target_indexes)}")
    
    # Check index health on startup
    try:
        available_indexes = await container.elasticsearch_service.check_index_health()
        print(f"✅ Available indexes: {', '.join(available_indexes)}")
    except Exception as e:
        print(f"⚠️ Warning: Could not check index health - {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("🛑 Shutting down Universal Elasticsearch Search API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 