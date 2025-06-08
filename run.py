#!/usr/bin/env python3
"""
Elasticsearch Name Search API
Run script for the FastAPI application
"""

import os
import uvicorn
from src.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Elasticsearch Name Search API")
    print(f"📍 Server will run on http://{host}:{port}")
    print(f"📚 API documentation available at http://{host}:{port}/docs")
    print(f"🔍 Interactive API at http://{host}:{port}/redoc")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        access_log=True
    ) 