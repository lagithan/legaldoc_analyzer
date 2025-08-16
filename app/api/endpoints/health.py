# backend/app/api/endpoints/health.py

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

# In-memory storage (should be same as documents)
documents_storage = {}

@router.get("/health")
async def basic_health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0",
        "documents_processed": len(documents_storage)
    }