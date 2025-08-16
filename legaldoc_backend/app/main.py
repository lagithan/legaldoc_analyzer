# backend/app/main.py

# ULTIMATE LEGAL DOCUMENT ANALYZER - Competition Winner!
# Integrating: Pre-trained Legal Models + ChromaDB + Gemini + Advanced Risk Assessment

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import configuration
from .config.settings import logger

# Import API routers
from .api.endpoints import documents, questions, analytics, health

# Initialize legal model manager and ChromaDB (singleton instances)
from .services.legal_model_manager import LegalModelManager
from .services.chromadb_manager import AdvancedChromaDBManager

# Initialize FastAPI
app = FastAPI(
    title="Ultimate AI Legal Doc Analyzer - Multi-Model Enhanced",
    description="ChromaDB + Legal Models + Gemini AI for comprehensive legal document analysis",
    version="8.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services on startup
legal_model_manager = LegalModelManager()
advanced_chroma_manager = AdvancedChromaDBManager()

# Include API routers
app.include_router(documents.router, tags=["Documents"])
app.include_router(questions.router, tags=["Questions"])
app.include_router(analytics.router, tags=["Analytics"])
app.include_router(health.router, tags=["Health"])

# Shared storage reference (in production, use proper database)
documents.documents_storage = {}
questions.documents_storage = documents.documents_storage
analytics.documents_storage = documents.documents_storage
health.documents_storage = documents.documents_storage

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")