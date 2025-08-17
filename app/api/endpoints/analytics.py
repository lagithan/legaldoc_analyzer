# backend/app/api/endpoints/analytics.py

from fastapi import APIRouter, HTTPException
import statistics
import logging

from ...models.response_models import AnalyticsResponse
from ...config.settings import AVAILABLE_LEGAL_MODELS

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage (should be same as documents)
documents_storage = {}

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get analytics for all processed documents"""
    try:
        # Initialize response fields
        total_documents = len(documents_storage)
        document_types = {}
        risk_distribution = {"low": 0, "medium": 0, "high": 0, "urgent": 0}
        confidence_scores = []
        total_requiring_lawyer = 0

        # Process each document in storage
        for doc_id, doc_info in documents_storage.items():
            analysis = doc_info["analysis"]

            # Document types
            doc_type = doc_info.get("document_type", "other")
            document_types[doc_type] = document_types.get(doc_type, 0) + 1

            # Risk distribution
            urgency = analysis.get("lawyer_urgency", "medium")
            if urgency in risk_distribution:
                risk_distribution[urgency] += 1

            # Confidence scores
            confidence = analysis.get("confidence_score", 0.0)
            confidence_scores.append(confidence)

            # Lawyer recommendation
            if analysis.get("lawyer_recommendation", False):
                total_requiring_lawyer += 1

        # Calculate average confidence (handle empty case)
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0

        # Log analytics for debugging
        logger.info(f"Analytics computed: total_documents={total_documents}, "
                    f"document_types={document_types}, risk_distribution={risk_distribution}, "
                    f"avg_confidence={avg_confidence}, total_requiring_lawyer={total_requiring_lawyer}")

        return AnalyticsResponse(
            total_documents=total_documents,
            document_types=document_types,
            risk_distribution=risk_distribution,
            avg_confidence=avg_confidence,
            total_requiring_lawyer=total_requiring_lawyer
        )

    except Exception as e:
        logger.error(f"Analytics endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing analytics: {str(e)}")

@router.get("/legal-models/status")
async def get_legal_models_status():
    """Get status of loaded legal models"""
    try:
        from ...services.legal_model_manager import LegalModelManager

        # Create instance to check model status
        legal_model_manager = LegalModelManager()

        # Check if the model is loaded
        model_loaded = legal_model_manager.model is not None
        tokenizer_loaded = legal_model_manager.tokenizer is not None
        
        # Get model name from the loaded model
        loaded_model_name = "Legal BERT Small (Fast)" if model_loaded else "None"
        
        # Determine status
        if model_loaded and tokenizer_loaded:
            status = "operational"
            loaded_models = [loaded_model_name]
        elif model_loaded or tokenizer_loaded:
            status = "partial"
            loaded_models = [loaded_model_name] if model_loaded else []
        else:
            status = "degraded"
            loaded_models = []

        return {
            "available_models": AVAILABLE_LEGAL_MODELS,
            "loaded_models": loaded_models,
            "device": str(legal_model_manager.device),
            "total_models": len(loaded_models),
            "status": status,
            "model_details": {
                "model_loaded": model_loaded,
                "tokenizer_loaded": tokenizer_loaded,
                "model_dimension": legal_model_manager.model_dimension,
                "current_model": loaded_model_name
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting legal models status: {str(e)}")
        return {
            "available_models": AVAILABLE_LEGAL_MODELS,
            "loaded_models": [],
            "device": "unknown",
            "total_models": 0,
            "status": "error",
            "error": str(e),
            "model_details": {
                "model_loaded": False,
                "tokenizer_loaded": False,
                "model_dimension": 0,
                "current_model": "Error loading model"
            }
        }