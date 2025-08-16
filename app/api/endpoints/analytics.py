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
    from ...services.legal_model_manager import LegalModelManager

    # This would ideally access the same instance used elsewhere
    legal_model_manager = LegalModelManager()

    return {
        "available_models": AVAILABLE_LEGAL_MODELS,
        "loaded_models": list(legal_model_manager.models.keys()),
        "device": str(legal_model_manager.device),
        "total_models": len(legal_model_manager.models),
        "status": "operational" if legal_model_manager.models else "degraded"
    }