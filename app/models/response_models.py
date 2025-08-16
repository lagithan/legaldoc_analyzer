# backend/app/models/response_models.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DocumentResponse(BaseModel):
    """Response model for document analysis"""
    id: str
    filename: str
    summary: str
    key_clauses: List[str]
    red_flags: List[str]
    confidence_score: float
    risk_score: float
    lawyer_recommendation: bool
    lawyer_urgency: str
    risk_breakdown: Dict[str, float]
    complexity_score: float
    suggested_questions: List[str]
    document_type: str
    estimated_reading_time: int
    ai_confidence_reasoning: str
    similar_documents: List[Dict[str, Any]] = []
    legal_model_analysis: Dict[str, Any] = {}
    legal_terminology_found: List[str] = []
    risk_indicators: List[str] = []
    urgency_signals: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "contract.pdf",
                "summary": "This is a service agreement...",
                "key_clauses": ["Payment terms", "Termination clause"],
                "red_flags": ["Unlimited liability clause"],
                "confidence_score": 0.85,
                "risk_score": 0.6,
                "lawyer_recommendation": True,
                "lawyer_urgency": "medium",
                "risk_breakdown": {
                    "financial_risk": 0.7,
                    "termination_risk": 0.4,
                    "liability_risk": 0.8,
                    "renewal_risk": 0.3,
                    "modification_risk": 0.5
                },
                "complexity_score": 0.75,
                "suggested_questions": ["What are the payment terms?"],
                "document_type": "service_agreement",
                "estimated_reading_time": 15,
                "ai_confidence_reasoning": "High confidence due to clear legal structure"
            }
        }

class QuestionResponse(BaseModel):
    """Response model for document questions"""
    answer: str
    confidence_score: float
    relevant_sections: List[str]
    follow_up_questions: List[str]
    legal_implications: str
    semantic_matches: List[Dict[str, Any]] = []
    legal_model_insights: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the contract, you are required to...",
                "confidence_score": 0.9,
                "relevant_sections": ["Section 3.1: Payment obligations"],
                "follow_up_questions": ["What happens if payment is late?"],
                "legal_implications": "This creates a binding obligation",
                "semantic_matches": [],
                "legal_model_insights": {}
            }
        }

class AnalyticsResponse(BaseModel):
    """Response model for analytics data"""
    total_documents: int
    document_types: Dict[str, int]
    risk_distribution: Dict[str, int]
    avg_confidence: float
    total_requiring_lawyer: int

    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 10,
                "document_types": {"contract": 5, "lease": 3, "nda": 2},
                "risk_distribution": {"low": 2, "medium": 5, "high": 3},
                "avg_confidence": 0.82,
                "total_requiring_lawyer": 7
            }
        }