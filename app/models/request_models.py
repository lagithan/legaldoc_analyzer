# backend/app/models/request_models.py

from pydantic import BaseModel
from typing import Optional

class QuestionRequest(BaseModel):
    """Request model for asking questions about a document"""
    document_id: str
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "question": "What are my obligations under this contract?"
            }
        }