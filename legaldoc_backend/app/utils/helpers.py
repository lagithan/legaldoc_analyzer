# backend/app/utils/helpers.py

from typing import List, Dict

def get_document_type_summary(documents: List[dict]) -> dict:
    """Get summary of document types"""
    type_counts = {}
    for doc in documents:
        doc_type = doc.get("document_type", "other")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    return type_counts

def get_risk_distribution(documents: List[dict]) -> dict:
    """Get risk level distribution"""
    risk_levels = {"low": 0, "medium": 0, "high": 0, "urgent": 0}
    for doc in documents:
        urgency = doc.get("lawyer_urgency", "medium")
        if urgency in risk_levels:
            risk_levels[urgency] += 1
    return risk_levels

def get_model_usage_stats(documents: List[dict]) -> dict:
    """Get legal model usage statistics"""
    model_usage = {}
    for doc in documents:
        model = doc.get("model_used", "unknown")
        model_usage[model] = model_usage.get(model, 0) + 1
    return model_usage