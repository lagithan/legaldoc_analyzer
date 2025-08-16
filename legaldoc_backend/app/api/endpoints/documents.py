# backend/app/api/endpoints/documents.py

from fastapi import APIRouter, File, UploadFile, HTTPException
import uuid
import logging
from datetime import datetime
from typing import List

from ...models.response_models import DocumentResponse
from ...services.document_processor import (
    extract_text_from_pdf,
    validate_legal_document_with_models,
    comprehensive_legal_analysis
)
from ...services.chromadb_manager import AdvancedChromaDBManager
from ...services.risk_assessment import calculate_overall_risk_from_breakdown
from ...utils.helpers import get_document_type_summary, get_risk_distribution, get_model_usage_stats

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage
documents_storage = {}

# Initialize ChromaDB manager
advanced_chroma_manager = AdvancedChromaDBManager()

@router.post("/upload-document", response_model=DocumentResponse)
async def upload_document_ultimate(file: UploadFile = File(...)):
    """
    Ultimate document upload with Legal Models + ChromaDB + Gemini integration
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if file.size and file.size > 25 * 1024 * 1024:  # 25MB for legal models
        raise HTTPException(status_code=400, detail="File size must be less than 25MB")

    try:
        logger.info(f"Processing document with ultimate system: {file.filename}")

        # Extract text from PDF
        file_content = await file.read()
        document_text = await extract_text_from_pdf(file_content)

        # STEP 1: ENHANCED VALIDATION WITH LEGAL MODELS
        is_legal, detected_type, rejection_reason = await validate_legal_document_with_models(document_text)

        if not is_legal:
            logger.info(f"Document rejected - not a legal document: {file.filename}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid Document Type",
                    "message": "This does not appear to be a legal document that requires legal analysis.",
                    "reason": rejection_reason,
                    "suggestion": "Please upload legal documents such as contracts, agreements, leases, terms of service, NDAs, or other documents with legal obligations and terms.",
                    "accepted_types": [
                        "Contracts and agreements", "Lease agreements", "Employment contracts",
                        "Service agreements", "Non-disclosure agreements (NDAs)", "Terms of service",
                        "Purchase agreements", "Loan documents", "Insurance policies", "Legal notices"
                    ]
                }
            )

        logger.info(f"Document validated as legal document type: {detected_type}")

        doc_id = str(uuid.uuid4())

        # STEP 2: COMPREHENSIVE ANALYSIS WITH ALL SYSTEMS
        analysis = await comprehensive_legal_analysis(document_text, file.filename)

        # Use detected type from validation if analysis doesn't have it
        if analysis.get('document_type') == 'other' and detected_type != 'not_legal':
            analysis['document_type'] = detected_type

        # Calculate risk score
        risk_score = calculate_overall_risk_from_breakdown(analysis['risk_breakdown'])

        # STEP 3: CHROMADB STORAGE WITH LEGAL EMBEDDINGS
        chroma_metadata = {
            'document_id': doc_id,
            'filename': file.filename,
            'document_type': analysis['document_type'],
            'risk_score': risk_score,
            'complexity_score': analysis['complexity_score'],
            'legal_terms_count': len(analysis.get('legal_terminology_found', [])),
            'risk_indicators_count': len(analysis.get('risk_indicators', [])),
            'urgency_signals_count': len(analysis.get('urgency_signals', [])),
            'created_at': datetime.now().isoformat()
        }

        # Store with legal model embeddings
        embedding_success = await advanced_chroma_manager.store_legal_document_embeddings(
            doc_id, document_text, analysis['legal_model_analysis'], chroma_metadata
        )

        # Find similar documents using legal embeddings
        similar_docs = advanced_chroma_manager.advanced_semantic_search(
            analysis['summary'], document_id=None, n_results=3
        )

        # Store comprehensive document metadata
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "text": document_text,
            "analysis": analysis,
            "risk_score": risk_score,
            "risk_breakdown": analysis['risk_breakdown'],
            "complexity_score": analysis['complexity_score'],
            "document_type": analysis['document_type'],
            "reading_time": analysis['estimated_reading_time'],
            "lawyer_reasoning": f"Urgency: {analysis['lawyer_urgency']} - {analysis['ai_confidence_reasoning']}",
            "validated": True,
            "validation_type": detected_type,
            "legal_model_analysis": analysis['legal_model_analysis'],
            "legal_terminology_found": analysis['legal_terminology_found'],
            "risk_indicators": analysis['risk_indicators'],
            "urgency_signals": analysis['urgency_signals'],
            "chromadb_stored": embedding_success,
            "similar_documents": similar_docs[:3],
            "created_at": datetime.now().isoformat(),
            "file_size": len(file_content)
        }
        documents_storage[doc_id] = doc_info

        logger.info(f"Ultimate document processing completed: {doc_id}")

        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            summary=analysis["summary"],
            key_clauses=analysis["key_clauses"],
            red_flags=analysis["red_flags"],
            confidence_score=analysis["confidence_score"],
            risk_score=risk_score,
            lawyer_recommendation=bool(analysis["lawyer_recommendation"]),  # Ensure boolean
            lawyer_urgency=str(analysis["lawyer_urgency"]),  # Ensure string
            risk_breakdown=analysis["risk_breakdown"],
            complexity_score=analysis["complexity_score"],
            suggested_questions=analysis["suggested_questions"],
            document_type=analysis["document_type"],
            estimated_reading_time=analysis["estimated_reading_time"],
            ai_confidence_reasoning=analysis["ai_confidence_reasoning"],
            similar_documents=doc_info["similar_documents"],
            legal_model_analysis=analysis["legal_model_analysis"],
            legal_terminology_found=analysis["legal_terminology_found"],
            risk_indicators=analysis["risk_indicators"],
            urgency_signals=analysis["urgency_signals"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ultimate processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/documents")
async def get_documents():
    """Get all processed documents with enhanced metadata"""
    documents = []
    for doc_id, doc_info in documents_storage.items():
        documents.append({
            "id": doc_id,
            "filename": doc_info["filename"],
            "summary": doc_info["analysis"]["summary"][:150] + "..." if len(doc_info["analysis"]["summary"]) > 150 else doc_info["analysis"]["summary"],
            "document_type": doc_info.get("document_type", "other"),
            "risk_score": doc_info.get("risk_score", 0.5),
            "complexity_score": doc_info.get("complexity_score", 0.5),
            "confidence_score": doc_info["analysis"]["confidence_score"],
            "lawyer_urgency": doc_info["analysis"].get("lawyer_urgency", "medium"),
            "reading_time": doc_info.get("reading_time", 5),
            "validated": doc_info.get("validated", True),
            "legal_terms_count": len(doc_info.get("legal_terminology_found", [])),
            "risk_indicators_count": len(doc_info.get("risk_indicators", [])),
            "urgency_signals_count": len(doc_info.get("urgency_signals", [])),
            "model_used": doc_info.get("legal_model_analysis", {}).get("model_used", "unknown"),
            "chromadb_stored": doc_info.get("chromadb_stored", False),
            "created_at": doc_info.get("created_at", "")
        })

    # Sort by creation date (most recent first)
    documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {
        "documents": documents,
        "total_count": len(documents),
        "by_type": get_document_type_summary(documents),
        "risk_distribution": get_risk_distribution(documents),
        "model_usage": get_model_usage_stats(documents)
    }

@router.get("/document/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Retrieve detailed document information with chunk metadata"""
    if document_id not in documents_storage:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        doc_info = documents_storage[document_id]
        analysis = doc_info["analysis"]

        # Fetch chunks from ChromaDB
        chunk_results = advanced_chroma_manager.document_collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        chunks = [
            {
                "text": chunk_results["documents"][i],
                "chunk_type": metadata["chunk_type"],
                "legal_complexity": metadata["legal_complexity"],
                "legal_terms_count": metadata["contains_legal_terms"],
                "risk_indicators_count": metadata["risk_indicators_count"]
            }
            for i, metadata in enumerate(chunk_results["metadatas"])
        ]

        # Aggregate chunk types
        chunk_types = list(set(chunk["chunk_type"] for chunk in chunks))

        return DocumentResponse(
            id=document_id,
            filename=doc_info["filename"],
            summary=analysis["summary"],
            key_clauses=analysis["key_clauses"],
            red_flags=analysis["red_flags"],
            confidence_score=analysis["confidence_score"],
            risk_score=doc_info["risk_score"],
            lawyer_recommendation=analysis["lawyer_recommendation"],
            lawyer_urgency=analysis["lawyer_urgency"],
            risk_breakdown=analysis["risk_breakdown"],
            complexity_score=analysis["complexity_score"],
            suggested_questions=analysis["suggested_questions"],
            document_type=analysis["document_type"],
            estimated_reading_time=analysis["estimated_reading_time"],
            ai_confidence_reasoning=analysis["ai_confidence_reasoning"],
            similar_documents=doc_info["similar_documents"],
            legal_model_analysis=analysis["legal_model_analysis"],
            legal_terminology_found=analysis["legal_terminology_found"],
            risk_indicators=analysis["risk_indicators"],
            urgency_signals=analysis["urgency_signals"]
        )
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.get("/export-report/{document_id}")
async def export_comprehensive_report(document_id: str):
    """Generate comprehensive analysis report for export"""
    if document_id not in documents_storage:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_info = documents_storage[document_id]
    analysis = doc_info["analysis"]

    report_content = f"""
COMPREHENSIVE LEGAL DOCUMENT ANALYSIS REPORT
===========================================

Document Information:
- Filename: {doc_info['filename']}
- Analysis Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
- Document ID: {document_id}
- File Size: {doc_info.get('file_size', 0):,} bytes

EXECUTIVE SUMMARY
{analysis['summary']}

RISK ASSESSMENT
Overall Risk Score: {doc_info.get('risk_score', 0.5):.1%} Risk Level
AI Confidence: {analysis['confidence_score']:.1%}
Legal Consultation Recommended: {'Yes' if analysis['lawyer_recommendation'] else 'No'}

Risk Factors Detected: {', '.join(doc_info.get('risk_indicators', []))}

KEY TERMS AND OBLIGATIONS
{chr(10).join(f"• {clause}" for clause in analysis['key_clauses'])}

RED FLAGS AND CONCERNS
{chr(10).join(f"⚠️ {flag}" for flag in analysis['red_flags'])}

RECOMMENDED QUESTIONS TO ASK
{chr(10).join(f"• {question}" for question in analysis.get('suggested_questions', []))}

RECOMMENDATIONS AND NEXT STEPS
1. Review all highlighted red flags carefully before signing
2. Consider negotiating concerning terms identified above
3. {"Strongly recommend consulting with a qualified attorney" if analysis['lawyer_recommendation'] else "Consider legal review for complex terms"}
4. Keep this analysis for your records and future reference

DISCLAIMER
This analysis is provided by AI for informational purposes only and does not constitute legal advice.
Always consult with a qualified attorney for legal matters requiring professional judgment.

Report Generated by AI Legal Doc Explainer v8.0
© 2025 - Competition Edition
"""

    return {
        "report": report_content,
        "filename": f"legal_analysis_{doc_info['filename'].replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
        "document_id": document_id,
        "generated_at": datetime.now().isoformat()
    }

@router.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its associated data"""
    if document_id not in documents_storage:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # Get document info before deletion
        doc_info = documents_storage[document_id]
        filename = doc_info.get("filename", "Unknown")

        # Delete from ChromaDB (embeddings and analysis)
        embedding_deleted = advanced_chroma_manager.delete_document_embeddings(document_id)

        # Delete from memory storage
        del documents_storage[document_id]

        logger.info(f"Document deleted successfully: {document_id} ({filename})")

        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "filename": filename,
            "chromadb_deleted": embedding_deleted,
            "deleted_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/suggest-questions/{document_id}")
async def suggest_questions(document_id: str):
    """Retrieve suggested questions for a document with robust error handling"""
    if document_id not in documents_storage:
        logger.error(f"Document {document_id} not found in storage")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        doc_info = documents_storage[document_id]
        analysis = doc_info.get("analysis", {})

        # Ensure suggested_questions exists in analysis
        suggested_questions = analysis.get("suggested_questions", [
            "What are the key obligations in this document?",
            "What are the potential risks or penalties?",
            "How can the agreement be terminated?",
            "Are there any specific compliance requirements?"
        ])

        # Validate questions are a list of strings
        if not isinstance(suggested_questions, list) or not all(isinstance(q, str) for q in suggested_questions):
            logger.warning(f"Invalid suggested_questions format for document {document_id}")
            suggested_questions = [
                "What are the main terms?",
                "What are the risks involved?",
                "How do I exit this agreement?"
            ]

        logger.info(f"Returning suggested questions for document {document_id}")
        return {
            "questions": suggested_questions,
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"Error retrieving suggested questions for {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving suggested questions: {str(e)}"
        )