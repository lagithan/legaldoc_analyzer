from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openai
import os
import uuid
import asyncio
from pathlib import Path
import json
import logging
from datetime import datetime
import requests
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Document processing
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="AI Legal Doc Explainer - Competition Edition",
    description="Advanced AI-powered legal document analyzer with RAG and advanced features",
    version="2.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai.api_key = OPENAI_API_KEY

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store_path = "./chroma_db"

# Pydantic models
class DocumentResponse(BaseModel):
    id: str
    filename: str
    summary: str
    key_clauses: List[str]
    red_flags: List[str]
    confidence_score: float
    risk_score: float
    risk_factors: List[str]
    lawyer_recommendation: bool
    suggested_questions: List[str]

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    confidence_score: float
    relevant_sections: List[str]
    lawyer_recommendation: bool

class ComparisonRequest(BaseModel):
    document_ids: List[str]

class ComparisonResponse(BaseModel):
    comparison_summary: str
    key_differences: List[str]
    risk_comparison: Dict[str, float]
    recommendations: List[str]

class AnalyticsResponse(BaseModel):
    total_documents: int
    document_types: Dict[str, int]
    risk_distribution: Dict[str, int]
    avg_confidence: float
    total_requiring_lawyer: int

# In-memory storage (use Redis or proper DB in production)
documents_storage = {}
question_history = {}

# Advanced GPT-4 Prompts
ADVANCED_ANALYSIS_PROMPT = """
You are a senior legal expert with 25+ years of experience analyzing complex legal documents. Provide comprehensive analysis with forensic precision.

DOCUMENT ANALYSIS FRAMEWORK:
1. **Executive Summary** (2-3 paragraphs in business language)
2. **Key Terms & Obligations** (6-10 most critical items)
3. **Risk Assessment** (identify concerning clauses, penalties, liability shifts)
4. **Red Flags** (auto-renewals, termination difficulties, one-sided terms)
5. **Confidence Level** (0.0-1.0 based on document clarity and complexity)
6. **Legal Consultation Necessity** (true/false with reasoning)

CRITICAL FOCUS AREAS:
- Contract duration, renewal mechanisms, and termination procedures
- Payment obligations, penalties, late fees, and financial consequences
- Liability allocation, limitations, and indemnification clauses
- Intellectual property rights and confidentiality provisions
- Performance metrics, SLAs, and compliance requirements
- Dispute resolution mechanisms and governing law
- Force majeure and risk allocation provisions
- Data privacy and security obligations

RISK SCORING CRITERIA:
- Auto-renewal clauses: +0.3 risk
- Penalty/liquidated damages: +0.4 risk
- One-sided termination rights: +0.5 risk
- Broad liability exposure: +0.6 risk
- Unclear performance metrics: +0.3 risk
- Restrictive confidentiality: +0.2 risk

Document Content: {document_text}

Return comprehensive JSON analysis with actionable business insights:
{{
    "summary": "Clear executive summary in business language...",
    "key_clauses": ["Critical obligation 1", "Key term 2", ...],
    "red_flags": ["Concerning clause 1", "Risk factor 2", ...],
    "confidence_score": 0.85,
    "risk_score": 0.65,
    "risk_factors": ["auto_renewal", "penalty_clauses", ...],
    "lawyer_recommendation": true,
    "business_impact": "High/Medium/Low impact assessment",
    "suggested_questions": ["What are termination conditions?", ...]
}}
"""

INTELLIGENT_QA_PROMPT = """
You are a legal counsel providing authoritative guidance on this specific legal document. Deliver precise, actionable answers.

RESPONSE FRAMEWORK:
1. **Direct Answer** - Address the question specifically and completely
2. **Legal Basis** - Reference exact document sections supporting your answer
3. **Business Implications** - Explain practical consequences and risks
4. **Confidence Assessment** - Rate answer certainty based on document clarity
5. **Action Items** - Suggest next steps or precautions if applicable

QUALITY STANDARDS:
- Base answers EXCLUSIVELY on provided document context
- If information is absent, clearly state "This information is not specified in the document"
- Translate legal language into business-friendly terms
- Highlight time-sensitive or high-risk elements
- Recommend legal consultation for complex interpretations

Document Context: {context}
User Question: {question}

Provide comprehensive legal guidance in JSON format:
{{
    "answer": "Complete, authoritative response addressing all aspects...",
    "confidence_score": 0.90,
    "relevant_sections": ["Exact quote from section 1", "Key clause from section 2"],
    "business_implications": "Practical impact and consequences...",
    "recommended_actions": ["Action step 1", "Precaution 2"],
    "lawyer_recommendation": false,
    "urgency_level": "High/Medium/Low"
}}
"""

DOCUMENT_COMPARISON_PROMPT = """
You are a contract negotiation expert comparing multiple legal documents. Provide strategic analysis for business decision-making.

COMPARISON FRAMEWORK:
1. **Overall Assessment** - Which document is more favorable and why
2. **Term-by-Term Analysis** - Key differences in obligations, rights, risks
3. **Risk Profile Comparison** - Relative risk levels and exposure areas
4. **Negotiation Opportunities** - Terms that could be improved or standardized
5. **Strategic Recommendations** - Business guidance for decision-making

Document 1: {doc1_summary}
Document 2: {doc2_summary}

Provide strategic comparison analysis:
{{
    "comparison_summary": "Executive overview of key differences...",
    "more_favorable": "Document 1/Document 2 with detailed reasoning",
    "key_differences": ["Major difference 1", "Critical variation 2", ...],
    "risk_comparison": {{"doc1_risk": 0.7, "doc2_risk": 0.4}},
    "negotiation_points": ["Term to negotiate 1", "Clause to modify 2", ...],
    "recommendations": ["Strategic recommendation 1", "Business advice 2", ...]
}}
"""

# Advanced helper functions
async def extract_text_from_pdf(file_content: bytes) -> str:
    """Enhanced PDF text extraction with error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                text += f"[Page {page_num + 1}] {page_text}\n"
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                continue
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

async def intelligent_chunk_text(text: str, document_type: str = "legal") -> List[Document]:
    """Advanced text chunking optimized for legal documents"""
    if document_type == "legal":
        # Legal document optimized chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Larger chunks for legal context
            chunk_overlap=300,  # More overlap for continuity
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            length_function=len,
        )
    else:
        # Standard chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    chunks = text_splitter.split_text(text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 50:  # Filter out tiny chunks
            documents.append(Document(
                page_content=chunk,
                metadata={"chunk_id": i, "document_type": document_type}
            ))
    
    return documents

def calculate_advanced_risk_score(document_text: str, analysis: dict) -> tuple[float, List[str]]:
    """Calculate comprehensive risk score with detailed factors"""
    risk_factors = {
        'auto_renewal': 0.3,
        'penalty_clauses': 0.4,
        'one_sided_terms': 0.5,
        'termination_difficulty': 0.4,
        'liability_shifts': 0.6,
        'unclear_language': 0.2,
        'broad_confidentiality': 0.3,
        'performance_penalties': 0.4,
        'indemnification': 0.5,
        'governing_law_issues': 0.2
    }
    
    detected_factors = []
    total_risk = 0.0
    text_lower = document_text.lower()
    
    # Auto-renewal detection
    auto_renewal_terms = ['automatically renew', 'auto-renew', 'shall renew', 'automatic extension']
    if any(term in text_lower for term in auto_renewal_terms):
        total_risk += risk_factors['auto_renewal']
        detected_factors.append('auto_renewal')
    
    # Penalty clause detection
    penalty_terms = ['penalty', 'fine', 'liquidated damages', 'late fee', 'breach penalty']
    if any(term in text_lower for term in penalty_terms):
        total_risk += risk_factors['penalty_clauses']
        detected_factors.append('penalty_clauses')
    
    # One-sided terms detection
    one_sided_terms = ['sole discretion', 'absolute right', 'without cause', 'immediate termination']
    if any(term in text_lower for term in one_sided_terms):
        total_risk += risk_factors['one_sided_terms']
        detected_factors.append('one_sided_terms')
    
    # Liability shift detection
    liability_terms = ['indemnify', 'hold harmless', 'liable for all', 'assume all risk']
    if any(term in text_lower for term in liability_terms):
        total_risk += risk_factors['liability_shifts']
        detected_factors.append('liability_shifts')
    
    # Additional risk factors from AI analysis
    if analysis.get('lawyer_recommendation', False):
        total_risk += 0.2
    
    if analysis.get('confidence_score', 1.0) < 0.6:
        total_risk += 0.3
        detected_factors.append('unclear_language')
    
    return min(total_risk, 1.0), detected_factors

async def analyze_document_with_advanced_gpt4(text: str) -> dict:
    """Advanced document analysis using GPT-4 with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Truncate text if too long
            analysis_text = text[:20000] if len(text) > 20000 else text
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior legal expert providing comprehensive document analysis."},
                    {"role": "user", "content": ADVANCED_ANALYSIS_PROMPT.format(document_text=analysis_text)}
                ],
                temperature=0.2,
                max_tokens=3000,
                timeout=30
            )
            
            result_text = response.choices[0].message.content
            analysis = json.loads(result_text)
            
            # Ensure all required fields are present
            required_fields = ['summary', 'key_clauses', 'red_flags', 'confidence_score', 'lawyer_recommendation']
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = [] if field in ['key_clauses', 'red_flags'] else False if field == 'lawyer_recommendation' else 0.5
            
            # Add default values for new fields
            if 'suggested_questions' not in analysis:
                analysis['suggested_questions'] = [
                    "What are the termination conditions?",
                    "What are my payment obligations?",
                    "Are there any penalties or fees?",
                    "What are the key risks I should know about?"
                ]
            
            return analysis
            
        except json.JSONDecodeError:
            logger.warning(f"JSON decode error on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return create_fallback_analysis()
        except Exception as e:
            logger.error(f"Analysis error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return create_fallback_analysis()
    
    return create_fallback_analysis()

def create_fallback_analysis() -> dict:
    """Fallback analysis when AI processing fails"""
    return {
        "summary": "Document uploaded and processed. Manual review recommended for detailed analysis.",
        "key_clauses": ["Document contains standard legal provisions", "Review recommended for specific terms"],
        "red_flags": ["Unable to automatically analyze - manual review needed"],
        "confidence_score": 0.3,
        "lawyer_recommendation": True,
        "suggested_questions": [
            "What are the main obligations in this document?",
            "Are there any termination clauses?",
            "What are the payment terms?",
            "Are there any penalty clauses?"
        ]
    }

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Enhanced request logging with performance monitoring"""
    start_time = datetime.now()
    client_ip = get_remote_address(request)
    
    logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# API Endpoints
@app.post("/upload-document", response_model=DocumentResponse)
@limiter.limit("20/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Advanced document upload and analysis"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File size must be less than 10MB")
    
    try:
        logger.info(f"Processing document: {file.filename}")
        
        # Read and extract text
        file_content = await file.read()
        document_text = await extract_text_from_pdf(file_content)
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Create optimized chunks
        chunks = await intelligent_chunk_text(document_text, "legal")
        
        # Create vector store
        doc_vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=f"{vector_store_path}/{doc_id}",
            collection_name=f"doc_{doc_id}"
        )
        
        # Advanced AI analysis
        analysis = await analyze_document_with_advanced_gpt4(document_text)
        
        # Calculate risk score
        risk_score, risk_factors = calculate_advanced_risk_score(document_text, analysis)
        
        # Store document information
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "text": document_text,
            "vector_store": doc_vector_store,
            "analysis": analysis,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "created_at": datetime.now().isoformat(),
            "file_size": len(file_content)
        }
        documents_storage[doc_id] = doc_info
        
        logger.info(f"Document processed successfully: {doc_id}")
        
        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            summary=analysis["summary"],
            key_clauses=analysis["key_clauses"],
            red_flags=analysis["red_flags"],
            confidence_score=analysis["confidence_score"],
            risk_score=risk_score,
            risk_factors=risk_factors,
            lawyer_recommendation=analysis["lawyer_recommendation"],
            suggested_questions=analysis.get("suggested_questions", [])
        )
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask-question", response_model=QuestionResponse)
@limiter.limit("30/minute")
async def ask_question(request: Request, question_request: QuestionRequest):
    """Enhanced Q&A with intelligent context retrieval"""
    if question_request.document_id not in documents_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        doc_info = documents_storage[question_request.document_id]
        vector_store = doc_info["vector_store"]
        
        # Intelligent retrieval with higher k value
        relevant_docs = vector_store.similarity_search(question_request.question, k=7)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Enhanced answer generation
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior legal counsel providing authoritative guidance."},
                {"role": "user", "content": INTELLIGENT_QA_PROMPT.format(
                    context=context, 
                    question=question_request.question
                )}
            ],
            temperature=0.1,
            max_tokens=2000,
            timeout=30
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Store question history
        if question_request.document_id not in question_history:
            question_history[question_request.document_id] = []
        
        question_history[question_request.document_id].append({
            "question": question_request.question,
            "answer": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return QuestionResponse(
            answer=result["answer"],
            confidence_score=result["confidence_score"],
            relevant_sections=result.get("relevant_sections", []),
            lawyer_recommendation=result["lawyer_recommendation"]
        )
        
    except Exception as e:
        logger.error(f"Question processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/compare-documents", response_model=ComparisonResponse)
@limiter.limit("10/minute")
async def compare_documents(request: Request, comparison_request: ComparisonRequest):
    """Advanced document comparison analysis"""
    doc_ids = comparison_request.document_ids
    
    if len(doc_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 documents required for comparison")
    
    if len(doc_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 documents can be compared at once")
    
    try:
        documents = []
        for doc_id in doc_ids:
            if doc_id not in documents_storage:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            documents.append(documents_storage[doc_id])
        
        # Prepare comparison data
        doc_summaries = []
        risk_comparison = {}
        
        for i, doc in enumerate(documents):
            doc_summaries.append(f"Document {i+1} ({doc['filename']}): {doc['analysis']['summary']}")
            risk_comparison[f"doc{i+1}_risk"] = doc['risk_score']
        
        # Generate comparison analysis
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a contract negotiation expert."},
                {"role": "user", "content": DOCUMENT_COMPARISON_PROMPT.format(
                    doc1_summary=doc_summaries[0],
                    doc2_summary=doc_summaries[1] if len(doc_summaries) > 1 else ""
                )}
            ],
            temperature=0.2,
            max_tokens=2500
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        return ComparisonResponse(
            comparison_summary=result["comparison_summary"],
            key_differences=result["key_differences"],
            risk_comparison=risk_comparison,
            recommendations=result["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing documents: {str(e)}")

@app.post("/batch-upload")
@limiter.limit("5/minute")
async def batch_upload_documents(request: Request, files: List[UploadFile] = File(...)):
    """Process multiple documents simultaneously"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 documents can be uploaded at once")
    
    results = []
    successful_uploads = 0
    
    for file in files:
        try:
            result = await upload_document(request, file)
            results.append({
                "filename": file.filename,
                "status": "success",
                "document_id": result.id,
                "risk_score": result.risk_score,
                "summary": result.summary[:150] + "..."
            })
            successful_uploads += 1
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "results": results,
        "total_files": len(files),
        "successful_uploads": successful_uploads,
        "failed_uploads": len(files) - successful_uploads
    }

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_comprehensive_analytics():
    """Comprehensive usage analytics and insights"""
    if not documents_storage:
        raise HTTPException(status_code=404, detail="No documents processed yet")
    
    total_docs = len(documents_storage)
    doc_types = {"lease": 0, "contract": 0, "agreement": 0, "terms": 0, "other": 0}
    risk_levels = {"low": 0, "medium": 0, "high": 0}
    total_confidence = 0
    lawyer_recommendations = 0
    
    for doc_info in documents_storage.values():
        # Document type classification
        filename = doc_info["filename"].lower()
        if "lease" in filename:
            doc_types["lease"] += 1
        elif "contract" in filename:
            doc_types["contract"] += 1
        elif "agreement" in filename:
            doc_types["agreement"] += 1
        elif "terms" in filename or "conditions" in filename:
            doc_types["terms"] += 1
        else:
            doc_types["other"] += 1
        
        # Risk level distribution
        risk_score = doc_info.get("risk_score", 0.5)
        if risk_score <= 0.3:
            risk_levels["low"] += 1
        elif risk_score <= 0.7:
            risk_levels["medium"] += 1
        else:
            risk_levels["high"] += 1
        
        # Confidence and lawyer recommendations
        total_confidence += doc_info["analysis"]["confidence_score"]
        if doc_info["analysis"]["lawyer_recommendation"]:
            lawyer_recommendations += 1
    
    return AnalyticsResponse(
        total_documents=total_docs,
        document_types=doc_types,
        risk_distribution=risk_levels,
        avg_confidence=total_confidence / total_docs,
        total_requiring_lawyer=lawyer_recommendations
    )

@app.get("/suggest-questions/{document_id}")
async def suggest_smart_questions(document_id: str):
    """Generate intelligent question suggestions based on document content"""
    if document_id not in documents_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = documents_storage[document_id]
    
    # Return pre-analyzed suggestions or generate new ones
    suggested_questions = doc_info["analysis"].get("suggested_questions", [])
    
    if not suggested_questions:
        # Generate based on document type and content
        filename = doc_info["filename"].lower()
        if "lease" in filename:
            suggested_questions = [
                "What are the monthly rent and payment terms?",
                "Can I terminate the lease early?",
                "What are the penalties for breaking the lease?",
                "Are pets allowed and what are the fees?",
                "Who is responsible for maintenance and repairs?",
                "Does the lease automatically renew?"
            ]
        elif "employment" in filename or "job" in filename:
            suggested_questions = [
                "What are my compensation and benefits?",
                "What is the termination clause?",
                "Are there non-compete restrictions?",
                "What are the confidentiality requirements?",
                "What are my vacation and leave policies?"
            ]
        else:
            suggested_questions = [
                "What are the key terms and obligations?",
                "How can this agreement be terminated?",
                "What are the payment terms?",
                "Are there any penalties or fees?",
                "What are the main risks I should know about?"
            ]
    
    return {"questions": suggested_questions, "document_id": document_id}

@app.get("/export-report/{document_id}")
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

Risk Factors Detected: {', '.join(doc_info.get('risk_factors', []))}

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

Report Generated by AI Legal Doc Explainer v2.0
© 2025 - Competition Edition
"""
    
    return {
        "report": report_content,
        "filename": f"legal_analysis_{doc_info['filename'].replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
        "document_id": document_id,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/documents")
async def get_all_documents():
    """Get comprehensive list of all uploaded documents"""
    documents = []
    for doc_id, doc_info in documents_storage.items():
        documents.append({
            "id": doc_id,
            "filename": doc_info["filename"],
            "summary": doc_info["analysis"]["summary"][:200] + "...",
            "risk_score": doc_info.get("risk_score", 0.5),
            "confidence_score": doc_info["analysis"]["confidence_score"],
            "lawyer_recommendation": doc_info["analysis"]["lawyer_recommendation"],
            "created_at": doc_info.get("created_at", ""),
            "question_count": len(question_history.get(doc_id, []))
        })
    
    return {
        "documents": documents,
        "total_count": len(documents),
        "last_updated": datetime.now().isoformat()
    }

@app.get("/document/{document_id}")
async def get_document_details(document_id: str):
    """Get comprehensive document details"""
    if document_id not in documents_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = documents_storage[document_id]
    analysis = doc_info["analysis"]
    
    return {
        "id": document_id,
        "filename": doc_info["filename"],
        "summary": analysis["summary"],
        "key_clauses": analysis["key_clauses"],
        "red_flags": analysis["red_flags"],
        "confidence_score": analysis["confidence_score"],
        "risk_score": doc_info.get("risk_score", 0.5),
        "risk_factors": doc_info.get("risk_factors", []),
        "lawyer_recommendation": analysis["lawyer_recommendation"],
        "suggested_questions": analysis.get("suggested_questions", []),
        "created_at": doc_info.get("created_at", ""),
        "file_size": doc_info.get("file_size", 0),
        "recent_questions": question_history.get(document_id, [])[-5:]  # Last 5 questions
    }

@app.get("/health")
async def basic_health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "documents_processed": len(documents_storage)
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with all system components"""
    try:
        # Check OpenAI API
        openai_status = "healthy"
        try:
            test_response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10
            )
            if not test_response.choices:
                openai_status = "degraded"
        except Exception as e:
            openai_status = f"unhealthy: {str(e)}"
        
        # Check vector database
        vector_db_status = "healthy"
        try:
            test_embedding = embeddings.embed_query("health check")
            if not test_embedding or len(test_embedding) == 0:
                vector_db_status = "unhealthy"
        except Exception as e:
            vector_db_status = f"unhealthy: {str(e)}"
        
        # System metrics
        memory_usage = "healthy"  # In production, add actual memory monitoring
        
        overall_status = "healthy"
        if "unhealthy" in [openai_status, vector_db_status]:
            overall_status = "unhealthy"
        elif "degraded" in [openai_status]:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {
                "api_server": "healthy",
                "openai_integration": openai_status,
                "vector_database": vector_db_status,
                "memory_usage": memory_usage
            },
            "metrics": {
                "documents_processed": len(documents_storage),
                "total_questions": sum(len(questions) for questions in question_history.values()),
                "avg_risk_score": sum(doc.get("risk_score", 0) for doc in documents_storage.values()) / max(len(documents_storage), 1)
            },
            "uptime": "calculated_uptime_here"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")