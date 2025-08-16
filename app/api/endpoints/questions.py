# backend/app/api/endpoints/questions.py

from fastapi import APIRouter, HTTPException
import json
import re
import logging

from ...models.request_models import QuestionRequest
from ...models.response_models import QuestionResponse
from ...services.chromadb_manager import AdvancedChromaDBManager
from ...services.legal_model_manager import LegalModelManager
from ...services.gemini_client import EnhancedGeminiClient
from ...services.document_processor import enhanced_context_search
from ...core.prompts import ENHANCED_QA_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
advanced_chroma_manager = AdvancedChromaDBManager()
legal_model_manager = LegalModelManager()
gemini_client = EnhancedGeminiClient()

# In-memory storage (should be same as documents)
documents_storage = {}

@router.post("/ask-question", response_model=QuestionResponse)
async def ask_question_ultimate(question_request: QuestionRequest):
    """Enhanced Q&A using ChromaDB semantic search + dual legal models + Gemini with comprehensive analysis"""
    if question_request.document_id not in documents_storage:
        logger.error(f"Document {question_request.document_id} not found")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        doc_info = documents_storage[question_request.document_id]

        logger.info(f"Processing question for document {question_request.document_id} with dual AI models")

        # Perform semantic search with ChromaDB
        semantic_results = advanced_chroma_manager.advanced_semantic_search(
            question_request.question,
            document_id=question_request.document_id,
            n_results=3
        )

        # Build semantic context from ChromaDB results
        semantic_context = ""
        semantic_matches = []
        for result in semantic_results:
            semantic_context += f"Relevant section (similarity: {result['similarity_score']:.2f}, complexity: {result['legal_complexity']:.2f}): {result['text']}\n\n"
            semantic_matches.append({
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "similarity_score": result["similarity_score"],
                "legal_complexity": result["legal_complexity"],
                "chunk_type": result["chunk_type"],
                "rank": result["rank"]
            })

        # Fallback to traditional search if no semantic results
        if not semantic_context:
            logger.warning(f"No semantic results for question on document {question_request.document_id}")
            semantic_context = enhanced_context_search(doc_info["text"], question_request.question)

        # Get comprehensive dual legal model insights for the question context
        logger.info("Analyzing question context with dual AI models...")
        legal_insights = legal_model_manager.analyze_legal_text(
            semantic_context, task="dual_analysis"
        )

        # Extract existing legal model analysis from document if available
        existing_analysis = doc_info.get("legal_model_analysis", {})

        # Prepare comprehensive model insights for Gemini
        comprehensive_insights = {
            "question_context_analysis": legal_insights,
            "document_analysis": existing_analysis,
            "models_used": legal_insights.get('models_used', ['unknown', 'unknown']),
            "model_agreement": legal_insights.get('confidence_indicators', {}).get('models_agreement', 0.5),
            "legal_complexity": legal_insights.get('legal_complexity', 0.5),
            "risk_indicators": legal_insights.get('risk_indicators', []),
            "urgency_signals": legal_insights.get('urgency_signals', []),
            "legal_terms_in_context": legal_insights.get('contains_legal_terms', []),
            "document_sections": legal_insights.get('document_sections', [])
        }

        # Create enhanced Q&A prompt with dual-model insights
        prompt = ENHANCED_QA_PROMPT.format(
            semantic_context=semantic_context,
            legal_model_insights=json.dumps(comprehensive_insights, indent=2),
            models_used=', '.join(legal_insights.get('models_used', ['unknown', 'unknown'])),
            model_agreement=f"{legal_insights.get('confidence_indicators', {}).get('models_agreement', 0.5):.1%}",
            legal_complexity=f"{legal_insights.get('legal_complexity', 0.5):.1%}",
            risk_indicators=', '.join(legal_insights.get('risk_indicators', [])),
            urgency_signals=', '.join(legal_insights.get('urgency_signals', [])),
            question=question_request.question
        )

        # Call Gemini API with enhanced error handling
        try:
            logger.info("Sending comprehensive analysis to Gemini for question answering...")
            response_text = await gemini_client.generate_content(prompt, max_tokens=800)

            # Robust JSON extraction using regex
            def extract_json(text):
                text = text.strip()
                # Find code block if present
                code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    # Find the largest valid JSON object {}
                    match = re.search(r'\{[\s\S]*\}', text)
                    if match:
                        json_str = match.group(0)
                    else:
                        return None

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error during extraction: {str(e)}")
                    return None

            result = extract_json(response_text)

            if result is None:
                raise json.JSONDecodeError("Failed to extract valid JSON", response_text, 0)

            logger.info("Gemini analysis completed successfully for question")

        except Exception as gemini_err:
            logger.error(f"Gemini API or JSON extraction failed: {str(gemini_err)}")
            # Enhanced fallback with dual-model insights
            models_used = legal_insights.get('models_used', ['legal model 1', 'legal model 2'])
            model_agreement = legal_insights.get('confidence_indicators', {}).get('models_agreement', 0.5)

            result = {
                "answer": f"Based on dual AI model analysis ({', '.join(models_used)} with {model_agreement:.1%} agreement), I can see relevant legal terms and patterns in your document, but encountered an error generating the specific answer. Please try rephrasing your question or ask about specific clauses.",
                "confidence_score": min(0.5 + (model_agreement * 0.2), 0.7),
                "relevant_sections": [section[:100] + "..." for section in semantic_context.split('\n\n')[:2] if section.strip()],
                "follow_up_questions": [
                    "Can you ask about a specific clause or term?",
                    "What specific obligations are you concerned about?",
                    "Would you like clarification on any particular section?"
                ],
                "legal_implications": f"Dual model analysis indicates {len(legal_insights.get('risk_indicators', []))} risk indicators. Professional legal review recommended for detailed guidance."
            }

        # Ensure all required fields are present
        result.setdefault("answer", "No answer generated. Please try again with a more specific question.")
        result.setdefault("confidence_score", 0.5)
        result.setdefault("relevant_sections", [])
        result.setdefault("follow_up_questions", [
            "What other aspects should I consider?",
            "Are there any exceptions to this?",
            "What are the practical implications?"
        ])
        result.setdefault("legal_implications", "Review with legal counsel recommended.")

        # Validate field types and enhance with model insights
        if not isinstance(result["confidence_score"], (int, float)):
            result["confidence_score"] = 0.5
        if not isinstance(result["relevant_sections"], list):
            result["relevant_sections"] = []
        if not isinstance(result["follow_up_questions"], list):
            result["follow_up_questions"] = ["What other questions do you have about this document?"]
        if not isinstance(result["legal_implications"], str):
            result["legal_implications"] = "Legal review advised."

        # Enhance answer with dual-model confidence information
        if "answer" in result and result["answer"]:
            models_info = f" (Analysis powered by {', '.join(legal_insights.get('models_used', []))}"
            if legal_insights.get('confidence_indicators', {}).get('models_agreement', 0) > 0.7:
                models_info += " with high model agreement)"
            else:
                models_info += ")"

            # Don't append if answer already mentions models
            if "model" not in result["answer"].lower():
                result["answer"] += models_info

        logger.info(f"Successfully processed question for document {question_request.document_id} with dual AI models")
        return QuestionResponse(
            answer=result["answer"],
            confidence_score=float(result["confidence_score"]),
            relevant_sections=result["relevant_sections"],
            follow_up_questions=result["follow_up_questions"],
            legal_implications=result["legal_implications"],
            semantic_matches=semantic_matches,
            legal_model_insights=comprehensive_insights
        )

    except Exception as e:
        logger.error(f"Error processing question for document {question_request.document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
            headers={"X-Error": "QuestionProcessingError"}
        )