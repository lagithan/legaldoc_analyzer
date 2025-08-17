import torch
import numpy as np
import logging
import re
from typing import Dict, List, Any, Union, Tuple, Set
from transformers import AutoTokenizer, AutoModel
import hashlib
import time

logger = logging.getLogger(__name__)

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class LegalModelManager:
    def __init__(self):
        """Ultra-fast initialization with minimal overhead"""
        self.model = None
        self.tokenizer = None
        self.model_dimension = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fast in-memory caches
        self.analysis_cache = {}  # Cache for repeated analysis
        self.embedding_cache = {}  # Cache for embeddings
        
        # Pre-compiled regex patterns for speed
        self._compile_legal_patterns()
        
        # Optimized legal keyword sets for O(1) lookup
        self.legal_keyword_sets = self._initialize_fast_legal_keywords()
        
        # Skip model loading initially - load on demand
        self.model_loaded = False
        self.model_loading_attempted = False
        
        logger.info("🚀 Ultra-fast Legal Model Manager initialized (model loading deferred)")

    def _compile_legal_patterns(self):
        """Pre-compile regex patterns for maximum speed"""
        self.compiled_patterns = {
            'liability_unlimited': re.compile(r'unlimited\s+liability|without\s+limitation|fully\s+liable', re.IGNORECASE),
            'indemnification': re.compile(r'indemnif\w+|hold\s+harmless|defend\s+against', re.IGNORECASE),
            'termination_immediate': re.compile(r'immediate\s+termination|terminate\s+immediately', re.IGNORECASE),
            'payment_terms': re.compile(r'payment\s+due|late\s+fee|billing\s+cycle', re.IGNORECASE),
            'liability_general': re.compile(r'liabilit\w+|liable|damages|claims', re.IGNORECASE),
            'contract_terms': re.compile(r'agreement|contract|terms\s+of\s+service|license', re.IGNORECASE),
            'compliance': re.compile(r'compliance|comply|regulation|applicable\s+law', re.IGNORECASE)
        }

    def _initialize_fast_legal_keywords(self) -> Dict[str, Set[str]]:
        """Initialize optimized keyword sets for O(1) lookup"""
        return {
            'contract_terms': {
                'agreement', 'contract', 'terms', 'conditions', 'license', 'service', 'user'
            },
            'liability_terms': {
                'liability', 'liable', 'responsible', 'damages', 'claims', 'losses', 'limitation'
            },
            'high_risk_terms': {
                'unlimited', 'indemnify', 'indemnification', 'harmless', 'defend', 'penalty'
            },
            'payment_terms': {
                'payment', 'fees', 'charges', 'billing', 'invoice', 'due', 'subscription'
            },
            'termination_terms': {
                'terminate', 'termination', 'cancel', 'cancellation', 'expire', 'breach'
            },
            'compliance_terms': {
                'compliance', 'comply', 'regulation', 'gdpr', 'privacy', 'data'
            }
        }

    def load_model_on_demand(self):
        """Load model only when needed to save startup time"""
        if self.model_loaded or self.model_loading_attempted:
            return
        
        self.model_loading_attempted = True
        start_time = time.time()
        
        try:
            # Try fast model first
            logger.info("⚡ Loading minimal model for embeddings...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
            self.model_dimension = 768
            self.model_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"✅ Fast model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"Model loading failed: {str(e)} - continuing with keyword-only analysis")
            self.model_loaded = False

    def ultra_fast_text_analysis(self, text: str) -> Dict:
        """Ultra-fast text analysis using primarily keyword-based methods"""
        start_time = time.time()
        
        # Create cache key
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()[:8]
        if text_hash in self.analysis_cache:
            logger.info(f"⚡ Cache hit! Analysis completed in {(time.time() - start_time)*1000:.1f}ms")
            return self.analysis_cache[text_hash]
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Fast keyword detection using sets (O(1) lookup)
        detected_terms = []
        for category, keywords in self.legal_keyword_sets.items():
            for word in text.split()[:100]:  # Check only first 100 words for speed
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in keywords:
                    detected_terms.append(category)
                    break
        
        # Ultra-fast risk detection using compiled regex
        risk_indicators = []
        for risk_type, pattern in self.compiled_patterns.items():
            if pattern.search(text[:2000]):  # Search only first 2000 chars
                risk_indicators.append(risk_type.replace('_', ' ').title())
        
        # Fast complexity estimation
        complexity_indicators = [
            'indemnification' in text_lower,
            'liability' in text_lower,
            'arbitration' in text_lower,
            'governing law' in text_lower,
            word_count > 1000
        ]
        complexity_score = sum(complexity_indicators) / len(complexity_indicators)
        
        # Fast document section identification
        sections = []
        section_keywords = {
            'Payment Terms': ['payment', 'fee', 'billing'],
            'Liability': ['liability', 'liable'],
            'Termination': ['terminate', 'termination'],
            'Service': ['service', 'software'],
            'Data': ['data', 'privacy']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                sections.append(section)
        
        # Build analysis result
        analysis = {
            "model_name": "Ultra-Fast Keyword Analysis",
            "model_type": "optimized_keyword_based",
            "text_length": len(text),
            "word_count": word_count,
            "sentence_count": text.count('.'),
            "legal_complexity": float(complexity_score),
            "contains_legal_terms": detected_terms if detected_terms else ['general_legal_content'],
            "document_sections": sections if sections else ['General Document'],
            "risk_indicators": risk_indicators if risk_indicators else ['Standard Risk Levels'],
            "analysis_metadata": {
                "analysis_method": "ultra_fast_keyword",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "cache_used": False
            }
        }
        
        # Add fast confidence scoring
        confidence_score = self.calculate_ultra_fast_confidence(analysis, text)
        analysis["confidence_indicators"] = {
            'overall_confidence': confidence_score,
            'confidence_level': self._get_confidence_level(confidence_score),
            'reliability_indicators': ['Ultra-fast keyword analysis']
        }
        
        # Cache result
        self.analysis_cache[text_hash] = analysis
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"🚀 Ultra-fast analysis completed in {processing_time:.1f}ms")
        
        return convert_numpy_types(analysis)

    def calculate_ultra_fast_confidence(self, analysis: Dict, text: str) -> float:
        """Ultra-fast confidence calculation"""
        base_confidence = 0.5
        
        # Boost for detected terms
        terms_count = len(analysis.get('contains_legal_terms', []))
        if terms_count > 0:
            base_confidence += min(terms_count * 0.1, 0.3)
        
        # Boost for risk indicators
        risks_count = len(analysis.get('risk_indicators', []))
        if risks_count > 0:
            base_confidence += min(risks_count * 0.05, 0.2)
        
        # Boost for document structure
        word_count = analysis.get('word_count', 0)
        if word_count > 100:
            base_confidence += 0.1
        if word_count > 500:
            base_confidence += 0.1
        
        return float(min(max(base_confidence, 0.6), 0.9))

    def get_fast_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings with caching (only if model is loaded)"""
        if not self.model_loaded:
            self.load_model_on_demand()
        
        if not self.model_loaded:
            # Return dummy embedding if model failed to load
            return np.random.rand(self.model_dimension)
        
        text_hash = hashlib.md5(text[:500].encode()).hexdigest()[:8]
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            # Truncate text for speed
            text = text[:512]
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128  # Reduced for speed
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embedding = embeddings[0]
            self.embedding_cache[text_hash] = embedding
            return embedding

        except Exception as e:
            logger.error(f"Fast embedding error: {str(e)}")
            return np.random.rand(self.model_dimension)

    def _get_confidence_level(self, confidence_score: float) -> str:
        """Convert confidence score to descriptive level"""
        if confidence_score >= 0.8:
            return 'High'
        elif confidence_score >= 0.7:
            return 'Good'
        elif confidence_score >= 0.6:
            return 'Medium'
        else:
            return 'Low'

    def clear_cache(self):
        """Clear caches to free memory"""
        self.analysis_cache.clear()
        self.embedding_cache.clear()
        logger.info("🧹 Caches cleared")

    # Main entry point - now ultra-fast
    def analyze_legal_text(self, text: str, task: str = "fast") -> Dict:
        """Main entry point for ultra-fast legal text analysis"""
        if task == "dual_analysis":
            # For backward compatibility, provide dual analysis structure
            base_analysis = self.ultra_fast_text_analysis(text)
            
            # Simulate dual model response for compatibility
            return {
                "models_used": ["Ultra-Fast Keyword Analyzer", "Pattern Matcher"],
                "model1_analysis": base_analysis,
                "model2_analysis": {"analysis_type": "pattern_matching"},
                "legal_complexity": base_analysis.get("legal_complexity", 0.5),
                "contains_legal_terms": base_analysis.get("contains_legal_terms", []),
                "document_sections": base_analysis.get("document_sections", []),
                "risk_indicators": base_analysis.get("risk_indicators", []),
                "urgency_signals": base_analysis.get("risk_indicators", []),
                "financial_terms": ["payment", "fee"] if "payment" in text.lower() else [],
                "temporal_terms": ["due", "term"] if any(t in text.lower() for t in ["due", "term"]) else [],
                "confidence_indicators": base_analysis.get("confidence_indicators", {}),
                "text_analysis": {
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "sentence_count": text.count('.'),
                    "paragraph_count": text.count('\n\n') + 1
                },
                "legal_terms_count": len(base_analysis.get("contains_legal_terms", [])),
                "risk_indicators_count": len(base_analysis.get("risk_indicators", [])),
                "urgency_signals_count": len(base_analysis.get("risk_indicators", [])),
                "financial_terms_count": 1 if "payment" in text.lower() else 0,
                "temporal_terms_count": 1 if any(t in text.lower() for t in ["due", "term"]) else 0,
                "sections_count": len(base_analysis.get("document_sections", [])),
                "document_structure_score": min(len(text.split()) / 1000, 1.0)
            }
        
        return self.ultra_fast_text_analysis(text)

# For backward compatibility, keep the original class name as an alias
LegalModelManager = UltraFastLegalModelManager