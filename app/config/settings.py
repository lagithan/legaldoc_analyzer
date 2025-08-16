# backend/app/config/settings.py

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# ================================
# LEGAL MODELS CONFIGURATION
# ================================

AVAILABLE_LEGAL_MODELS = {
    "legal_bert_base": {
        "model_id": "nlpaueb/legal-bert-base-uncased",
        "description": "BERT trained on 12GB of legal documents",
        "size": "110M parameters",
        "speed": "medium"
    },
    "legal_bert_small": {
        "model_id": "nlpaueb/legal-bert-small-uncased",
        "description": "Lightweight Legal BERT (33% size)",
        "size": "36M parameters",
        "speed": "fast"
    },
    "casehold_legal_bert": {
        "model_id": "casehold/legalbert",
        "description": "BERT trained on Harvard Law case corpus",
        "size": "110M parameters",
        "speed": "medium"
    }
}

# Gemini Configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_HEADERS = {
    'Content-Type': 'application/json',
    'X-goog-api-key': GEMINI_API_KEY
}