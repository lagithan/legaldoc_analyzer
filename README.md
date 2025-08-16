# Ultimate Legal Document Analyzer - Backend

A comprehensive legal document analysis system integrating pre-trained Legal Models, ChromaDB vector database, and Gemini AI for advanced legal document processing.

## 🏗️ Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py             # Configuration and environment variables
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py       # Pydantic request models
│   │   └── response_models.py      # Pydantic response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── legal_model_manager.py  # Legal-BERT model management
│   │   ├── chromadb_manager.py     # Vector database operations
│   │   ├── gemini_client.py        # Gemini AI integration
│   │   ├── risk_assessment.py      # Risk analysis and lawyer recommendations
│   │   └── document_processor.py   # Document processing and analysis
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py        # Document upload and management
│   │   │   ├── questions.py        # Q&A functionality
│   │   │   ├── analytics.py        # Analytics and reporting
│   │   │   └── health.py           # Health check endpoints
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py              # Utility functions
│   └── core/
│       ├── __init__.py
│       └── prompts.py              # AI prompts and templates
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Features

### Advanced AI Integration
- **Legal-BERT Models**: Specialized BERT models trained on legal documents
- **ChromaDB with FAISS**: Vector database for semantic search with FAISS backend
- **Gemini AI**: Advanced language model for document analysis and Q&A

### Comprehensive Analysis
- **Risk Assessment**: Multi-dimensional risk scoring with pattern recognition
- **Legal Complexity**: Document complexity estimation using legal terminology
- **Lawyer Recommendations**: Intelligent recommendations with urgency levels
- **Semantic Search**: Advanced context-aware question answering

### Document Processing
- **PDF Extraction**: Robust PDF text extraction
- **Legal-Aware Chunking**: Smart document chunking respecting legal structure
- **Document Validation**: AI-powered legal document type detection
- **Similarity Detection**: Find similar documents using legal embeddings

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB+ recommended for legal models)
- GPU support optional (CUDA-compatible for faster processing)

### Environment Variables
Create a `.env` file in the backend directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🛠️ Installation

1. **Clone and navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   python -m app.main
   # Or using uvicorn directly:
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🔧 Configuration

### Legal Models
The system automatically loads these legal models:
- `nlpaueb/legal-bert-base-uncased` - Primary legal analysis model
- `nlpaueb/legal-bert-small-uncased` - Fast processing model
- Fallback to `google-bert/bert-base-uncased` if legal models fail

### ChromaDB Setup
- Uses FAISS backend for efficient vector operations
- Persistent storage in `./chroma_db` directory
- Automatic collection management for documents and analysis

## 📡 API Endpoints

### Documents
- `POST /upload-document` - Upload and analyze legal documents
- `GET /documents` - List all processed documents
- `GET /document/{document_id}` - Get detailed document information
- `GET /suggest-questions/{document_id}` - Get suggested questions

### Questions & Analysis
- `POST /ask-question` - Ask questions about documents
- `GET /analytics` - Get system analytics and statistics
- `GET /legal-models/status` - Check legal model status

### Health & Monitoring
- `GET /health` - Basic health check

## 🏛️ Architecture

### Service Layer
- **LegalModelManager**: Manages Legal-BERT models and embeddings
- **AdvancedChromaDBManager**: Handles vector database operations
- **EnhancedGeminiClient**: Manages Gemini AI API calls
- **AdvancedRiskAssessment**: Multi-dimensional risk analysis
- **LawyerRecommendationEngine**: Intelligent legal consultation recommendations

### Data Flow
1. **Document Upload** → PDF extraction → Legal validation → Analysis
2. **AI Analysis** → Legal models + Gemini AI → Risk assessment → Storage
3. **Vector Storage** → ChromaDB with legal embeddings → Semantic search
4. **Q&A Processing** → Semantic search → Legal context → AI response

## 🔍 Usage Examples

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload-document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@contract.pdf"
```

### Ask Question
```bash
curl -X POST "http://localhost:8000/ask-question" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "uuid", "question": "What are the termination conditions?"}'
```

## 🧪 Testing

Run health check to verify system status:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/legal-models/status
```

## 📊 Performance

### Model Loading
- Legal-BERT Base: ~110M parameters, ~440MB memory
- Legal-BERT Small: ~36M parameters, ~144MB memory
- ChromaDB with FAISS: Optimized for fast similarity search

### Recommended Limits
- Document size: 25MB max
- Processing time: 30-60 seconds per document
- Concurrent uploads: 5-10 depending on hardware

## 🚨 Error Handling

The system includes comprehensive error handling:
- **Fallback Models**: Automatic fallback if legal models fail
- **Robust JSON Parsing**: Multiple parsing strategies for AI responses
- **Graceful Degradation**: Continues operation with reduced functionality
- **Detailed Logging**: Comprehensive logging for debugging

## 🔒 Security Considerations

- Input validation for all file uploads
- File size and type restrictions
- API rate limiting considerations
- Secure environment variable handling

## 📈 Scaling

For production deployment:
- Use proper database instead of in-memory storage
- Implement Redis for caching
- Add authentication and authorization
- Configure load balancing for multiple instances
- Monitor GPU memory usage for model inference

## 🐛 Troubleshooting

### Common Issues

1. **Legal models not loading**
   - Check internet connection for model download
   - Verify sufficient memory (4GB+ recommended)
   - Check logs for specific error messages

2. **ChromaDB errors**
   - Ensure `./chroma_db` directory is writable
   - Check FAISS installation: `pip install faiss-cpu`

3. **Gemini API errors**
   - Verify `GEMINI_API_KEY` in `.env` file
   - Check API quota and rate limits

## 📝 License

This project is part of a competition submission. All rights reserved.

## 🤝 Contributing

This is a competition entry. For questions or issues, please refer to the competition documentation.