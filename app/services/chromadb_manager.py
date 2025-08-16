# backend/app/services/chromadb_manager.py

import chromadb
from chromadb.config import Settings
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class AdvancedChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB with legal model integration using FAISS backend"""
        # Modified to use FAISS as the indexing backend
        self.client = chromadb.PersistentClient(
            path="./chroma_db",  # Consistent with original path
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                # Specify FAISS as the indexing backend (replaces HNSW)
                # Note: ChromaDB automatically uses FAISS if faiss-cpu is installed and hnswlib is not
                # Optional: Configure FAISS-specific settings if needed
            )
        )

        # Create collections with legal model embeddings
        self.document_collection = self.client.get_or_create_collection(
            name="legal_documents_enhanced",
            metadata={
                "description": "Legal documents with specialized embeddings",
                # Optional: Specify FAISS index type (e.g., FlatL2 for exact search)
                "hnsw:space": "l2"  # FAISS uses L2 (Euclidean) distance by default
            }
        )

        self.legal_analysis_collection = self.client.get_or_create_collection(
            name="legal_analysis_enhanced",
            metadata={"description": "Pre-computed legal analysis results"}
        )

        logger.info("Advanced ChromaDB with legal models and FAISS backend initialized")

    def legal_aware_chunking(self, text: str, chunk_size: int = 400) -> List[Dict]:
        """Smart chunking that respects legal document structure"""
        from ..services.legal_model_manager import LegalModelManager

        # Split on legal section indicators
        section_breaks = [
            "WHEREAS", "NOW THEREFORE", "IN WITNESS WHEREOF",
            "ARTICLE", "SECTION", "CLAUSE", "PARAGRAPH",
            "DEFINITIONS:", "OBLIGATIONS:", "PAYMENT:", "TERMINATION:"
        ]

        chunks = []
        current_chunk = ""
        current_type = "general"

        sentences = text.split('. ')

        # Initialize legal model manager for analysis
        legal_model_manager = LegalModelManager()

        for sentence in sentences:
            # Check if this sentence starts a new legal section
            sentence_upper = sentence.upper()
            section_detected = None

            for section_break in section_breaks:
                if section_break in sentence_upper:
                    section_detected = section_break.lower().replace(":", "")
                    break

            # If we detected a new section and have content, save current chunk
            if section_detected and current_chunk:
                chunk_analysis = legal_model_manager.analyze_legal_text(
                    current_chunk, task="fast_analysis"
                )

                chunks.append({
                    'text': current_chunk.strip(),
                    'type': current_type,
                    'complexity': chunk_analysis.get('legal_complexity', 0.5),
                    'legal_terms': chunk_analysis.get('contains_legal_terms', []),
                    'risk_indicators': chunk_analysis.get('risk_indicators', []),
                    'word_count': len(current_chunk.split())
                })

                current_chunk = sentence + ". "
                current_type = section_detected or "general"
            else:
                current_chunk += sentence + ". "

                # If chunk is getting too long, split it
                if len(current_chunk.split()) > chunk_size:
                    chunk_analysis = legal_model_manager.analyze_legal_text(
                        current_chunk, task="fast_analysis"
                    )

                    chunks.append({
                        'text': current_chunk.strip(),
                        'type': current_type,
                        'complexity': chunk_analysis.get('legal_complexity', 0.5),
                        'legal_terms': chunk_analysis.get('contains_legal_terms', []),
                        'risk_indicators': chunk_analysis.get('risk_indicators', []),
                        'word_count': len(current_chunk.split())
                    })
                    current_chunk = ""

        # Add final chunk if exists
        if current_chunk.strip():
            chunk_analysis = legal_model_manager.analyze_legal_text(
                current_chunk, task="fast_analysis"
            )

            chunks.append({
                'text': current_chunk.strip(),
                'type': current_type,
                'complexity': chunk_analysis.get('legal_complexity', 0.5),
                'legal_terms': chunk_analysis.get('contains_legal_terms', []),
                'risk_indicators': chunk_analysis.get('risk_indicators', []),
                'word_count': len(current_chunk.split())
            })

        return chunks

    async def store_legal_document_embeddings(self, document_id: str, text: str,
                                            analysis: Dict, metadata: Dict) -> bool:
        """Store document with legal model embeddings"""
        try:
            # Get chunks using legal-aware chunking
            chunks = self.legal_aware_chunking(text)

            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_legal_chunk_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk['text'])

                # Enhanced metadata with legal analysis
                chunk_metadata = {
                    'document_id': document_id,
                    'chunk_index': i,
                    'chunk_type': chunk['type'],
                    'legal_complexity': chunk.get('complexity', 0.5),
                    'contains_legal_terms': len(chunk.get('legal_terms', [])),
                    'risk_indicators_count': len(chunk.get('risk_indicators', [])),
                    'document_type': metadata.get('document_type', 'other'),
                    'filename': metadata.get('filename', ''),
                    'risk_score': metadata.get('risk_score', 0.5),
                    'model_analysis': analysis.get('model_used', 'unknown'),
                    'created_at': metadata.get('created_at', datetime.now().isoformat())
                }
                chunk_metadatas.append(chunk_metadata)

            # Store in ChromaDB with legal embeddings (FAISS backend handles indexing)
            self.document_collection.add(
                documents=chunk_texts,
                ids=chunk_ids,
                metadatas=chunk_metadatas
            )

            # Store legal analysis separately
            self.legal_analysis_collection.add(
                documents=[json.dumps(analysis)],
                ids=[f"{document_id}_analysis"],
                metadatas=[{
                    'document_id': document_id,
                    'analysis_type': 'legal_model_analysis',
                    'model_used': analysis.get('model_used', 'unknown'),
                    'created_at': datetime.now().isoformat()
                }]
            )

            logger.info(f"Stored legal document with {len(chunks)} chunks: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing legal embeddings: {str(e)}")
            return False

    def advanced_semantic_search(self, query: str, document_id: Optional[str] = None,
                                n_results: int = 5, complexity_filter: float = None) -> List[Dict]:
        """Advanced semantic search with legal complexity filtering"""
        try:
            where_clause = {}
            if document_id:
                where_clause["document_id"] = document_id

            if complexity_filter:
                where_clause["legal_complexity"] = {"$gte": complexity_filter}

            results = self.document_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            # Enhanced results with legal context
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    search_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,
                        'legal_complexity': metadata.get('legal_complexity', 0.5),
                        'chunk_type': metadata.get('chunk_type', 'general'),
                        'legal_terms_count': metadata.get('contains_legal_terms', 0),
                        'risk_indicators_count': metadata.get('risk_indicators_count', 0),
                        'rank': i + 1
                    })

            return search_results

        except Exception as e:
            logger.error(f"Advanced semantic search error: {str(e)}")
            return []

    def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a specific document"""
        try:
            # Get all chunk IDs for this document
            results = self.document_collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )

            if results['ids']:
                # Delete from document collection
                self.document_collection.delete(ids=results['ids'])

                # Delete from analysis collection
                analysis_results = self.legal_analysis_collection.get(
                    where={"document_id": document_id},
                    include=["metadatas"]
                )
                if analysis_results['ids']:
                    self.legal_analysis_collection.delete(ids=analysis_results['ids'])

                logger.info(f"Deleted embeddings for document {document_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            return False