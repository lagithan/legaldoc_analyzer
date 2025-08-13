export type DocumentResponse = {
  id: string;
  filename: string;
  summary: string;
  key_clauses: string[];
  red_flags: string[];
  confidence_score: number;
  risk_score: number;
  lawyer_recommendation: boolean;
  lawyer_urgency: 'low' | 'medium' | 'high' | 'urgent';
  risk_breakdown: Record<string, number>;
  complexity_score: number;
  suggested_questions: string[];
  document_type: string;
  estimated_reading_time: number;
  ai_confidence_reasoning: string;
  similar_documents: Array<{
    document_id: string;
    filename: string;
    document_type: string;
    similarity_score: number;
    risk_score: number;
    complexity_score: number;
    relevant_text: string;
  }>;
  legal_model_analysis: Record<string, any>;
  legal_terminology_found: string[];
  risk_indicators: string[];
  urgency_signals: string[];
  created_at: string;
  file_size: number;
  recent_questions: Array<{ question: string; answer: string; timestamp: string }>;
  chunk_types: string[];
  chunks: Array<{
    text: string;
    chunk_type: string;
    legal_complexity: number;
    legal_terms_count: number;
    risk_indicators_count: number;
  }>;
};

export type DocumentsListResponse = {
  documents: Array<{
    id: string;
    filename: string;
    summary: string;
    risk_score: number;
    confidence_score: number;
    lawyer_recommendation: boolean;
    created_at: string;
    question_count: number;
    document_type: string;
    complexity_score: number;
    lawyer_urgency: string;
    legal_terms_count: number;
    risk_indicators_count: number;
    urgency_signals_count: number;
    model_used: string;
    chromadb_stored: boolean;
  }>;
  total_count: number;
  by_type: Record<string, number>;
  risk_distribution: Record<string, number>;
  model_usage: Record<string, number>;
};

export type QuestionResponse = {
  answer: string;
  confidence_score: number;
  relevant_sections: Array<{
    text: string;
    chunk_type: string;
    legal_complexity: number;
    legal_terms_count: number;
    risk_indicators_count: number;
  }>;
  follow_up_questions: string[];
  legal_implications: string;
  semantic_matches: Array<{
    text: string;
    similarity_score: number;
    legal_complexity: number;
    chunk_type: string;
    rank: number;
  }>;
  legal_model_insights: Record<string, any>;
};

export type AnalyticsResponse = {
  total_documents: number;
  document_types: Record<string, number>;
  risk_distribution: Record<string, number>;
  avg_confidence: number;
  total_requiring_lawyer: number;
};

export type CompareResponse = {
  comparison_summary: string;
  key_differences: string[];
  risk_comparison: Record<string, number>;
  recommendations: string[];
};

export type SimilarDocumentsResponse = {
  query: string;
  similar_documents: Array<{
    document_id: string;
    filename: string;
    document_type: string;
    similarity_score: number;
    risk_score: number;
    complexity_score: number;
    relevant_text: string;
  }>;
  total_found: number;
};

const BASE = "http://localhost:8000";

export async function uploadDocument(file: File): Promise<DocumentResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload-document`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listDocuments(): Promise<DocumentsListResponse> {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDocument(id: string): Promise<DocumentResponse> {
  const res = await fetch(`${BASE}/document/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function askQuestion(document_id: string, question: string): Promise<QuestionResponse> {
  const res = await fetch(`${BASE}/ask-question`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id, question }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getAnalytics(): Promise<AnalyticsResponse> {
  const res = await fetch(`${BASE}/analytics`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function compareDocuments(document_ids: string[]): Promise<CompareResponse> {
  const res = await fetch(`${BASE}/compare-documents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_ids }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function exportReport(id: string) {
  const res = await fetch(`${BASE}/export-report/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function suggestQuestions(id: string): Promise<{ questions: string[]; document_id: string }> {
  const res = await fetch(`${BASE}/suggest-questions/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSimilarDocuments(id: string, complexity_filter?: number): Promise<SimilarDocumentsResponse> {
  const url = complexity_filter
    ? `${BASE}/similar-documents/${id}?complexity_filter=${complexity_filter}`
    : `${BASE}/similar-documents/${id}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}