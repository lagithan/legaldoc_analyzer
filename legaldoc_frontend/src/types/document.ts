export interface DocumentResponse {
  id: string;
  filename: string;
  summary: string;
  key_clauses: string[];
  red_flags: string[];
  confidence_score: number;
  risk_score: number;
  lawyer_recommendation: boolean;
  lawyer_urgency: 'low' | 'medium' | 'high' | 'urgent';
  risk_breakdown: RiskBreakdown;
  complexity_score: number;
  suggested_questions: string[];
  document_type: DocumentType;
  estimated_reading_time: number;
  ai_confidence_reasoning: string;
  similar_documents: SimilarDocument[];
  legal_model_analysis: LegalModelAnalysis;
  legal_terminology_found: string[];
  risk_indicators: string[];
  urgency_signals: string[];
}

export interface RiskBreakdown {
  financial_risk: number;
  termination_risk: number;
  liability_risk: number;
  renewal_risk: number;
  modification_risk: number;
}

export interface SimilarDocument {
  text: string;
  similarity_score: number;
  legal_complexity: number;
  chunk_type: string;
  rank: number;
}

export interface LegalModelAnalysis {
  model_used: string;
  text_length: number;
  embedding_dimension: number;
  legal_complexity: number;
  contains_legal_terms: string[];
  document_sections: string[];
  risk_indicators: string[];
  urgency_signals: string[];
}

export type DocumentType =
  | 'lease_agreement'
  | 'employment_contract'
  | 'service_agreement'
  | 'nda'
  | 'terms_of_service'
  | 'purchase_agreement'
  | 'loan_agreement'
  | 'partnership_agreement'
  | 'vendor_contract'
  | 'licensing_agreement'
  | 'insurance_policy'
  | 'warranty'
  | 'other_legal'
  | 'other';

export interface DocumentListItem {
  id: string;
  filename: string;
  summary: string;
  document_type: DocumentType;
  risk_score: number;
  complexity_score: number;
  confidence_score: number;
  lawyer_urgency: string;
  reading_time: number;
  validated: boolean;
  legal_terms_count: number;
  risk_indicators_count: number;
  urgency_signals_count: number;
  model_used: string;
  chromadb_stored: boolean;
  created_at: string;
}

export interface DocumentsResponse {
  documents: DocumentListItem[];
  total_count: number;
  by_type: Record<string, number>;
  risk_distribution: Record<string, number>;
  model_usage: Record<string, number>;
}

export interface AnalyticsResponse {
  total_documents: number;
  document_types: Record<string, number>;
  risk_distribution: Record<string, number>;
  avg_confidence: number;
  total_requiring_lawyer: number;
}