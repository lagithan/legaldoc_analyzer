export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  document_id?: string;
  confidence_score?: number;
  relevant_sections?: string[];
  legal_implications?: string;
  follow_up_questions?: string[];
}

export interface QuestionRequest {
  document_id: string;
  question: string;
}

export interface QuestionResponse {
  answer: string;
  confidence_score: number;
  relevant_sections: string[];
  follow_up_questions: string[];
  legal_implications: string;
  semantic_matches: SemanticMatch[];
  legal_model_insights: Record<string, any>;
}

export interface SemanticMatch {
  text: string;
  similarity_score: number;
  legal_complexity: number;
  chunk_type: string;
  rank: number;
}

export interface SuggestedQuestionsResponse {
  questions: string[];
  document_id: string;
}

export interface ChatSession {
  document_id: string;
  document_name: string;
  messages: ChatMessage[];
  is_active: boolean;
  last_updated: Date;
}