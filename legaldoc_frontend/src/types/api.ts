export interface ApiResponse<T> {
  data?: T;
  error?: string;
  success: boolean;
  message?: string;
}

export interface ApiError {
  error: string;
  message: string;
  suggestion?: string;
  accepted_types?: string[];
  reason?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  documents_processed: number;
  components?: {
    api_server: string;
    openai_integration: string;
    vector_database: string;
    memory_usage: string;
  };
  metrics?: {
    documents_processed: number;
    total_questions: number;
    avg_risk_score: number;
  };
}

export interface LegalModelsStatus {
  available_models: Record<string, {
    model_id: string;
    description: string;
    size: string;
    speed: string;
  }>;
  loaded_models: string[];
  device: string;
  total_models: number;
  status: 'operational' | 'degraded';
}

export interface UploadProgress {
  file_name: string;
  progress: number;
  status: 'uploading' | 'processing' | 'analyzing' | 'complete' | 'error';
  error?: string;
}