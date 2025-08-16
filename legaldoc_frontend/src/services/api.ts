import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { API_BASE_URL } from '../utils/constants';
import type { ApiResponse, ApiError } from '../types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 60000, // 60 seconds for large file uploads
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('[API] Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`[API] Response ${response.status} from ${response.config.url}`);
        return response;
      },
      (error: AxiosError) => {
        console.error('[API] Response error:', error);
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: AxiosError): ApiError {
    if (error.response) {
      // Server responded with error status
      const data = error.response.data as any;

      if (data?.detail) {
        // FastAPI error format
        if (typeof data.detail === 'object') {
          return {
            error: data.detail.error || 'Request failed',
            message: data.detail.message || 'An error occurred',
            suggestion: data.detail.suggestion,
            accepted_types: data.detail.accepted_types,
            reason: data.detail.reason
          };
        } else {
          return {
            error: 'Request failed',
            message: data.detail
          };
        }
      }

      return {
        error: 'Request failed',
        message: data?.message || error.message || 'An unexpected error occurred'
      };
    } else if (error.request) {
      // Network error
      return {
        error: 'Network error',
        message: 'Unable to connect to the server. Please check your connection.'
      };
    } else {
      // Other error
      return {
        error: 'Client error',
        message: error.message || 'An unexpected error occurred'
      };
    }
  }

  // Generic HTTP methods
  async get<T>(url: string, params?: any): Promise<T> {
    const response = await this.client.get(url, { params });
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.post(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any): Promise<T> {
    const response = await this.client.put(url, data);
    return response.data;
  }

  async delete<T>(url: string): Promise<T> {
    const response = await this.client.delete(url);
    return response.data;
  }

  // File upload method
  async uploadFile<T>(
    url: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  // Health check
  async healthCheck() {
    try {
      const response = await this.get('/health');
      return { success: true, data: response };
    } catch (error) {
      return { success: false, error };
    }
  }
}

// Create and export singleton instance
export const apiClient = new ApiClient();
export default apiClient;