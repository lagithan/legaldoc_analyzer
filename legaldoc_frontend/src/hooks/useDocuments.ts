import { useState, useEffect, useCallback } from 'react';
import { documentService } from '../services/documentService';
import type { DocumentResponse, DocumentsResponse, DocumentListItem } from '../types/document';

export interface UseDocumentsReturn {
  documents: DocumentListItem[];
  currentDocument: DocumentResponse | null;
  loading: boolean;
  error: string | null;
  uploading: boolean;
  uploadProgress: number;
  totalCount: number;
  byType: Record<string, number>;
  riskDistribution: Record<string, number>;
  modelUsage: Record<string, number>;

  // Actions
  fetchDocuments: () => Promise<void>;
  fetchDocument: (id: string) => Promise<void>;
  uploadDocument: (file: File) => Promise<DocumentResponse | null>;
  deleteDocument: (id: string) => Promise<boolean>;
  clearError: () => void;
  clearCurrentDocument: () => void;
}

export const useDocuments = (): UseDocumentsReturn => {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [currentDocument, setCurrentDocument] = useState<DocumentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [byType, setByType] = useState<Record<string, number>>({});
  const [riskDistribution, setRiskDistribution] = useState<Record<string, number>>({});
  const [modelUsage, setModelUsage] = useState<Record<string, number>>({});

  // Fetch all documents
  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await documentService.getDocuments();
      setDocuments(response.documents);
      setTotalCount(response.total_count);
      setByType(response.by_type);
      setRiskDistribution(response.risk_distribution);
      setModelUsage(response.model_usage);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch documents');
      console.error('Error fetching documents:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch specific document
  const fetchDocument = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);

    try {
      const document = await documentService.getDocument(id);
      setCurrentDocument(document);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch document');
      console.error('Error fetching document:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Upload document
  const uploadDocument = useCallback(async (file: File): Promise<DocumentResponse | null> => {
    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const response = await documentService.uploadDocument(
        file,
        (progress) => setUploadProgress(progress)
      );

      // Refresh documents list
      await fetchDocuments();

      return response;
    } catch (err: any) {
      setError(err.message || 'Failed to upload document');
      console.error('Error uploading document:', err);
      return null;
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [fetchDocuments]);

  // Delete document
  const deleteDocument = useCallback(async (id: string): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      await documentService.deleteDocument(id);

      // Remove from local state
      setDocuments(prev => prev.filter(doc => doc.id !== id));

      // Clear current document if it was deleted
      if (currentDocument?.id === id) {
        setCurrentDocument(null);
      }

      // Update counts
      setTotalCount(prev => prev - 1);

      return true;
    } catch (err: any) {
      setError(err.message || 'Failed to delete document');
      console.error('Error deleting document:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, [currentDocument]);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Clear current document
  const clearCurrentDocument = useCallback(() => {
    setCurrentDocument(null);
  }, []);

  // Load documents on mount
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return {
    documents,
    currentDocument,
    loading,
    error,
    uploading,
    uploadProgress,
    totalCount,
    byType,
    riskDistribution,
    modelUsage,

    // Actions
    fetchDocuments,
    fetchDocument,
    uploadDocument,
    deleteDocument,
    clearError,
    clearCurrentDocument,
  };
};