import { useState, useEffect, useCallback } from 'react';
import { documentService } from '../services/documentService';
import type { AnalyticsResponse, LegalModelsStatus } from '../types/document';

export interface UseAnalyticsReturn {
  analytics: AnalyticsResponse | null;
  legalModelsStatus: LegalModelsStatus | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchAnalytics: () => Promise<void>;
  fetchLegalModelsStatus: () => Promise<void>;
  refreshAll: () => Promise<void>;
  clearError: () => void;
}

export const useAnalytics = (): UseAnalyticsReturn => {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [legalModelsStatus, setLegalModelsStatus] = useState<LegalModelsStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch analytics
  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await documentService.getAnalytics();
      setAnalytics(response);
    } catch (err: any) {
      console.log('Analytics fetch error:', err);

      // If it's a "no documents" error, set empty analytics instead of error
      if (err.message?.includes('No documents') || err.message?.includes('404')) {
        console.log('No documents found, setting empty analytics');
        setAnalytics({
          total_documents: 0,
          document_types: {},
          risk_distribution: { low: 0, medium: 0, high: 0, urgent: 0 },
          avg_confidence: 0,
          total_requiring_lawyer: 0
        });
      } else {
        setError(err.message || 'Failed to fetch analytics');
        console.error('Error fetching analytics:', err);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch legal models status
  const fetchLegalModelsStatus = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await documentService.getLegalModelsStatus();
      setLegalModelsStatus(response);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch legal models status');
      console.error('Error fetching legal models status:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Refresh all data
  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [analyticsResponse, modelsResponse] = await Promise.all([
        documentService.getAnalytics().catch(err => {
          console.log('Analytics error in refreshAll:', err);
          // Return empty analytics if no documents
          if (err.message?.includes('No documents') || err.message?.includes('404')) {
            return {
              total_documents: 0,
              document_types: {},
              risk_distribution: { low: 0, medium: 0, high: 0, urgent: 0 },
              avg_confidence: 0,
              total_requiring_lawyer: 0
            };
          }
          throw err;
        }),
        documentService.getLegalModelsStatus(),
      ]);

      setAnalytics(analyticsResponse);
      setLegalModelsStatus(modelsResponse);
    } catch (err: any) {
      setError(err.message || 'Failed to refresh data');
      console.error('Error refreshing analytics data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Load data on mount
  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  return {
    analytics,
    legalModelsStatus,
    loading,
    error,

    // Actions
    fetchAnalytics,
    fetchLegalModelsStatus,
    refreshAll,
    clearError,
  };
};