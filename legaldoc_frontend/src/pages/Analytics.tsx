import React from 'react';
import { RefreshCw, AlertCircle, Upload } from 'lucide-react';
import AnalyticsDashboard from '../components/analytics/AnalyticsDashboard';
import Button from '../components/common/Button';
import { useAnalytics } from '../hooks/useAnalytics';

const Analytics: React.FC = () => {
  const {
    analytics,
    legalModelsStatus,
    loading,
    error,
    refreshAll,
    clearError
  } = useAnalytics();

  const handleRefresh = async () => {
    clearError();
    await refreshAll();
  };

  const handleUploadRedirect = () => {
    window.location.href = '/documents';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-black">Analytics</h1>
          <p className="text-gray-600 mt-1">
            Insights and analytics for your legal documents
          </p>
        </div>

        <div className="flex items-center space-x-3">
          {analytics && analytics.total_documents === 0 && (
            <Button
              variant="primary"
              onClick={handleUploadRedirect}
              icon={<Upload className="w-4 h-4" />}
            >
              Upload Documents
            </Button>
          )}

          <Button
            variant="secondary"
            onClick={handleRefresh}
            loading={loading}
            icon={<RefreshCw className="w-4 h-4" />}
          >
            Refresh Data
          </Button>
        </div>
      </div>

      {/* Analytics Dashboard */}
      <AnalyticsDashboard
        analytics={analytics}
        legalModelsStatus={legalModelsStatus}
        loading={loading}
      />

      {/* Error Display */}
      {error && (
        <div className="fixed bottom-4 right-4 max-w-sm z-40">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-red-600" />
                <span className="text-sm text-red-800">{error}</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearError}
                className="p-1 text-red-600 hover:text-red-800"
              >
                ×
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;