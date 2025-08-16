import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload,
  FileText,
  TrendingUp,
  AlertTriangle,
  MessageSquare,
  Plus,
  ArrowRight
} from 'lucide-react';
import DocumentUpload from '../components/documents/DocumentUpload';
import DocumentCard from '../components/documents/DocumentCard';
import StatsCard from '../components/analytics/StatsCard';
import RiskChart from '../components/analytics/RiskChart';
import Card, { CardHeader, CardTitle, CardContent } from '../components/common/Card';
import Button from '../components/common/Button';
import LoadingSpinner, { Skeleton } from '../components/common/LoadingSpinner';
import Modal from '../components/common/Modal';
import { useDocuments } from '../hooks/useDocuments';
import { useChat } from '../hooks/useChat';
import { cn } from '../utils/cn';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [showUploadModal, setShowUploadModal] = useState(false);

  const {
    documents,
    loading,
    error,
    uploading,
    uploadProgress,
    totalCount,
    riskDistribution,
    uploadDocument,
    deleteDocument,
    clearError
  } = useDocuments();

  const { openChat } = useChat();

  // Get recent documents (last 5)
  const recentDocuments = documents.slice(0, 5);

  // Calculate quick stats
  const highRiskCount = (riskDistribution.high || 0) + (riskDistribution.urgent || 0);
  const avgRiskScore = documents.length > 0
    ? documents.reduce((sum, doc) => sum + doc.risk_score, 0) / documents.length
    : 0;

  const handleUpload = async (file: File) => {
    const result = await uploadDocument(file);
    if (result) {
      setShowUploadModal(false);
      // Navigate to the uploaded document
      navigate(`/documents/${result.id}`);
    }
  };

  const handleViewDocument = (document: any) => {
    navigate(`/documents/${document.id}`);
  };

  const handleChatWithDocument = (document: any) => {
    console.log('Opening chat for document:', document.filename);
    openChat(document.id, document.filename);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-black">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Welcome to your AI-powered legal document analyzer
          </p>
        </div>

        <Button
          variant="primary"
          size="lg"
          onClick={() => setShowUploadModal(true)}
          icon={<Plus className="w-5 h-5" />}
          loading={uploading}
        >
          Upload Document
        </Button>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Documents"
          value={totalCount}
          description="Legal documents analyzed"
          icon={FileText}
          color="info"
        />

        <StatsCard
          title="Average Risk Score"
          value={`${Math.round(avgRiskScore * 100)}%`}
          description="Across all documents"
          icon={TrendingUp}
          color={avgRiskScore > 0.6 ? "warning" : "success"}
        />

        <StatsCard
          title="AI Assistant"
          value="Ready"
          description="Ask questions about your docs"
          icon={MessageSquare}
          color="success"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Recent Documents */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>Recent Documents</span>
                </CardTitle>

                {totalCount > 5 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => navigate('/documents')}
                    icon={<ArrowRight className="w-4 h-4" />}
                  >
                    View All
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {loading && documents.length === 0 ? (
                <div className="space-y-4">
                  {Array.from({ length: 3 }).map((_, index) => (
                    <div key={index} className="p-4 border border-pearl-gray rounded-lg">
                      <Skeleton lines={3} />
                    </div>
                  ))}
                </div>
              ) : recentDocuments.length > 0 ? (
                <div className="space-y-4">
                  {recentDocuments.map((document) => (
                    <DocumentCard
                      key={document.id}
                      document={document}
                      onView={() => handleViewDocument(document)}
                      onChat={() => handleChatWithDocument(document)}
                      onDelete={() => deleteDocument(document.id)}
                      className="w-full"
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p className="text-lg font-medium mb-2">No documents yet</p>
                  <p className="text-sm mb-4">Upload your first legal document to get started</p>
                  <Button
                    variant="primary"
                    onClick={() => setShowUploadModal(true)}
                    icon={<Upload className="w-4 h-4" />}
                  >
                    Upload Document
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => setShowUploadModal(true)}
                  icon={<Upload className="w-4 h-4" />}
                >
                  Upload New Document
                </Button>

                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => navigate('/documents')}
                  icon={<FileText className="w-4 h-4" />}
                >
                  Browse All Documents
                </Button>

                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => navigate('/analytics')}
                  icon={<TrendingUp className="w-4 h-4" />}
                >
                  View Analytics
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Tips */}
          <Card>
            <CardHeader>
              <CardTitle>💡 Tips</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm text-gray-600">
                <div className="flex items-start space-x-2">
                  <span className="text-accent-blue">•</span>
                  <span>Upload clear, high-quality PDF documents for best analysis results</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-accent-blue">•</span>
                  <span>Use the AI chat assistant to ask specific questions about your documents</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-accent-blue">•</span>
                  <span>Pay special attention to documents marked as "High Risk" or "Urgent"</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-accent-blue">•</span>
                  <span>Consider legal consultation for complex agreements</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Upload Modal */}
      <Modal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        title="Upload Legal Document"
        description="Upload a PDF legal document for AI-powered analysis"
        size="lg"
      >
        <DocumentUpload
          onUpload={handleUpload}
          uploading={uploading}
          uploadProgress={uploadProgress}
        />
      </Modal>

      {/* Error Display */}
      {error && (
        <div className="fixed bottom-4 right-4 max-w-sm">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-red-600" />
                <span className="text-sm text-red-800">{error}</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearError}
                className="p-1"
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

export default Dashboard;