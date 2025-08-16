import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, FileText, AlertCircle } from 'lucide-react';
import DocumentList from '../components/documents/DocumentList';
import DocumentUpload from '../components/documents/DocumentUpload';
import Button from '../components/common/Button';
import Modal from '../components/common/Modal';
import { useDocuments } from '../hooks/useDocuments';
import { useChat } from '../hooks/useChat';
import { cn } from '../utils/cn';

const Documents: React.FC = () => {
  const navigate = useNavigate();
  const [showUploadModal, setShowUploadModal] = useState(false);

  const {
    documents,
    loading,
    error,
    uploading,
    uploadProgress,
    totalCount,
    uploadDocument,
    deleteDocument,
    clearError
  } = useDocuments();

  const { openChat } = useChat();

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
    openChat(document.id, document.filename);
  };

  const handleDeleteDocument = async (documentId: string) => {
    return await deleteDocument(documentId);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-black">Documents</h1>
          <p className="text-gray-600 mt-1">
            Manage and analyze your legal documents
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

      {/* Summary Stats */}
      {totalCount > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg border border-pearl-gray p-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-accent-blue" />
              </div>
              <div>
                <div className="text-2xl font-bold text-black">{totalCount}</div>
                <div className="text-sm text-gray-600">Total Documents</div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-pearl-gray p-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-success-green" />
              </div>
              <div>
                <div className="text-2xl font-bold text-black">
                  {documents.filter(d => d.risk_score <= 0.3).length}
                </div>
                <div className="text-sm text-gray-600">Low Risk</div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-pearl-gray p-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-danger-red" />
              </div>
              <div>
                <div className="text-2xl font-bold text-black">
                  {documents.filter(d => d.risk_score > 0.6).length}
                </div>
                <div className="text-sm text-gray-600">High Risk</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Document List */}
      <DocumentList
        documents={documents}
        loading={loading}
        onView={handleViewDocument}
        onChat={handleChatWithDocument}
        onDelete={handleDeleteDocument}
      />

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

export default Documents;