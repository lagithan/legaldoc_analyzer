import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AlertCircle, Download } from 'lucide-react';
import DocumentDetails from '../components/documents/DocumentDetails';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useDocuments } from '../hooks/useDocuments';
import { useChat } from '../hooks/useChat';
import { documentService } from '../services/documentService';
import { downloadTextAsFile } from '../utils/helpers';

const DocumentView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [exportLoading, setExportLoading] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportSuccess, setExportSuccess] = useState(false);

  const {
    currentDocument,
    loading,
    error,
    fetchDocument,
    clearError,
    clearCurrentDocument
  } = useDocuments();

  const { openChat, forceShowChat } = useChat();

  // Fetch document on mount
  useEffect(() => {
    if (id) {
      fetchDocument(id);
    }

    // Cleanup on unmount
    return () => {
      clearCurrentDocument();
    };
  }, [id, fetchDocument, clearCurrentDocument]);

  const handleBack = () => {
    navigate('/documents');
  };

  const handleChat = () => {
    if (currentDocument) {
      console.log('DocumentView: Opening chat for document:', {
        id: currentDocument.id,
        filename: currentDocument.filename
      });

      // First try to open the chat normally
      openChat(currentDocument.id, currentDocument.filename);

      // Immediately force show with explicit parameters (bypass state timing issues)
      console.log('DocumentView: Immediately forcing chat visibility');
      forceShowChat(currentDocument.id, currentDocument.filename);

      // Also force show via event as backup
      setTimeout(() => {
        console.log('DocumentView: Forcing chat to show via event with document info');
        window.dispatchEvent(new CustomEvent('forceShowChat', {
          detail: {
            documentId: currentDocument.id,
            documentName: currentDocument.filename
          }
        }));
      }, 100);

      console.log('DocumentView: Chat open commands sent');
    } else {
      console.error('DocumentView: No current document available for chat');
    }
  };

  const handleExport = async () => {
    if (!currentDocument) {
      console.error('No current document available for export');
      setExportError('No document available for export');
      return;
    }

    console.log('Starting export for document:', currentDocument.id);
    setExportLoading(true);
    setExportError(null);
    setExportSuccess(false);

    try {
      console.log('Calling export service...');
      const reportData = await documentService.exportReport(currentDocument.id);
      console.log('Export service returned:', reportData);

      // Download the report
      console.log('Starting download...');
      downloadTextAsFile(
        reportData.report,
        reportData.filename
      );

      console.log('Report exported and downloaded successfully');
      setExportSuccess(true);
      setTimeout(() => setExportSuccess(false), 3000); // Hide success message after 3 seconds
    } catch (error: any) {
      console.error('Export failed with error:', error);
      setExportError(error.message || 'Failed to export report. Please try again.');
    } finally {
      setExportLoading(false);
    }
  };

  // Loading state
  if (loading && !currentDocument) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <LoadingSpinner size="lg" text="Loading document..." />
      </div>
    );
  }

  // Error state
  if (error && !currentDocument) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="w-16 h-16 text-red-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Error Loading Document
        </h3>
        <p className="text-gray-600 mb-6">{error}</p>
        <button
          onClick={handleBack}
          className="btn-primary"
        >
          Back to Documents
        </button>
      </div>
    );
  }

  // Document not found
  if (!loading && !currentDocument) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Document Not Found
        </h3>
        <p className="text-gray-600 mb-6">
          The document you're looking for doesn't exist or has been deleted.
        </p>
        <button
          onClick={handleBack}
          className="btn-primary"
        >
          Back to Documents
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {currentDocument && (
        <DocumentDetails
          document={currentDocument}
          onBack={handleBack}
          onChat={handleChat}
          onExport={handleExport}
        />
      )}

      {/* Success/Error Display */}
      {(error || exportError || exportSuccess) && (
        <div className="fixed bottom-4 right-4 max-w-sm z-40">
          <div className={`border rounded-lg p-4 shadow-lg ${
            exportSuccess
              ? 'bg-green-50 border-green-200'
              : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {exportSuccess ? (
                  <>
                    <div className="w-4 h-4 text-green-600">✓</div>
                    <span className="text-sm text-green-800">Report exported successfully!</span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-4 h-4 text-red-600" />
                    <span className="text-sm text-red-800">{error || exportError}</span>
                  </>
                )}
              </div>
              <button
                onClick={() => {
                  clearError();
                  setExportError(null);
                  setExportSuccess(false);
                }}
                className={`p-1 hover:opacity-80 ${
                  exportSuccess ? 'text-green-600' : 'text-red-600'
                }`}
              >
                ×
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Loading Overlay */}
      {exportLoading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 shadow-lg">
            <LoadingSpinner size="lg" text="Generating report..." />
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentView;