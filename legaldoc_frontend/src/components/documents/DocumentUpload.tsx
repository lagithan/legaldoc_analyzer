import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import Card from '../common/Card';
import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';
import { cn } from '../../utils/cn';
import { validateFile, formatFileSize } from '../../utils/helpers';
import { FILE_UPLOAD } from '../../utils/constants';

export interface DocumentUploadProps {
  onUpload: (file: File) => Promise<void>;
  uploading?: boolean;
  uploadProgress?: number;
  className?: string;
}


const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUpload,
  uploading = false,
  uploadProgress = 0,
  className
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validationWarning, setValidationWarning] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setValidationError(null);
    setValidationWarning(null);

    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];

    // Basic file validation
    const validation = validateFile(file);
    if (!validation.isValid) {
      setValidationError(validation.error || 'Invalid file');
      return;
    }


    try {
      await onUpload(file);
    } catch (error: any) {
      setValidationError(error.message || 'Upload failed');
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    maxSize: FILE_UPLOAD.MAX_SIZE,
    disabled: uploading,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    onDropAccepted: () => setDragActive(false),
    onDropRejected: (fileRejections) => {
      setDragActive(false);
      if (fileRejections.length > 0) {
        const rejection = fileRejections[0];
        if (rejection.errors.length > 0) {
          setValidationError(rejection.errors[0].message);
        }
      }
    }
  });

  return (
    <Card
      className={cn('border-2 border-dashed transition-all duration-200', className)}
    >
      <div
        {...getRootProps()}
        className={cn(
          'text-center py-12 px-6 cursor-pointer transition-colors duration-200',
          isDragActive || dragActive
            ? 'border-accent-blue bg-blue-50'
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50',
          uploading && 'cursor-not-allowed opacity-50'
        )}
      >
        <input {...getInputProps()} />

        {uploading ? (
          <div className="space-y-4">
            <LoadingSpinner size="lg" />
            <div className="space-y-2">
              <p className="text-sm text-gray-600">Uploading and analyzing legal document...</p>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-accent-blue h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-xs text-gray-500">{uploadProgress}% complete</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
              {isDragActive || dragActive ? (
                <Upload className="w-8 h-8 text-accent-blue" />
              ) : (
                <FileText className="w-8 h-8 text-gray-400" />
              )}
            </div>

            <div className="space-y-2">
              <h3 className="text-lg font-medium text-black">
                {isDragActive || dragActive
                  ? 'Drop your legal document here'
                  : 'Upload Legal Document Only'
                }
              </h3>
              <p className="text-sm text-gray-600">
                Upload legal documents such as contracts, agreements, leases, NDAs, terms of service, etc.
              </p>
              <p className="text-xs text-gray-500">
                Supports PDF files up to {formatFileSize(FILE_UPLOAD.MAX_SIZE)}
              </p>
            </div>

            <Button variant="primary" size="lg" className="mt-4">
              <Upload className="w-4 h-4 mr-2" />
              Select Legal Document
            </Button>
          </div>
        )}

        {validationError && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2 text-red-700">
              <XCircle className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm">{validationError}</span>
            </div>
          </div>
        )}

        {validationWarning && !validationError && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center space-x-2 text-yellow-700">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm">{validationWarning}</span>
            </div>
          </div>
        )}
      </div>

      {/* Legal Document Requirements */}
      <div className="border-t border-pearl-gray p-4 bg-gray-50">
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-success-green" />
              <span>Legal Documents Only</span>
            </div>
            <div className="flex items-center space-x-4">
              <span>Max size: {formatFileSize(FILE_UPLOAD.MAX_SIZE)}</span>
              <span>AI-powered validation</span>
            </div>
          </div>

          <div className="text-xs text-gray-600">
            <div className="font-medium mb-1">✅ Accepted Documents:</div>
            <div className="grid grid-cols-2 gap-1 text-xs">
              <span>• Contracts & Agreements</span>
              <span>• Employment Contracts</span>
              <span>• Lease Agreements</span>
              <span>• Service Agreements</span>
              <span>• Non-Disclosure Agreements</span>
              <span>• Terms of Service</span>
              <span>• Purchase Agreements</span>
              <span>• Legal Policies</span>
            </div>
          </div>

          <div className="text-xs text-gray-600">
            <div className="font-medium mb-1 text-red-600">❌ Not Accepted:</div>
            <div className="text-red-600">
              Personal documents, resumes, reports, manuals, books, articles, invoices, or any non-legal content
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default DocumentUpload;