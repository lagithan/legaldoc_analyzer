import React, { useState, useMemo } from 'react';
import { Search, SortAsc, SortDesc, Grid, List } from 'lucide-react';
import DocumentCard from './DocumentCard';
import Button from '../common/Button';
import LoadingSpinner, { Skeleton } from '../common/LoadingSpinner';
import { ConfirmModal } from '../common/Modal';
import { cn } from '../../utils/cn';
import { debounce } from '../../utils/helpers';
import type { DocumentListItem } from '../../types/document';

export interface DocumentListProps {
  documents: DocumentListItem[];
  loading?: boolean;
  onView?: (document: DocumentListItem) => void;
  onChat?: (document: DocumentListItem) => void;
  onDelete?: (documentId: string) => Promise<boolean>;
  className?: string;
}

type SortField = 'created_at' | 'filename' | 'risk_score' | 'confidence_score';
type SortOrder = 'asc' | 'desc';
type ViewMode = 'grid' | 'list';

const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  loading = false,
  onView,
  onChat,
  onDelete,
  className
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<SortField>('created_at');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [filterRisk, setFilterRisk] = useState<string>('all');
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Debounced search
  const debouncedSearch = useMemo(
    () => debounce((term: string) => setSearchTerm(term), 300),
    []
  );

  // Filtered and sorted documents
  const filteredDocuments = useMemo(() => {
    let filtered = documents;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(doc =>
        doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
        doc.summary.toLowerCase().includes(searchTerm.toLowerCase()) ||
        doc.document_type.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Risk filter
    if (filterRisk !== 'all') {
      filtered = filtered.filter(doc => {
        const riskLevel = doc.risk_score <= 0.3 ? 'low' :
                         doc.risk_score <= 0.6 ? 'medium' : 'high';
        return riskLevel === filterRisk;
      });
    }

    // Sorting
    filtered.sort((a, b) => {
      let aValue: any = a[sortField];
      let bValue: any = b[sortField];

      if (sortField === 'created_at') {
        aValue = new Date(aValue).getTime();
        bValue = new Date(bValue).getTime();
      }

      if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [documents, searchTerm, sortField, sortOrder, filterRisk]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const handleDelete = async (documentId: string) => {
    setDocumentToDelete(documentId);
    setDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!documentToDelete || !onDelete) return;

    setDeleting(true);
    try {
      const success = await onDelete(documentToDelete);
      if (success) {
        setDeleteModalOpen(false);
        setDocumentToDelete(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    } finally {
      setDeleting(false);
    }
  };

  if (loading && documents.length === 0) {
    return (
      <div className={cn('space-y-6', className)}>
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="space-y-4 p-6 border border-pearl-gray rounded-xl">
              <Skeleton lines={3} />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-8 w-24" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('space-y-6', className)}>
      {/* Header and Controls */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-black">
            Documents ({filteredDocuments.length})
          </h2>

          <div className="flex items-center space-x-2">
            <Button
              variant={viewMode === 'grid' ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setViewMode('grid')}
              icon={<Grid className="w-4 h-4" />}
            />
            <Button
              variant={viewMode === 'list' ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setViewMode('list')}
              icon={<List className="w-4 h-4" />}
            />
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center space-x-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search documents..."
              className="input-field pl-10"
              onChange={(e) => debouncedSearch(e.target.value)}
            />
          </div>

          <select
            value={filterRisk}
            onChange={(e) => setFilterRisk(e.target.value)}
            className="input-field w-40"
          >
            <option value="all">All Risk Levels</option>
            <option value="low">Low Risk</option>
            <option value="medium">Medium Risk</option>
            <option value="high">High Risk</option>
          </select>

          <Button
            variant="secondary"
            size="sm"
            onClick={() => handleSort('created_at')}
            icon={sortField === 'created_at' && sortOrder === 'desc' ? <SortDesc className="w-4 h-4" /> : <SortAsc className="w-4 h-4" />}
          >
            Date
          </Button>

          <Button
            variant="secondary"
            size="sm"
            onClick={() => handleSort('risk_score')}
            icon={sortField === 'risk_score' && sortOrder === 'desc' ? <SortDesc className="w-4 h-4" /> : <SortAsc className="w-4 h-4" />}
          >
            Risk
          </Button>
        </div>
      </div>

      {/* Document Grid/List */}
      {filteredDocuments.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg mb-2">No documents found</div>
          <p className="text-gray-500">
            {searchTerm || filterRisk !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Upload your first legal document to get started'
            }
          </p>
        </div>
      ) : (
        <div className={cn(
          viewMode === 'grid'
            ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6'
            : 'space-y-4'
        )}>
          {filteredDocuments.map((document) => (
            <DocumentCard
              key={document.id}
              document={document}
              onView={() => onView?.(document)}
              onChat={() => onChat?.(document)}
              onDelete={() => handleDelete(document.id)}
              className={viewMode === 'list' ? 'max-w-none' : ''}
            />
          ))}
        </div>
      )}

      {/* Loading indicator for additional items */}
      {loading && documents.length > 0 && (
        <div className="flex justify-center py-8">
          <LoadingSpinner text="Loading more documents..." />
        </div>
      )}

      {/* Delete Confirmation Modal */}
      <ConfirmModal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        onConfirm={confirmDelete}
        title="Delete Document"
        message="Are you sure you want to delete this document? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        loading={deleting}
      />
    </div>
  );
};

export default DocumentList;