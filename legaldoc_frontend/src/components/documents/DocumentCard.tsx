import React from 'react';
import { FileText, Clock, AlertTriangle, Trash2 } from 'lucide-react';
import Card, { CardContent, CardFooter } from '../common/Card';
import Button from '../common/Button';
import { RiskBadge, LawyerUrgencyBadge } from '../common/Badge';
import { cn } from '../../utils/cn';
import {
  formatTimeAgo,
  getDocumentTypeLabel,
  truncateText,
  formatConfidenceScore
} from '../../utils/helpers';
import type { DocumentListItem } from '../../types/document';

export interface DocumentCardProps {
  document: DocumentListItem;
  onView?: () => void;
  onChat?: () => void;
  onDelete?: () => void;
  className?: string;
}

const DocumentCard: React.FC<DocumentCardProps> = ({
  document,
  onView,
  onDelete,
  className
}) => {
  return (
    <Card
      hover={!!onView}
      className={cn('transition-all duration-200', className)}
      onClick={onView}
    >
      <CardContent className="space-y-4">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3 flex-1 min-w-0">
            <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-gray-600" />
            </div>

            <div className="flex-1 min-w-0">
              <h3 className="font-medium text-black truncate" title={document.filename}>
                {document.filename}
              </h3>
              <p className="text-xs text-gray-500 mt-1">
                {getDocumentTypeLabel(document.document_type)}
              </p>
            </div>
          </div>

          <div className="flex-shrink-0 ml-2">
            <RiskBadge riskScore={document.risk_score} size="sm" />
          </div>
        </div>

        {/* Summary */}
        <div className="space-y-2">
          <p className="text-sm text-gray-700 line-clamp-2">
            {truncateText(document.summary, 120)}
          </p>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div className="space-y-1">
            <div className="flex items-center text-gray-500">
              <AlertTriangle className="w-3 h-3 mr-1" />
              <span>Risk Score</span>
            </div>
            <div className="font-medium text-black">
              {Math.round(document.risk_score * 100)}%
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center text-gray-500">
              <Clock className="w-3 h-3 mr-1" />
              <span>Reading Time</span>
            </div>
            <div className="font-medium text-black">
              {document.reading_time}s
            </div>
          </div>
        </div>

        {/* Legal Analysis Info */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">AI Confidence:</span>
            <span className="font-medium text-black">
              {formatConfidenceScore(document.confidence_score)}
            </span>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Legal Terms:</span>
            <span className="font-medium text-black">
              {document.legal_terms_count} detected
            </span>
          </div>

          {document.risk_indicators_count > 0 && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-500">Risk Indicators:</span>
              <span className="font-medium text-orange-600">
                {document.risk_indicators_count} found
              </span>
            </div>
          )}
        </div>

        {/* Lawyer Recommendation */}
        <div className="pt-2 border-t border-pearl-gray">
          <LawyerUrgencyBadge
            urgency={document.lawyer_urgency as any}
            size="sm"
          />
        </div>
      </CardContent>

      <CardFooter className="flex items-center justify-between">
        <div className="text-xs text-gray-500">
          {formatTimeAgo(document.created_at)}
        </div>

        <div className="flex items-center space-x-2">
          {onDelete && (
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              className="p-2 text-danger-red hover:text-red-600"
              title="Delete document"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
};

export default DocumentCard;