import React, { useState } from 'react';
import {
  FileText,
  Clock,
  Brain,
  AlertTriangle,
  CheckCircle,
  MessageSquare,
  Download,
  Share,
  ArrowLeft
} from 'lucide-react';
import Card, { CardHeader, CardTitle, CardContent } from '../common/Card';
import Button from '../common/Button';
import { RiskBadge, LawyerUrgencyBadge } from '../common/Badge';
import RiskMeter from './RiskMeter';
import { cn } from '../../utils/cn';
import {
  formatDate,
  getDocumentTypeLabel,
  formatConfidenceScore,
  formatPercentage
} from '../../utils/helpers';
import type { DocumentResponse } from '../../types/document';
import ChatAssistant from '../chat/ChatAssistant';

export interface DocumentDetailsProps {
  document: DocumentResponse;
  onBack?: () => void;
  onChat?: () => void;
  onExport?: () => void;
  className?: string;
}

const DocumentDetails: React.FC<DocumentDetailsProps> = ({
  document,
  onBack,
  onChat,
  onExport,
  className
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis' | 'risks'>('overview');
  const [exportLoading, setExportLoading] = useState(false);

  const handleExport = async () => {
    if (!onExport) return;
    setExportLoading(true);
    try {
      await onExport();
    } catch (error) {
      console.error('Export error in component:', error);
    } finally {
      setExportLoading(false);
    }
  };

  const handleChatOpen = () => {
    console.log('Chat button clicked for document:', document.filename);
    if (onChat) {
      onChat();
    } else {
      console.error('onChat handler not provided');
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: FileText },
    { id: 'analysis', label: 'AI Analysis', icon: Brain },
    { id: 'risks', label: 'Risk Assessment', icon: AlertTriangle }
  ];

  return (
    <div className={cn('space-y-6', className)}>
      {/* Header */}
       <ChatAssistant onchat={handleChatOpen} />
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {onBack && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onBack}
              icon={<ArrowLeft className="w-4 h-4" />}
            />
          )}

          <div>
            <h1 className="text-2xl font-bold text-black">
              {document.filename}
            </h1>
            <p className="text-green-600 mt-1  text-lg">
              Document Type - {getDocumentTypeLabel(document.document_type)}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <Button
            variant="primary"
            onClick={onChat}
            icon={<MessageSquare className="w-4 h-4" />}
          >
            Ask Questions
          </Button>

          <Button
            variant="secondary"
            onClick={handleExport}
            loading={exportLoading}
            disabled={!onExport}
            icon={<Download className="w-4 h-4" />}
          >
            Export Report
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={async () => {
              const shareData = {
                title: `Legal Analysis: ${document.filename}`,
                text: `AI-powered analysis of ${document.filename} - Risk Score: ${Math.round(document.risk_score * 100)}%`,
                url: window.location.href
              };

              try {
                if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
                  await navigator.share(shareData);
                } else {
                  // Fallback to clipboard
                  await navigator.clipboard.writeText(window.location.href);

                  // Show a temporary notification
                  const button = event?.currentTarget as HTMLButtonElement;
                  const originalText = button.textContent;
                  button.textContent = 'Link copied!';
                  setTimeout(() => {
                    if (button) button.textContent = originalText;
                  }, 2000);
                }
              } catch (error) {
                console.error('Share failed:', error);
                // Final fallback - just copy to clipboard
                try {
                  await navigator.clipboard.writeText(window.location.href);
                  alert('Document link copied to clipboard!');
                } catch (clipboardError) {
                  console.error('Clipboard fallback failed:', clipboardError);
                  alert('Unable to share or copy link. Please copy the URL manually.');
                }
              }
            }}
            icon={<Share className="w-4 h-4" />}
            title="Share document"
          />
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card padding="sm">
          <div className="text-center">
            <div className="text-2xl font-bold text-black">
              {Math.round(document.risk_score * 100)}%
            </div>
            <div className="text-sm text-gray-600 mt-1">Risk Score</div>
            <div className="mt-2">
              <RiskBadge riskScore={document.risk_score} size="sm" />
            </div>
          </div>
        </Card>

        <Card padding="sm">
          <div className="text-center">
            <div className="text-2xl font-bold text-black">
              {Math.round(document.confidence_score * 100)}%
            </div>
            <div className="text-sm text-gray-600 mt-1">AI Confidence</div>
            <div className="mt-2 flex justify-center">
              <CheckCircle className="w-5 h-5 text-success-green" />
            </div>
          </div>
        </Card>

        <Card padding="sm">
          <div className="text-center">
            <div className="text-2xl font-bold text-black">
              {document.estimated_reading_time}s
            </div>
            <div className="text-sm text-gray-600 mt-1">Reading Time</div>
            <div className="mt-2 flex justify-center">
              <Clock className="w-5 h-5 text-gray-400" />
            </div>
          </div>
        </Card>

        <Card padding="sm">
          <div className="text-center">
            <div className="text-2xl font-bold text-black">
              Lawyer Review
            </div>
            <div className="mt-2">
              <LawyerUrgencyBadge urgency={document.lawyer_urgency} size="sm" />
            </div>
          </div>
        </Card>
      </div>

      {/* Tabs */}
      <div className="border-b border-pearl-gray">
        <nav className="flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  'flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors',
                  activeTab === tab.id
                    ? 'border-black text-black'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                )}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <>
            {/* Summary */}
            <Card>
              <CardHeader>
                <CardTitle>Executive Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 leading-relaxed">
                  {document.summary}
                </p>
              </CardContent>
            </Card>

            {/* Key Clauses */}
            <Card>
              <CardHeader>
                <CardTitle>Key Terms & Obligations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {document.key_clauses.map((clause, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                      <div className="flex-shrink-0 w-6 h-6 bg-black text-white rounded-full flex items-center justify-center text-xs font-medium">
                        {index + 1}
                      </div>
                      <p className="text-gray-700 flex-1">{clause}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Red Flags */}
            {document.red_flags.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <AlertTriangle className="w-5 h-5 text-danger-red" />
                    <span>Red Flags & Concerns</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {document.red_flags.map((flag, index) => (
                      <div key={index} className="flex items-start space-x-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                        <AlertTriangle className="w-5 h-5 text-danger-red flex-shrink-0 mt-0.5" />
                        <p className="text-red-800 flex-1">{flag}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {activeTab === 'analysis' && (
          <>
            {/* AI Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>AI Confidence & Reasoning</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                    <div>
                      <div className="font-medium text-black">
                        {formatConfidenceScore(document.confidence_score)}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        Analysis Confidence
                      </div>
                    </div>
                    <Brain className="w-8 h-8 text-accent-blue" />
                  </div>

                  
                </div>
              </CardContent>
            </Card>

            {/* Legal Model Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>Legal Model Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-black mb-3">Legal Terms Detected</h4>
                    <div className="space-y-2">
                      {document.legal_terminology_found.slice(0, 10).map((term, index) => (
                        <div key={index} className="inline-block bg-gray-100 px-2 py-1 rounded text-sm text-gray-700 mr-2 mb-2">
                          {term}
                        </div>
                      ))}
                      {document.legal_terminology_found.length > 10 && (
                        <div className="text-sm text-gray-500">
                          +{document.legal_terminology_found.length - 10} more terms
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              </CardContent>
            </Card>
          </>
        )}

        {activeTab === 'risks' && (
          <>
            {/* Risk Meter */}
            <Card>
              <CardHeader>
                <CardTitle>Risk Assessment</CardTitle>
              </CardHeader>
              <CardContent>
                <RiskMeter
                  riskScore={document.risk_score}
                  riskBreakdown={document.risk_breakdown}
                  showBreakdown={true}
                  size="lg"
                />
              </CardContent>
            </Card>

            {/* Risk Indicators */}
            {document.risk_indicators.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Risk Indicators Found</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {document.risk_indicators.map((indicator, index) => (
                      <div key={index} className="flex items-center space-x-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
                        <AlertTriangle className="w-5 h-5 text-orange-600 flex-shrink-0" />
                        <span className="text-orange-800 capitalize">{indicator.replace('_', ' ')}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Urgency Signals */}
            {document.urgency_signals.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Urgency Signals</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {document.urgency_signals.map((signal, index) => (
                      <div key={index} className="flex items-center space-x-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                        <div className="text-lg">🚨</div>
                        <span className="text-red-800 capitalize">{signal.replace('_', ' ')}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>

      {/* Suggested Questions */}
      <Card>
        <CardHeader>
          <CardTitle>Suggested Questions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {document.suggested_questions.map((question, index) => (
              <button
                key={index}
                onClick={() => {
                  console.log('Suggested question clicked:', question);
                  // First open the chat
                  handleChatOpen();
                  // Then send the question after a small delay to ensure chat is open
                  setTimeout(() => {
                    console.log('Sending chat message event for:', question);
                    window.dispatchEvent(new CustomEvent('sendChatMessage', {
                      detail: { message: question }
                    }));
                  }, 500);
                }}
                className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <div className="flex items-center space-x-2">
                  <MessageSquare className="w-4 h-4 text-accent-blue flex-shrink-0" />
                  <span className="text-sm text-gray-700">{question}</span>
                </div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DocumentDetails;