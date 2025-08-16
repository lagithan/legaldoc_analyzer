import React from 'react';
import {
  FileText,
  AlertTriangle,
  Brain,
  Shield,
  TrendingUp,
  BarChart3
} from 'lucide-react';
import StatsCard, { MiniStatsCard } from './StatsCard';
import RiskChart, { DonutChart } from './RiskChart';
import Card, { CardHeader, CardTitle, CardContent } from '../common/Card';
import LoadingSpinner, { Skeleton } from '../common/LoadingSpinner';
import { cn } from '../../utils/cn';
import { formatPercentage } from '../../utils/helpers';
import type { AnalyticsResponse, LegalModelsStatus } from '../../types/document';

export interface AnalyticsDashboardProps {
  analytics: AnalyticsResponse | null;
  legalModelsStatus: LegalModelsStatus | null;
  loading?: boolean;
  className?: string;
}

const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({
  analytics,
  legalModelsStatus,
  loading = false,
  className
}) => {
  if (loading && !analytics) {
    return (
      <div className={cn('space-y-6', className)}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="p-6 border border-pearl-gray rounded-xl">
              <Skeleton lines={3} />
            </div>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {Array.from({ length: 2 }).map((_, index) => (
            <div key={index} className="p-6 border border-pearl-gray rounded-xl">
              <Skeleton lines={5} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className={cn('text-center py-12', className)}>
        <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Analytics Data</h3>
        <p className="text-gray-600">Upload some documents to see analytics and insights.</p>
      </div>
    );
  }

  // Check if analytics exists but has no documents
  if (analytics && analytics.total_documents === 0) {
    return (
      <div className={cn('space-y-6', className)}>
        {/* Empty State for Zero Documents */}
        <div className="text-center py-12">
          <div className="mx-auto w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mb-6">
            <BarChart3 className="w-12 h-12 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No Documents Yet</h3>
          <p className="text-gray-600 mb-6 max-w-md mx-auto">
            Upload your first legal document to start generating analytics and insights about your document portfolio.
          </p>
          <div className="flex justify-center">
            <button
              onClick={() => window.location.href = '/documents'}
              className="btn-primary"
            >
              Upload First Document
            </button>
          </div>
        </div>

        {/* System Status - Still show this even with no documents */}
        {legalModelsStatus && (
          <div className="bg-white rounded-xl border border-pearl-gray p-6">
            <h3 className="text-lg font-semibold text-black mb-4 flex items-center">
              <div className="w-8 h-8 bg-success-green rounded-full flex items-center justify-center mr-3">
                <div className="w-3 h-3 bg-white rounded-full"></div>
              </div>
              AI Legal System Ready
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-black">
                  {legalModelsStatus.total_models}
                </div>
                <div className="text-sm text-gray-600">Legal Models Loaded</div>
              </div>

              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-black">
                  {legalModelsStatus.device}
                </div>
                <div className="text-sm text-gray-600">Processing Device</div>
              </div>

              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className={cn(
                  'text-2xl font-bold',
                  legalModelsStatus.status === 'operational' ? 'text-success-green' : 'text-warning-yellow'
                )}>
                  {legalModelsStatus.status}
                </div>
                <div className="text-sm text-gray-600">System Status</div>
              </div>
            </div>
            <p className="text-sm text-gray-600 mt-4 text-center">
              Ready to analyze legal documents with AI-powered Legal-BERT and Gemini models
            </p>
          </div>
        )}
      </div>
    );
  }

  // Calculate risk level percentages
  const totalDocs = analytics.total_documents;
  const riskPercentages = Object.entries(analytics.risk_distribution).reduce((acc, [key, value]) => {
    acc[key] = totalDocs > 0 ? (value / totalDocs) * 100 : 0;
    return acc;
  }, {} as Record<string, number>);

  // Get most common document type
  const mostCommonType = Object.entries(analytics.document_types).reduce((a, b) =>
    analytics.document_types[a[0]] > analytics.document_types[b[0]] ? a : b
  );

  return (
    <div className={cn('space-y-6', className)}>
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Documents"
          value={analytics.total_documents}
          description="Legal documents analyzed"
          icon={FileText}
          color="info"
        />

        <StatsCard
          title="Avg Confidence"
          value={`${Math.round(analytics.avg_confidence * 100)}%`}
          description="AI analysis confidence"
          icon={Brain}
          color="success"
        />

        <StatsCard
          title="Require Lawyer"
          value={analytics.total_requiring_lawyer}
          description={`${Math.round((analytics.total_requiring_lawyer / Math.max(totalDocs, 1)) * 100)}% of documents`}
          icon={AlertTriangle}
          color="warning"
        />

        <StatsCard
          title="Legal Models"
          value={legalModelsStatus?.total_models || 0}
          description={`Status: ${legalModelsStatus?.status || 'Unknown'}`}
          icon={Shield}
          color={legalModelsStatus?.status === 'operational' ? 'success' : 'warning'}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5" />
              <span>Risk Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <DonutChart
              data={analytics.risk_distribution}
              size={200}
            />
          </CardContent>
        </Card>

        {/* Document Types */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="w-5 h-5" />
              <span>Document Types</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RiskChart
              data={analytics.document_types}
              title=""
            />
          </CardContent>
        </Card>
      </div>

      {/* Detailed Risk Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>Risk Analysis Details</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(analytics.risk_distribution).map(([level, count]) => {
              const percentage = riskPercentages[level];
              let color: 'success' | 'warning' | 'danger' = 'success';
              let description = '';

              switch (level) {
                case 'low':
                  color = 'success';
                  description = 'Low risk documents';
                  break;
                case 'medium':
                  color = 'warning';
                  description = 'Medium risk documents';
                  break;
                case 'high':
                case 'urgent':
                  color = 'danger';
                  description = 'High risk documents';
                  break;
              }

              return (
                <MiniStatsCard
                  key={level}
                  label={description}
                  value={`${count} (${Math.round(percentage)}%)`}
                  color={color}
                />
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* System Status */}
      {legalModelsStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="w-5 h-5" />
              <span>Legal AI System Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-black">
                    {legalModelsStatus.total_models}
                  </div>
                  <div className="text-sm text-gray-600">Models Loaded</div>
                </div>

                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-black">
                    {legalModelsStatus.device}
                  </div>
                  <div className="text-sm text-gray-600">Device</div>
                </div>

                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className={cn(
                    'text-2xl font-bold',
                    legalModelsStatus.status === 'operational' ? 'text-success-green' : 'text-warning-yellow'
                  )}>
                    {legalModelsStatus.status}
                  </div>
                  <div className="text-sm text-gray-600">Status</div>
                </div>
              </div>

              {/* Available Models */}
              <div className="space-y-2">
                <h4 className="font-medium text-black">Available Legal Models:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {Object.entries(legalModelsStatus.available_models).map(([key, model]) => (
                    <div key={key} className="p-3 bg-gray-50 rounded-lg">
                      <div className="font-medium text-sm text-black">{model.description}</div>
                      <div className="text-xs text-gray-600 mt-1">
                        {model.size} • Speed: {model.speed}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {legalModelsStatus.loaded_models.includes(key) ? '✅ Loaded' : '⏳ Available'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Insights */}
      <Card>
        <CardHeader>
          <CardTitle>Key Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {totalDocs > 0 && (
              <>
                <div className="flex items-center space-x-2 text-sm">
                  <span className="w-2 h-2 bg-accent-blue rounded-full"></span>
                  <span>
                    Most common document type: <strong>{mostCommonType[0].replace('_', ' ')}</strong> ({mostCommonType[1]} documents)
                  </span>
                </div>

                <div className="flex items-center space-x-2 text-sm">
                  <span className="w-2 h-2 bg-success-green rounded-full"></span>
                  <span>
                    Average AI confidence: <strong>{formatPercentage(analytics.avg_confidence)}</strong>
                  </span>
                </div>

                <div className="flex items-center space-x-2 text-sm">
                  <span className="w-2 h-2 bg-warning-yellow rounded-full"></span>
                  <span>
                    <strong>{formatPercentage(analytics.total_requiring_lawyer / totalDocs)}</strong> of documents recommend legal consultation
                  </span>
                </div>
              </>
            )}

            {legalModelsStatus && (
              <div className="flex items-center space-x-2 text-sm">
                <span className={cn(
                  'w-2 h-2 rounded-full',
                  legalModelsStatus.status === 'operational' ? 'bg-success-green' : 'bg-warning-yellow'
                )}></span>
                <span>
                  Legal AI system is <strong>{legalModelsStatus.status}</strong> with {legalModelsStatus.total_models} models ready
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AnalyticsDashboard;