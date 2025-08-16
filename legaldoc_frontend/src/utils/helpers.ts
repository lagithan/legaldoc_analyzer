import { RISK_LEVELS, DOCUMENT_TYPES, LAWYER_URGENCY } from './constants';
import type { DocumentType } from '../types/document';

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch {
    return 'Unknown date';
  }
};

export const formatTimeAgo = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMs = now.getTime() - date.getTime();
    const diffInMinutes = Math.floor(diffInMs / (1000 * 60));
    const diffInHours = Math.floor(diffInMinutes / 60);
    const diffInDays = Math.floor(diffInHours / 24);

    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInHours < 24) return `${diffInHours}h ago`;
    if (diffInDays < 7) return `${diffInDays}d ago`;

    return formatDate(dateString);
  } catch {
    return 'Unknown time';
  }
};

export const getRiskLevel = (score: number): keyof typeof RISK_LEVELS => {
  if (score <= RISK_LEVELS.low.threshold) return 'low';
  if (score <= RISK_LEVELS.medium.threshold) return 'medium';
  if (score <= RISK_LEVELS.high.threshold) return 'high';
  return 'urgent';
};

export const getRiskColor = (score: number): string => {
  const level = getRiskLevel(score);
  return RISK_LEVELS[level].color;
};

export const getRiskBadgeClasses = (score: number): string => {
  const level = getRiskLevel(score);
  const config = RISK_LEVELS[level];
  return `${config.color} ${config.bgColor} ${config.borderColor} border px-2 py-1 rounded-full text-xs font-medium`;
};

export const getDocumentTypeLabel = (type: DocumentType): string => {
  return DOCUMENT_TYPES[type] || 'Unknown Document';
};

export const getLawyerUrgencyConfig = (urgency: string) => {
  return LAWYER_URGENCY[urgency as keyof typeof LAWYER_URGENCY] || LAWYER_URGENCY.medium;
};

export const formatPercentage = (value: number): string => {
  return `${Math.round(value * 100)}%`;
};

export const formatConfidenceScore = (score: number): string => {
  const percentage = Math.round(score * 100);
  if (percentage >= 85) return `${percentage}% (High Confidence)`;
  if (percentage >= 65) return `${percentage}% (Medium Confidence)`;
  return `${percentage}% (Low Confidence)`;
};

export const truncateText = (text: string, maxLength: number = 150): string => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trim() + '...';
};

export const capitalizeFirst = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};

export const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

export const downloadTextAsFile = (content: string, filename: string): void => {
  try {
    console.log('Starting file download:', { filename, contentLength: content.length });

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    console.log('Blob created:', { size: blob.size, type: blob.type });

    const url = URL.createObjectURL(blob);
    console.log('Object URL created:', url);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';

    document.body.appendChild(a);
    console.log('Download link added to DOM');

    a.click();
    console.log('Download triggered');

    // Cleanup
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log('Download cleanup completed');
  } catch (error) {
    console.error('Download failed:', error);
    // Fallback - try to open in new window
    try {
      const dataUri = 'data:text/plain;charset=utf-8,' + encodeURIComponent(content);
      const newWindow = window.open(dataUri, '_blank');
      if (!newWindow) {
        alert('Download failed. Please allow popups and try again.');
      }
    } catch (fallbackError) {
      console.error('Fallback download also failed:', fallbackError);
      alert('Download failed. Please try again or contact support.');
    }
  }
};

export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(null, args), delay);
  };
};

export const getInitials = (name: string): string => {
  return name
    .split(' ')
    .map(word => word.charAt(0).toUpperCase())
    .join('')
    .slice(0, 2);
};

export const validateFile = (file: File): { isValid: boolean; error?: string } => {
  // Check file type
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    return { isValid: false, error: 'Only PDF files are supported' };
  }

  // Check file size (25MB limit)
  if (file.size > 25 * 1024 * 1024) {
    return { isValid: false, error: 'File size must be less than 25MB' };
  }

  return { isValid: true };
};

export const calculateOverallRisk = (riskBreakdown: Record<string, number>): number => {
  const weights = {
    financial_risk: 0.25,
    termination_risk: 0.20,
    liability_risk: 0.25,
    renewal_risk: 0.15,
    modification_risk: 0.15
  };

  let totalRisk = 0;
  Object.entries(weights).forEach(([key, weight]) => {
    totalRisk += (riskBreakdown[key] || 0) * weight;
  });

  return Math.min(totalRisk, 1.0);
};