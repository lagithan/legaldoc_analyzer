import React from 'react';
import { AlertTriangle, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import { cn } from '../../utils/cn';
import { formatPercentage } from '../../utils/helpers';
import { RISK_CATEGORIES } from '../../utils/constants';
import type { RiskBreakdown } from '../../types/document';

export interface RiskMeterProps {
  riskScore: number;
  riskBreakdown?: RiskBreakdown;
  showBreakdown?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const RiskMeter: React.FC<RiskMeterProps> = ({
  riskScore,
  riskBreakdown,
  showBreakdown = false,
  size = 'md',
  className
}) => {
  const getRiskLevel = (score: number) => {
    if (score <= 0.3) return { level: 'low', color: 'text-success-green', bgColor: 'bg-success-green' };
    if (score <= 0.6) return { level: 'medium', color: 'text-warning-yellow', bgColor: 'bg-warning-yellow' };
    if (score <= 0.8) return { level: 'high', color: 'text-orange-600', bgColor: 'bg-orange-600' };
    return { level: 'urgent', color: 'text-danger-red', bgColor: 'bg-danger-red' };
  };

  const getRiskIcon = (score: number) => {
    if (score <= 0.3) return CheckCircle;
    if (score <= 0.6) return AlertCircle;
    if (score <= 0.8) return AlertTriangle;
    return XCircle;
  };

  const getRiskLabel = (score: number) => {
    if (score <= 0.3) return 'Low Risk';
    if (score <= 0.6) return 'Medium Risk';
    if (score <= 0.8) return 'High Risk';
    return 'Urgent Review';
  };

  const risk = getRiskLevel(riskScore);
  const RiskIcon = getRiskIcon(riskScore);

  const sizes = {
    sm: {
      meter: 'h-2',
      container: 'w-24',
      icon: 'w-4 h-4',
      text: 'text-xs'
    },
    md: {
      meter: 'h-3',
      container: 'w-32',
      icon: 'w-5 h-5',
      text: 'text-sm'
    },
    lg: {
      meter: 'h-4',
      container: 'w-48',
      icon: 'w-6 h-6',
      text: 'text-base'
    }
  };

  const sizeConfig = sizes[size];

  return (
    <div className={cn('space-y-3', className)}>
      {/* Main Risk Meter */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <RiskIcon className={cn(sizeConfig.icon, risk.color)} />
            <span className={cn('font-medium', sizeConfig.text, risk.color)}>
              {getRiskLabel(riskScore)}
            </span>
          </div>
          <span className={cn('font-bold', sizeConfig.text, risk.color)}>
            {formatPercentage(riskScore)}
          </span>
        </div>

        {/* Progress Bar */}
        <div className={cn('bg-gray-200 rounded-full overflow-hidden', sizeConfig.meter)}>
          <div
            className={cn('h-full transition-all duration-500 ease-out', risk.bgColor)}
            style={{ width: `${Math.min(riskScore * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Risk Breakdown */}
      {showBreakdown && riskBreakdown && (
        <div className="space-y-3 pt-2 border-t border-pearl-gray">
          <h4 className={cn('font-medium text-gray-900', sizeConfig.text)}>
            Risk Breakdown
          </h4>

          <div className="space-y-2">
            {Object.entries(riskBreakdown).map(([category, score]) => {
              const categoryConfig = RISK_CATEGORIES[category as keyof typeof RISK_CATEGORIES];
              const categoryRisk = getRiskLevel(score);

              if (!categoryConfig) return null;

              return (
                <div key={category} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{categoryConfig.icon}</span>
                      <span className={cn('text-gray-700', sizeConfig.text)}>
                        {categoryConfig.label}
                      </span>
                    </div>
                    <span className={cn('font-medium', sizeConfig.text, categoryRisk.color)}>
                      {formatPercentage(score)}
                    </span>
                  </div>

                  {/* Mini progress bar */}
                  <div className={cn('bg-gray-200 rounded-full', size === 'lg' ? 'h-2' : 'h-1.5')}>
                    <div
                      className={cn('h-full rounded-full transition-all duration-300', categoryRisk.bgColor)}
                      style={{ width: `${Math.min(score * 100, 100)}%` }}
                    />
                  </div>

                  {size === 'lg' && (
                    <p className="text-xs text-gray-500 mt-1">
                      {categoryConfig.description}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

// Circular Risk Meter variant
export interface CircularRiskMeterProps {
  riskScore: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
}

export const CircularRiskMeter: React.FC<CircularRiskMeterProps> = ({
  riskScore,
  size = 120,
  strokeWidth = 8,
  className
}) => {
  const risk = React.useMemo(() => {
    if (riskScore <= 0.3) return { color: '#10B981', textColor: 'text-success-green' };
    if (riskScore <= 0.6) return { color: '#F59E0B', textColor: 'text-warning-yellow' };
    if (riskScore <= 0.8) return { color: '#F97316', textColor: 'text-orange-600' };
    return { color: '#EF4444', textColor: 'text-danger-red' };
  }, [riskScore]);

  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (riskScore * circumference);

  return (
    <div className={cn('relative inline-flex items-center justify-center', className)}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#E5E7EB"
          strokeWidth={strokeWidth}
          fill="transparent"
        />

        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={risk.color}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-500 ease-out"
        />
      </svg>

      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={cn('text-2xl font-bold', risk.textColor)}>
          {Math.round(riskScore * 100)}%
        </span>
        <span className="text-xs text-gray-500 mt-1">
          Risk
        </span>
      </div>
    </div>
  );
};

export default RiskMeter;