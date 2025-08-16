import React from 'react';
import { LucideIcon } from 'lucide-react';
import Card, { CardContent } from '../common/Card';
import { cn } from '../../utils/cn';

export interface StatsCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: LucideIcon;
  change?: {
    value: number;
    period: string;
    trend: 'up' | 'down' | 'neutral';
  };
  color?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  className?: string;
}

const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  description,
  icon: Icon,
  change,
  color = 'default',
  className
}) => {
  const colors = {
    default: {
      icon: 'text-gray-600 bg-gray-100',
      change: 'text-gray-600'
    },
    success: {
      icon: 'text-success-green bg-green-100',
      change: 'text-success-green'
    },
    warning: {
      icon: 'text-warning-yellow bg-yellow-100',
      change: 'text-warning-yellow'
    },
    danger: {
      icon: 'text-danger-red bg-red-100',
      change: 'text-danger-red'
    },
    info: {
      icon: 'text-accent-blue bg-blue-100',
      change: 'text-accent-blue'
    }
  };

  const colorConfig = colors[color];

  const getTrendIcon = (trend: 'up' | 'down' | 'neutral') => {
    switch (trend) {
      case 'up':
        return '↗️';
      case 'down':
        return '↘️';
      default:
        return '→';
    }
  };

  const getTrendColor = (trend: 'up' | 'down' | 'neutral') => {
    switch (trend) {
      case 'up':
        return 'text-success-green';
      case 'down':
        return 'text-danger-red';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <Card className={cn('hover:shadow-medium transition-shadow duration-200', className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-600 uppercase tracking-wide">
              {title}
            </p>
            <p className="text-3xl font-bold text-black mt-2">
              {typeof value === 'number' ? value.toLocaleString() : value}
            </p>

            {description && (
              <p className="text-sm text-gray-500 mt-1">
                {description}
              </p>
            )}

            {change && (
              <div className="flex items-center mt-2 space-x-1">
                <span className={cn('text-sm font-medium', getTrendColor(change.trend))}>
                  {getTrendIcon(change.trend)} {Math.abs(change.value)}%
                </span>
                <span className="text-sm text-gray-500">
                  {change.period}
                </span>
              </div>
            )}
          </div>

          {Icon && (
            <div className={cn(
              'flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center',
              colorConfig.icon
            )}>
              <Icon className="w-6 h-6" />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Mini stats card for smaller spaces
export interface MiniStatsCardProps {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  color?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  className?: string;
}

export const MiniStatsCard: React.FC<MiniStatsCardProps> = ({
  label,
  value,
  icon: Icon,
  color = 'default',
  className
}) => {
  const colors = {
    default: 'text-gray-600',
    success: 'text-success-green',
    warning: 'text-warning-yellow',
    danger: 'text-danger-red',
    info: 'text-accent-blue'
  };

  return (
    <div className={cn('text-center p-4 rounded-lg bg-gray-50', className)}>
      {Icon && (
        <div className="flex justify-center mb-2">
          <Icon className={cn('w-8 h-8', colors[color])} />
        </div>
      )}
      <div className="text-2xl font-bold text-black">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
      <div className="text-sm text-gray-600 mt-1">
        {label}
      </div>
    </div>
  );
};

export default StatsCard;