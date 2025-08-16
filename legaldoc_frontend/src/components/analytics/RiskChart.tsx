import React from 'react';
import { cn } from '../../utils/cn';

export interface RiskChartProps {
  data: Record<string, number>;
  title?: string;
  className?: string;
}

const RiskChart: React.FC<RiskChartProps> = ({
  data,
  title = "Risk Distribution",
  className
}) => {
  const total = Object.values(data).reduce((sum, value) => sum + value, 0);

  const colors = {
    low: 'bg-success-green',
    medium: 'bg-warning-yellow',
    high: 'bg-danger-red',
    urgent: 'bg-red-700'
  };

  const labels = {
    low: 'Low Risk',
    medium: 'Medium Risk',
    high: 'High Risk',
    urgent: 'Urgent Review'
  };

  if (total === 0) {
    return (
      <div className={cn('text-center py-8 text-gray-500', className)}>
        <p>No data available</p>
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {title && (
        <h3 className="text-lg font-semibold text-black">{title}</h3>
      )}

      {/* Bar Chart */}
      <div className="space-y-3">
        {Object.entries(data).map(([key, value]) => {
          const percentage = total > 0 ? (value / total) * 100 : 0;
          const colorClass = colors[key as keyof typeof colors] || 'bg-gray-400';
          const label = labels[key as keyof typeof labels] || key;

          return (
            <div key={key} className="space-y-1">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-700">{label}</span>
                <span className="font-medium text-black">
                  {value} ({Math.round(percentage)}%)
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={cn('h-3 rounded-full transition-all duration-500', colorClass)}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div className="text-sm text-gray-600 pt-2 border-t border-gray-200">
        Total documents: {total}
      </div>
    </div>
  );
};

// Donut Chart variant
export interface DonutChartProps {
  data: Record<string, number>;
  size?: number;
  className?: string;
}

export const DonutChart: React.FC<DonutChartProps> = ({
  data,
  size = 200,
  className
}) => {
  const total = Object.values(data).reduce((sum, value) => sum + value, 0);

  const colors = {
    low: '#10B981',
    medium: '#F59E0B',
    high: '#EF4444',
    urgent: '#B91C1C'
  };

  const labels = {
    low: 'Low Risk',
    medium: 'Medium Risk',
    high: 'High Risk',
    urgent: 'Urgent Review'
  };

  if (total === 0) {
    return (
      <div className={cn('flex items-center justify-center', className)} style={{ height: size }}>
        <p className="text-gray-500">No data</p>
      </div>
    );
  }

  const radius = (size - 40) / 2;
  const circumference = 2 * Math.PI * radius;
  let currentAngle = 0;

  return (
    <div className={cn('flex items-center space-x-6', className)}>
      {/* SVG Chart */}
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {Object.entries(data).map(([key, value], index) => {
            if (value === 0) return null;

            const percentage = value / total;
            const strokeDasharray = circumference * percentage;
            const strokeDashoffset = -circumference * currentAngle;
            const color = colors[key as keyof typeof colors] || '#6B7280';

            currentAngle += percentage;

            return (
              <circle
                key={key}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="transparent"
                stroke={color}
                strokeWidth="20"
                strokeDasharray={`${strokeDasharray} ${circumference}`}
                strokeDashoffset={strokeDashoffset}
                className="transition-all duration-500"
              />
            );
          })}
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-2xl font-bold text-black">{total}</div>
          <div className="text-sm text-gray-600">Documents</div>
        </div>
      </div>

      {/* Legend */}
      <div className="space-y-2">
        {Object.entries(data).map(([key, value]) => {
          if (value === 0) return null;

          const percentage = total > 0 ? (value / total) * 100 : 0;
          const color = colors[key as keyof typeof colors] || '#6B7280';
          const label = labels[key as keyof typeof labels] || key;

          return (
            <div key={key} className="flex items-center space-x-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-sm text-gray-700">
                {label}: {value} ({Math.round(percentage)}%)
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default RiskChart;