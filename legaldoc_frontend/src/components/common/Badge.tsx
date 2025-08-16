import React from 'react';
import { cn } from '../../utils/cn';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  dot?: boolean;
}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({
    children,
    className,
    variant = 'default',
    size = 'md',
    icon,
    dot = false,
    ...props
  }, ref) => {
    const baseClasses = 'inline-flex items-center font-medium rounded-full border';

    const variants = {
      default: 'bg-gray-50 text-gray-700 border-gray-200',
      success: 'bg-green-50 text-green-700 border-green-200',
      warning: 'bg-yellow-50 text-yellow-700 border-yellow-200',
      danger: 'bg-red-50 text-red-700 border-red-200',
      info: 'bg-blue-50 text-blue-700 border-blue-200',
      secondary: 'bg-pearl-white text-black border-pearl-gray'
    };

    const sizes = {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-2.5 py-1 text-xs',
      lg: 'px-3 py-1.5 text-sm'
    };

    const dotSizes = {
      sm: 'w-1.5 h-1.5',
      md: 'w-2 h-2',
      lg: 'w-2.5 h-2.5'
    };

    const iconSizes = {
      sm: 'w-3 h-3',
      md: 'w-4 h-4',
      lg: 'w-4 h-4'
    };

    const dotColors = {
      default: 'bg-gray-400',
      success: 'bg-green-500',
      warning: 'bg-yellow-500',
      danger: 'bg-red-500',
      info: 'bg-blue-500',
      secondary: 'bg-gray-500'
    };

    return (
      <span
        ref={ref}
        className={cn(
          baseClasses,
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {dot && (
          <span
            className={cn(
              'rounded-full mr-1.5',
              dotSizes[size],
              dotColors[variant]
            )}
          />
        )}

        {icon && (
          <span className={cn('mr-1', iconSizes[size])}>
            {icon}
          </span>
        )}

        {children}
      </span>
    );
  }
);

Badge.displayName = 'Badge';

// Risk Badge specifically for document risk levels
export interface RiskBadgeProps extends Omit<BadgeProps, 'variant'> {
  riskScore: number;
  showScore?: boolean;
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({
  riskScore,
  showScore = false,
  className,
  ...props
}) => {
  let variant: BadgeProps['variant'] = 'success';
  let label = 'Low Risk';

  if (riskScore <= 0.3) {
    variant = 'success';
    label = 'Low Risk';
  } else if (riskScore <= 0.6) {
    variant = 'warning';
    label = 'Medium Risk';
  } else if (riskScore <= 0.8) {
    variant = 'danger';
    label = 'High Risk';
  } else {
    variant = 'danger';
    label = 'Urgent Review';
  }

  return (
    <Badge
      variant={variant}
      className={className}
      {...props}
    >
      {label}
      {showScore && ` (${Math.round(riskScore * 100)}%)`}
    </Badge>
  );
};

// Lawyer Urgency Badge
export interface LawyerUrgencyBadgeProps extends Omit<BadgeProps, 'variant'> {
  urgency: 'low' | 'medium' | 'high' | 'urgent';
}

export const LawyerUrgencyBadge: React.FC<LawyerUrgencyBadgeProps> = ({
  urgency,
  className,
  ...props
}) => {
  const urgencyConfig = {
    low: { variant: 'success' as const, label: 'No needed', icon: '✓' },
    medium: { variant: 'warning' as const, label: 'Consider Review', icon: '⚠️' },
    high: { variant: 'danger' as const, label: 'Recommended', icon: '⚡' },
    urgent: { variant: 'danger' as const, label: 'Required', icon: '🚨' }
  };

  const config = urgencyConfig[urgency] || urgencyConfig.medium;

  return (
    <Badge
      variant={config.variant}
      className={className}
      {...props}
    >
      <span className="mr-1">{config.icon}</span>
      {config.label}
    </Badge>
  );
};

// Status Badge for various statuses
export interface StatusBadgeProps extends Omit<BadgeProps, 'variant'> {
  status: 'active' | 'inactive' | 'pending' | 'processing' | 'completed' | 'error';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className,
  ...props
}) => {
  const statusConfig = {
    active: { variant: 'success' as const, label: 'Active' },
    inactive: { variant: 'secondary' as const, label: 'Inactive' },
    pending: { variant: 'warning' as const, label: 'Pending' },
    processing: { variant: 'info' as const, label: 'Processing' },
    completed: { variant: 'success' as const, label: 'Completed' },
    error: { variant: 'danger' as const, label: 'Error' }
  };

  const config = statusConfig[status] || statusConfig.pending;

  return (
    <Badge
      variant={config.variant}
      dot
      className={className}
      {...props}
    >
      {config.label}
    </Badge>
  );
};

export default Badge;