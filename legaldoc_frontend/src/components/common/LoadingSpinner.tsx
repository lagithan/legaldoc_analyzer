import React from 'react';
import { cn } from '../../utils/cn';

export interface LoadingSpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'spinner' | 'dots' | 'pulse';
  color?: 'primary' | 'secondary' | 'white';
  text?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'spinner',
  color = 'primary',
  text,
  className,
  ...props
}) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const colors = {
    primary: 'border-black border-t-transparent',
    secondary: 'border-gray-400 border-t-transparent',
    white: 'border-white border-t-transparent'
  };

  const textSizes = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
    xl: 'text-lg'
  };

  if (variant === 'spinner') {
    return (
      <div className={cn('flex flex-col items-center justify-center space-y-2', className)} {...props}>
        <div
          className={cn(
            'border-2 rounded-full animate-spin',
            sizes[size],
            colors[color]
          )}
        />
        {text && (
          <span className={cn('text-gray-600', textSizes[size])}>
            {text}
          </span>
        )}
      </div>
    );
  }

  if (variant === 'dots') {
    const dotSize = {
      sm: 'w-1 h-1',
      md: 'w-2 h-2',
      lg: 'w-3 h-3',
      xl: 'w-4 h-4'
    };

    return (
      <div className={cn('flex flex-col items-center justify-center space-y-2', className)} {...props}>
        <div className="flex space-x-1">
          <div
            className={cn(
              'bg-gray-400 rounded-full animate-pulse',
              dotSize[size]
            )}
            style={{ animationDelay: '0ms' }}
          />
          <div
            className={cn(
              'bg-gray-400 rounded-full animate-pulse',
              dotSize[size]
            )}
            style={{ animationDelay: '150ms' }}
          />
          <div
            className={cn(
              'bg-gray-400 rounded-full animate-pulse',
              dotSize[size]
            )}
            style={{ animationDelay: '300ms' }}
          />
        </div>
        {text && (
          <span className={cn('text-gray-600', textSizes[size])}>
            {text}
          </span>
        )}
      </div>
    );
  }

  if (variant === 'pulse') {
    return (
      <div className={cn('flex flex-col items-center justify-center space-y-2', className)} {...props}>
        <div
          className={cn(
            'bg-gray-300 rounded-full animate-pulse',
            sizes[size]
          )}
        />
        {text && (
          <span className={cn('text-gray-600 animate-pulse', textSizes[size])}>
            {text}
          </span>
        )}
      </div>
    );
  }

  return null;
};

// Skeleton loading component for content placeholders
export const Skeleton: React.FC<{
  className?: string;
  lines?: number;
  height?: string;
}> = ({ className, lines = 1, height = 'h-4' }) => {
  return (
    <div className={cn('animate-pulse', className)}>
      {Array.from({ length: lines }).map((_, index) => (
        <div
          key={index}
          className={cn(
            'bg-gray-200 rounded',
            height,
            index > 0 ? 'mt-2' : '',
            index === lines - 1 && lines > 1 ? 'w-3/4' : 'w-full'
          )}
        />
      ))}
    </div>
  );
};

// Full page loading overlay
export const LoadingOverlay: React.FC<{
  isVisible: boolean;
  text?: string;
}> = ({ isVisible, text = 'Loading...' }) => {
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 shadow-lg">
        <LoadingSpinner size="lg" text={text} />
      </div>
    </div>
  );
};

export default LoadingSpinner;