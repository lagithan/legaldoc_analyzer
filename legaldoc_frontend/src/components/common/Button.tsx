import React from 'react';
import { cn } from '../../utils/cn';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({
    children,
    className,
    variant = 'primary',
    size = 'md',
    loading = false,
    disabled,
    icon,
    iconPosition = 'left',
    type = 'button',
    ...props
  }, ref) => {
    const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
      primary: 'bg-black text-pearl-white hover:bg-dark-gray focus:ring-gray-500',
      secondary: 'bg-pearl-white text-black border border-gray-300 hover:bg-light-gray focus:ring-gray-500',
      danger: 'bg-danger-red text-white hover:bg-red-600 focus:ring-red-500',
      ghost: 'text-black hover:bg-light-gray focus:ring-gray-500',
      outline: 'border border-black text-black hover:bg-black hover:text-pearl-white focus:ring-gray-500'
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-sm',
      lg: 'px-6 py-3 text-base'
    };

    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        type={type}
        disabled={isDisabled}
        className={cn(
          baseClasses,
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {loading && (
          <div className="mr-2">
            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {!loading && icon && iconPosition === 'left' && (
          <span className={cn('mr-2', children ? 'mr-2' : 'mr-0')}>
            {icon}
          </span>
        )}

        {children}

        {!loading && icon && iconPosition === 'right' && (
          <span className={cn('ml-2', children ? 'ml-2' : 'ml-0')}>
            {icon}
          </span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;