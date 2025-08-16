import React from 'react';
import { FileText, Brain, Shield } from 'lucide-react';
import { cn } from '../../utils/cn';

export interface HeaderProps {
  className?: string;
}

const Header: React.FC<HeaderProps> = ({ className }) => {
  return (
    <header className={cn('bg-white border-b border-pearl-gray', className)}>
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="relative">
                <FileText className="w-8 h-8 text-black" />
                <Brain className="w-4 h-4 text-accent-blue absolute -top-1 -right-1" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-black">
                  AI Legal Doc Explainer
                </h1>
                <p className="text-xs text-gray-600">
                  Powered by Legal-BERT & Gemini AI
                </p>
              </div>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="flex items-center space-x-4">
            {/* AI Status */}
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-success-green rounded-full animate-pulse" />
                <span className="text-xs text-gray-600">AI Active</span>
              </div>
            </div>

            {/* Legal Models Indicator */}
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-success-green" />
              <span className="text-xs text-gray-600">Legal Models Ready</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;