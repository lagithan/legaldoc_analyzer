import React, { useState } from 'react';
import {
  User,
  Bot,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  FileText
} from 'lucide-react';
import Button from '../common/Button';
import { cn } from '../../utils/cn';
import { formatTimeAgo, formatConfidenceScore } from '../../utils/helpers';
import type { ChatMessage as ChatMessageType } from '../../types/chat';

export interface ChatMessageProps {
  message: ChatMessageType;
  onSuggestedQuestionClick?: (question: string) => void;
  className?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onSuggestedQuestionClick,
  className
}) => {
  const [copied, setCopied] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const isUser = message.role === 'user';
  const isAI = message.role === 'assistant';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy message:', error);
    }
  };

  return (
    <div className={cn('flex space-x-3', isUser && 'flex-row-reverse space-x-reverse', className)}>
      {/* Avatar */}
      <div className={cn(
        'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
        isUser ? 'bg-black text-white' : 'bg-gray-100 text-gray-600'
      )}>
        {isUser ? (
          <User className="w-4 h-4" />
        ) : (
          <Bot className="w-4 h-4" />
        )}
      </div>

      {/* Message Content */}
      <div className={cn(
        'flex-1 min-w-0',
        isUser ? 'text-right' : 'text-left'
      )}>
        {/* Message Bubble */}
        <div className={cn(
          'inline-block max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl',
          isUser
            ? 'chat-bubble-user'
            : 'chat-bubble-ai'
        )}>
          <p className="whitespace-pre-wrap break-words">
            {message.content}
          </p>
        </div>

        {/* AI Message Additional Info */}
        {isAI && (message.confidence_score || message.relevant_sections || message.legal_implications) && (
          <div className="mt-2 space-y-2">
            {/* Confidence Score */}
            {message.confidence_score && (
              <div className="flex items-center space-x-2 text-xs text-gray-500">
                <AlertCircle className="w-3 h-3" />
                <span>Confidence: {formatConfidenceScore(message.confidence_score)}</span>
              </div>
            )}

            {/* Toggle Details */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs p-1 h-auto"
              icon={showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            >
              {showDetails ? 'Hide details' : 'Show details'}
            </Button>

            {/* Expandable Details */}
            {showDetails && (
              <div className="space-y-3 p-3 bg-gray-50 rounded-lg text-xs">
                {/* Relevant Sections */}
                {message.relevant_sections && message.relevant_sections.length > 0 && (
                  <div>
                    <div className="flex items-center space-x-1 font-medium text-gray-700 mb-2">
                      <FileText className="w-3 h-3" />
                      <span>Relevant sections:</span>
                    </div>
                    <div className="space-y-1">
                      {message.relevant_sections.map((section, index) => (
                        <div key={index} className="pl-2 border-l-2 border-gray-200">
                          <p className="text-gray-600 italic">"{section}"</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Legal Implications */}
                {message.legal_implications && (
                  <div>
                    <div className="font-medium text-gray-700 mb-1">Legal implications:</div>
                    <p className="text-gray-600">{message.legal_implications}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Follow-up Questions */}
        {isAI && message.follow_up_questions && message.follow_up_questions.length > 0 && (
          <div className="mt-3 space-y-2">
            <div className="text-xs text-gray-500 font-medium">Follow-up questions:</div>
            <div className="space-y-1">
              {message.follow_up_questions.map((question, index) => (
                <button
                  key={index}
                  onClick={() => onSuggestedQuestionClick?.(question)}
                  className="block w-full text-left text-xs p-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message Footer */}
        <div className={cn(
          'flex items-center space-x-2 mt-2',
          isUser ? 'justify-end' : 'justify-start'
        )}>
          <span className="text-xs text-gray-500">
            {formatTimeAgo(message.timestamp.toISOString())}
          </span>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="p-1 h-auto opacity-0 group-hover:opacity-100 transition-opacity"
            icon={copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            title="Copy message"
          />
        </div>
      </div>
    </div>
  );
};

// Typing indicator component
export const TypingIndicator: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={cn('flex space-x-3', className)}>
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
        <Bot className="w-4 h-4 text-gray-600" />
      </div>

      <div className="chat-bubble-ai">
        <div className="loading-dots">
          <div></div>
          <div></div>
          <div></div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;