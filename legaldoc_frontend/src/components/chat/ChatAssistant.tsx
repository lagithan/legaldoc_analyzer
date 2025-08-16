import React, { useEffect, useRef } from 'react';
import {
  MessageSquare,
  X,
  Trash2,
  FileText,
  AlertCircle
} from 'lucide-react';
import ChatMessage, { TypingIndicator } from './ChatMessage';
import ChatInput from './ChatInput';
import Button from '../common/Button';
import Card, { CardHeader, CardTitle, CardContent } from '../common/Card';
import { useChat } from '../../hooks/useChat';
import { cn } from '../../utils/cn';

export interface ChatAssistantProps {
  className?: string;
  onchat?: () => void;
}

const ChatAssistant: React.FC<ChatAssistantProps> = ({ className,onchat }) => {
  const {
    messages,
    isLoading,
    error,
    isOpen,
    currentDocumentId,
    currentDocumentName,
    suggestedQuestions,
    closeChat,
    sendMessage,
    clearMessages,
    clearError,
    selectSuggestedQuestion,
    forceShowChat
  } = useChat();

  const [isMinimized, setIsMinimized] = React.useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Debug chat state - only log key changes
  React.useEffect(() => {
    console.log('ChatAssistant: Chat state changed:', {
      isOpen,
      currentDocumentId: currentDocumentId ? 'exists' : 'null',
      shouldRenderChat: currentDocumentId !== null && isOpen
    });
  }, [isOpen, currentDocumentId]);

  // Define handleSendMessage early to avoid hoisting issues
  const handleSendMessage = async (message: string) => {
    console.log('ChatAssistant: Sending message:', message);
    clearError();
    await sendMessage(message);
  };

  const handleSuggestedQuestionClick = (question: string) => {
    selectSuggestedQuestion(question);
  };

  const handleClose = () => {
    console.log('ChatAssistant: Closing chat');
    closeChat();
    setIsMinimized(false);
  };

  const handleClearChat = () => {
    clearMessages();
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current && !isMinimized) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading, isMinimized]);

  // Handle chat opening from external components
  useEffect(() => {
    console.log('ChatAssistant: Chat opening effect triggered', { isOpen, isMinimized });
    if (isOpen && isMinimized) {
      console.log('ChatAssistant: Expanding minimized chat');
      setIsMinimized(false);
    }
  }, [isOpen, isMinimized]);

  // Listen for external message sending events
  useEffect(() => {
    const handleExternalMessage = (event: CustomEvent) => {
      if (isOpen && currentDocumentId && event.detail?.message) {
        console.log('ChatAssistant: Processing external message:', event.detail.message);
        handleSendMessage(event.detail.message);
      }
    };

    const handleForceShow = (event: CustomEvent) => {
      const { documentId, documentName } = event.detail || {};
      forceShowChat(documentId, documentName);
    };

    window.addEventListener('sendChatMessage', handleExternalMessage as EventListener);
    window.addEventListener('forceShowChat', handleForceShow as EventListener);

    return () => {
      window.removeEventListener('sendChatMessage', handleExternalMessage as EventListener);
      window.removeEventListener('forceShowChat', handleForceShow as EventListener);
    };
  }, [isOpen, currentDocumentId, handleSendMessage, forceShowChat]);

  // Chat should render if there's a document AND the chat is open
  const shouldRenderChat = currentDocumentId !== null && isOpen;

  // Chat button (when should not render chat)
  if (!shouldRenderChat) {
    return (
      <div className={cn('fixed bottom-6 right-6 z-50', className)}>
        <Button
          variant="primary"
          size="lg"
          onClick={() => onchat ? onchat() : null}
          className="rounded-full shadow-strong hover:shadow-medium transition-shadow duration-200 p-4"
          icon={<MessageSquare className="w-6 h-6" />}
          title="Open AI Chat Assistant"
        />
      </div>
    );
  }

  // Chat panel (when should render chat)
  return (
    <div
      className={cn(
        'fixed bottom-6 right-6 z-50 transition-all duration-300 ease-out',
        isMinimized ? 'w-80' : 'w-96 sm:w-[400px]',
        className
      )}
    >
      <Card
        className={cn(
          'shadow-strong border-2 border-gray-200 animate-slide-in',
          isMinimized ? 'h-16' : 'h-[600px]'
        )}
        padding="none"
      >
        {/* Header */}
        <CardHeader className="p-4 pb-2 border-b border-pearl-gray">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 min-w-0 flex-1">
              <div className="flex-shrink-0 w-8 h-8 bg-accent-blue rounded-full flex items-center justify-center">
                <MessageSquare className="w-4 h-4 text-white" />
              </div>
              <div className="min-w-0 flex-1">
                <CardTitle className="text-sm truncate">AI Legal Assistant</CardTitle>
                {currentDocumentName && (
                  <div className="flex items-center space-x-1 text-xs text-gray-500">
                    <FileText className="w-3 h-3" />
                    <span className="truncate">{currentDocumentName}</span>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center space-x-1 ml-2">
              {!isMinimized && messages.length > 1 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClearChat}
                  className="p-1.5"
                  icon={<Trash2 className="w-4 h-4" />}
                  title="Clear chat"
                />
              )}

              <Button
                variant="ghost"
                size="sm"
                onClick={handleClose}
                className="p-1.5"
                icon={<X className="w-4 h-4" />}
                title="Close chat"
              />
            </div>
          </div>
        </CardHeader>

        {/* Chat Content */}
        {!isMinimized && (
          <CardContent className="flex flex-col h-[520px] p-0">
            {/* Messages Area */}
            <div
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide"
            >
              {!currentDocumentId ? (
                <div className="text-center py-8 text-gray-500">
                  <MessageSquare className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p className="text-sm">
                    Select a document to start asking questions
                  </p>
                </div>
              ) : (
                <>
                  {messages.map((message) => (
                    <div key={message.id} className="group">
                      <ChatMessage
                        message={message}
                        onSuggestedQuestionClick={handleSuggestedQuestionClick}
                      />
                    </div>
                  ))}

                  {isLoading && (
                    <TypingIndicator />
                  )}

                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Error Display */}
            {error && (
              <div className="mx-4 mb-2">
                <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span className="text-sm flex-1">{error}</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearError}
                    className="p-1"
                    icon={<X className="w-3 h-3" />}
                  />
                </div>
              </div>
            )}

            {/* Input Area */}
            <div className="border-t border-pearl-gray p-4">
              <ChatInput
                onSendMessage={handleSendMessage}
                loading={isLoading}
                disabled={!currentDocumentId}
                placeholder={
                  currentDocumentId
                    ? "Ask a question about this document..."
                    : "Select a document first..."
                }
                suggestedQuestions={suggestedQuestions}
                onSuggestedQuestionClick={handleSuggestedQuestionClick}
              />
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  );
};

export default ChatAssistant;