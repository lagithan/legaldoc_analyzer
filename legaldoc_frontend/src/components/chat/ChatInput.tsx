import React, { useState, useRef, useEffect } from "react";
import { Send, Loader2, Lightbulb } from "lucide-react";
import Button from "../common/Button";
import { cn } from "../../utils/cn";

export interface ChatInputProps {
  onSendMessage: (message: string) => void;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  suggestedQuestions?: string[];
  onSuggestedQuestionClick?: (question: string) => void;
  className?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  loading = false,
  disabled = false,
  placeholder = "Ask a question about this document...",
  suggestedQuestions = [],
  onSuggestedQuestionClick,
  className,
}) => {
  const [message, setMessage] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  // Focus textarea when not loading
  useEffect(() => {
    if (!loading && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [loading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const trimmedMessage = message.trim();
    if (!trimmedMessage || loading || disabled) return;

    onSendMessage(trimmedMessage);
    setMessage("");
    setShowSuggestions(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSuggestedQuestionClick = (question: string) => {
    setMessage(question);
    setShowSuggestions(false);
    onSuggestedQuestionClick?.(question);
  };

  const isDisabled = loading || disabled;
  const canSend = message.trim().length > 0 && !isDisabled;

  return (
    <div className={cn("space-y-3", className)}>
      {/* Suggested Questions */}
      {suggestedQuestions.length > 0 && (
        <div className="space-y-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSuggestions(!showSuggestions)}
            className="text-xs h-auto p-2"
            icon={<Lightbulb className="w-3 h-3" />}
          >
            {showSuggestions ? "Hide suggestions" : "Show suggested questions"}
          </Button>

          {showSuggestions && (
            <div className="grid gap-2">
              {suggestedQuestions.slice(0, 4).map((question, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestedQuestionClick(question)}
                  className="text-left text-xs p-2 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors"
                  disabled={isDisabled}
                >
                  {question}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="space-y-2">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isDisabled}
            rows={1}
            className={cn(
              "w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-accent-blue focus:border-transparent outline-none transition-all duration-200",
              "min-h-[48px] max-h-32",
              isDisabled && "opacity-50 cursor-not-allowed"
            )}
            style={{
              minHeight: "48px",
              maxHeight: "128px",
            }}
          />

          {/* Send Button */}
          <div className="absolute right-2 bottom-3.5">
            <Button
              type="submit"
              variant="primary"
              size="sm"
              disabled={!canSend}
              loading={loading}
              className="p-2 rounded-lg"
              icon={
                loading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )
              }
              title="Send message"
            />
          </div>
        </div>

        {/* Character count and hints */}
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-4">
            <span>{message.length}/500 characters</span>
            {message.length > 450 && (
              <span className="text-orange-600">
                Character limit approaching
              </span>
            )}
          </div>

          <span className="hidden sm:block">
            Press Enter to send, Shift+Enter for new line
          </span>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;
