import { useState, useCallback, useRef, useEffect } from 'react';
import { chatService } from '../services/chatService';
import type { ChatMessage, ChatSession, QuestionResponse } from '../types/chat';
import { generateId } from '../utils/helpers';

export interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  isOpen: boolean;
  currentDocumentId: string | null;
  currentDocumentName: string | null;
  suggestedQuestions: string[];

  // Actions
  openChat: (documentId: string, documentName: string) => void;
  closeChat: () => void;
  sendMessage: (message: string) => Promise<void>;
  clearMessages: () => void;
  clearError: () => void;
  loadSuggestedQuestions: () => Promise<void>;
  selectSuggestedQuestion: (question: string) => void;
  forceShowChat: (documentId?: string, documentName?: string) => void; // Updated signature
}

export const useChat = (): UseChatReturn => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [isVisible, setIsVisible] = useState(false); // New state for actual visibility
  const [currentDocumentId, setCurrentDocumentId] = useState<string | null>(null);
  const [currentDocumentName, setCurrentDocumentName] = useState<string | null>(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);

  // Store chat sessions for different documents
  const chatSessionsRef = useRef<Map<string, ChatSession>>(new Map());

  // Debug state changes
  useEffect(() => {
    console.log('useChat: State changed:', {
      isOpen,
      isVisible,
      currentDocumentId,
      currentDocumentName,
      messageCount: messages.length
    });
  }, [isOpen, isVisible, currentDocumentId, currentDocumentName, messages.length]);

  // Open chat for specific document
  const openChat = useCallback((documentId: string, documentName: string) => {
    console.log('useChat: Opening chat for:', { documentId, documentName });
    console.log('useChat: Current state before opening:', {
      isOpen,
      isVisible,
      currentDocumentId,
      currentDocumentName
    });

    // Set all states
    setCurrentDocumentId(documentId);
    setCurrentDocumentName(documentName);
    setIsOpen(true);
    setIsVisible(true); // Explicitly set visibility
    setError(null);

    console.log('useChat: All states set - isOpen and isVisible should be true');

    // Load existing chat session or create new one
    const existingSession = chatSessionsRef.current.get(documentId);
    if (existingSession) {
      console.log('useChat: Loading existing chat session');
      setMessages(existingSession.messages);
    } else {
      console.log('useChat: Creating new chat session');
      setMessages([]);
      // Add welcome message
      const welcomeMessage: ChatMessage = {
        id: generateId(),
        content: `Hi! I'm your AI legal assistant. I've analyzed "${documentName}" and I'm ready to answer your questions about the document. What would you like to know?`,
        role: 'assistant',
        timestamp: new Date(),
        document_id: documentId,
      };
      console.log('useChat: Adding welcome message:', welcomeMessage);
      setMessages([welcomeMessage]);
    }

    // Load suggested questions
    console.log('useChat: Loading suggested questions');
    loadSuggestedQuestionsInternal(documentId);

    console.log('useChat: openChat completed');
  }, []);

  // Close chat
  const closeChat = useCallback(() => {
    console.log('useChat: Closing chat for document:', currentDocumentId);

    // Save current session before closing
    if (currentDocumentId && currentDocumentName) {
      chatSessionsRef.current.set(currentDocumentId, {
        document_id: currentDocumentId,
        document_name: currentDocumentName,
        messages,
        is_active: false,
        last_updated: new Date(),
      });
    }

    setIsOpen(false);
    setIsVisible(false);
    // Keep document info for potential reopening, but clear suggested questions
    setSuggestedQuestions([]);
    console.log('useChat: Chat closed - isOpen and isVisible set to false');
  }, [currentDocumentId, currentDocumentName, messages]);

  // Load suggested questions
  const loadSuggestedQuestionsInternal = useCallback(async (documentId: string) => {
    try {
      const response = await chatService.getSuggestedQuestions(documentId);
      setSuggestedQuestions(response.questions);
    } catch (err: any) {
      console.error('Error loading suggested questions:', err);
      // Set default questions on error
      setSuggestedQuestions([
        "What are my main obligations in this document?",
        "Are there any risks I should be aware of?",
        "Can I terminate this agreement early?",
        "What are the payment terms?"
      ]);
    }
  }, []);

  const loadSuggestedQuestions = useCallback(async () => {
    if (currentDocumentId) {
      await loadSuggestedQuestionsInternal(currentDocumentId);
    }
  }, [currentDocumentId, loadSuggestedQuestionsInternal]);

  // Send message
  const sendMessage = useCallback(async (messageContent: string) => {
    console.log('useChat: sendMessage called with:', messageContent);

    if (!currentDocumentId) {
      console.error('useChat: No current document ID');
      setError('No document selected. Please select a document first.');
      return;
    }

    if (!messageContent.trim()) {
      console.error('useChat: Empty message content');
      return;
    }

    // Validate question
    const validation = chatService.validateQuestion(messageContent);
    if (!validation.isValid) {
      console.error('useChat: Message validation failed:', validation.error);
      setError(validation.error || 'Invalid question');
      return;
    }

    console.log('useChat: Starting message send process');
    setIsLoading(true);
    setError(null);

    // Add user message
    const userMessage: ChatMessage = {
      id: generateId(),
      content: messageContent.trim(),
      role: 'user',
      timestamp: new Date(),
      document_id: currentDocumentId,
    };

    console.log('useChat: Adding user message to state');
    setMessages(prev => [...prev, userMessage]);

    try {
      console.log('useChat: Calling chat service with:', {
        documentId: currentDocumentId,
        question: messageContent.trim()
      });

      // Get AI response
      const response: QuestionResponse = await chatService.askQuestion(
        currentDocumentId,
        messageContent.trim()
      );

      console.log('useChat: Received AI response:', response);

      // Add AI response message
      const aiMessage: ChatMessage = {
        id: generateId(),
        content: response.answer,
        role: 'assistant',
        timestamp: new Date(),
        document_id: currentDocumentId,
        confidence_score: response.confidence_score,
        relevant_sections: response.relevant_sections,
        legal_implications: response.legal_implications,
        follow_up_questions: response.follow_up_questions,
      };

      console.log('useChat: Adding AI message to state');
      setMessages(prev => [...prev, aiMessage]);

      // Update suggested questions with follow-ups
      if (response.follow_up_questions && response.follow_up_questions.length > 0) {
        console.log('useChat: Updating suggested questions with follow-ups');
        setSuggestedQuestions(response.follow_up_questions);
      }

    } catch (err: any) {
      console.error('useChat: Error during message processing:', err);
      setError(err.message || 'Failed to get response');

      // Add error message
      const errorMessage: ChatMessage = {
        id: generateId(),
        content: 'I apologize, but I encountered an error processing your question. Please try again or rephrase your question.',
        role: 'assistant',
        timestamp: new Date(),
        document_id: currentDocumentId,
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      console.log('useChat: Message send process completed');
      setIsLoading(false);
    }
  }, [currentDocumentId]);

  // Select suggested question
  const selectSuggestedQuestion = useCallback((question: string) => {
    sendMessage(question);
  }, [sendMessage]);

  // Clear messages
  const clearMessages = useCallback(() => {
    if (currentDocumentId) {
      // Keep welcome message
      const welcomeMessage: ChatMessage = {
        id: generateId(),
        content: `Chat cleared. I'm ready to answer more questions about "${currentDocumentName}".`,
        role: 'assistant',
        timestamp: new Date(),
        document_id: currentDocumentId,
      };
      setMessages([welcomeMessage]);

      // Remove from stored sessions
      chatSessionsRef.current.delete(currentDocumentId);
    }
  }, [currentDocumentId, currentDocumentName]);

  // Force show chat (for when it's already open but not visible)
  const forceShowChat = useCallback((documentId?: string, documentName?: string) => {
    console.log('useChat: forceShowChat called with params:', { documentId, documentName });
    console.log('useChat: Current state:', { currentDocumentId, currentDocumentName, isOpen, isVisible });

    // Use passed parameters or current state
    const docId = documentId || currentDocumentId;
    const docName = documentName || currentDocumentName;

    if (docId && docName) {
      console.log('useChat: Forcing chat visibility for document:', { docId, docName });
      setCurrentDocumentId(docId);
      setCurrentDocumentName(docName);
      setIsOpen(true);
      setIsVisible(true);
      console.log('useChat: Chat forced to be visible');
    } else {
      console.log('useChat: No document info available to show chat for');
    }
  }, [currentDocumentId, currentDocumentName, isOpen, isVisible]);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    isOpen: isOpen && isVisible, // Chat is only truly open if both flags are true
    currentDocumentId,
    currentDocumentName,
    suggestedQuestions,

    // Actions
    openChat,
    closeChat,
    sendMessage,
    clearMessages,
    clearError,
    loadSuggestedQuestions,
    selectSuggestedQuestion,
    forceShowChat,
  };
};