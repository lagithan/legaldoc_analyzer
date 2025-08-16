import apiClient from './api';
import type { QuestionRequest, QuestionResponse, SuggestedQuestionsResponse } from '../types/chat';

export class ChatService {
  // Ask question about document
  async askQuestion(documentId: string, question: string): Promise<QuestionResponse> {
    const request: QuestionRequest = {
      document_id: documentId,
      question: question.trim()
    };

    return apiClient.post<QuestionResponse>('/ask-question', request);
  }

  // Get suggested questions for document
  async getSuggestedQuestions(documentId: string): Promise<SuggestedQuestionsResponse> {
    return apiClient.get<SuggestedQuestionsResponse>(`/suggest-questions/${documentId}`);
  }

  // Validate question before sending
  validateQuestion(question: string): { isValid: boolean; error?: string } {
    const trimmed = question.trim();

    if (!trimmed) {
      return { isValid: false, error: 'Please enter a question' };
    }

    if (trimmed.length < 3) {
      return { isValid: false, error: 'Question is too short' };
    }

    if (trimmed.length > 500) {
      return { isValid: false, error: 'Question is too long (max 500 characters)' };
    }

    return { isValid: true };
  }

  // Generate follow-up questions based on context
  generateFollowUpQuestions(context: string, currentQuestion: string): string[] {
    const followUps = [
      "Can you explain this in simpler terms?",
      "What are the specific consequences?",
      "Are there any exceptions to this rule?",
      "How does this compare to standard practices?",
      "What should I do to protect myself?",
      "When does this take effect?",
      "Who is responsible for enforcement?",
      "What are my options if I disagree?"
    ];

    // Return a subset of relevant follow-ups
    return followUps.slice(0, 3);
  }
}

// Create and export singleton instance
export const chatService = new ChatService();
export default chatService;