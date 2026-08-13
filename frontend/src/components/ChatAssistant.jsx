import React, { useState, useRef, useEffect } from 'react';
import '../styles/ChatAssistant.css';
import { getEndpoint, getTunnelHeaders } from '../apiConfig';

const STARTER_QUESTIONS = [
  'What does a normal chest X-ray look like?',
  'How can tuberculosis appear on a chest X-ray?',
  'What are common pleural findings on a chest X-ray?',
  'What does the model confidence score mean?',
];

/**
 * ChatAssistant - Multi-turn conversational assistant for medical X-ray questions
 * 
 * Features:
 * - Maintains conversation history
 * - Answers based on medical knowledge base only
 * - Asks clarifying questions
 * - System restrictions enforced server-side
 */
export default function ChatAssistant() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hello! I\'m a medical AI assistant specialized in chest X-ray analysis. I can answer questions about cardiac pathology, chronic lung disease, pleural pathology, tuberculosis, and normal chest X-ray findings. How can I help you today?',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (messageToSend = input) => {
    const text = messageToSend.trim();
    if (!text || loading) return;

    const userMessage = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setError('');
    setLoading(true);

    try {
      const response = await fetch(getEndpoint('/api/chat-assistant'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getTunnelHeaders() },
        body: JSON.stringify({
          message: text,
          conversation_history: messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
          })),
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.status === 'ok') {
        const assistantMessage = {
          role: 'assistant',
          content: data.response,
          clarifying_question: data.clarifying_question || false,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        throw new Error('Unexpected response format');
      }
    } catch (err) {
      setError(err.message || 'Failed to get response from assistant');
      console.error('Chat error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-assistant">
      <div className="chat-header">
        <h2>Medical AI Assistant</h2>
        <p className="chat-subtitle">
          Ask about chest X-ray findings, pathologies, and analysis
        </p>
      </div>

      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? '👤' : '🩺'}
            </div>
            <div className="message-content">
              <p>{msg.content}</p>
              {msg.clarifying_question && (
                <span className="clarifying-badge">Asking for clarification</span>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-avatar">🩺</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {error && <div className="chat-error">{error}</div>}

      <div className="chat-starters" aria-label="Suggested questions">
        <span>Try asking</span>
        <div>
          {STARTER_QUESTIONS.map((question) => (
            <button key={question} type="button" onClick={() => handleSendMessage(question)} disabled={loading}>
              {question}
            </button>
          ))}
        </div>
      </div>

      <div className="chat-input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about chest X-ray findings, TB, cardiac conditions, pleural pathology, lung disease..."
          rows="3"
          disabled={loading}
        />
        <button type="button" onClick={() => handleSendMessage()} disabled={loading || !input.trim()}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>

      <div className="chat-info">
        <p>
          <strong>Note:</strong> This assistant answers only about medical chest X-ray content.
          It does not diagnose, prescribe treatment, or replace radiologist review.
        </p>
      </div>
    </div>
  );
}
