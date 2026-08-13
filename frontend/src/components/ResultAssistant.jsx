import React, { useState } from 'react';

const QUESTIONS = [
  'Why did the model predict this?',
  'Which regions received the most graph attention?',
  'What does the confidence score mean?',
  'What are the main signs for this class?',
  'Should this result be reviewed?',
];

export default function ResultAssistant({ results, onAsk, messages = [], isLoading }) {
  const [question, setQuestion] = useState('');

  const ask = (value) => {
    const text = value.trim();
    if (!text || !results || isLoading) return;
    onAsk(text);
    setQuestion('');
  };

  return (
    <aside className="result-assistant" aria-label="Result Assistant">
      <div className="result-assistant-header">
        <div>
          <div className="result-assistant-kicker">AI support</div>
          <h2>Result Assistant</h2>
        </div>
        <span className="result-assistant-status">{results ? 'Context ready' : 'Awaiting result'}</span>
      </div>

      {!results ? (
        <p className="result-assistant-empty">Analyze a chest X-ray to ask questions about the model result.</p>
      ) : (
        <>
          <div className="result-assistant-context">
            <span>Current prediction</span>
            <strong>{results.prediction || results.probabilities?.[0]?.label}</strong>
            <small>{((results.confidence || 0) * 100).toFixed(1)}% confidence</small>
          </div>
          <div className="result-assistant-label">Ask about this result</div>
          <div className="result-assistant-questions">
            {QUESTIONS.map((item) => (
              <button key={item} type="button" onClick={() => ask(item)} disabled={isLoading}>{item}</button>
            ))}
          </div>
          <div className="result-assistant-messages">
            {messages.map((message, index) => (
              <div className="result-assistant-message" key={`${message.question}-${index}`}>
                <div className="result-assistant-question">{message.question}</div>
                <div className="result-assistant-answer">{message.answer}</div>
              </div>
            ))}
          </div>
          <form className="result-assistant-form" onSubmit={(event) => { event.preventDefault(); ask(question); }}>
            <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about this result…" disabled={isLoading} />
            <button type="submit" disabled={isLoading || !question.trim()}>{isLoading ? '…' : 'Ask'}</button>
          </form>
        </>
      )}
    </aside>
  );
}
