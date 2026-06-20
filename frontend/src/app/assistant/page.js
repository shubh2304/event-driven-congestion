'use client';

import { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '@/lib/api';
import PageTransition from '@/components/PageTransition';
import ChatMessage, { TypingIndicator } from '@/components/ChatMessage';
import { IconSend, IconChat, IconRefresh } from '@/components/SvgIcons';

const WELCOME_MESSAGE = {
  role: 'assistant',
  content: `**ASTRAM Traffic Assistant — What I Can Do:**

**Predict congestion impact** — Give me event details and I'll predict priority, road closure risk, duration, and recommend resources.
> *"accident at 12.97, 77.59 at 5pm"*
> *"waterlogging near Central Zone 2 at 8am on Monday"*

**Show hotspots** — I'll show you the most congestion-dense areas.
> *"show hotspots"* or *"hotspots in North Zone 1"*

**Compare corridors/zones** — Compare risk levels between corridors.
> *"compare Mysore Road vs Hosur Road"*

**Explain recommendations** — I'll explain how the system makes decisions.
> *"why is priority high for protests?"*

**Model performance** — Get current model accuracy metrics.
> *"model accuracy"* or *"how good are the predictions?"*`,
  timestamp: ''
};

const QUICK_ACTIONS = [
  'Show hotspots',
  'Model accuracy',
  'Compare Mysore Road vs Hosur Road',
  'How are recommendations generated?',
  'Help',
];

export default function AssistantPage() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    setMessages([
      {
        ...WELCOME_MESSAGE,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages, isLoading]);

  const handleSend = async (text) => {
    const messageText = text || input.trim();
    if (!messageText || isLoading) return;

    const userMsg = {
      role: 'user',
      content: messageText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await sendChatMessage(messageText);
      const assistantMsg = {
        role: 'assistant',
        content: res.response,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `An error occurred: ${err.message}. Please try again.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([
      {
        ...WELCOME_MESSAGE,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
  };

  return (
    <PageTransition>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-header__title">ASTRAM Traffic Assistant</h1>
          <p className="page-header__desc">
            Ask questions in natural language to get predictions, explore hotspots,
            compare corridors, and understand how the models work.
          </p>
        </div>
        <button className="btn btn--ghost" onClick={handleClear} id="clear-chat-btn">
          <IconRefresh size={16} />
          Clear Chat
        </button>
      </div>

      <div className="chat-container" id="chat-container">
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <ChatMessage key={i} {...msg} />
          ))}
          {isLoading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-quick-actions">
          {QUICK_ACTIONS.map((action, i) => (
            <button
              key={i}
              className="quick-action-btn"
              onClick={() => handleSend(action)}
              disabled={isLoading}
            >
              {action}
            </button>
          ))}
        </div>

        <div className="chat-input-bar">
          <input
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about traffic events, hotspots, or predictions..."
            disabled={isLoading}
            id="chat-input"
          />
          <button
            className="chat-send-btn"
            onClick={() => handleSend()}
            disabled={isLoading || !input.trim()}
            id="chat-send-btn"
          >
            <IconSend size={18} />
          </button>
        </div>
      </div>
    </PageTransition>
  );
}
