import { useState, useRef, useEffect } from 'react'
import SourceViewer from './SourceViewer'
import { useChat } from '../hooks/useChat'

function MetaBadge({ label, value, color }) {
  return (
    <span style={{
      fontSize: '11px', padding: '2px 8px',
      borderRadius: '99px', border: `1px solid ${color}20`,
      background: `${color}12`, color: color,
      fontWeight: 500,
    }}>
      {label}{value !== undefined ? `: ${value}` : ''}
    </span>
  )
}

function MessageMeta({ msg }) {
  if (!msg.latency && !msg.isGeneral && !msg.cached && !msg.chunksUsed) return null

  return (
    <div style={{
      display: 'flex', flexWrap: 'wrap', gap: '6px',
      marginTop: '10px', alignItems: 'center',
    }}>
      {msg.cached && (
        <MetaBadge label="⚡ cached" color="#16a34a" />
      )}
      {msg.isGeneral && (
        <MetaBadge label="general knowledge" color="#7c3aed" />
      )}
      {!msg.isGeneral && msg.chunksUsed > 0 && (
        <MetaBadge label="chunks used" value={msg.chunksUsed} color="#2563eb" />
      )}
      {msg.threshold !== undefined && !msg.isGeneral && (
        <MetaBadge label="threshold" value={`${(msg.threshold * 100).toFixed(0)}%`} color="#d97706" />
      )}
      {msg.mempalaceData?.from_memory && (
        <MetaBadge
          label={` MemPalace (${msg.mempalaceData.prior_turns} prior turns)`}
          color="#0891b2"
        />
      )}
      {msg.mempalaceData?.boost_used && (
        <MetaBadge label="memory boosted" color="#0891b2" />
      )}
      {msg.latency && (
        <span style={{ fontSize: '11px', color: '#9ca3af', marginLeft: '2px' }}>
          {msg.latency}ms
        </span>
      )}
    </div>
  )
}

function TypingDots() {
  return (
    <span style={{ display: 'inline-flex', gap: '4px', padding: '4px 0' }}>
      {[0, 1, 2].map(i => (
        <span key={i} style={{
          width: '6px', height: '6px', borderRadius: '50%',
          background: '#9ca3af', display: 'inline-block',
          animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
        }} />
      ))}
      <style>{`@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}`}</style>
    </span>
  )
}

export default function ChatInterface() {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const { messages, isLoading, sources, sendMessage, clearChat } = useChat()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    sendMessage(input)
    setInput('')
    inputRef.current?.focus()
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>

        {messages.length === 0 && (
          <div style={{ textAlign: 'center', marginTop: '60px', color: '#9ca3af' }}>
            <div style={{ fontSize: '44px', marginBottom: '12px' }}></div>
            <p style={{ fontSize: '17px', fontWeight: 500, color: '#6b7280', margin: '0 0 6px' }}>
              Ask anything about your documents
            </p>
            <p style={{ fontSize: '13px', margin: '0 0 20px' }}>
              Document questions show sources + page numbers. General questions answer directly.
            </p>
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
              {['What is monocistronic?', 'Explain Hardy-Weinberg', 'What is RAG?'].map((q, i) => (
                <button key={i} onClick={() => sendMessage(q)} style={{
                  padding: '7px 14px', borderRadius: '99px',
                  border: '1px solid #e5e7eb', background: '#f9fafb',
                  fontSize: '12px', cursor: 'pointer', color: '#374151',
                }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} style={{
            display: 'flex',
            justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            marginBottom: '14px', alignItems: 'flex-start',
          }}>
            {msg.role === 'assistant' && (
              <div style={{
                width: '30px', height: '30px', borderRadius: '50%',
                background: '#eff6ff', border: '1px solid #bfdbfe',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '14px', flexShrink: 0, marginRight: '8px', marginTop: '4px',
              }}></div>
            )}

            <div style={{
              maxWidth: '78%',
              padding: '12px 16px',
              borderRadius: msg.role === 'user'
                ? '18px 18px 4px 18px' : '4px 18px 18px 18px',
              background: msg.role === 'user' ? '#2563eb' : '#ffffff',
              color: msg.role === 'user' ? '#ffffff' : '#1f2937',
              border: msg.role === 'assistant' ? '1px solid #e5e7eb' : 'none',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
              fontSize: '14px', lineHeight: 1.7,
            }}>
              {msg.role === 'assistant' ? (
                <>
                  {msg.loading && !msg.content
                    ? <TypingDots />
                    : <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                  }
                  {msg.error && (
                    <div style={{
                      marginTop: '8px', padding: '8px 10px',
                      background: '#fef2f2', border: '1px solid #fecaca',
                      borderRadius: '6px', fontSize: '12px', color: '#dc2626',
                    }}>
                      {msg.content}
                    </div>
                  )}
                  <MessageMeta msg={msg} />
                </>
              ) : (
                <p style={{ margin: 0 }}>{msg.content}</p>
              )}
            </div>

            {msg.role === 'user' && (
              <div style={{
                width: '30px', height: '30px', borderRadius: '50%',
                background: '#2563eb', color: 'white',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '13px', flexShrink: 0, marginLeft: '8px', marginTop: '4px',
              }}>👤</div>
            )}
          </div>
        ))}

        {sources.length > 0 && <SourceViewer sources={sources} />}
        <div ref={bottomRef} />
      </div>

      <div style={{
        borderTop: '1px solid #e5e7eb', padding: '12px 20px', background: '#ffffff',
      }}>
        {isLoading && (
          <div style={{
            fontSize: '12px', color: '#6b7280', marginBottom: '8px',
            display: 'flex', alignItems: 'center', gap: '6px',
          }}>
            <span style={{
              width: '7px', height: '7px', borderRadius: '50%', background: '#3b82f6',
              display: 'inline-block', animation: 'pulse 1s ease-in-out infinite',
            }} />
            <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}`}</style>
            Searching documents and generating answer...
          </div>
        )}
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '8px' }}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) handleSubmit(e) }}
            placeholder="Ask about your documents or anything else..."
            disabled={isLoading}
            style={{
              flex: 1, padding: '10px 16px', border: '1px solid #d1d5db',
              borderRadius: '12px', fontSize: '14px', outline: 'none',
              background: isLoading ? '#f9fafb' : '#ffffff',
            }}
            onFocus={e => e.target.style.borderColor = '#3b82f6'}
            onBlur={e => e.target.style.borderColor = '#d1d5db'}
          />
          <button type="submit" disabled={isLoading || !input.trim()} style={{
            padding: '10px 20px',
            background: isLoading || !input.trim() ? '#93c5fd' : '#2563eb',
            color: '#fff', border: 'none', borderRadius: '12px',
            fontSize: '14px', fontWeight: 500, cursor: 'pointer',
          }}>
            {isLoading ? '...' : 'Send'}
          </button>
          <button type="button" onClick={clearChat} style={{
            padding: '10px 14px', background: '#fff', color: '#6b7280',
            border: '1px solid #d1d5db', borderRadius: '12px',
            fontSize: '14px', cursor: 'pointer',
          }}>
            Clear
          </button>
        </form>
      </div>
    </div>
  )
}