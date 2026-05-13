import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import SourceViewer from './SourceViewer'
import { useChat } from '../hooks/useChat'

export default function ChatInterface() {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const { messages, isLoading, sources, sendMessage, clearChat } = useChat()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim()) return
    sendMessage(input)
    setInput('')
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-5xl mb-4">💬</p>
            <p className="text-lg font-medium">Ask anything about your documents</p>
            <p className="text-sm mt-1">Upload a PDF or Excel file above, then ask a question</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm
              ${msg.role === 'user'
                ? 'bg-blue-600 text-white rounded-br-sm'
                : 'bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm'
              }`}
            >
              {msg.role === 'assistant' ? (
                <>
                  <ReactMarkdown>{msg.content || (msg.loading ? '▋' : '')}</ReactMarkdown>
                  {msg.latency && (
                    <p className="text-xs text-gray-400 mt-2">
                      {msg.cached ? '⚡ cached · ' : ''}{msg.latency}ms
                    </p>
                  )}
                  {msg.error && <p className="text-red-500 text-xs mt-1">Something went wrong</p>}
                </>
              ) : (
                <p>{msg.content}</p>
              )}
            </div>
          </div>
        ))}

        {/* Sources panel below the latest answer */}
        {sources.length > 0 && <SourceViewer sources={sources} />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 bg-white p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 border border-gray-300 rounded-xl text-sm
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
              disabled:bg-gray-50 disabled:text-gray-400"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium
              hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? '...' : 'Send'}
          </button>
          <button
            type="button"
            onClick={clearChat}
            className="px-3 py-2.5 border border-gray-300 text-gray-600 rounded-xl text-sm
              hover:bg-gray-50 transition-colors"
          >
            Clear
          </button>
        </form>
      </div>
    </div>
  )
}