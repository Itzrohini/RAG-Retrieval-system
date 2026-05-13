import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { streamQuery } from '../services/api'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [sources, setSources] = useState([])
  const sessionId = useRef(uuidv4())

  const sendMessage = useCallback(async (question) => {
    if (!question.trim() || isLoading) return

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setIsLoading(true)
    setSources([])

    // Add empty assistant message that we'll fill token by token
    setMessages(prev => [...prev, { role: 'assistant', content: '', loading: true }])

    await streamQuery(
      question,
      sessionId.current,
      // onToken — append each token to the last message
      (token) => {
        setMessages(prev => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          updated[updated.length - 1] = { ...last, content: last.content + token }
          return updated
        })
      },
      // onSources
      (newSources) => setSources(newSources),
      // onDone
      (meta) => {
        setIsLoading(false)
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            loading: false,
            cached: meta?.cached,
            latency: meta?.latency_ms,
          }
          return updated
        })
      },
      // onError
      (err) => {
        setIsLoading(false)
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: `Error: ${err}`,
            loading: false,
            error: true,
          }
          return updated
        })
      },
    )
  }, [isLoading])

  const clearChat = useCallback(() => {
    setMessages([])
    setSources([])
    sessionId.current = uuidv4()
  }, [])

  return { messages, isLoading, sources, sendMessage, clearChat, sessionId: sessionId.current }
}