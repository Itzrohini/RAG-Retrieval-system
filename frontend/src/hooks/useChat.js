import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [sources, setSources] = useState([])
  const [queryMeta, setQueryMeta] = useState(null)
  const sessionId = useRef(uuidv4())
  const requestStartTime = useRef(null)

  const sendMessage = useCallback(async (question) => {
    if (!question.trim() || isLoading) return

    setMessages(prev => [...prev, { role: 'user', content: question }])
    setIsLoading(true)
    setSources([])   // always clear sources at the start of a new question
    setQueryMeta(null)
    setMessages(prev => [...prev, {
      role: 'assistant', content: '', loading: true
    }])

    requestStartTime.current = performance.now()

    try {
      const response = await fetch('/api/v1/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          session_id: sessionId.current,
          top_k: 5,
          use_cache: true,
        }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const lines = decoder.decode(value).split('\n')
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const parsed = JSON.parse(line.slice(6))

            if (parsed.type === 'token') {
              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                updated[updated.length - 1] = {
                  ...last,
                  content: last.content + parsed.data
                }
                return updated
              })

            } else if (parsed.type === 'sources') {
              // Backend only sends this event when is_general=False and
              // relevant chunks exist — so we can safely set them here.
              setSources(parsed.data)

            } else if (parsed.type === 'mempalace') {
              setQueryMeta(prev => ({ ...prev, mempalace: parsed.data }))

            } else if (parsed.type === 'done') {
              setIsLoading(false)
              setQueryMeta(parsed.data)

              // If the final answer is general knowledge, clear any stale
              // sources that might have been set from a previous question.
              if (parsed.data?.is_general) {
                setSources([])
              }

              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  loading: false,
                  latency: parsed.data?.latency_ms,
                  isGeneral: parsed.data?.is_general,
                  chunksUsed: parsed.data?.chunks_used,
                  threshold: parsed.data?.threshold,
                  mempalaceData: parsed.data?.mempalace,
                }
                return updated
              })

            } else if (parsed.type === 'cached') {
              const realLatency = Math.round(performance.now() - requestStartTime.current)

              // Only show cached sources when the cached answer was NOT general.
              // is_general is now stored in the cache on the backend.
              const cachedIsGeneral = parsed.data.is_general ?? false
              const cachedSources = cachedIsGeneral ? [] : (parsed.data.sources || [])

              setSources(cachedSources)

              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: parsed.data.answer,
                  loading: false,
                  cached: true,
                  latency: realLatency,
                  isGeneral: cachedIsGeneral,
                  chunksUsed: cachedSources.length,
                }
                return updated
              })
              setIsLoading(false)

            } else if (parsed.type === 'error') {
              throw new Error(parsed.data)
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      setIsLoading(false)
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: `Error: ${err.message}`,
          loading: false,
          error: true,
        }
        return updated
      })
    }
  }, [isLoading])

  const clearChat = useCallback(() => {
    setMessages([])
    setSources([])
    setQueryMeta(null)
    sessionId.current = uuidv4()
  }, [])

  return {
    messages, isLoading, sources, queryMeta,
    sendMessage, clearChat,
    sessionId: sessionId.current
  }
}