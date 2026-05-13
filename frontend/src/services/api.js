const BASE_URL = '/api/v1'

export const uploadDocument = async (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${BASE_URL}/ingest`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const err = await response.json()
    throw new Error(err.detail || 'Upload failed')
  }
  return response.json()
}

export const streamQuery = async (question, sessionId, onToken, onSources, onDone, onError) => {
  try {
    const response = await fetch(`${BASE_URL}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        session_id: sessionId,
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
          if (parsed.type === 'token') onToken(parsed.data)
          else if (parsed.type === 'sources') onSources(parsed.data)
          else if (parsed.type === 'done') onDone(parsed.data)
          else if (parsed.type === 'error') onError(parsed.data)
          else if (parsed.type === 'cached') {
            onSources(parsed.data.sources)
            onToken(parsed.data.answer)
            onDone({ cached: true })
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    onError(err.message)
  }
}

export const checkHealth = async () => {
  const r = await fetch('/health')
  return r.json()
}