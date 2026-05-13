import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

export default function FileUpload({ onUploadComplete }) {
  const [uploading, setUploading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [error, setError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    setError(null)
    setUploading(true)
    for (const file of acceptedFiles) {
      try {
        const formData = new FormData()
        formData.append('file', file)
        const response = await fetch('/api/v1/ingest', {
          method: 'POST',
          body: formData,
        })
        if (!response.ok) {
          const err = await response.json()
          throw new Error(err.detail || 'Upload failed')
        }
        const result = await response.json()
        setUploadedFiles(prev => [...prev, { name: file.name, ...result }])
        onUploadComplete?.(result)
      } catch (err) {
        setError(err.message)
      }
    }
    setUploading(false)
  }, [onUploadComplete])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
    },
    maxSize: 50 * 1024 * 1024,
  })

  return (
    <div style={{ width: '100%' }}>
      <div
        {...getRootProps()}
        style={{
          border: `2px dashed ${isDragActive ? '#3b82f6' : '#cbd5e1'}`,
          borderRadius: '12px',
          padding: '32px',
          textAlign: 'center',
          cursor: 'pointer',
          background: isDragActive ? '#eff6ff' : '#f8fafc',
          transition: 'all 0.2s',
        }}
      >
        <input {...getInputProps()} />
        <div style={{ fontSize: '40px', marginBottom: '12px' }}>📄</div>
        {uploading ? (
          <p style={{ color: '#3b82f6', fontWeight: 500, margin: 0 }}>
            Uploading and processing...
          </p>
        ) : isDragActive ? (
          <p style={{ color: '#3b82f6', fontWeight: 500, margin: 0 }}>
            Drop files here
          </p>
        ) : (
          <>
            <p style={{ color: '#374151', fontWeight: 500, margin: 0 }}>
              Drag and drop PDF or Excel files here
            </p>
            <p style={{ color: '#9ca3af', fontSize: '13px', marginTop: '6px', marginBottom: 0 }}>
              or click to browse — max 50MB
            </p>
          </>
        )}
      </div>

      {error && (
        <div style={{
          marginTop: '12px', padding: '12px',
          background: '#fef2f2', border: '1px solid #fecaca',
          borderRadius: '8px', color: '#dc2626', fontSize: '13px',
        }}>
          {error}
        </div>
      )}

      {uploadedFiles.length > 0 && (
        <div style={{ marginTop: '16px' }}>
          {uploadedFiles.map((f, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '10px 14px', background: '#f0fdf4',
              border: '1px solid #bbf7d0', borderRadius: '8px',
              fontSize: '13px', marginBottom: '6px',
            }}>
              <span style={{ color: '#16a34a' }}>✓</span>
              <span style={{ fontWeight: 500, color: '#15803d' }}>{f.name}</span>
              <span style={{ color: '#16a34a', marginLeft: 'auto' }}>
                {f.status || 'processing'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}