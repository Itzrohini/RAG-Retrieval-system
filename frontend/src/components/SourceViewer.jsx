export default function SourceViewer({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <div style={{ marginTop: '16px' }}>
      <p style={{
        fontSize: '11px', fontWeight: 600, color: '#6b7280',
        textTransform: 'uppercase', letterSpacing: '0.05em',
        marginBottom: '8px', margin: '0 0 8px 0',
      }}>
        Sources used
      </p>
      {sources.map((s, i) => (
        <div key={i} style={{
          padding: '10px 14px', background: '#f9fafb',
          border: '1px solid #e5e7eb', borderRadius: '8px',
          fontSize: '12px', marginBottom: '6px',
        }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            marginBottom: '4px',
          }}>
            <span style={{ fontWeight: 600, color: '#374151' }}>
              {s.file_name}
              {s.page_number ? ` · Page ${s.page_number}` : ''}
              {s.sheet_name ? ` · Sheet: ${s.sheet_name}` : ''}
            </span>
            <span style={{ color: '#3b82f6', fontFamily: 'monospace' }}>
              {(s.score * 100).toFixed(1)}%
            </span>
          </div>
          <p style={{ color: '#6b7280', margin: 0, lineHeight: 1.5 }}>
            {s.text ? s.text.slice(0, 200) + '...' : ''}
          </p>
        </div>
      ))}
    </div>
  )
}