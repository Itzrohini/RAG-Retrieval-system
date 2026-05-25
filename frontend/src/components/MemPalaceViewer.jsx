import { useState, useEffect } from 'react'

/**
 * MemPalaceViewer — shows the live graph state for a session.
 *
 * Rooms   = topic clusters (e.g. "activity 3.13", "magnesium oxide")
 * Drawers = entities within rooms (e.g. "ionic bond", "nacl")
 * Items   = chunk IDs stored in drawers (actual PDF chunks retrieved)
 */

function Badge({ children, bg = '#1F3A5F', fg = '#3B82F6' }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      padding: '1px 7px', borderRadius: 4,
      background: bg, color: fg,
      fontSize: 10, fontWeight: 600,
      fontFamily: "'IBM Plex Mono', monospace",
      border: `1px solid ${fg}22`,
    }}>
      {children}
    </span>
  )
}

function SectionHeader({ icon, label, count, color = '#3B82F6' }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      marginBottom: 8,
    }}>
      <span style={{ fontSize: 14 }}>{icon}</span>
      <span style={{
        fontSize: 11, fontWeight: 600, color: '#8B949E',
        textTransform: 'uppercase', letterSpacing: '0.06em',
        fontFamily: "'IBM Plex Mono', monospace",
      }}>
        {label}
      </span>
      <span style={{
        marginLeft: 'auto',
        fontSize: 10, fontWeight: 700,
        color, fontFamily: "'IBM Plex Mono', monospace",
      }}>
        {count}
      </span>
    </div>
  )
}

function RoomCard({ room }) {
  return (
    <div style={{
      padding: '8px 12px',
      background: '#0D2030',
      border: '1px solid #1F3A5F',
      borderRadius: 6, marginBottom: 6,
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <span style={{
        fontSize: 12, color: '#93C5FD',
        fontFamily: "'IBM Plex Mono', monospace",
        fontWeight: 500,
      }}>
        🏛 {room.label}
      </span>
      <Badge bg='#0D2030' fg='#58A6FF'>
        {room.visits} visit{room.visits !== 1 ? 's' : ''}
      </Badge>
    </div>
  )
}

function DrawerCard({ drawer }) {
  return (
    <div style={{
      padding: '6px 12px',
      background: '#161B22',
      border: '1px solid #21262D',
      borderRadius: 6, marginBottom: 4,
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <span style={{ fontSize: 12, color: '#E6EDF3' }}>
         {drawer.label}
      </span>
      <Badge bg='#1C1438' fg='#A371F7'>
        ×{drawer.frequency}
      </Badge>
    </div>
  )
}

function ItemCard({ item }) {
  const short = item.chunk_id.slice(0, 16) + '…'
  return (
    <div style={{
      padding: '5px 10px',
      background: '#0D1117',
      border: '1px solid #21262D',
      borderRadius: 6, marginBottom: 4,
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <span style={{
        fontSize: 11, color: '#6E7681',
        fontFamily: "'IBM Plex Mono', monospace",
      }}>
         {short}
      </span>
      <Badge bg='#0D2818' fg='#3FB950'>
        {item.hits} hit{item.hits !== 1 ? 's' : ''}
      </Badge>
    </div>
  )
}

function EpisodeCard({ ep, index }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{
      border: '1px solid #21262D',
      borderRadius: 6, marginBottom: 6, overflow: 'hidden',
    }}>
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          padding: '8px 12px', cursor: 'pointer',
          background: '#161B22',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}
      >
        <span style={{ fontSize: 12, color: '#E6EDF3' }}>
          Turn {index + 1}: {ep.question.slice(0, 50)}{ep.question.length > 50 ? '…' : ''}
        </span>
        <span style={{ fontSize: 10, color: '#6E7681' }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div style={{ padding: '10px 12px', background: '#0D1117' }}>
          <div style={{ marginBottom: 6 }}>
            <span style={{
              fontSize: 10, color: '#6E7681',
              textTransform: 'uppercase', letterSpacing: '0.05em',
              fontFamily: "'IBM Plex Mono', monospace",
            }}>
              Entities extracted
            </span>
            <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {ep.entities.length > 0
                ? ep.entities.map((e, i) => (
                    <Badge key={i} bg='#1C1208' fg='#D29922'>{e}</Badge>
                  ))
                : <span style={{ fontSize: 11, color: '#6E7681' }}>none</span>
              }
            </div>
          </div>
          <div>
            <span style={{
              fontSize: 10, color: '#6E7681',
              textTransform: 'uppercase', letterSpacing: '0.05em',
              fontFamily: "'IBM Plex Mono', monospace",
            }}>
              Chunks retrieved
            </span>
            <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {ep.chunk_ids.length > 0
                ? ep.chunk_ids.map((id, i) => (
                    <Badge key={i} bg='#0D2818' fg='#3FB950'>
                      {id.slice(0, 12)}…
                    </Badge>
                  ))
                : <span style={{ fontSize: 11, color: '#6E7681' }}>none</span>
              }
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function MemPalaceViewer({ sessionId }) {
  const [open, setOpen] = useState(false)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('rooms') // rooms | drawers | items | episodes

  const fetch_data = async () => {
    if (!sessionId) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`/api/v1/memory/${sessionId}`)
      const json = await res.json()
      if (json.ok) setData(json)
      else setError(json.error)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Refresh when panel opens
  useEffect(() => {
    if (open) fetch_data()
  }, [open, sessionId])

  const tabs = [
    { id: 'rooms',    label: ' Rooms',    count: data?.graph?.rooms?.length ?? 0 },
    { id: 'drawers',  label: ' Drawers',  count: data?.graph?.drawers?.length ?? 0 },
    { id: 'items',    label: ' Items',    count: data?.graph?.items?.length ?? 0 },
    { id: 'episodes', label: ' Episodes', count: data?.episodes?.length ?? 0 },
  ]

  return (
    <div style={{ marginTop: 12 }}>
      {/* Toggle button */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 12px',
          background: open ? '#161B22' : 'transparent',
          border: '1px solid #21262D',
          borderRadius: 6, cursor: 'pointer',
          color: '#6E7681', fontSize: 11, fontWeight: 600,
          fontFamily: "'IBM Plex Mono', monospace",
          letterSpacing: '0.03em',
          transition: 'all .15s',
          width: '100%', justifyContent: 'space-between',
        }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = '#3B82F6'; e.currentTarget.style.color = '#E6EDF3' }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = '#21262D'; e.currentTarget.style.color = '#6E7681' }}
      >
        <span> MemPalace — session memory graph</span>
        <span style={{ fontSize: 10 }}>{open ? '▲ hide' : '▼ inspect'}</span>
      </button>

      {/* Panel */}
      {open && (
        <div style={{
          marginTop: 6,
          background: '#161B22',
          border: '1px solid #21262D',
          borderRadius: 8,
          overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            padding: '10px 14px',
            borderBottom: '1px solid #21262D',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {data && (
                <>
                  <Badge bg='#1F3A5F' fg='#3B82F6'>
                    {data.turns} turn{data.turns !== 1 ? 's' : ''}
                  </Badge>
                  <Badge bg='#0D2030' fg='#58A6FF'>
                    {data.graph?.total_nodes ?? 0} nodes
                  </Badge>
                  <Badge bg='#1C1208' fg='#D29922'>
                    {data.graph?.total_edges ?? 0} edges
                  </Badge>
                </>
              )}
            </div>
            <button
              onClick={fetch_data}
              disabled={loading}
              style={{
                padding: '3px 10px', fontSize: 11,
                background: 'transparent', color: '#6E7681',
                border: '1px solid #21262D', borderRadius: 4,
                cursor: loading ? 'not-allowed' : 'pointer',
                fontFamily: "'IBM Plex Mono', monospace",
              }}
            >
              {loading ? '…' : '↻ refresh'}
            </button>
          </div>

          {/* Tabs */}
          <div style={{
            display: 'flex', borderBottom: '1px solid #21262D',
            background: '#0D1117',
          }}>
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  flex: 1, padding: '8px 4px',
                  background: activeTab === tab.id ? '#161B22' : 'transparent',
                  border: 'none',
                  borderBottom: activeTab === tab.id ? '2px solid #3B82F6' : '2px solid transparent',
                  color: activeTab === tab.id ? '#E6EDF3' : '#6E7681',
                  fontSize: 11, cursor: 'pointer',
                  fontFamily: "'IBM Plex Mono', monospace",
                  transition: 'all .15s',
                }}
              >
                {tab.label}
                <span style={{
                  marginLeft: 4, fontSize: 10,
                  color: activeTab === tab.id ? '#3B82F6' : '#6E7681',
                }}>
                  ({tab.count})
                </span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div style={{ padding: '12px 14px', maxHeight: 320, overflowY: 'auto' }}>
            {error && (
              <div style={{
                padding: '8px 12px', background: '#2D1117',
                border: '1px solid #F8514944', borderRadius: 6,
                fontSize: 12, color: '#F85149',
              }}>
                {error}
              </div>
            )}

            {!loading && !error && data && (
              <>
                {activeTab === 'rooms' && (
                  <>
                    <SectionHeader icon='🏛' label='Rooms — topic clusters' count={data.graph.rooms.length} color='#58A6FF' />
                    {data.graph.rooms.length === 0
                      ? <p style={{ fontSize: 12, color: '#6E7681' }}>No rooms yet — ask a question first.</p>
                      : data.graph.rooms.map((r, i) => <RoomCard key={i} room={r} />)
                    }
                  </>
                )}

                {activeTab === 'drawers' && (
                  <>
                    <SectionHeader icon='🗂' label='Drawers — entities' count={data.graph.drawers.length} color='#A371F7' />
                    {data.graph.drawers.length === 0
                      ? <p style={{ fontSize: 12, color: '#6E7681' }}>No entities extracted yet.</p>
                      : data.graph.drawers.map((d, i) => <DrawerCard key={i} drawer={d} />)
                    }
                  </>
                )}

                {activeTab === 'items' && (
                  <>
                    <SectionHeader icon='📄' label='Items — retrieved chunks' count={data.graph.items.length} color='#3FB950' />
                    {data.graph.items.length === 0
                      ? <p style={{ fontSize: 12, color: '#6E7681' }}>No chunks stored yet.</p>
                      : data.graph.items.map((item, i) => <ItemCard key={i} item={item} />)
                    }
                  </>
                )}

                {activeTab === 'episodes' && (
                  <>
                    <SectionHeader icon=' ' label='Recent episodes' count={data.episodes.length} color='#D29922' />
                    {data.episodes.length === 0
                      ? <p style={{ fontSize: 12, color: '#6E7681' }}>No episodes yet.</p>
                      : [...data.episodes].reverse().map((ep, i) => (
                          <EpisodeCard key={i} ep={ep} index={data.episodes.length - 1 - i} />
                        ))
                    }
                  </>
                )}
              </>
            )}

            {loading && (
              <p style={{ fontSize: 12, color: '#6E7681', textAlign: 'center', padding: '20px 0' }}>
                Loading graph…
              </p>
            )}
          </div>

          {/* Footer */}
          <div style={{
            padding: '8px 14px',
            borderTop: '1px solid #21262D',
            fontSize: 10, color: '#6E7681',
            fontFamily: "'IBM Plex Mono', monospace",
            background: '#0D1117',
          }}>
            session: {sessionId?.slice(0, 20)}…
          </div>
        </div>
      )}
    </div>
  )
}