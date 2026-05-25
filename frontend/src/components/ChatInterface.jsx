import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import SourceViewer from './SourceViewer'
import MemPalaceViewer from './MemPalaceViewer'
import { useChat } from '../hooks/useChat'

const C = {
  bg:'#0D1117',surface:'#161B22',border:'#21262D',
  accent:'#2563EB',accentLt:'#3B82F6',muted:'#6E7681',
  text:'#E6EDF3',textSub:'#8B949E',user:'#1D4ED8',
}

const css = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:#0D1117;color:#E6EDF3;font-family:'IBM Plex Sans',sans-serif;}
  ::-webkit-scrollbar{width:4px;}
  ::-webkit-scrollbar-thumb{background:#21262D;border-radius:2px;}
  @keyframes fadeSlide{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
  @keyframes spin{to{transform:rotate(360deg)}}
  .msg-in{animation:fadeSlide .2s ease both;}
  .cursor::after{content:'▋';animation:blink 1s infinite;font-size:.85em;}
`

function Tag({ children, bg='#1F3A5F', fg='#3B82F6' }) {
  return (
    <span style={{
      display:'inline-flex',alignItems:'center',gap:4,
      padding:'2px 8px',borderRadius:4,background:bg,color:fg,
      fontSize:11,fontWeight:500,fontFamily:"'IBM Plex Mono',monospace",
      border:`1px solid ${fg}22`,letterSpacing:'0.02em',
    }}>{children}</span>
  )
}

function Spinner() {
  return <span style={{
    display:'inline-block',width:13,height:13,
    border:'2px solid #21262D',borderTopColor:'#3B82F6',
    borderRadius:'50%',animation:'spin .7s linear infinite',
  }}/>
}

function MessageMeta({ msg }) {
  if (!msg.latency && !msg.chunksUsed && !msg.cached) return null
  return (
    <div style={{
      display:'flex',flexWrap:'wrap',gap:6,
      marginTop:10,paddingTop:10,
      borderTop:'1px solid #21262D',alignItems:'center',
    }}>
      {msg.cached && <Tag bg='#0D2818' fg='#3FB950'>cached</Tag>}
      {msg.isGeneral && <Tag bg='#1C1438' fg='#A371F7'>general knowledge</Tag>}
      {!msg.isGeneral && msg.chunksUsed > 0 && <Tag>{msg.chunksUsed} chunks</Tag>}
      {msg.threshold !== undefined && !msg.isGeneral && (
        <Tag bg='#1C1208' fg='#D29922'>threshold {(msg.threshold*100).toFixed(0)}%</Tag>
      )}
      {msg.mempalaceData?.from_memory && (
        <Tag bg='#0D2030' fg='#58A6FF'>memory · {msg.mempalaceData.prior_turns} turns</Tag>
      )}
      {msg.latency && (
        <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:'#6E7681',marginLeft:'auto'}}>
          {msg.latency}ms
        </span>
      )}
    </div>
  )
}

const mdComponents = {
  p:({children})=><p style={{margin:'0 0 10px',lineHeight:1.75,color:'#E6EDF3'}}>{children}</p>,
  strong:({children})=><strong style={{fontWeight:600,color:'#F0F6FC'}}>{children}</strong>,
  em:({children})=><em style={{fontStyle:'italic',color:'#8B949E'}}>{children}</em>,
  ul:({children})=><ul style={{margin:'6px 0 10px',paddingLeft:20}}>{children}</ul>,
  ol:({children})=><ol style={{margin:'6px 0 10px',paddingLeft:20}}>{children}</ol>,
  li:({children})=><li style={{margin:'4px 0',lineHeight:1.65,color:'#E6EDF3'}}>{children}</li>,
  code:({inline,children})=>inline
    ?<code style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:'0.85em',padding:'1px 5px',borderRadius:3,background:'#1C2128',color:'#79C0FF',border:'1px solid #21262D'}}>{children}</code>
    :<pre style={{margin:'8px 0',padding:'12px 14px',borderRadius:6,background:'#161B22',border:'1px solid #21262D',overflow:'auto'}}><code style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:13,color:'#79C0FF',lineHeight:1.6}}>{children}</code></pre>,
  blockquote:({children})=><blockquote style={{borderLeft:'3px solid #3B82F6',margin:'8px 0',paddingLeft:14,color:'#8B949E'}}>{children}</blockquote>,
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className="msg-in" style={{
      display:'flex',flexDirection:'column',
      alignItems:isUser?'flex-end':'flex-start',marginBottom:20,
    }}>
      <div style={{
        display:'flex',alignItems:'center',gap:8,marginBottom:6,
        flexDirection:isUser?'row-reverse':'row',
      }}>
        <div style={{
          width:24,height:24,borderRadius:4,
          background:isUser?'#1D4ED8':'#161B22',
          border:`1px solid ${isUser?'#2563EB44':'#21262D'}`,
          display:'flex',alignItems:'center',justifyContent:'center',
          fontSize:10,fontWeight:600,
          color:isUser?'#93C5FD':'#6E7681',
          fontFamily:"'IBM Plex Mono',monospace",
        }}>{isUser?'U':'AI'}</div>
        <span style={{fontSize:11,color:'#6E7681',fontWeight:500}}>
          {isUser?'You':'Assistant'}
        </span>
      </div>
      <div style={{
        maxWidth:'82%',
        padding:isUser?'10px 14px':'14px 16px',
        borderRadius:isUser?'12px 4px 12px 12px':'4px 12px 12px 12px',
        background:isUser?'#1D4ED8':'#161B22',
        border:`1px solid ${isUser?'#2563EB55':'#21262D'}`,
        fontSize:14,lineHeight:1.7,
        color:isUser?'#EFF6FF':'#E6EDF3',
      }}>
        {isUser ? (
          <span>{msg.content}</span>
        ) : (
          <>
            {msg.loading && !msg.content ? (
              <div style={{display:'flex',alignItems:'center',gap:10}}>
                <Spinner/>
                <span style={{fontSize:12,color:'#6E7681'}}>Retrieving and generating…</span>
              </div>
            ) : (
              <div className={msg.loading?'cursor':''}>
                <ReactMarkdown components={mdComponents}>{msg.content||''}</ReactMarkdown>
              </div>
            )}
            {msg.error && (
              <div style={{marginTop:8,padding:'8px 12px',background:'#2D1117',border:'1px solid #F8514944',borderRadius:6,fontSize:12,color:'#F85149'}}>
                {msg.content}
              </div>
            )}
            {!msg.loading && <MessageMeta msg={msg}/>}
          </>
        )}
      </div>
    </div>
  )
}

function EmptyState({ onSend }) {
  const suggestions = ['What is convergent evolution?','Explain Problem 6.10','Summarise Chapter 4']
  const caps = [['Vector Search','HNSW dense'],['Sparse BM42','Qdrant'],['BM25','MongoDB'],['MemPalace','Session memory'],['Reranker','BGE cross-encoder']]
  return (
    <div style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',padding:'40px 24px',gap:28}}>
      <div style={{textAlign:'center'}}>
        <div style={{
          width:52,height:52,borderRadius:12,margin:'0 auto 16px',
          background:'linear-gradient(135deg,#2563EB,#1E40AF)',
          border:'1px solid #2563EB44',
          display:'flex',alignItems:'center',justifyContent:'center',
        }}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M12 3L3 8.5V15.5L12 21L21 15.5V8.5L12 3Z" stroke="#93C5FD" strokeWidth="1.5" fill="none"/>
            <circle cx="12" cy="12" r="3" fill="#93C5FD"/>
          </svg>
        </div>
        <h2 style={{fontSize:20,fontWeight:600,color:'#E6EDF3',letterSpacing:'-0.02em',marginBottom:6}}>RAG Retrieval System</h2>
        <p style={{fontSize:13,color:'#6E7681',lineHeight:1.6}}>
          Hybrid search across your documents.<br/>
          Exact identifiers, semantic queries, and conversational follow-ups.
        </p>
      </div>
      <div style={{display:'flex',gap:8,flexWrap:'wrap',justifyContent:'center'}}>
        {caps.map(([l,s])=>(
          <div key={l} style={{padding:'6px 12px',borderRadius:6,background:'#161B22',border:'1px solid #21262D',textAlign:'center'}}>
            <div style={{fontSize:11,fontWeight:600,color:'#E6EDF3'}}>{l}</div>
            <div style={{fontSize:10,color:'#6E7681',marginTop:1}}>{s}</div>
          </div>
        ))}
      </div>
      <div style={{display:'flex',gap:8,flexWrap:'wrap',justifyContent:'center'}}>
        {suggestions.map(q=>(
          <button key={q} onClick={()=>onSend(q)} style={{
            padding:'7px 14px',borderRadius:6,
            border:'1px solid #21262D',background:'transparent',
            color:'#8B949E',fontSize:12,cursor:'pointer',
            fontFamily:"'IBM Plex Sans',sans-serif",transition:'all .15s',
          }}
          onMouseEnter={e=>{e.target.style.borderColor='#3B82F6';e.target.style.color='#E6EDF3'}}
          onMouseLeave={e=>{e.target.style.borderColor='#21262D';e.target.style.color='#8B949E'}}
          >{q}</button>
        ))}
      </div>
    </div>
  )
}

export default function ChatInterface() {
  const [input,setInput]=useState('')
  const bottomRef=useRef(null)
  const inputRef=useRef(null)
  const {messages,isLoading,sources,sendMessage,clearChat,sessionId}=useChat()

  useEffect(()=>{bottomRef.current?.scrollIntoView({behavior:'smooth'})},[messages])

  const handleSubmit=(e)=>{
    e?.preventDefault()
    if(!input.trim()||isLoading) return
    sendMessage(input.trim())
    setInput('')
    inputRef.current?.focus()
  }

  const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant')
  const showSources = sources.length > 0 && !lastAssistantMsg?.isGeneral

  return (
    <>
      <style>{css}</style>
      <div style={{display:'flex',flexDirection:'column',height:'100%',background:'#0D1117',fontFamily:"'IBM Plex Sans',sans-serif"}}>
        <div style={{flex:1,overflowY:'auto',padding:'24px 0'}}>
          <div style={{maxWidth:760,margin:'0 auto',padding:'0 20px'}}>
            {messages.length===0
              ? <EmptyState onSend={q=>sendMessage(q)}/>
              : messages.map((msg,i)=><Message key={i} msg={msg}/>)
            }

            {/* Sources — only for document-grounded answers */}
            {showSources && (
              <div style={{marginTop:4}}>
                <SourceViewer sources={sources}/>
              </div>
            )}

            {/* MemPalace debug panel — always visible once chat starts */}
            {messages.length > 0 && (
              <MemPalaceViewer sessionId={sessionId} />
            )}

            <div ref={bottomRef}/>
          </div>
        </div>
        <div style={{borderTop:'1px solid #21262D',background:'#161B22',padding:'14px 20px'}}>
          <div style={{maxWidth:760,margin:'0 auto'}}>
            {isLoading && (
              <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:10,fontSize:11,color:'#6E7681',fontFamily:"'IBM Plex Mono',monospace"}}>
                <Spinner/> searching · retrieving · generating
              </div>
            )}
            <form onSubmit={handleSubmit} style={{display:'flex',gap:8}}>
              <input
                ref={inputRef} value={input}
                onChange={e=>setInput(e.target.value)}
                onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey)handleSubmit(e)}}
                placeholder="Ask a question about your documents…"
                disabled={isLoading}
                style={{
                  flex:1,padding:'10px 14px',background:'#0D1117',
                  border:'1px solid #21262D',borderRadius:8,fontSize:14,
                  color:'#E6EDF3',outline:'none',
                  fontFamily:"'IBM Plex Sans',sans-serif",transition:'border-color .15s',
                }}
                onFocus={e=>e.target.style.borderColor='#3B82F6'}
                onBlur={e=>e.target.style.borderColor='#21262D'}
              />
              <button type="submit" disabled={isLoading||!input.trim()} style={{
                padding:'10px 18px',
                background:isLoading||!input.trim()?'#1C2128':'#2563EB',
                color:isLoading||!input.trim()?'#6E7681':'#EFF6FF',
                border:`1px solid ${isLoading||!input.trim()?'#21262D':'#2563EB'}`,
                borderRadius:8,fontSize:13,fontWeight:500,
                cursor:isLoading||!input.trim()?'not-allowed':'pointer',
                fontFamily:"'IBM Plex Sans',sans-serif",whiteSpace:'nowrap',
              }}>Send</button>
              <button type="button" onClick={clearChat} style={{
                padding:'10px 14px',background:'transparent',color:'#6E7681',
                border:'1px solid #21262D',borderRadius:8,fontSize:13,
                cursor:'pointer',fontFamily:"'IBM Plex Sans',sans-serif",
              }}
              onMouseEnter={e=>e.target.style.color='#E6EDF3'}
              onMouseLeave={e=>e.target.style.color='#6E7681'}
              >Clear</button>
            </form>
          </div>
        </div>
      </div>
    </>
  )
}
