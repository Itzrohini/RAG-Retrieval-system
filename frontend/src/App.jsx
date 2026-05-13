import { useState } from 'react'
import FileUpload from './components/FileUpload'
import ChatInterface from './components/ChatInterface'

export default function App() {
  const [uploadedDocs, setUploadedDocs] = useState([])
  const [showUpload, setShowUpload] = useState(true)

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">RAG System</h1>
          <p className="text-xs text-gray-500">Hybrid Retrieval · MemPalace · BGE-M3</p>
        </div>
        <div className="flex items-center gap-3">
          {uploadedDocs.length > 0 && (
            <span className="text-xs text-green-700 bg-green-100 px-3 py-1 rounded-full">
              {uploadedDocs.length} doc{uploadedDocs.length > 1 ? 's' : ''} loaded
            </span>
          )}
          <button
            onClick={() => setShowUpload(v => !v)}
            className="text-sm px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            {showUpload ? 'Hide upload' : 'Upload files'}
          </button>
        </div>
      </header>

      {/* Upload panel */}
      {showUpload && (
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <FileUpload onUploadComplete={(doc) => {
            setUploadedDocs(prev => [...prev, doc])
            setShowUpload(false)
          }} />
        </div>
      )}

      {/* Chat — takes remaining height */}
      <div className="flex-1 overflow-hidden">
        <ChatInterface />
      </div>
    </div>
  )
}