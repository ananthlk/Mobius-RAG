import { useState } from 'react'
import './DocumentInputTab.css'

interface DocumentInputTabProps {
  onUpload: (file: File) => Promise<void>
  uploading: boolean
  error: string | null
}

export function DocumentInputTab({ onUpload, uploading, error }: DocumentInputTabProps) {
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleUpload = async () => {
    if (file) {
      await onUpload(file)
      setFile(null)
      // Reset file input
      const fileInput = document.getElementById('file-input') as HTMLInputElement
      if (fileInput) {
        fileInput.value = ''
      }
    }
  }

  return (
    <div className="document-input-tab">
      <div className="input-methods">
        {/* Upload Method */}
        <div className="input-method-card active">
          <h3 className="method-title">Upload Document</h3>
          <p className="method-description">Upload a PDF document from your computer</p>
          
          <div
            className={`upload-area ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              onChange={handleFileChange}
              id="file-input"
              accept=".pdf"
              disabled={uploading}
            />
            <label htmlFor="file-input" className="file-label">
              {file ? file.name : 'Choose a file or drag it here'}
            </label>
            {file && (
              <div className="file-info">
                <span className="file-size">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
            )}
            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="btn btn-primary upload-button"
            >
              {uploading ? 'Uploading…' : 'Upload'}
            </button>
          </div>
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {/* Google Drive Method - Coming Soon */}
        <div className="input-method-card disabled">
          <h3 className="method-title">Google Drive</h3>
          <p className="method-description">Import documents from Google Drive</p>
          <div className="coming-soon-badge">Coming Soon</div>
        </div>

        {/* SFTP to GCP Method - Coming Soon */}
        <div className="input-method-card disabled">
          <h3 className="method-title">SFTP to GCP</h3>
          <p className="method-description">Sync documents via SFTP to Google Cloud Storage</p>
          <div className="coming-soon-badge">Coming Soon</div>
        </div>
      </div>
    </div>
  )
}
