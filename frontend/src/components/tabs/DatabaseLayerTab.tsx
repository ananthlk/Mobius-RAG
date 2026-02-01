import { useState, useEffect } from 'react'
import { API_BASE } from '../../config'
import './DatabaseLayerTab.css'

interface DatabaseLayerTabProps {
  isActive?: boolean
}

export function DatabaseLayerTab({ isActive = true }: DatabaseLayerTabProps) {
  const [dbTables, setDbTables] = useState<any[]>([])
  const [selectedTable, setSelectedTable] = useState<string | null>(null)
  const [tableSchema, setTableSchema] = useState<any>(null)
  const [tableRecords, setTableRecords] = useState<any[]>([])
  const [tableTotal, setTableTotal] = useState(0)
  const [tablePage, setTablePage] = useState(0)
  const [editingRecord, setEditingRecord] = useState<any>(null)
  const [viewingRecord, setViewingRecord] = useState<any>(null)
  const [sqlQuery, setSqlQuery] = useState('')
  const [sqlResults, setSqlResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const loadDbTables = async () => {
    try {
      const response = await fetch(`${API_BASE}/admin/db/tables`)
      if (response.ok) {
        const data = await response.json()
        setDbTables(data.tables || [])
      }
    } catch (err) {
      setError('Failed to load tables')
    }
  }

  const selectTable = async (tableName: string) => {
    setSelectedTable(tableName)
    setTablePage(0)
    setEditingRecord(null)
    setViewingRecord(null)
    
    try {
      const schemaResponse = await fetch(`${API_BASE}/admin/db/tables/${tableName}/schema`)
      if (schemaResponse.ok) {
        const schemaData = await schemaResponse.json()
        setTableSchema(schemaData)
      }
    } catch (err) {
      console.error('Failed to load schema:', err)
    }
    
    await loadTableRecords(tableName)
  }

  const loadTableRecords = async (tableName: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/admin/db/tables/${tableName}/records?limit=100&offset=${tablePage * 100}`
      )
      if (response.ok) {
        const data = await response.json()
        setTableRecords(data.records || [])
        setTableTotal(data.total || 0)
      }
    } catch (err) {
      setError('Failed to load records')
    }
  }

  useEffect(() => {
    loadDbTables()
  }, [])

  // Refresh table list when tab becomes active
  useEffect(() => {
    if (isActive) {
      loadDbTables()
    }
  }, [isActive])

  useEffect(() => {
    if (selectedTable) {
      loadTableRecords(selectedTable)
    }
  }, [selectedTable, tablePage])

  const handleSaveRecord = async (data: any) => {
    try {
      if (editingRecord.id) {
        const response = await fetch(
          `${API_BASE}/admin/db/tables/${selectedTable}/records/${editingRecord.id}`,
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
          }
        )
        if (!response.ok) throw new Error(await response.text())
      } else {
        const response = await fetch(
          `${API_BASE}/admin/db/tables/${selectedTable}/records`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
          }
        )
        if (!response.ok) throw new Error(await response.text())
      }
      setEditingRecord(null)
      await loadTableRecords(selectedTable!)
      // Refresh table list to update row counts
      await loadDbTables()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save record')
    }
  }

  const handleDeleteRecord = async (recordId: any) => {
    if (!confirm('Delete this record?')) return
    
    try {
      if (selectedTable === 'documents') {
        const response = await fetch(
          `${API_BASE}/admin/db/documents/${recordId}/delete-cascade`,
          { method: 'POST' }
        )
        if (!response.ok) {
          const errorData = await response.json().catch(() => null)
          throw new Error(errorData?.detail || await response.text())
        }
      } else {
        const response = await fetch(
          `${API_BASE}/admin/db/tables/${selectedTable}/records/${recordId}`,
          { method: 'DELETE' }
        )
        if (!response.ok) {
          const errorData = await response.json().catch(() => null)
          throw new Error(errorData?.detail || await response.text())
        }
      }
      await loadTableRecords(selectedTable!)
      // Refresh table list to update row counts
      await loadDbTables()
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete record')
    }
  }

  const formatFieldValue = (value: any, type: string): string => {
    if (value === null || value === undefined) return '—'
    if (type.includes('JSONB')) {
      return JSON.stringify(value, null, 2)
    }
    if (type.includes('DateTime')) {
      return new Date(value).toLocaleString()
    }
    return String(value)
  }

  const formatFieldValueForTable = (value: any): string => {
    if (value === null || value === undefined) return '—'
    const str = String(value)
    return str.length > 50 ? str.substring(0, 50) + '...' : str
  }

  return (
    <div className="database-layer-tab">
      <div className="db-admin-container">
        {/* Left Sidebar - Tables */}
        <div className="db-admin-sidebar">
          <div className="db-sidebar-header">
            <h4>Tables</h4>
            <button
              onClick={loadDbTables}
              className="btn btn-sm btn-secondary"
              title="Refresh table list and counts"
            >
              ↻
            </button>
          </div>
          <div className="db-table-list">
            {dbTables.map((table) => (
              <button
                key={table.name}
                className={`db-table-item ${selectedTable === table.name ? 'active' : ''}`}
                onClick={() => selectTable(table.name)}
              >
                <span>{table.name}</span>
                <span className="db-table-count">{table.row_count}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Main Area */}
        <div className="db-admin-main">
          {selectedTable ? (
            <>
              <div className="db-table-header">
                <h3>{selectedTable}</h3>
                <div className="db-table-actions">
                  <button onClick={() => setEditingRecord({})} className="btn btn-success">
                    + New Record
                  </button>
                </div>
              </div>

              {tableSchema && (
                <div className="db-schema-info">
                  <strong>Schema:</strong>
                  <ul>
                    {tableSchema.columns.map((col: any) => (
                      <li key={col.name}>
                        <code>{col.name}</code> ({col.type})
                        {col.primary_key && ' [PK]'}
                        {col.foreign_key && ` [FK → ${col.foreign_key.table}]`}
                        {!col.nullable && ' NOT NULL'}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {error && (
                <div className="error-message">{error}</div>
              )}

              {editingRecord !== null ? (
                <RecordEditor
                  table={selectedTable}
                  schema={tableSchema}
                  record={editingRecord}
                  onSave={handleSaveRecord}
                  onCancel={() => setEditingRecord(null)}
                />
              ) : viewingRecord ? (
                <RecordViewer
                  table={selectedTable}
                  schema={tableSchema}
                  record={viewingRecord}
                  onClose={() => setViewingRecord(null)}
                  onEdit={() => {
                    setViewingRecord(null)
                    setEditingRecord(viewingRecord)
                  }}
                  onDelete={handleDeleteRecord}
                />
              ) : (
                <>
                  <div className="db-records-table-wrap">
                    <table className="db-records-table">
                      <thead>
                        <tr>
                          {tableSchema?.columns.map((col: any) => (
                            <th key={col.name}>{col.name}</th>
                          ))}
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tableRecords.length === 0 ? (
                          <tr>
                            <td colSpan={(tableSchema?.columns.length || 0) + 1} className="empty-state">
                              No records found
                            </td>
                          </tr>
                        ) : (
                          tableRecords.map((record, idx) => {
                            const pkCol = tableSchema?.columns.find((c: any) => c.primary_key)
                            const recordId = pkCol ? record[pkCol.name] : idx
                            return (
                              <tr key={recordId || idx}>
                                {tableSchema?.columns.map((col: any) => (
                                  <td key={col.name} title={formatFieldValue(record[col.name], col.type)}>
                                    {formatFieldValueForTable(record[col.name])}
                                  </td>
                                ))}
                                <td className="col-actions">
                                  <div className="action-buttons">
                                    <button
                                      onClick={() => setViewingRecord(record)}
                                      className="btn btn-sm btn-secondary"
                                    >
                                      View
                                    </button>
                                    <button
                                      onClick={() => setEditingRecord(record)}
                                      className="btn btn-sm btn-secondary"
                                    >
                                      Edit
                                    </button>
                                    <button
                                      onClick={() => handleDeleteRecord(recordId)}
                                      className="btn btn-sm btn-danger"
                                    >
                                      Delete
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            )
                          })
                        )}
                      </tbody>
                    </table>
                  </div>

                  <div className="db-pagination">
                    <button
                      onClick={() => setTablePage(Math.max(0, tablePage - 1))}
                      disabled={tablePage === 0}
                      className="btn btn-secondary"
                    >
                      Previous
                    </button>
                    <span>
                      Page {tablePage + 1} ({(tablePage * 100) + 1}-{Math.min((tablePage + 1) * 100, tableTotal)} of {tableTotal})
                    </span>
                    <button
                      onClick={() => setTablePage(tablePage + 1)}
                      disabled={(tablePage + 1) * 100 >= tableTotal}
                      className="btn btn-secondary"
                    >
                      Next
                    </button>
                  </div>
                </>
              )}
            </>
          ) : (
            <p className="no-table-selected">Select a table from the left to view records.</p>
          )}

          {/* SQL Executor */}
          <div className="db-sql-executor">
            <h4>Execute SQL</h4>
            <textarea
              value={sqlQuery}
              onChange={(e) => setSqlQuery(e.target.value)}
              placeholder="SELECT * FROM documents LIMIT 10;"
              className="sql-textarea"
            />
            <button
              onClick={async () => {
                try {
                  const response = await fetch(`${API_BASE}/admin/db/execute`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sql: sqlQuery })
                  })
                  if (!response.ok) throw new Error(await response.text())
                  const data = await response.json()
                  setSqlResults(data)
                } catch (err) {
                  setError(err instanceof Error ? err.message : 'Failed to execute SQL')
                }
              }}
              className="btn btn-primary"
            >
              Execute
            </button>
            {sqlResults && (
              <div className="db-sql-results">
                <p><strong>Results:</strong> {sqlResults.count || sqlResults.affected_rows || 0} row(s)</p>
                {sqlResults.records && sqlResults.records.length > 0 && (
                  <div className="db-records-table-wrap">
                    <table className="db-records-table">
                      <thead>
                        <tr>
                          {sqlResults.columns.map((col: string) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sqlResults.records.map((record: any, idx: number) => (
                          <tr key={idx}>
                            {sqlResults.columns.map((col: string) => (
                              <td key={col}>{String(record[col] || '—')}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// Record Editor Component
function RecordEditor({ table: _table, schema, record, onSave, onCancel }: {
  table: string
  schema: any
  record: any
  onSave: (data: any) => Promise<void>
  onCancel: () => void
}) {
  const [formData, setFormData] = useState<any>(record || {})
  const [saving, setSaving] = useState(false)

  return (
    <div className="db-record-editor">
      <h4>{record.id ? 'Edit' : 'Create'} Record</h4>
      <form
        onSubmit={async (e) => {
          e.preventDefault()
          setSaving(true)
          try {
            await onSave(formData)
          } finally {
            setSaving(false)
          }
        }}
      >
        {schema?.columns.map((col: any) => {
          if (col.primary_key && record.id) {
            return (
              <div key={col.name} className="form-field">
                <label>
                  {col.name} (Primary Key)
                </label>
                <input
                  type="text"
                  value={String(formData[col.name] || '')}
                  disabled
                  className="form-input disabled"
                />
              </div>
            )
          }
          return (
            <div key={col.name} className="form-field">
              <label>
                {col.name}
                {!col.nullable && <span className="required"> *</span>}
                {col.foreign_key && <span className="foreign-key"> (FK: {col.foreign_key.table})</span>}
              </label>
              {col.type.includes('TEXT') || col.type.includes('JSONB') ? (
                <textarea
                  value={col.type.includes('JSONB') 
                    ? (typeof formData[col.name] === 'string' ? formData[col.name] : JSON.stringify(formData[col.name] || {}, null, 2))
                    : String(formData[col.name] || '')}
                  onChange={(e) => {
                    let value: any = e.target.value
                    if (col.type.includes('JSONB')) {
                      try {
                        value = JSON.parse(value)
                      } catch {
                        // Keep as string if invalid JSON
                      }
                    }
                    setFormData({ ...formData, [col.name]: value })
                  }}
                  required={!col.nullable}
                  className="form-textarea"
                />
              ) : (
                <input
                  type={col.type.includes('INTEGER') ? 'number' : col.type.includes('DateTime') ? 'datetime-local' : 'text'}
                  value={String(formData[col.name] || '')}
                  onChange={(e) => {
                    let value: any = e.target.value
                    if (col.type.includes('INTEGER')) {
                      value = value ? parseInt(value, 10) : null
                    }
                    setFormData({ ...formData, [col.name]: value })
                  }}
                  required={!col.nullable}
                  className="form-input"
                />
              )}
            </div>
          )
        })}
        <div className="form-actions">
          <button type="submit" className="btn btn-primary" disabled={saving}>
            {saving ? 'Saving...' : 'Save'}
          </button>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
        </div>
      </form>
    </div>
  )
}

// Record Viewer Component
function RecordViewer({ table, schema, record, onClose, onEdit, onDelete }: {
  table: string
  schema: any
  record: any
  onClose: () => void
  onEdit: () => void
  onDelete: (recordId: any) => void
}) {
  const [expandedJsonb, setExpandedJsonb] = useState<Set<string>>(new Set())

  const toggleJsonb = (fieldName: string) => {
    setExpandedJsonb(prev => {
      const next = new Set(prev)
      if (next.has(fieldName)) {
        next.delete(fieldName)
      } else {
        next.add(fieldName)
      }
      return next
    })
  }

  const formatValue = (value: any, type: string): string => {
    if (value === null || value === undefined) return '—'
    if (type.includes('JSONB')) {
      return JSON.stringify(value, null, 2)
    }
    if (type.includes('DateTime')) {
      return new Date(value).toLocaleString()
    }
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No'
    }
    return String(value)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const pkCol = schema?.columns.find((c: any) => c.primary_key)
  const recordId = pkCol ? record[pkCol.name] : null

  return (
    <div className="record-viewer">
      <div className="record-viewer-header">
        <h3>Record Details</h3>
        <button onClick={onClose} className="close-btn">×</button>
      </div>
      <div className="record-viewer-content">
        <div className="record-breadcrumb">
          {table} {recordId && `> ${recordId}`}
        </div>
        <div className="record-fields">
          {schema?.columns.map((col: any) => {
            const value = record[col.name]
            const isJsonb = col.type.includes('JSONB')
            const isExpanded = expandedJsonb.has(col.name)
            
            return (
              <div key={col.name} className="record-field">
                <div className="field-label-row">
                  <label className="field-label">{col.name}</label>
                  <div className="field-actions">
                    {isJsonb && (
                      <button
                        onClick={() => toggleJsonb(col.name)}
                        className="btn-icon"
                      >
                        {isExpanded ? '▼' : '▶'}
                      </button>
                    )}
                    <button
                      onClick={() => copyToClipboard(formatValue(value, col.type))}
                      className="btn-icon"
                      title="Copy to clipboard"
                    >
                      📋
                    </button>
                  </div>
                </div>
                <div className="field-value">
                  {isJsonb ? (
                    <div className="jsonb-field">
                      {isExpanded ? (
                        <pre className="jsonb-content">{formatValue(value, col.type)}</pre>
                      ) : (
                        <div className="jsonb-collapsed">
                          {typeof value === 'object' ? `{${Object.keys(value || {}).length} keys}` : 'Click to expand'}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="field-value-text">{formatValue(value, col.type)}</div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
        <div className="record-viewer-actions">
          <button onClick={onEdit} className="btn btn-primary">
            Edit
          </button>
          <button
            onClick={() => {
              if (confirm('Delete this record?')) {
                onDelete(recordId)
                onClose()
              }
            }}
            className="btn btn-danger"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}
