import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

describe('App', () => {
  it('renders Mobius RAG header and upload section', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: /mobius rag/i })).toBeInTheDocument()
    expect(screen.getByText(/upload a document to extract eligibility rules/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /upload/i })).toBeInTheDocument()
  })

  it('has collapsible section for Upload', () => {
    render(<App />)
    expect(screen.getByText(/1\. upload/i)).toBeInTheDocument()
  })
})
