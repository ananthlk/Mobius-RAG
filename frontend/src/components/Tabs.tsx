import type { ReactNode } from 'react'
import './Tabs.css'

interface TabsProps {
  activeTab: string
  onTabChange: (tabId: string) => void
  children: ReactNode
}

interface TabListProps {
  children: ReactNode
}

interface TabProps {
  id: string
  isActive: boolean
  onClick: () => void
  children: ReactNode
}

interface TabPanelsProps {
  children: ReactNode
}

interface TabPanelProps {
  id: string
  isActive: boolean
  children: ReactNode
}

export function Tabs({ children }: TabsProps) {
  return (
    <div className="tabs-container">
      {children}
    </div>
  )
}

export function TabList({ children }: TabListProps) {
  return (
    <div className="tab-list">
      {children}
    </div>
  )
}

export function Tab({ id, isActive, onClick, children }: TabProps) {
  return (
    <button
      className={`tab ${isActive ? 'tab-active' : ''}`}
      onClick={onClick}
      role="tab"
      aria-selected={isActive}
      aria-controls={`panel-${id}`}
      id={`tab-${id}`}
    >
      {children}
    </button>
  )
}

export function TabPanels({ children }: TabPanelsProps) {
  return (
    <div className="tab-panels">
      {children}
    </div>
  )
}

export function TabPanel({ id, isActive, children }: TabPanelProps) {
  return (
    <div
      className={`tab-panel ${isActive ? 'tab-panel-active' : ''}`}
      id={`panel-${id}`}
      role="tabpanel"
      aria-labelledby={`tab-${id}`}
      hidden={!isActive}
    >
      {children}
    </div>
  )
}
