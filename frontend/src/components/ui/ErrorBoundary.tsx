import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  label?: string
}

interface State {
  hasError: boolean
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback
      return (
        <div style={{
          padding: 'var(--space-4)',
          color: 'var(--text-muted)',
          fontSize: 'var(--text-xs)',
          fontFamily: 'var(--font-mono)',
          textAlign: 'center',
        }}>
          {this.props.label ?? 'Failed to load'}
        </div>
      )
    }
    return this.props.children
  }
}
