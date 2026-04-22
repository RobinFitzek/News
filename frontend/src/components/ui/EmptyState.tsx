import type { ReactNode } from 'react'
import styles from './EmptyState.module.css'

interface EmptyStateProps {
  message: string
  hint?: string
  action?: ReactNode
}

export function EmptyState({ message, hint, action }: EmptyStateProps) {
  return (
    <div className={styles.container}>
      <p className={styles.message}>{message}</p>
      {hint && <p className={styles.hint}>{hint}</p>}
      {action && <div className={styles.action}>{action}</div>}
    </div>
  )
}
