import clsx from 'clsx'
import styles from './Badge.module.css'
import type { ReactNode } from 'react'

interface BadgeProps {
  variant?: 'success' | 'danger' | 'neutral' | 'warning' | 'ghost' | 'gold'
  size?: 'xs' | 'sm'
  children: ReactNode
  className?: string
}

export function Badge({ variant = 'ghost', size = 'xs', children, className }: BadgeProps) {
  return (
    <span className={clsx(styles.badge, styles[variant], styles[size], className)}>
      {children}
    </span>
  )
}
