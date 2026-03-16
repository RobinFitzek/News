import { motion } from 'framer-motion'
import type { SchedulerState } from '@/types/api'
import styles from './StatusDot.module.css'
import clsx from 'clsx'

interface StatusDotProps {
  status: SchedulerState | 'active' | 'success' | 'danger' | 'warning'
  size?: 'sm' | 'md'
}

const pulseVariants = {
  scanning: {
    scale: [1, 1.5, 1],
    opacity: [1, 0.5, 1],
    transition: { repeat: Infinity, duration: 1.2, ease: 'easeInOut' as const },
  },
  running: {
    scale: [1, 1.2, 1],
    opacity: [1, 0.7, 1],
    transition: { repeat: Infinity, duration: 2, ease: 'easeInOut' as const },
  },
  active: {
    scale: [1, 1.2, 1],
    opacity: [1, 0.7, 1],
    transition: { repeat: Infinity, duration: 2, ease: 'easeInOut' as const },
  },
  success: {
    scale: [1, 1.2, 1],
    opacity: [1, 0.7, 1],
    transition: { repeat: Infinity, duration: 2.5, ease: 'easeInOut' as const },
  },
  stopped: { scale: 1, opacity: 0.5 },
  sleeping: {
    opacity: [1, 0.3, 1],
    transition: { repeat: Infinity, duration: 3, ease: 'easeInOut' as const },
  },
  idle: {
    opacity: [1, 0.4, 1],
    transition: { repeat: Infinity, duration: 3, ease: 'easeInOut' as const },
  },
  danger: { scale: 1, opacity: 0.9 },
  warning: {
    scale: [1, 1.1, 1],
    transition: { repeat: Infinity, duration: 1.5, ease: 'easeInOut' as const },
  },
}

export function StatusDot({ status, size = 'sm' }: StatusDotProps) {
  return (
    <motion.span
      className={clsx(styles.dot, styles[status], size === 'md' && styles.md)}
      animate={pulseVariants[status] ?? { scale: 1 }}
    />
  )
}
