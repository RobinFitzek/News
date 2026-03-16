import { motion } from 'framer-motion'
import clsx from 'clsx'
import styles from './ProgressBar.module.css'

interface ProgressBarProps {
  value: number  // 0-100
  variant?: 'default' | 'success' | 'danger' | 'warning'
  height?: number
  className?: string
  animated?: boolean
}

export function ProgressBar({
  value,
  variant = 'default',
  height = 2,
  className,
  animated = true,
}: ProgressBarProps) {
  const clamped = Math.min(100, Math.max(0, value))

  return (
    <div
      className={clsx(styles.track, className)}
      style={{ height }}
      role="progressbar"
      aria-valuenow={clamped}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <motion.div
        className={clsx(styles.fill, styles[variant])}
        initial={animated ? { scaleX: 0 } : false}
        animate={{ scaleX: clamped / 100 }}
        style={{ transformOrigin: 'left' }}
        transition={{ duration: 0.6, ease: [0.34, 1.2, 0.64, 1] }}
      />
    </div>
  )
}
