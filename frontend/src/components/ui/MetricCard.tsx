import clsx from 'clsx'
import styles from './MetricCard.module.css'

interface MetricCardProps {
  label: string
  value: string | number
  delta?: string
  deltaSign?: 'positive' | 'negative' | 'neutral'
  mono?: boolean
  large?: boolean
  className?: string
}

export function MetricCard({
  label,
  value,
  delta,
  deltaSign = 'neutral',
  mono = false,
  large = false,
  className,
}: MetricCardProps) {
  return (
    <div className={clsx(styles.metric, className)}>
      <div className={clsx(styles.value, large && styles.large, mono && styles.mono)}>
        {value}
      </div>
      {delta && (
        <div className={clsx(styles.delta, styles[deltaSign])}>
          {deltaSign === 'positive' && '▲ '}
          {deltaSign === 'negative' && '▼ '}
          {delta}
        </div>
      )}
      <div className={styles.label}>{label}</div>
    </div>
  )
}
