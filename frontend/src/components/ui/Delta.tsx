import clsx from 'clsx'
import styles from './Delta.module.css'

interface DeltaProps {
  value: number | string
  sign?: 'positive' | 'negative' | 'neutral'
  showArrow?: boolean
  className?: string
}

export function Delta({ value, sign, showArrow = true, className }: DeltaProps) {
  const resolved = sign ?? (
    typeof value === 'number'
      ? value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral'
      : 'neutral'
  )

  return (
    <span className={clsx(styles.delta, styles[resolved], className)}>
      {showArrow && resolved === 'positive' && '▲ '}
      {showArrow && resolved === 'negative' && '▼ '}
      {value}
    </span>
  )
}
