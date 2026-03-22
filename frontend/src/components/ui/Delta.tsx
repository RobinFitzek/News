import clsx from 'clsx'
import styles from './Delta.module.css'

interface DeltaProps {
  value: number | string
  sign?: 'positive' | 'negative' | 'neutral'
  showArrow?: boolean
  className?: string
}

function ArrowUp() {
  return (
    <svg
      width="8"
      height="10"
      viewBox="0 0 8 10"
      className={styles.arrow}
      aria-hidden="true"
    >
      <polygon points="0,7 4,1 8,7" fill="currentColor" />
      <rect x="3" y="6" width="2" height="4" fill="currentColor" />
    </svg>
  )
}

function ArrowDown() {
  return (
    <svg
      width="8"
      height="10"
      viewBox="0 0 8 10"
      className={styles.arrow}
      aria-hidden="true"
    >
      <polygon points="0,3 4,9 8,3" fill="currentColor" />
      <rect x="3" y="0" width="2" height="4" fill="currentColor" />
    </svg>
  )
}

export function Delta({ value, sign, showArrow = true, className }: DeltaProps) {
  const resolved = sign ?? (
    typeof value === 'number'
      ? value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral'
      : 'neutral'
  )

  return (
    <span className={clsx(styles.delta, styles[resolved], className)}>
      {showArrow && resolved === 'positive' && <ArrowUp />}
      {showArrow && resolved === 'negative' && <ArrowDown />}
      {value}
    </span>
  )
}
