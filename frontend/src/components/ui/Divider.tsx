import styles from './Divider.module.css'
import clsx from 'clsx'

interface DividerProps {
  vertical?: boolean
  className?: string
}

export function Divider({ vertical = false, className }: DividerProps) {
  return <div className={clsx(styles.divider, vertical && styles.vertical, className)} />
}
