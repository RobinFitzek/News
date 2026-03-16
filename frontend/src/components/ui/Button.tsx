import { motion } from 'framer-motion'
import clsx from 'clsx'
import { Spinner } from './Spinner'
import styles from './Button.module.css'
import type { ReactNode, ButtonHTMLAttributes } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'success'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  children: ReactNode
}

export function Button({
  variant = 'secondary',
  size = 'md',
  loading = false,
  disabled,
  children,
  className,
  type = 'button',
  onClick,
}: ButtonProps) {
  return (
    <motion.button
      whileTap={{ scale: disabled || loading ? 1 : 0.96 }}
      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      className={clsx(styles.btn, styles[variant], styles[size], className)}
      disabled={disabled || loading}
      type={type as 'button' | 'submit' | 'reset'}
      onClick={onClick}
    >
      {loading && <Spinner size="sm" />}
      {children}
    </motion.button>
  )
}
