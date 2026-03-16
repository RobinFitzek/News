import { useState, useRef, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './NavGroup.module.css'
import clsx from 'clsx'
import type { ReactNode } from 'react'

export interface NavItem {
  to?: string
  label: string
  onClick?: () => void
  danger?: boolean
  divider?: boolean
}

interface NavGroupProps {
  label: string
  items: NavItem[]
  icon?: ReactNode
}

const dropdownVariants = {
  open: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.18, ease: [0.34, 1.2, 0.64, 1] as const },
  },
  closed: {
    opacity: 0,
    y: -6,
    transition: { duration: 0.12, ease: [0.4, 0, 1, 1] as const },
  },
}

export function NavGroup({ label, items }: NavGroupProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const location = useLocation()

  const isActive = items.some(
    item => item.to && location.pathname.startsWith(item.to) && item.to !== '/'
  )

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Close on route change
  useEffect(() => { setOpen(false) }, [location.pathname])

  return (
    <div
      ref={ref}
      className={styles.group}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        className={clsx(styles.trigger, isActive && styles.active)}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        {label}
        <svg
          width="10" height="6" viewBox="0 0 10 6" fill="none"
          className={clsx(styles.chevron, open && styles.chevronOpen)}
        >
          <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className={styles.dropdown}
            variants={dropdownVariants}
            initial="closed"
            animate="open"
            exit="closed"
          >
            {items.map((item, i) => {
              if (item.divider) return <div key={i} className={styles.divider} />

              if (item.onClick) {
                return (
                  <button
                    key={item.label}
                    className={clsx(styles.item, item.danger && styles.danger)}
                    onClick={() => { item.onClick?.(); setOpen(false) }}
                  >
                    {item.label}
                  </button>
                )
              }

              return (
                <Link
                  key={item.label}
                  to={item.to!}
                  className={clsx(
                    styles.item,
                    item.danger && styles.danger,
                    location.pathname === item.to && styles.activeItem
                  )}
                  onClick={() => setOpen(false)}
                >
                  {item.label}
                </Link>
              )
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
