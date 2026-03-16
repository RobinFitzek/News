import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Modal } from '@/components/ui/Modal'
import { Kbd } from '@/components/ui/Kbd'
import styles from './KbdOverlay.module.css'

const shortcuts = [
  { keys: ['g', 'd'], label: 'Dashboard',     to: '/' },
  { keys: ['g', 'w'], label: 'Watchlist',     to: '/watchlist' },
  { keys: ['g', 'p'], label: 'Portfolio',     to: '/portfolio' },
  { keys: ['g', 'a'], label: 'Analyze Stock', to: '/analyze' },
  { keys: ['g', 's'], label: 'Settings',      to: '/settings' },
  { keys: ['g', 'l'], label: 'Learning',      to: '/learning' },
]

export function KbdOverlay() {
  const [open, setOpen] = useState(false)
  const [pending, setPending] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>

    const handler = (e: KeyboardEvent) => {
      // Ignore when typing in inputs
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) return

      if (e.key === '?') { setOpen(o => !o); return }
      if (e.key === 'Escape') { setOpen(false); setPending(null); return }

      if (pending === 'g') {
        const shortcut = shortcuts.find(s => s.keys[1] === e.key)
        if (shortcut) { navigate(shortcut.to) }
        setPending(null)
        clearTimeout(timer)
        return
      }

      if (e.key === 'g') {
        setPending('g')
        timer = setTimeout(() => setPending(null), 1500)
      }
    }

    document.addEventListener('keydown', handler)
    return () => {
      document.removeEventListener('keydown', handler)
      clearTimeout(timer)
    }
  }, [pending, navigate])

  return (
    <>
      {/* Pending key indicator */}
      <AnimatePresence>
        {pending && (
          <motion.div
            className={styles.pendingKey}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
          >
            <Kbd>{pending}</Kbd>
            <span className={styles.pendingHint}>then…</span>
          </motion.div>
        )}
      </AnimatePresence>

      <Modal open={open} onClose={() => setOpen(false)} title="Keyboard Shortcuts" size="sm">
        <div className={styles.list}>
          {shortcuts.map(s => (
            <div key={s.keys.join('')} className={styles.row}>
              <div className={styles.keys}>
                {s.keys.map((k, i) => (
                  <span key={k} className={styles.keyWrap}>
                    <Kbd>{k}</Kbd>
                    {i < s.keys.length - 1 && <span className={styles.then}>then</span>}
                  </span>
                ))}
              </div>
              <span className={styles.keyLabel}>{s.label}</span>
            </div>
          ))}
          <div className={styles.divider} />
          <div className={styles.row}>
            <Kbd>?</Kbd>
            <span className={styles.keyLabel}>Toggle this overlay</span>
          </div>
          <div className={styles.row}>
            <Kbd>Esc</Kbd>
            <span className={styles.keyLabel}>Close / Cancel</span>
          </div>
        </div>
      </Modal>
    </>
  )
}
