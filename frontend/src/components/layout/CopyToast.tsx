import { motion, AnimatePresence } from 'framer-motion'
import { useState, useCallback } from 'react'
import styles from './CopyToast.module.css'

// Simple copy-to-clipboard utility exposed on window
let _show: ((text: string) => void) | null = null

export function useCopyToClipboard() {
  return useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      _show?.(text)
    } catch {
      // fallback
      const ta = document.createElement('textarea')
      ta.value = text
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      _show?.(text)
    }
  }, [])
}

export function CopyToast() {
  const [text, setText] = useState<string | null>(null)

  // Register show handler
  _show = (t: string) => {
    setText(t)
    setTimeout(() => setText(null), 2000)
  }

  return (
    <AnimatePresence>
      {text && (
        <motion.div
          className={styles.toast}
          initial={{ y: 40, opacity: 0, x: '-50%' }}
          animate={{ y: 0, opacity: 1, x: '-50%' }}
          exit={{ y: 40, opacity: 0, x: '-50%' }}
          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        >
          <span className={styles.icon}>✓</span>
          <span className={styles.text}>
            <strong>{text}</strong> copied
          </span>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
