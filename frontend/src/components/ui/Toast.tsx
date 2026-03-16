import { motion, AnimatePresence } from 'framer-motion'
import { useToastStore } from '@/stores/toastStore'
import styles from './Toast.module.css'
import clsx from 'clsx'

export function ToastContainer() {
  const { toasts, removeToast } = useToastStore()

  return (
    <div className={styles.container} aria-live="polite">
      <AnimatePresence initial={false}>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            className={clsx(styles.toast, styles[t.type])}
            initial={{ opacity: 0, x: 40, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 40, scale: 0.9 }}
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            onClick={() => removeToast(t.id)}
          >
            <span className={styles.icon}>
              {t.type === 'success' && '✓'}
              {t.type === 'error'   && '✕'}
              {t.type === 'warning' && '!'}
              {t.type === 'info'    && 'i'}
            </span>
            <span className={styles.message}>{t.message}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
