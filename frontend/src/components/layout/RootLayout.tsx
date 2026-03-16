import { Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Luminary } from './Luminary'
import { Navbar } from './Navbar'
import { CopyToast } from './CopyToast'
import { KbdOverlay } from './KbdOverlay'
import { ToastContainer } from '@/components/ui/Toast'
import styles from './RootLayout.module.css'

export function RootLayout() {
  const location = useLocation()

  return (
    <div className={styles.root}>
      {/* Atmospheric layers */}
      <Luminary />
      <div className={styles.shell} aria-hidden="true" />

      {/* Navigation */}
      <Navbar />

      {/* Page content with transitions */}
      <AnimatePresence mode="wait" initial={false}>
        <motion.main
          key={location.pathname}
          className={styles.main}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{
            duration: 0.25,
            ease: [0.34, 1.2, 0.64, 1],
          }}
        >
          <div className="main-container">
            <Outlet />
          </div>
        </motion.main>
      </AnimatePresence>

      {/* Global overlays */}
      <ToastContainer />
      <CopyToast />
      <KbdOverlay />
    </div>
  )
}
