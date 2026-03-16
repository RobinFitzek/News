import { Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Luminary } from './Luminary'
import { Sidebar } from './Sidebar'
import { CopyToast } from './CopyToast'
import { KbdOverlay } from './KbdOverlay'
import { ToastContainer } from '@/components/ui/Toast'
import { useThemeStore } from '@/stores/themeStore'
import styles from './RootLayout.module.css'

export function RootLayout() {
  const location = useLocation()
  const sidebarExpanded = useThemeStore(s => s.sidebarExpanded)

  return (
    <div className={styles.root}>
      {/* Atmospheric layers */}
      <Luminary />
      <div className={styles.shell} aria-hidden="true" />

      {/* Sidebar navigation */}
      <Sidebar />

      {/* Main content — offset by sidebar width */}
      <motion.div
        className={styles.body}
        animate={{ marginLeft: sidebarExpanded ? 220 : 56 }}
        transition={{ type: 'spring', stiffness: 320, damping: 32 }}
      >
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
      </motion.div>

      {/* Global overlays */}
      <ToastContainer />
      <CopyToast />
      <KbdOverlay />
    </div>
  )
}
