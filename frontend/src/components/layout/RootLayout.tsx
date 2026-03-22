import { Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect } from 'react'
import { Luminary } from './Luminary'
import { RadianceProvider } from './RadianceProvider'
import { Sidebar } from './Sidebar'
import { CopyToast } from './CopyToast'
import { KbdOverlay } from './KbdOverlay'
import { ToastContainer } from '@/components/ui/Toast'
import { MercuryLoading } from '@/components/ui/MercuryLoading'
import { useThemeStore } from '@/stores/themeStore'
import { useThemeInit } from '@/hooks/useTheme'
import styles from './RootLayout.module.css'

export function RootLayout() {
  useThemeInit()
  const location = useLocation()
  const sidebarExpanded = useThemeStore(s => s.sidebarExpanded)
  const showLoadingScreen = useThemeStore(s => s.showLoadingScreen)
  const [appReady, setAppReady] = useState(false)
  const [loadingDone, setLoadingDone] = useState(() => !showLoadingScreen)

  // Mark app as ready after a minimum display time for the loading experience
  useEffect(() => {
    if (!showLoadingScreen) return
    const timer = setTimeout(() => setAppReady(true), 2800)
    return () => clearTimeout(timer)
  }, [showLoadingScreen])

  return (
    <div className={styles.root}>
      {/* Mercury diffusion loading screen — first visit only */}
      {!loadingDone && (
        <MercuryLoading ready={appReady} onDone={() => setLoadingDone(true)} />
      )}

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
            <RadianceProvider>
              <div className="main-container">
                <Outlet />
              </div>
            </RadianceProvider>
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
