import { NavLink, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useMutation } from '@tanstack/react-query'
import { useThemeStore } from '@/stores/themeStore'
import { queryClient } from '@/api/queryClient'
import { StatusPill } from './StatusPill'
import api from '@/api/client'
import styles from './Sidebar.module.css'

// ── SVG icons ─────────────────────────────────────
function Icon({ d, d2 }: { d: string; d2?: string }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true" className={styles.icon}>
      <path d={d} stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      {d2 && <path d={d2} stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />}
    </svg>
  )
}

const ICONS = {
  dashboard: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  watchlist: 'M2 12C2 6.477 6.477 2 12 2s10 4.477 10 10-4.477 10-10 10S2 17.523 2 12zm10-4v4l3 3',
  discover: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm3.5 6.5-5 5m0-5 5 5',
  analyze: 'M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18',
  discoveries: 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z',
  topPicks: 'M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01',
  portfolio: 'M21 12V7a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7',
  paperTrading: 'M9 19V6l12-3v13M9 19c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm12-3c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2z',
  insider: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2m16-7 2 2 4-4M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
  trust: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
  learning: 'M12 2a7 7 0 0 1 7 7c0 3.87-3.13 7-7 7s-7-3.13-7-7a7 7 0 0 1 7-7zm0 16v4m-4 0h8',
  crosscheck: 'M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 0 0 1.946-.806 3.42 3.42 0 0 1 4.438 0 3.42 3.42 0 0 0 1.946.806 3.42 3.42 0 0 1 3.138 3.138 3.42 3.42 0 0 0 .806 1.946 3.42 3.42 0 0 1 0 4.438 3.42 3.42 0 0 0-.806 1.946 3.42 3.42 0 0 1-3.138 3.138 3.42 3.42 0 0 0-1.946.806 3.42 3.42 0 0 1-4.438 0 3.42 3.42 0 0 0-1.946-.806 3.42 3.42 0 0 1-3.138-3.138 3.42 3.42 0 0 0-.806-1.946 3.42 3.42 0 0 1 0-4.438 3.42 3.42 0 0 0 .806-1.946 3.42 3.42 0 0 1 3.138-3.138z',
  journal: 'M4 6h16M4 12h16M4 18h7',
  history: 'M12 8v4l3 3m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0z',
  sector: 'M3 3v18h18M9 9v9m4-5v5m4-9v9',
  geoHistory: 'M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z',
  backtest: 'M13 2L3 14h9l-1 8 10-12h-9l1-8z',
  logs: 'M9 12h6m-6 4h6m2 5H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5.586a1 1 0 0 1 .707.293l5.414 5.414a1 1 0 0 1 .293.707V19a2 2 0 0 1-2 2z',
  macro: 'M2 20h20M6 16V8m4 8V4m4 12V8m4 8V6',
  scenarios: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10zM12 8v4m0 4h.01',
  corporate: 'M3 21h18M3 10h18M3 10l9-8 9 8M5 10v11m14-11v11m-10-7h2v7H9z',
  graveyard: 'M8 2v4m8-4v4m-9 4h10M5 6h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z',
  compare: 'M16 3h5v5M8 3H3v5m18 8v5h-5M3 16v5h5M21 3l-7 7M3 3l7 7m4 4 7 7M3 21l7-7',
  architecture: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  graham: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm-1 14V8m-3 4h8',
  fearGreed: 'M12 2a10 10 0 0 1 10 10c0 2.4-.85 4.6-2.26 6.33M12 6v6l4 2M4.93 4.93l1.41 1.41',
  politicians: 'M17 20h5v-2a3 3 0 0 0-5.36-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 0 1 5.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 0 1 9.288 0M15 7a3 3 0 1 1-6 0 3 3 0 0 1 6 0z',
  lstm: 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-1',
  settings: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z',
  logout: 'M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9',
  chevronLeft: 'M15 18l-6-6 6-6',
  chevronRight: 'M9 18l6-6-6-6',
}

interface NavItem {
  to: string
  label: string
  icon: keyof typeof ICONS
}

const NAV_GROUPS: { items: NavItem[] }[] = [
  {
    items: [
      { to: '/',              label: 'Dashboard',     icon: 'dashboard' },
      { to: '/watchlist',     label: 'Watchlist',     icon: 'watchlist' },
      { to: '/discover',      label: 'Discover',      icon: 'discover' },
      { to: '/discoveries',   label: 'Discoveries',   icon: 'discoveries' },
      { to: '/top-picks',     label: 'Top Picks',     icon: 'topPicks' },
      { to: '/analyze',       label: 'Analyze',       icon: 'analyze' },
    ],
  },
  {
    items: [
      { to: '/portfolio',       label: 'Portfolio',     icon: 'portfolio' },
      { to: '/paper-trading',   label: 'Paper Trading', icon: 'paperTrading' },
      { to: '/insider-activity',label: 'Insider',       icon: 'insider' },
    ],
  },
  {
    items: [
      { to: '/macro',             label: 'Macro',            icon: 'macro' },
      { to: '/corporate-actions', label: 'Corp. Actions',    icon: 'corporate' },
      { to: '/scenarios',         label: 'Scenarios',        icon: 'scenarios' },
      { to: '/sector-screen',     label: 'Sectors',          icon: 'sector' },
      { to: '/geo-history',       label: 'Geo History',      icon: 'geoHistory' },
      { to: '/graveyard',         label: 'Graveyard',        icon: 'graveyard' },
    ],
  },
  {
    items: [
      { to: '/trust',       label: 'Trust Score',  icon: 'trust' },
      { to: '/learning',    label: 'Learning',     icon: 'learning' },
      { to: '/crosscheck',  label: 'Fact-Check',   icon: 'crosscheck' },
      { to: '/backtest',    label: 'Backtest',     icon: 'backtest' },
      { to: '/stock/compare', label: 'Compare',    icon: 'compare' },
    ],
  },
  {
    items: [
      { to: '/graham',             label: 'Graham Value',   icon: 'graham' },
      { to: '/fear-greed',         label: 'Fear & Greed',   icon: 'fearGreed' },
      { to: '/politician-trades',  label: 'Senate Trades',  icon: 'politicians' },
      { to: '/lstm',               label: 'LSTM Model',     icon: 'lstm' },
    ],
  },
  {
    items: [
      { to: '/journal',       label: 'Journal',      icon: 'journal' },
      { to: '/history',       label: 'History',      icon: 'history' },
      { to: '/logs',          label: 'Logs',         icon: 'logs' },
      { to: '/architecture',  label: 'How it Works', icon: 'architecture' },
    ],
  },
]

function useLogout() {
  const navigate = useNavigate()
  return useMutation({
    mutationFn: () => api.post('/logout'),
    onSettled: () => {
      queryClient.clear()
      navigate('/login')
    },
  })
}

export function Sidebar() {
  const expanded = useThemeStore(s => s.sidebarExpanded)
  const toggleSidebar = useThemeStore(s => s.toggleSidebar)
  const logout = useLogout()

  return (
    <motion.aside
      className={styles.sidebar}
      animate={{ width: expanded ? 220 : 56 }}
      transition={{ type: 'spring', stiffness: 320, damping: 32 }}
    >
      {/* Logo */}
      <div className={styles.logoRow}>
        <NavLink to="/" className={styles.logoMark} aria-label="Dashboard">
          S
        </NavLink>
        <AnimatePresence>
          {expanded && (
            <motion.span
              className={styles.logoText}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.2 }}
            >
              Stockholm
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Status pill */}
      <div className={styles.statusRow}>
        <StatusPill compact={!expanded} />
      </div>

      {/* Nav groups */}
      <nav className={styles.nav} aria-label="Main navigation">
        {NAV_GROUPS.map((group, gi) => (
          <div key={gi} className={styles.group}>
            {group.items.map(item => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  [styles.navItem, isActive ? styles.active : ''].filter(Boolean).join(' ')
                }
                title={!expanded ? item.label : undefined}
              >
                <Icon d={ICONS[item.icon]} />
                <AnimatePresence>
                  {expanded && (
                    <motion.span
                      className={styles.navLabel}
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -6 }}
                      transition={{ duration: 0.18 }}
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      {/* Bottom pinned */}
      <div className={styles.bottom}>
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            [styles.navItem, isActive ? styles.active : ''].filter(Boolean).join(' ')
          }
          title={!expanded ? 'Settings' : undefined}
        >
          <Icon d={ICONS.settings} />
          <AnimatePresence>
            {expanded && (
              <motion.span
                className={styles.navLabel}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
              >
                Settings
              </motion.span>
            )}
          </AnimatePresence>
        </NavLink>

        <button
          type="button"
          className={styles.navItem}
          onClick={() => logout.mutate()}
          title={!expanded ? 'Sign out' : undefined}
        >
          <Icon d={ICONS.logout} />
          <AnimatePresence>
            {expanded && (
              <motion.span
                className={styles.navLabel}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
              >
                Sign out
              </motion.span>
            )}
          </AnimatePresence>
        </button>

        {/* Collapse toggle */}
        <button
          type="button"
          className={styles.toggleBtn}
          onClick={toggleSidebar}
          aria-label={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
          title={expanded ? 'Collapse' : 'Expand'}
        >
          <Icon d={expanded ? ICONS.chevronLeft : ICONS.chevronRight} />
        </button>
      </div>
    </motion.aside>
  )
}
