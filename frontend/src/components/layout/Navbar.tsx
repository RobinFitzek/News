import { Link, useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { NavGroup } from './NavGroup'
import { StatusPill } from './StatusPill'
import { queryClient } from '@/api/queryClient'
import api from '@/api/client'
import styles from './Navbar.module.css'

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

export function Navbar() {
  const logout = useLogout()

  return (
    <nav className={styles.nav}>
      <div className={styles.container}>
        {/* Logo */}
        <Link to="/" className={styles.logo}>
          Stockholm
          <span className={styles.logoDot} />
        </Link>

        {/* Status pill — center */}
        <StatusPill />

        {/* Nav menu */}
        <div className={styles.menu}>
          <Link to="/" className={styles.link}>Dashboard</Link>

          <NavGroup
            label="Monitor"
            items={[
              { to: '/discoveries', label: 'Auto-Discovery' },
              { to: '/watchlist',   label: 'Watchlist' },
              { to: '/top-picks',   label: 'Top Picks' },
              { divider: true,      label: '' },
              { to: '/insider-activity', label: 'Insider Activity' },
            ]}
          />

          <NavGroup
            label="Portfolio"
            items={[
              { to: '/portfolio',     label: 'Portfolio' },
              { to: '/paper-trading', label: 'Paper Trading' },
            ]}
          />

          <NavGroup
            label="Intelligence"
            items={[
              { to: '/trust',      label: 'Trust Score' },
              { to: '/learning',   label: 'Learning' },
              { to: '/crosscheck', label: 'Fact-Check' },
              { divider: true,     label: '' },
              { to: '/history',    label: 'History' },
              { to: '/geo-history',label: 'Geo History' },
            ]}
          />

          <NavGroup
            label="Tools"
            items={[
              { to: '/analyze',      label: 'Analyze Stock' },
              { to: '/stock/compare',label: 'Compare Stocks' },
              { to: '/sector-screen',label: 'Sector Screen' },
              { divider: true,       label: '' },
              { to: '/discover',     label: 'AI Discover' },
              { to: '/backtest',     label: 'Backtest' },
              { to: '/journal',      label: 'Trade Journal' },
            ]}
          />

          <NavGroup
            label="System"
            items={[
              { to: '/settings',      label: 'Settings' },
              { to: '/logs',          label: 'Logs' },
              { to: '/architecture',  label: 'How it Works' },
              { divider: true,        label: '' },
              { label: 'Logout', onClick: () => logout.mutate(), danger: true },
            ]}
          />
        </div>
      </div>
    </nav>
  )
}
