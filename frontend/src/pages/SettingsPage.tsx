import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { PageHeader } from '@/components/layout/PageHeader'
import { PanelAPIConnections } from '@/components/settings/panels/PanelAPIConnections'
import { PanelScheduler } from '@/components/settings/panels/PanelScheduler'
import { PanelAnalysis } from '@/components/settings/panels/PanelAnalysis'
import { PanelBudget } from '@/components/settings/panels/PanelBudget'
import { PanelSecurity } from '@/components/settings/panels/PanelSecurity'
import { PanelAppearance } from '@/components/settings/panels/PanelAppearance'
import { PanelPlugins } from '@/components/settings/panels/PanelPlugins'
import styles from './SettingsPage.module.css'
import clsx from 'clsx'

const PANELS = [
  { id: 'api',        label: 'API Connections',  icon: '⬡' },
  { id: 'scheduler',  label: 'Scheduler',         icon: '◷' },
  { id: 'analysis',   label: 'Analysis',          icon: '◈' },
  { id: 'budget',     label: 'Budget',            icon: '◎' },
  { id: 'security',   label: 'Security',          icon: '◉' },
  { id: 'appearance', label: 'Appearance',        icon: '◐' },
  { id: 'plugins',    label: 'Plugins',           icon: '◧' },
] as const

type PanelId = typeof PANELS[number]['id']

const panelComponents: Record<PanelId, React.ComponentType> = {
  api:        PanelAPIConnections,
  scheduler:  PanelScheduler,
  analysis:   PanelAnalysis,
  budget:     PanelBudget,
  security:   PanelSecurity,
  appearance: PanelAppearance,
  plugins:    PanelPlugins,
}

const panelVariants = {
  initial: { opacity: 0, x: 16 },
  animate: { opacity: 1, x: 0, transition: { duration: 0.2, ease: [0.34, 1.2, 0.64, 1] as const } },
  exit:    { opacity: 0, x: -8, transition: { duration: 0.12 } },
}

import type React from 'react'

export function SettingsPage() {
  const [activePanel, setActivePanel] = useState<PanelId>('api')
  const [search, setSearch] = useState('')

  const filtered = PANELS.filter(p =>
    p.label.toLowerCase().includes(search.toLowerCase())
  )

  const ActivePanel = panelComponents[activePanel]

  return (
    <>
      <PageHeader title="Settings" subtitle="Configure Stockholm" />

      <div className={styles.layout}>
        {/* Sidebar */}
        <aside className={styles.sidebar}>
          <div className={styles.search}>
            <input
              className={styles.searchInput}
              placeholder="Search settings…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <nav className={styles.nav}>
            {filtered.map(panel => (
              <button
                key={panel.id}
                className={clsx(styles.navItem, activePanel === panel.id && styles.active)}
                onClick={() => setActivePanel(panel.id)}
              >
                <span className={styles.navIcon}>{panel.icon}</span>
                {panel.label}
              </button>
            ))}
          </nav>
        </aside>

        {/* Panel content */}
        <main className={styles.panel}>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={activePanel}
              variants={panelVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              <ActivePanel />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </>
  )
}
