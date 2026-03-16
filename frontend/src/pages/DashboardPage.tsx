import { motion } from 'framer-motion'
import { PageHeader } from '@/components/layout/PageHeader'
import { SystemCommandCenter } from '@/components/dashboard/SystemCommandCenter'
import { MarketRegimeCard } from '@/components/dashboard/MarketRegimeCard'
import { BenchmarkCard } from '@/components/dashboard/BenchmarkCard'
import { IntelStrip } from '@/components/dashboard/IntelStrip'
import { EconomicCalendarCard } from '@/components/dashboard/EconomicCalendarCard'
import { GeoRadarCard } from '@/components/dashboard/GeoRadarCard'
import styles from './DashboardPage.module.css'

const containerVariants = {
  animate: {
    transition: { staggerChildren: 0.07, delayChildren: 0.1 },
  },
}

const itemVariants = {
  initial: { opacity: 0, y: 20 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: [0.34, 1.2, 0.64, 1] as const },
  },
}

export function DashboardPage() {
  return (
    <>
      <PageHeader
        title="Dashboard"
        subtitle="Autonomous investment intelligence"
      />

      {/* System command center */}
      <SystemCommandCenter />

      {/* Primary row — market + portfolio */}
      <motion.div
        className={styles.primaryRow}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={itemVariants}>
          <MarketRegimeCard />
        </motion.div>
        <motion.div variants={itemVariants}>
          <BenchmarkCard />
        </motion.div>
      </motion.div>

      {/* Intelligence summary strip */}
      <IntelStrip />

      {/* Context row — sector + calendar */}
      <motion.div
        className={styles.contextRow}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={itemVariants}>
          <EconomicCalendarCard />
        </motion.div>
        <motion.div variants={itemVariants}>
          {/* SectorMomentumCard — placeholder until built */}
          <div className={styles.sectorPlaceholder}>
            <span className={styles.sectorLabel}>Sector Momentum</span>
            <p className={styles.sectorSub}>Loading sector data…</p>
          </div>
        </motion.div>
      </motion.div>

      {/* Geopolitical radar */}
      <GeoRadarCard />

      {/* Bottom padding */}
      <div className={styles.bottomSpacer} />
    </>
  )
}
