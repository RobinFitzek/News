import { motion } from 'framer-motion'
import { PageHeader } from '@/components/layout/PageHeader'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'
import { SystemCommandCenter } from '@/components/dashboard/SystemCommandCenter'
import { MarketRegimeCard } from '@/components/dashboard/MarketRegimeCard'
import { BenchmarkCard } from '@/components/dashboard/BenchmarkCard'
import { IntelStrip } from '@/components/dashboard/IntelStrip'
import { EconomicCalendarCard } from '@/components/dashboard/EconomicCalendarCard'
import { SectorMomentumCard } from '@/components/dashboard/SectorMomentumCard'
import { AutoTradeCard } from '@/components/dashboard/AutoTradeCard'
import { GeoRadarCard } from '@/components/dashboard/GeoRadarCard'
import { FearGreedDashCard } from '@/components/dashboard/FearGreedDashCard'
import { GrahamDashCard } from '@/components/dashboard/GrahamDashCard'
import { LSTMSignalsDashCard } from '@/components/dashboard/LSTMSignalsDashCard'
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
      <ErrorBoundary label="System command center unavailable">
        <SystemCommandCenter />
      </ErrorBoundary>

      {/* Primary row — market regime + benchmark */}
      <motion.div
        className={styles.primaryRow}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="Market regime unavailable">
            <MarketRegimeCard />
          </ErrorBoundary>
        </motion.div>
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="Benchmark unavailable">
            <BenchmarkCard />
          </ErrorBoundary>
        </motion.div>
      </motion.div>

      {/* Intelligence summary strip */}
      <ErrorBoundary label="Intel strip unavailable">
        <IntelStrip />
      </ErrorBoundary>

      {/* Context row — sector + calendar */}
      <motion.div
        className={styles.contextRow}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="Economic calendar unavailable">
            <EconomicCalendarCard />
          </ErrorBoundary>
        </motion.div>
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="Sector momentum unavailable">
            <SectorMomentumCard />
          </ErrorBoundary>
        </motion.div>
      </motion.div>

      {/* Sentiment + value row */}
      <motion.div
        className={styles.contextRow}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="Fear & Greed unavailable">
            <FearGreedDashCard />
          </ErrorBoundary>
        </motion.div>
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="Graham screen unavailable">
            <GrahamDashCard />
          </ErrorBoundary>
        </motion.div>
      </motion.div>

      {/* AI signals + geo row */}
      <motion.div
        className={styles.contextRow}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="LSTM signals unavailable">
            <LSTMSignalsDashCard />
          </ErrorBoundary>
        </motion.div>
        <motion.div variants={itemVariants}>
          <ErrorBoundary label="Geo radar unavailable">
            <GeoRadarCard />
          </ErrorBoundary>
        </motion.div>
      </motion.div>

      {/* Auto-trade row */}
      <motion.div
        className={styles.contextRow}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={itemVariants} style={{ gridColumn: '1 / -1' }}>
          <ErrorBoundary label="Auto-trade unavailable">
            <AutoTradeCard />
          </ErrorBoundary>
        </motion.div>
      </motion.div>

      {/* Bottom padding */}
      <div className={styles.bottomSpacer} />
    </>
  )
}
