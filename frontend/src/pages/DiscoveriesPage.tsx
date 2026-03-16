import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import clsx from 'clsx'
import {
  useDiscoveries,
  usePromoteDiscovery,
  useDismissDiscovery,
} from '@/api/endpoints/discovery'
import type { Discovery, DiscoveryStatus } from '@/api/endpoints/discovery'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './DiscoveriesPage.module.css'

// ── Animation variants ────────────────────────────────────────────────────────

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.05 } },
}

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.35, ease: [0.34, 1.2, 0.64, 1] as number[] },
  },
}

// ── Status filter config ──────────────────────────────────────────────────────

type StatusFilter = 'all' | DiscoveryStatus

const STATUS_FILTERS: { value: StatusFilter; label: string }[] = [
  { value: 'all',       label: 'All' },
  { value: 'new',       label: 'New' },
  { value: 'screened',  label: 'Screened' },
  { value: 'promoted',  label: 'Promoted' },
  { value: 'dismissed', label: 'Dismissed' },
]

function statusBadgeVariant(
  status: DiscoveryStatus
): 'neutral' | 'warning' | 'success' | 'ghost' {
  switch (status) {
    case 'new':       return 'neutral'
    case 'screened':  return 'warning'
    case 'promoted':  return 'success'
    case 'dismissed': return 'ghost'
  }
}

// ── Discovery card ────────────────────────────────────────────────────────────

interface DiscoveryCardProps {
  discovery: Discovery
  onPromote: (id: number) => Promise<void>
  onDismiss: (id: number) => Promise<void>
  isPromoting: boolean
  isDismissing: boolean
}

function DiscoveryCard({
  discovery,
  onPromote,
  onDismiss,
  isPromoting,
  isDismissing,
}: DiscoveryCardProps) {
  const { id, ticker, name, status, discovery_score, signal, reason, discovered_at } = discovery
  const canAct = status !== 'promoted' && status !== 'dismissed'

  const scoreWidth = `${Math.min(100, Math.max(0, discovery_score))}%`

  const formattedDate = (() => {
    try {
      return new Date(discovered_at).toLocaleDateString('en-SE', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      })
    } catch {
      return discovered_at
    }
  })()

  return (
    <motion.div variants={itemVariants}>
      <Card animate={false}>
        {/* Header: ticker + badge */}
        <div className={styles.cardHeader}>
          <div className={styles.cardHeaderLeft}>
            <span className={styles.ticker}>{ticker}</span>
            <span className={styles.name}>{name}</span>
          </div>
          <Badge variant={statusBadgeVariant(status)}>
            {status}
          </Badge>
        </div>

        {/* Signal label */}
        {signal && (
          <div className={styles.signal}>{signal}</div>
        )}

        {/* Score bar */}
        <div className={styles.scoreRow}>
          <span className={styles.scoreLabel}>Score</span>
          <div className={styles.scoreBarTrack}>
            <div className={styles.scoreBarFill} style={{ width: scoreWidth }} />
          </div>
          <span className={styles.scoreValue}>{discovery_score}/100</span>
        </div>

        {/* Reason */}
        {reason && <p className={styles.reason}>{reason}</p>}

        {/* Footer: date + action buttons */}
        <div className={styles.cardFooter}>
          <span className={styles.discoveredAt}>{formattedDate}</span>
          {canAct && (
            <div className={styles.cardActions}>
              <Button
                variant="primary"
                size="sm"
                loading={isPromoting}
                disabled={isDismissing}
                onClick={() => onPromote(id)}
              >
                Promote
              </Button>
              <Button
                variant="ghost"
                size="sm"
                loading={isDismissing}
                disabled={isPromoting}
                onClick={() => onDismiss(id)}
              >
                Dismiss
              </Button>
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export function DiscoveriesPage() {
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [pendingPromote, setPendingPromote] = useState<number | null>(null)
  const [pendingDismiss, setPendingDismiss] = useState<number | null>(null)

  const { data, isLoading, isError } = useDiscoveries('all')
  const promoteMut = usePromoteDiscovery()
  const dismissMut = useDismissDiscovery()
  const { addToast } = useToastStore()

  const discoveries = data?.discoveries ?? []
  const stats = data?.stats

  const filtered: Discovery[] =
    statusFilter === 'all'
      ? discoveries
      : discoveries.filter(d => d.status === statusFilter)

  async function handlePromote(id: number) {
    setPendingPromote(id)
    try {
      await promoteMut.mutateAsync(id)
      addToast('Discovery promoted to watchlist', 'success')
    } catch {
      addToast('Failed to promote discovery', 'error')
    } finally {
      setPendingPromote(null)
    }
  }

  async function handleDismiss(id: number) {
    setPendingDismiss(id)
    try {
      await dismissMut.mutateAsync(id)
      addToast('Discovery dismissed', 'info')
    } catch {
      addToast('Failed to dismiss discovery', 'error')
    } finally {
      setPendingDismiss(null)
    }
  }

  const lastRun = stats?.last_run ?? null

  return (
    <>
      <PageHeader
        title="Auto-Discovery"
        subtitle="Automated stock opportunity scanner"
      />

      {/* Stats row */}
      <div className={styles.statsGrid}>
        <Card animate delay={0.05}>
          <MetricCard
            label="Discovered (7d)"
            value={stats?.week_total ?? '—'}
            mono
            large
          />
        </Card>
        <Card animate delay={0.1}>
          <MetricCard
            label="Promoted (7d)"
            value={stats?.week_promoted ?? '—'}
            mono
            large
          />
        </Card>
        <Card animate delay={0.15}>
          <MetricCard
            label="Pending Review"
            value={stats !== undefined ? (stats.total_new + stats.total_screened) : '—'}
            mono
            large
          />
        </Card>
        <Card animate delay={0.2}>
          <MetricCard
            label="Dismissed"
            value={stats?.total_dismissed ?? '—'}
            mono
            large
          />
        </Card>
      </div>

      {/* Last run bar */}
      {lastRun && (
        <div className={styles.lastRunBar}>
          <span className={styles.lastRunLabel}>Last run</span>
          <span className={styles.lastRunItem}>
            <span>{new Date(lastRun.run_at).toLocaleString('en-SE')}</span>
          </span>
          <span className={styles.lastRunItem}>
            <span className={styles.scoreLabel}>Scanned</span>
            <span className={styles.lastRunValue}>&nbsp;{lastRun.tickers_scanned}</span>
          </span>
          <span className={styles.lastRunItem}>
            <span className={styles.scoreLabel}>Found</span>
            <span className={styles.lastRunValue}>&nbsp;{lastRun.discoveries_found}</span>
          </span>
          <span className={styles.lastRunItem}>
            <span className={styles.scoreLabel}>Duration</span>
            <span className={styles.lastRunValue}>&nbsp;{lastRun.duration_seconds}s</span>
          </span>
          <span className={styles.lastRunSpacer} />
          <Badge variant={lastRun.errors ? 'danger' : 'success'}>
            {lastRun.errors ? 'Errors' : 'OK'}
          </Badge>
        </div>
      )}

      {/* Status filter tabs */}
      <div className={styles.filters}>
        {STATUS_FILTERS.map(f => (
          <button
            key={f.value}
            className={clsx(styles.filterTab, statusFilter === f.value && styles.activeTab)}
            onClick={() => setStatusFilter(f.value)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {isLoading ? (
        <div className={styles.loading}>
          <Spinner size="lg" />
        </div>
      ) : isError ? (
        <div className={styles.errorState}>Failed to load discoveries.</div>
      ) : (
        <AnimatePresence mode="wait">
          <motion.div
            key={statusFilter}
            className={styles.grid}
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {filtered.length === 0 ? (
              <div className={styles.emptyState}>
                No discoveries for this filter.
              </div>
            ) : (
              filtered.map(discovery => (
                <DiscoveryCard
                  key={discovery.id}
                  discovery={discovery}
                  onPromote={handlePromote}
                  onDismiss={handleDismiss}
                  isPromoting={pendingPromote === discovery.id}
                  isDismissing={pendingDismiss === discovery.id}
                />
              ))
            )}
          </motion.div>
        </AnimatePresence>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
