import { motion } from 'framer-motion'
import clsx from 'clsx'
import { useInsiderActivity, useScanInsiderActivity } from '@/api/endpoints/insider'
import type { InsiderSignal } from '@/api/endpoints/insider'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './InsiderActivityPage.module.css'

function formatNumber(n: number): string {
  return new Intl.NumberFormat('en-US').format(n)
}

function formatValue(n: number): string {
  if (n >= 1_000_000_000) return `$${(n / 1_000_000_000).toFixed(2)}B`
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`
  return `$${formatNumber(n)}`
}

function significanceTier(score: number): 'High' | 'Med' | 'Low' {
  if (score >= 7) return 'High'
  if (score >= 4) return 'Med'
  return 'Low'
}

function formatDate(d: string): string {
  return new Date(d).toLocaleDateString('sv-SE', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

function SignificanceDisplay({ score }: { score: number }) {
  const tier = significanceTier(score)
  const cls = tier === 'High' ? styles.sigHigh : tier === 'Med' ? styles.sigMed : styles.sigLow
  return (
    <div className={styles.significance}>
      <span className={clsx(styles.sigValue, cls)}>
        {score.toFixed(1)}<span className={styles.muted}>/10</span>
      </span>
    </div>
  )
}

function sortBySignificance(signals: InsiderSignal[]): InsiderSignal[] {
  return [...signals].sort((a, b) => b.significance_score - a.significance_score)
}

export function InsiderActivityPage() {
  const { data, isLoading } = useInsiderActivity()
  const scanMut = useScanInsiderActivity()
  const { addToast } = useToastStore()

  const signals = sortBySignificance(data?.signals ?? [])
  const buyCount = signals.filter(s => s.transaction_type === 'BUY').length
  const sellCount = signals.filter(s => s.transaction_type === 'SELL').length

  async function handleScan() {
    try {
      await scanMut.mutateAsync()
      addToast('Insider activity scan complete', 'success')
    } catch {
      addToast('Scan failed — please try again', 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="Insider Activity"
        subtitle="Track insider buying and selling patterns"
        actions={
          <Button
            variant="primary"
            size="md"
            loading={scanMut.isPending}
            onClick={handleScan}
          >
            Scan Now
          </Button>
        }
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Stats row */}
          <div className={styles.statsRow}>
            <Card delay={0}>
              <MetricCard
                label="Total Signals"
                value={signals.length}
                mono
              />
            </Card>
            <Card delay={0.05} glow="positive">
              <MetricCard
                label="BUY Transactions"
                value={buyCount}
                mono
              />
            </Card>
            <Card delay={0.1} glow="negative">
              <MetricCard
                label="SELL Transactions"
                value={sellCount}
                mono
              />
            </Card>
          </div>

          {/* Signals table */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Insider Signals</div>

            {signals.length === 0 ? (
              <Card>
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>🔍</div>
                  <div className={styles.emptyText}>
                    No insider signals found.<br />
                    Run a scan to discover insider transactions.
                  </div>
                </div>
              </Card>
            ) : (
              <Card className={styles.tableCard}>
                <div className={styles.tableWrapper}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Insider</th>
                        <th>Role</th>
                        <th>Transaction</th>
                        <th>Shares</th>
                        <th>Value</th>
                        <th>Date</th>
                        <th>Significance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.map((sig, i) => (
                        <motion.tr
                          key={`${sig.ticker}-${sig.insider_name}-${sig.date}-${i}`}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.04, duration: 0.3 }}
                          className={styles.row}
                        >
                          <td>
                            <span className={styles.ticker}>{sig.ticker}</span>
                          </td>
                          <td>
                            <span className={styles.insiderName}>{sig.insider_name}</span>
                          </td>
                          <td>
                            <span className={styles.role}>{sig.role}</span>
                          </td>
                          <td>
                            <Badge
                              variant={sig.transaction_type === 'BUY' ? 'success' : 'danger'}
                            >
                              {sig.transaction_type}
                            </Badge>
                          </td>
                          <td>
                            <span className={styles.mono}>{formatNumber(sig.shares)}</span>
                          </td>
                          <td>
                            <span className={styles.mono}>{formatValue(sig.value)}</span>
                          </td>
                          <td>
                            <span className={styles.timestamp}>{formatDate(sig.date)}</span>
                          </td>
                          <td>
                            <SignificanceDisplay score={sig.significance_score} />
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}
          </div>
        </>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
