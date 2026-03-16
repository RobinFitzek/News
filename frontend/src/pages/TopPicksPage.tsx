import { motion } from 'framer-motion'
import clsx from 'clsx'
import { useTopPicks } from '@/api/endpoints/topPicks'
import type { TopPick } from '@/api/endpoints/topPicks'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import styles from './TopPicksPage.module.css'

function accuracyTier(acc: number): 'Green' | 'Amber' | 'Red' {
  if (acc >= 80) return 'Green'
  if (acc >= 60) return 'Amber'
  return 'Red'
}

function signalVariant(signal: string): 'success' | 'danger' | 'neutral' {
  const s = signal.toUpperCase()
  if (s === 'BUY') return 'success'
  if (s === 'SELL') return 'danger'
  return 'neutral'
}

function formatTs(ts: string): string {
  return new Date(ts).toLocaleString('sv-SE', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function AccuracyCell({ pick }: { pick: TopPick }) {
  const tier = accuracyTier(pick.accuracy)
  const valueClass = styles[`accuracy${tier}`]
  const fillClass = styles[`fill${tier}`]
  return (
    <div className={styles.accuracyCell}>
      <span className={clsx(styles.accuracyValue, valueClass)}>
        {pick.accuracy.toFixed(1)}%
      </span>
      <div className={styles.accuracyBar}>
        <div
          className={clsx(styles.accuracyFill, fillClass)}
          style={{ width: `${Math.min(pick.accuracy, 100)}%` }}
        />
      </div>
    </div>
  )
}

export function TopPicksPage() {
  const { data, isLoading } = useTopPicks()

  const topPicks = data?.top_picks ?? []
  const recentSignals = data?.recent_signals ?? []
  const learningStats = data?.learning_stats
  const totalTrusted = data?.total_trusted ?? 0

  const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000
  const recentFiltered = recentSignals.filter(
    s => new Date(s.timestamp).getTime() >= sevenDaysAgo
  )

  return (
    <>
      <PageHeader
        title="Top Picks"
        subtitle="Highest-accuracy prediction track record"
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Stats row */}
          <div className={styles.statsRow}>
            <Card delay={0}>
              <MetricCard
                label="Trusted Tickers"
                value={totalTrusted}
                mono
              />
            </Card>
            <Card delay={0.05}>
              <MetricCard
                label="Accuracy Rate"
                value={learningStats ? `${(learningStats.accuracy_rate * 100).toFixed(1)}%` : '—'}
                mono
              />
            </Card>
            <Card delay={0.1}>
              <MetricCard
                label="Total Predictions"
                value={learningStats ? learningStats.total_predictions.toLocaleString() : '—'}
                mono
              />
            </Card>
          </div>

          {/* Top Picks table */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Top Picks Ranked</div>
            <Card className={styles.tableCard}>
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Ticker</th>
                      <th>Predictions</th>
                      <th>Accuracy</th>
                      <th>Avg Confidence</th>
                      <th>Last Signal</th>
                      <th>Last Date</th>
                      <th>Win Streak</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topPicks.map((pick, i) => (
                      <motion.tr
                        key={pick.ticker}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.04, duration: 0.3 }}
                        className={styles.row}
                      >
                        <td>
                          <span className={styles.rank}>{i + 1}</span>
                        </td>
                        <td>
                          <span className={styles.ticker}>{pick.ticker}</span>
                        </td>
                        <td>
                          <span className={styles.mono}>{pick.predictions_count}</span>
                        </td>
                        <td>
                          <AccuracyCell pick={pick} />
                        </td>
                        <td>
                          <span className={styles.mono}>{pick.avg_confidence.toFixed(1)}%</span>
                        </td>
                        <td>
                          <Badge variant={signalVariant(pick.last_signal)}>
                            {pick.last_signal}
                          </Badge>
                        </td>
                        <td>
                          <span className={styles.timestamp}>
                            {pick.last_signal_date
                              ? new Date(pick.last_signal_date).toLocaleDateString('sv-SE')
                              : '—'}
                          </span>
                        </td>
                        <td>
                          <span
                            className={clsx(
                              styles.streak,
                              pick.win_streak > 0 ? styles.streakPositive : styles.streakNeutral
                            )}
                          >
                            {pick.win_streak > 0 ? `🔥 ${pick.win_streak}` : pick.win_streak}
                          </span>
                        </td>
                      </motion.tr>
                    ))}
                    {topPicks.length === 0 && (
                      <tr>
                        <td colSpan={8}>
                          <div style={{ padding: 'var(--space-8)', textAlign: 'center', color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
                            No top picks yet.
                          </div>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          {/* Recent High-Confidence Signals */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>
              Recent High-Confidence Signals — Last 7 days
            </div>
            {recentFiltered.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
                No recent signals in the past 7 days.
              </div>
            ) : (
              <div className={styles.signalsGrid}>
                {recentFiltered.map((sig, i) => (
                  <motion.div
                    key={`${sig.ticker}-${sig.timestamp}-${i}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.04, duration: 0.3 }}
                  >
                    <Card className={styles.signalCard} delay={i * 0.04}>
                      <div className={styles.signalCardTicker}>{sig.ticker}</div>
                      <div className={styles.signalCardMeta}>
                        <Badge variant={signalVariant(sig.signal)}>{sig.signal}</Badge>
                        <span className={styles.signalConfidence}>
                          {sig.confidence.toFixed(1)}%
                        </span>
                      </div>
                      <div className={styles.signalTime}>{formatTs(sig.timestamp)}</div>
                    </Card>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
