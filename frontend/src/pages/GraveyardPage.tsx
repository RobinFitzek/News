import { useGraveyardPerformance } from '@/api/endpoints/graveyard'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import styles from './GraveyardPage.module.css'

function pnlColor(pct: number): string {
  if (pct > 5) return '#ef4444'   // stock went up = bad removal
  if (pct < -5) return '#10b981'  // stock went down = good removal
  return 'var(--text-muted)'
}

function verdictLabel(pct: number): string {
  if (pct > 5) return 'Missed Gains'
  if (pct < -5) return 'Good Call'
  return 'Flat'
}

export function GraveyardPage() {
  const { data, isLoading } = useGraveyardPerformance()
  const results = data?.results ?? []

  const goodCalls = results.filter(r => r.change_pct < -5).length
  const missedGains = results.filter(r => r.change_pct > 5).length
  const winRate = results.length > 0 ? Math.round((goodCalls / results.length) * 100) : 0

  return (
    <>
      <PageHeader
        title="Graveyard"
        subtitle="Removed tickers and their post-removal performance"
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Summary metrics */}
          <div className={styles.metricsRow}>
            <MetricCard label="Total Removed" value={results.length} />
            <MetricCard label="Good Calls" value={goodCalls} />
            <MetricCard label="Missed Gains" value={missedGains} />
            <MetricCard label="Win Rate" value={`${winRate}%`} />
          </div>

          {/* Results table */}
          <Card className={styles.tableCard} delay={0.1}>
            {results.length === 0 ? (
              <div className={styles.emptyState}>No removed tickers found.</div>
            ) : (
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Reason</th>
                      <th>Removed</th>
                      <th>Price at Removal</th>
                      <th>Current Price</th>
                      <th>Change</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map(r => (
                      <tr key={r.ticker} className={styles.row}>
                        <td>
                          <span className={styles.ticker}>{r.ticker}</span>
                        </td>
                        <td>
                          <span className={styles.reason}>{r.reason || '—'}</span>
                        </td>
                        <td>
                          <span className={styles.muted}>
                            {r.removed_at ? new Date(r.removed_at).toLocaleDateString() : '—'}
                          </span>
                        </td>
                        <td className={styles.price}>${r.removal_price.toFixed(2)}</td>
                        <td className={styles.price}>${r.current_price.toFixed(2)}</td>
                        <td>
                          <span
                            className={styles.change}
                            style={{ color: pnlColor(r.change_pct) }}
                          >
                            {r.change_pct > 0 ? '+' : ''}{r.change_pct.toFixed(1)}%
                          </span>
                        </td>
                        <td>
                          <Badge
                            variant={r.change_pct < -5 ? 'success' : r.change_pct > 5 ? 'warning' : 'ghost'}
                          >
                            {verdictLabel(r.change_pct)}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
