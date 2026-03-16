import { useState } from 'react'
import { motion } from 'framer-motion'
import { useWatchlist, useAddToWatchlist, useRemoveFromWatchlist } from '@/api/endpoints/watchlist'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Delta } from '@/components/ui/Delta'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import type { WatchlistTier, SignalType } from '@/types/api'
import styles from './WatchlistPage.module.css'
import clsx from 'clsx'

const TIERS: { value: WatchlistTier | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'core', label: 'Core Holdings' },
  { value: 'swing', label: 'Swing Trades' },
  { value: 'research', label: 'Research' },
  { value: 'earnings_play', label: 'Earnings Play' },
]

function SignalBadge({ signal }: { signal: SignalType }) {
  if (!signal) return null
  const v = signal === 'BUY' ? 'success' : signal === 'SELL' ? 'danger' : 'neutral'
  return <Badge variant={v}>{signal}</Badge>
}

export function WatchlistPage() {
  const { data: items, isLoading } = useWatchlist()
  const addMut = useAddToWatchlist()
  const removeMut = useRemoveFromWatchlist()
  const { addToast } = useToastStore()

  const [tier, setTier] = useState<WatchlistTier | 'all'>('all')
  const [newTicker, setNewTicker] = useState('')
  const [newName, setNewName] = useState('')

  const filtered = (items ?? []).filter(
    item => tier === 'all' || item.tier === tier
  )

  async function handleAdd() {
    if (!newTicker.trim()) return
    try {
      await addMut.mutateAsync({ ticker: newTicker.trim().toUpperCase(), name: newName.trim() })
      addToast(`${newTicker.toUpperCase()} added to watchlist`, 'success')
      setNewTicker(''); setNewName('')
    } catch {
      addToast('Failed to add ticker', 'error')
    }
  }

  async function handleRemove(ticker: string) {
    try {
      await removeMut.mutateAsync(ticker)
      addToast(`${ticker} removed`, 'info')
    } catch {
      addToast('Failed to remove ticker', 'error')
    }
  }

  const staleColor = (days: number | null) =>
    days === null ? '' : days <= 2 ? 'positive' : days <= 7 ? 'warning' : 'negative'

  return (
    <>
      <PageHeader
        title="Watchlist"
        subtitle="Monitor and track your target securities"
      />

      {/* Add ticker form */}
      <Card className={styles.addCard}>
        <div className={styles.addForm}>
          <input
            className={styles.input}
            placeholder="Ticker (e.g. AAPL)"
            value={newTicker}
            onChange={e => setNewTicker(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAdd()}
          />
          <input
            className={styles.input}
            placeholder="Name (optional)"
            value={newName}
            onChange={e => setNewName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAdd()}
          />
          <Button
            variant="primary"
            size="md"
            loading={addMut.isPending}
            onClick={handleAdd}
            disabled={!newTicker.trim()}
          >
            Add
          </Button>
        </div>
      </Card>

      {/* Tier filters */}
      <div className={styles.filters}>
        {TIERS.map(t => (
          <button
            key={t.value}
            className={clsx(styles.filterTab, tier === t.value && styles.activeTab)}
            onClick={() => setTier(t.value)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Table */}
      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <Card className={styles.tableCard}>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Name</th>
                  <th>Tier</th>
                  <th>Signal</th>
                  <th>Geo Risk</th>
                  <th>Last Analyzed</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((item, i) => (
                  <motion.tr
                    key={item.ticker}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.03, duration: 0.3 }}
                    className={styles.row}
                  >
                    <td>
                      <span className={styles.ticker}>{item.ticker}</span>
                    </td>
                    <td>
                      <span className={styles.name}>{item.name}</span>
                    </td>
                    <td>
                      <Badge variant="ghost">{item.tier.replace('_', ' ')}</Badge>
                    </td>
                    <td>
                      {item.signal ? (
                        <div className={styles.signalCell}>
                          <SignalBadge signal={item.signal} />
                          {item.confidence && (
                            <span className={styles.confidence}>{item.confidence}%</span>
                          )}
                        </div>
                      ) : (
                        <span className={styles.muted}>—</span>
                      )}
                    </td>
                    <td>
                      {item.geo_risk_score !== null ? (
                        <Delta
                          value={String(item.geo_risk_score)}
                          sign={item.geo_risk_score >= 7 ? 'negative' : item.geo_risk_score >= 4 ? 'neutral' : 'positive'}
                          showArrow={false}
                        />
                      ) : (
                        <span className={styles.muted}>—</span>
                      )}
                    </td>
                    <td>
                      {item.days_since_analysis !== null ? (
                        <span className={`${styles.stale} ${styles[staleColor(item.days_since_analysis)]}`}>
                          {item.days_since_analysis === 0 ? 'Today' : `${item.days_since_analysis}d ago`}
                        </span>
                      ) : (
                        <span className={styles.muted}>Never</span>
                      )}
                    </td>
                    <td>
                      <div className={styles.actions}>
                        <a href={`/analyze?ticker=${item.ticker}`}>
                          <Button variant="secondary" size="sm">Analyze</Button>
                        </a>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemove(item.ticker)}
                        >
                          Remove
                        </Button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>

            {filtered.length === 0 && (
              <div className={styles.emptyState}>
                <p>No tickers in this category.</p>
              </div>
            )}
          </div>
        </Card>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
