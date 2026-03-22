import { useState, useMemo, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useWatchlist, useAddToWatchlist, useRemoveFromWatchlist } from '@/api/endpoints/watchlist'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { SignalGlyph } from '@/components/ui/SignalGlyph'
import { Delta } from '@/components/ui/Delta'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import type { WatchlistTier, SignalType } from '@/types/api'
import styles from './WatchlistPage.module.css'
import clsx from 'clsx'

// ── Sort/Filter types ────────────────────────────────────────────────────────

type SortKey = 'ticker' | 'tier' | 'signal_age' | 'geo_risk'
type SortDir = 'asc' | 'desc'
type FilterMode = 'all' | 'alerts_only' | 'stale_only' | 'discovered'

const TIERS: { value: WatchlistTier | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'core', label: 'Core Holdings' },
  { value: 'swing', label: 'Swing Trades' },
  { value: 'research', label: 'Research' },
  { value: 'earnings_play', label: 'Earnings Play' },
]

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'ticker', label: 'Ticker' },
  { value: 'tier', label: 'Tier' },
  { value: 'signal_age', label: 'Signal Age' },
  { value: 'geo_risk', label: 'Geo Risk' },
]

const FILTER_OPTIONS: { value: FilterMode; label: string }[] = [
  { value: 'all', label: 'Show All' },
  { value: 'alerts_only', label: 'Alerts Only' },
  { value: 'stale_only', label: 'Stale (>5d)' },
]

// ── localStorage persistence ─────────────────────────────────────────────────

function loadPref<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(`watchlist_${key}`)
    return v ? JSON.parse(v) : fallback
  } catch { return fallback }
}

function savePref(key: string, value: unknown) {
  try { localStorage.setItem(`watchlist_${key}`, JSON.stringify(value)) } catch {}
}

// ── Subcomponents ────────────────────────────────────────────────────────────

function SignalBadge({ signal }: { signal: SignalType }) {
  if (!signal) return null
  const v = signal === 'BUY' ? 'success' : signal === 'SELL' ? 'danger' : 'neutral'
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
      <SignalGlyph signal={signal} size={14} />
      <Badge variant={v}>{signal}</Badge>
    </span>
  )
}

// ── Main Component ───────────────────────────────────────────────────────────

export function WatchlistPage() {
  const { data: items, isLoading } = useWatchlist()
  const addMut = useAddToWatchlist()
  const removeMut = useRemoveFromWatchlist()
  const { addToast } = useToastStore()

  const [tier, setTier] = useState<WatchlistTier | 'all'>(() => loadPref('tier', 'all'))
  const [sortKey, setSortKey] = useState<SortKey>(() => loadPref('sortKey', 'ticker'))
  const [sortDir, setSortDir] = useState<SortDir>(() => loadPref('sortDir', 'asc'))
  const [filterMode, setFilterMode] = useState<FilterMode>(() => loadPref('filterMode', 'all'))
  const [newTicker, setNewTicker] = useState('')
  const [newName, setNewName] = useState('')

  // Persist preferences
  const updateTier = useCallback((v: WatchlistTier | 'all') => { setTier(v); savePref('tier', v) }, [])
  const updateSort = useCallback((key: SortKey) => {
    setSortKey(prev => {
      if (prev === key) {
        const newDir = sortDir === 'asc' ? 'desc' : 'asc'
        setSortDir(newDir)
        savePref('sortDir', newDir)
        return prev
      }
      setSortDir('asc')
      savePref('sortDir', 'asc')
      savePref('sortKey', key)
      return key
    })
  }, [sortDir])
  const updateFilter = useCallback((v: FilterMode) => { setFilterMode(v); savePref('filterMode', v) }, [])

  // Filter + sort pipeline
  const processed = useMemo(() => {
    let list = [...(items ?? [])]

    // Tier filter
    if (tier !== 'all') list = list.filter(i => i.tier === tier)

    // Mode filter
    if (filterMode === 'alerts_only') {
      list = list.filter(i => i.signal && (i.signal.includes('BUY') || i.signal.includes('SELL')))
    } else if (filterMode === 'stale_only') {
      list = list.filter(i => i.days_since_analysis === null || i.days_since_analysis > 5)
    }

    // Sort
    list.sort((a, b) => {
      let cmp = 0
      switch (sortKey) {
        case 'ticker':
          cmp = a.ticker.localeCompare(b.ticker)
          break
        case 'tier':
          cmp = (a.tier ?? '').localeCompare(b.tier ?? '')
          break
        case 'signal_age':
          cmp = (a.days_since_analysis ?? 999) - (b.days_since_analysis ?? 999)
          break
        case 'geo_risk':
          cmp = (b.geo_risk_score ?? 0) - (a.geo_risk_score ?? 0)
          break
      }
      return sortDir === 'desc' ? -cmp : cmp
    })

    return list
  }, [items, tier, sortKey, sortDir, filterMode])

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

  const sortArrow = (key: SortKey) =>
    sortKey === key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''

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

      {/* Controls row: tier filters + sort + filter mode */}
      <div className={styles.controlsRow}>
        {/* Tier filters */}
        <div className={styles.filters}>
          {TIERS.map(t => (
            <button
              key={t.value}
              className={clsx(styles.filterTab, tier === t.value && styles.activeTab)}
              onClick={() => updateTier(t.value)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Sort + Filter controls */}
        <div className={styles.sortFilterRow}>
          <select
            className={styles.sortSelect}
            value={filterMode}
            onChange={e => updateFilter(e.target.value as FilterMode)}
          >
            {FILTER_OPTIONS.map(f => (
              <option key={f.value} value={f.value}>{f.label}</option>
            ))}
          </select>
          <select
            className={styles.sortSelect}
            value={sortKey}
            onChange={e => updateSort(e.target.value as SortKey)}
          >
            {SORT_OPTIONS.map(s => (
              <option key={s.value} value={s.value}>Sort: {s.label}</option>
            ))}
          </select>
          <button
            className={styles.sortDirBtn}
            onClick={() => {
              const d = sortDir === 'asc' ? 'desc' : 'asc'
              setSortDir(d)
              savePref('sortDir', d)
            }}
            title={sortDir === 'asc' ? 'Ascending' : 'Descending'}
          >
            {sortDir === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* Count badge */}
      <div className={styles.countRow}>
        <span className={styles.countLabel}>
          {processed.length} ticker{processed.length !== 1 ? 's' : ''}
          {tier !== 'all' && ` in ${tier.replace('_', ' ')}`}
          {filterMode !== 'all' && ` · ${filterMode.replace('_', ' ')}`}
        </span>
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
                  <th className={styles.sortableHeader} onClick={() => updateSort('ticker')}>
                    Ticker{sortArrow('ticker')}
                  </th>
                  <th>Name</th>
                  <th className={styles.sortableHeader} onClick={() => updateSort('tier')}>
                    Tier{sortArrow('tier')}
                  </th>
                  <th>Signal</th>
                  <th className={styles.sortableHeader} onClick={() => updateSort('geo_risk')}>
                    Geo Risk{sortArrow('geo_risk')}
                  </th>
                  <th className={styles.sortableHeader} onClick={() => updateSort('signal_age')}>
                    Last Analyzed{sortArrow('signal_age')}
                  </th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {processed.map((item, i) => (
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
                        <a href={`/stock/${item.ticker}`}>
                          <Button variant="secondary" size="sm">View</Button>
                        </a>
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

            {processed.length === 0 && (
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
