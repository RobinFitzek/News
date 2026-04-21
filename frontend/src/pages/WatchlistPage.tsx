import { useState, useMemo, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  useWatchlist,
  useAddToWatchlist,
  useRemoveFromWatchlist,
  useSaveWatchlistNote,
  useUpdateWatchlistTier,
} from '@/api/endpoints/watchlist'
import { useGrahamScreen } from '@/api/endpoints/graham'
import api from '@/api/client'
import { queryClient } from '@/api/queryClient'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Modal } from '@/components/ui/Modal'
import { SignalGlyph } from '@/components/ui/SignalGlyph'
import { Delta } from '@/components/ui/Delta'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import type { WatchlistTier, SignalType } from '@/types/api'
import styles from './WatchlistPage.module.css'
import clsx from 'clsx'

// ── Sort/Filter types ────────────────────────────────────────────────────────

type SortKey = 'ticker' | 'tier' | 'signal_age' | 'geo_risk' | 'performance' | 'volatility'
type SortDir = 'asc' | 'desc'
type FilterMode = 'all' | 'alerts_only' | 'stale_only' | 'discovered'

const TIERS: { value: WatchlistTier | 'all'; label: string }[] = [
  { value: 'all',          label: 'All' },
  { value: 'core',         label: 'Core Holdings' },
  { value: 'swing',        label: 'Swing Trades' },
  { value: 'research',     label: 'Research' },
  { value: 'earnings_play', label: 'Earnings Play' },
]

const TIER_OPTIONS: { value: WatchlistTier; label: string }[] = [
  { value: 'core',          label: 'Core Holdings' },
  { value: 'swing',         label: 'Swing Trades' },
  { value: 'research',      label: 'Research' },
  { value: 'earnings_play', label: 'Earnings Play' },
]

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'ticker',     label: 'Ticker' },
  { value: 'tier',       label: 'Tier' },
  { value: 'performance', label: 'Performance' },
  { value: 'signal_age', label: 'Signal Age' },
  { value: 'geo_risk',   label: 'Geo Risk' },
  { value: 'volatility', label: 'Volatility' },
]

const FILTER_OPTIONS: { value: FilterMode; label: string }[] = [
  { value: 'all',        label: 'Show All' },
  { value: 'alerts_only', label: 'Alerts Only' },
  { value: 'stale_only', label: 'Stale (>5d)' },
  { value: 'discovered', label: 'Discovered' },
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

// ── Confirm dialog state ─────────────────────────────────────────────────────

interface ConfirmState {
  open: boolean
  title: string
  message: string
  variant: 'danger' | 'warning'
  onConfirm: () => void
}

const CONFIRM_CLOSED: ConfirmState = {
  open: false, title: '', message: '', variant: 'danger', onConfirm: () => {},
}

// ── Main Component ───────────────────────────────────────────────────────────

export function WatchlistPage() {
  const { data: items, isLoading } = useWatchlist()
  const { data: grahamScreen } = useGrahamScreen(0.2, true)
  const addMut = useAddToWatchlist()
  const removeMut = useRemoveFromWatchlist()
  const noteMut = useSaveWatchlistNote()
  const tierMut = useUpdateWatchlistTier()
  const { addToast } = useToastStore()

  // Graham map for O(1) lookup
  const grahamMap = useMemo(() => {
    const m = new Map<string, { buy_signal: boolean; upside_pct: number | null; intrinsic_value: number | null }>()
    for (const r of grahamScreen?.results ?? []) {
      m.set(r.ticker, { buy_signal: r.buy_signal, upside_pct: r.upside_pct, intrinsic_value: r.intrinsic_value })
    }
    return m
  }, [grahamScreen])

  // Filters / sort prefs
  const [tier, setTier] = useState<WatchlistTier | 'all'>(() => loadPref('tier', 'all'))
  const [sortKey, setSortKey] = useState<SortKey>(() => loadPref('sortKey', 'ticker'))
  const [sortDir, setSortDir] = useState<SortDir>(() => loadPref('sortDir', 'asc'))
  const [filterMode, setFilterMode] = useState<FilterMode>(() => loadPref('filterMode', 'all'))
  const [search, setSearch] = useState('')

  // Add-form state
  const [newTicker, setNewTicker] = useState('')
  const [newName, setNewName] = useState('')
  const [newTier, setNewTier] = useState<WatchlistTier>('research')

  // Note editing
  const [editingNote, setEditingNote] = useState<string | null>(null)
  const [noteText, setNoteText] = useState('')

  // Inline tier editing
  const [editingTier, setEditingTier] = useState<string | null>(null)

  // Confirm dialog
  const [confirm, setConfirm] = useState<ConfirmState>(CONFIRM_CLOSED)

  // ── Preference helpers ──────────────────────────────────────────────────────

  const updateTier = useCallback((v: WatchlistTier | 'all') => { setTier(v); savePref('tier', v) }, [])
  const updateSort = useCallback((key: SortKey) => {
    setSortKey(prev => {
      if (prev === key) {
        const newDir = sortDir === 'asc' ? 'desc' : 'asc'
        setSortDir(newDir); savePref('sortDir', newDir)
        return prev
      }
      setSortDir('asc'); savePref('sortDir', 'asc'); savePref('sortKey', key)
      return key
    })
  }, [sortDir])
  const updateFilter = useCallback((v: FilterMode) => { setFilterMode(v); savePref('filterMode', v) }, [])

  // ── Filter + sort pipeline ──────────────────────────────────────────────────

  const processed = useMemo(() => {
    let list = [...(items ?? [])]

    if (tier !== 'all') list = list.filter(i => i.tier === tier)

    if (filterMode === 'alerts_only')
      list = list.filter(i => i.signal && (i.signal.includes('BUY') || i.signal.includes('SELL')))
    else if (filterMode === 'stale_only')
      list = list.filter(i => i.days_since_analysis === null || i.days_since_analysis > 5)
    else if (filterMode === 'discovered')
      list = list.filter(i => i.signal === 'BUY' || i.signal === 'WATCH')

    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter(i =>
        i.ticker.toLowerCase().includes(q) ||
        (i.name ?? '').toLowerCase().includes(q)
      )
    }

    list.sort((a, b) => {
      let cmp = 0
      switch (sortKey) {
        case 'ticker':      cmp = a.ticker.localeCompare(b.ticker); break
        case 'tier':        cmp = (a.tier ?? '').localeCompare(b.tier ?? ''); break
        case 'performance': cmp = (a.confidence ?? 0) - (b.confidence ?? 0); break
        case 'signal_age':  cmp = (a.days_since_analysis ?? 999) - (b.days_since_analysis ?? 999); break
        case 'geo_risk':    cmp = (b.geo_risk_score ?? 0) - (a.geo_risk_score ?? 0); break
        case 'volatility':  cmp = (b.geo_risk_score ?? 0) - (a.geo_risk_score ?? 0); break
      }
      return sortDir === 'desc' ? -cmp : cmp
    })

    return list
  }, [items, tier, sortKey, sortDir, filterMode, search])

  // ── Handlers ────────────────────────────────────────────────────────────────

  async function handleAdd() {
    if (!newTicker.trim()) return
    try {
      await addMut.mutateAsync({ ticker: newTicker.trim().toUpperCase(), name: newName.trim(), tier: newTier })
      addToast(`${newTicker.toUpperCase()} added to watchlist`, 'success')
      setNewTicker(''); setNewName('')
    } catch {
      addToast('Failed to add ticker', 'error')
    }
  }

  function confirmRemove(ticker: string) {
    setConfirm({
      open: true,
      title: 'Remove from Watchlist',
      message: `Remove ${ticker} from your watchlist? This cannot be undone.`,
      variant: 'danger',
      onConfirm: async () => {
        try {
          await removeMut.mutateAsync(ticker)
          addToast(`${ticker} removed`, 'info')
        } catch { addToast('Failed to remove ticker', 'error') }
      },
    })
  }

  function confirmArchive(ticker: string) {
    setConfirm({
      open: true,
      title: 'Archive Ticker',
      message: `Archive ${ticker}? It will be hidden from your watchlist but data is kept.`,
      variant: 'warning',
      onConfirm: async () => {
        try {
          await api.post(`/watchlist/archive/${ticker}`)
          queryClient.invalidateQueries({ queryKey: ['watchlist'] })
          addToast(`${ticker} archived`, 'info')
        } catch { addToast('Failed to archive ticker', 'error') }
      },
    })
  }

  async function handleSaveNote(ticker: string) {
    try {
      await noteMut.mutateAsync({ ticker, note: noteText })
      addToast('Note saved', 'success')
      setEditingNote(null); setNoteText('')
    } catch { addToast('Failed to save note', 'error') }
  }

  async function handleTierChange(ticker: string, tier: WatchlistTier) {
    try {
      await tierMut.mutateAsync({ ticker, tier })
      addToast(`${ticker} moved to ${tier.replace('_', ' ')}`, 'success')
      setEditingTier(null)
    } catch { addToast('Failed to update tier', 'error') }
  }

  const staleColor = (days: number | null) =>
    days === null ? '' : days <= 2 ? 'positive' : days <= 7 ? 'warning' : 'negative'

  const sortArrow = (key: SortKey) =>
    sortKey === key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''

  return (
    <>
      <PageHeader title="Watchlist" subtitle="Monitor and track your target securities" />

      {/* ── Add ticker form ── */}
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
          <select
            className={styles.sortSelect}
            value={newTier}
            onChange={e => setNewTier(e.target.value as WatchlistTier)}
            title="Tier"
          >
            {TIER_OPTIONS.map(t => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
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

      {/* ── Controls row ── */}
      <div className={styles.controlsRow}>
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

        <div className={styles.sortFilterRow}>
          {/* Search box */}
          <input
            className={clsx(styles.sortSelect, styles.searchInput)}
            placeholder="Search ticker / name…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
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
              setSortDir(d); savePref('sortDir', d)
            }}
            title={sortDir === 'asc' ? 'Ascending' : 'Descending'}
          >
            {sortDir === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* ── Count ── */}
      <div className={styles.countRow}>
        <span className={styles.countLabel}>
          {processed.length} ticker{processed.length !== 1 ? 's' : ''}
          {tier !== 'all' && ` in ${tier.replace('_', ' ')}`}
          {filterMode !== 'all' && ` · ${filterMode.replace('_', ' ')}`}
          {search && ` · matching "${search}"`}
        </span>
      </div>

      {/* ── Table ── */}
      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <Card className={styles.tableCard}>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.sortableHeader} onClick={() => updateSort('ticker')}>Ticker{sortArrow('ticker')}</th>
                  <th>Name</th>
                  <th className={styles.sortableHeader} onClick={() => updateSort('tier')}>Tier{sortArrow('tier')}</th>
                  <th>Signal</th>
                  <th>Graham IV</th>
                  <th className={styles.sortableHeader} onClick={() => updateSort('geo_risk')}>Geo Risk{sortArrow('geo_risk')}</th>
                  <th className={styles.sortableHeader} onClick={() => updateSort('signal_age')}>Last Analyzed{sortArrow('signal_age')}</th>
                  <th>Note</th>
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
                    {/* Ticker — clickable to stock detail */}
                    <td>
                      <a href={`/stock/${item.ticker}`} className={styles.tickerLink}>
                        {item.ticker}
                      </a>
                    </td>

                    <td><span className={styles.name}>{item.name}</span></td>

                    {/* Tier — click to edit inline */}
                    <td>
                      {editingTier === item.ticker ? (
                        <select
                          className={clsx(styles.sortSelect, styles.tierSelect)}
                          defaultValue={item.tier}
                          autoFocus
                          onChange={e => handleTierChange(item.ticker, e.target.value as WatchlistTier)}
                          onBlur={() => setEditingTier(null)}
                        >
                          {TIER_OPTIONS.map(t => (
                            <option key={t.value} value={t.value}>{t.label}</option>
                          ))}
                        </select>
                      ) : (
                        <button
                          className={styles.tierBadgeBtn}
                          onClick={() => setEditingTier(item.ticker)}
                          title="Click to change tier"
                        >
                          <Badge variant="ghost">{item.tier.replace('_', ' ')}</Badge>
                        </button>
                      )}
                    </td>

                    <td>
                      {item.signal ? (
                        <div className={styles.signalCell}>
                          <SignalBadge signal={item.signal} />
                          {item.confidence && <span className={styles.confidence}>{item.confidence}%</span>}
                        </div>
                      ) : <span className={styles.muted}>—</span>}
                    </td>

                    <td>
                      {(() => {
                        const g = grahamMap.get(item.ticker)
                        if (!g || g.intrinsic_value === null) return <span className={styles.muted}>—</span>
                        return (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <Badge variant={g.buy_signal ? 'success' : 'ghost'}>{g.buy_signal ? 'BUY' : 'HOLD'}</Badge>
                            {g.upside_pct !== null && (
                              <span style={{
                                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
                                color: g.upside_pct >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)',
                                fontWeight: 600,
                              }}>
                                {g.upside_pct >= 0 ? '+' : ''}{g.upside_pct.toFixed(0)}%
                              </span>
                            )}
                          </div>
                        )
                      })()}
                    </td>

                    <td>
                      {item.geo_risk_score !== null ? (
                        <Delta
                          value={String(item.geo_risk_score)}
                          sign={item.geo_risk_score >= 7 ? 'negative' : item.geo_risk_score >= 4 ? 'neutral' : 'positive'}
                          showArrow={false}
                        />
                      ) : <span className={styles.muted}>—</span>}
                    </td>

                    <td>
                      {item.days_since_analysis !== null ? (
                        <span className={`${styles.stale} ${styles[staleColor(item.days_since_analysis)]}`}>
                          {item.days_since_analysis === 0 ? 'Today' : `${item.days_since_analysis}d ago`}
                        </span>
                      ) : <span className={styles.muted}>Never</span>}
                    </td>

                    {/* Note preview + inline editor */}
                    <td className={styles.noteCell}>
                      {editingNote === item.ticker ? (
                        <div className={styles.noteEditor}>
                          <input
                            className={styles.input}
                            placeholder="Add a note…"
                            value={noteText}
                            onChange={e => setNoteText(e.target.value)}
                            onKeyDown={e => {
                              if (e.key === 'Enter') handleSaveNote(item.ticker)
                              if (e.key === 'Escape') setEditingNote(null)
                            }}
                            autoFocus
                          />
                          <Button variant="primary" size="sm" onClick={() => handleSaveNote(item.ticker)}>Save</Button>
                          <Button variant="ghost" size="sm" onClick={() => setEditingNote(null)}>✕</Button>
                        </div>
                      ) : (
                        <button
                          className={styles.notePreview}
                          onClick={() => { setEditingNote(item.ticker); setNoteText(item.note ?? '') }}
                          title={item.note ?? 'Add note'}
                        >
                          {item.note
                            ? <span className={styles.noteText}>{item.note.length > 40 ? item.note.slice(0, 40) + '…' : item.note}</span>
                            : <span className={styles.notePlaceholder}>+ note</span>
                          }
                        </button>
                      )}
                    </td>

                    <td>
                      <div className={styles.actions}>
                        <a href={`/analyze?ticker=${item.ticker}`}>
                          <Button variant="secondary" size="sm">Analyze</Button>
                        </a>
                        <Button variant="ghost" size="sm" onClick={() => confirmArchive(item.ticker)}>Archive</Button>
                        <Button variant="ghost" size="sm" onClick={() => confirmRemove(item.ticker)}>Remove</Button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>

            {processed.length === 0 && (
              <div className={styles.emptyState}>
                {search ? `No tickers matching "${search}".` : 'No tickers in this category.'}
              </div>
            )}
          </div>
        </Card>
      )}

      {/* ── Confirm dialog ── */}
      <Modal
        open={confirm.open}
        onClose={() => setConfirm(CONFIRM_CLOSED)}
        title={confirm.title}
        size="sm"
      >
        <p className={styles.confirmMessage}>{confirm.message}</p>
        <div className={styles.modalActions}>
          <Button variant="ghost" size="md" onClick={() => setConfirm(CONFIRM_CLOSED)}>Cancel</Button>
          <Button
            variant={confirm.variant === 'danger' ? 'danger' : 'secondary'}
            size="md"
            onClick={() => { confirm.onConfirm(); setConfirm(CONFIRM_CLOSED) }}
          >
            Confirm
          </Button>
        </div>
      </Modal>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
