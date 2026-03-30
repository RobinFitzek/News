import { useState } from 'react'
import { motion } from 'framer-motion'
import { usePoliticianTrades, usePoliticianTopTickers } from '@/api/endpoints/politicians'
import type { PoliticianTrade, TopTicker } from '@/api/endpoints/politicians'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import styles from './PoliticianTradesPage.module.css'

const DAYS_OPTIONS = [
  { value: 14, label: '14 days' },
  { value: 30, label: '30 days' },
  { value: 60, label: '60 days' },
  { value: 90, label: '90 days' },
]

function fmtAmount(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}K`
  return `$${v.toFixed(0)}`
}

function formatDate(d: string): string {
  return new Date(d).toLocaleDateString('sv-SE', {
    year: 'numeric', month: 'short', day: 'numeric',
  })
}

function txVariant(t: PoliticianTrade): 'success' | 'danger' | 'neutral' {
  if (t.is_buy) return 'success'
  if (t.is_sell) return 'danger'
  return 'neutral'
}

function txLabel(t: PoliticianTrade): string {
  if (t.is_buy) return 'BUY'
  if (t.is_sell) return 'SELL'
  return t.tx_type.toUpperCase() || 'OTHER'
}

export function PoliticianTradesPage() {
  const [days, setDays] = useState(30)
  const [filterTicker, setFilterTicker] = useState('')
  const [appliedFilter, setAppliedFilter] = useState<string | undefined>(undefined)

  const { data: tradesData, isLoading: tradesLoading } = usePoliticianTrades(appliedFilter, days)
  const { data: topData, isLoading: topLoading } = usePoliticianTopTickers(90, 20)

  const trades = tradesData?.trades ?? []
  const topTickers = topData?.tickers ?? []

  const buyCount = trades.filter(t => t.is_buy).length
  const sellCount = trades.filter(t => t.is_sell).length
  const uniqueSenators = new Set(trades.map(t => t.senator)).size

  function handleFilter() {
    setAppliedFilter(filterTicker.trim().toUpperCase() || undefined)
  }

  return (
    <>
      <PageHeader
        title="Politician Trades"
        subtitle="U.S. Senate financial disclosure data (45-day filing lag applies)"
      />

      {/* Controls */}
      <div className={styles.controls}>
        <div className={styles.filterRow}>
          <input
            className={styles.tickerInput}
            placeholder="Filter by ticker"
            value={filterTicker}
            onChange={e => setFilterTicker(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleFilter()}
          />
          <button className={styles.filterBtn} onClick={handleFilter}>Filter</button>
          {appliedFilter && (
            <button
              className={styles.clearBtn}
              onClick={() => { setAppliedFilter(undefined); setFilterTicker('') }}
            >
              Clear
            </button>
          )}
        </div>
        <div className={styles.daysRow}>
          {DAYS_OPTIONS.map(o => (
            <button
              key={o.value}
              className={days === o.value ? styles.dayBtnActive : styles.dayBtn}
              onClick={() => setDays(o.value)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stats row */}
      <div className={styles.statsRow}>
        <Card delay={0}>
          <MetricCard label="Trades Found" value={tradesData?.count ?? '—'} mono />
        </Card>
        <Card delay={0.05} glow="positive">
          <MetricCard label="Buy Trades" value={buyCount} mono />
        </Card>
        <Card delay={0.1} glow="negative">
          <MetricCard label="Sell Trades" value={sellCount} mono />
        </Card>
        <Card delay={0.15}>
          <MetricCard label="Senators" value={uniqueSenators} mono />
        </Card>
      </div>

      <div className={styles.grid}>
        {/* Top Tickers panel */}
        <div>
          <div className={styles.sectionTitle}>Most Active Tickers (90d)</div>
          {topLoading ? (
            <Card><div className={styles.loading}><Spinner size="md" /></div></Card>
          ) : (
            <Card className={styles.topCard}>
              {topTickers.map((t, i) => (
                <motion.div
                  key={t.ticker}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className={styles.topRow}
                  onClick={() => { setFilterTicker(t.ticker); setAppliedFilter(t.ticker) }}
                >
                  <div className={styles.topRank}>{i + 1}</div>
                  <div className={styles.topTicker}>{t.ticker}</div>
                  <div className={styles.topStats}>
                    <span className={styles.positive}>{t.buy_count}↑</span>
                    <span className={styles.negative}>{t.sell_count}↓</span>
                    <span className={styles.muted}>{t.unique_senators} senators</span>
                  </div>
                  <div className={styles.topVolume}>{fmtAmount(t.total_volume_mid)}</div>
                </motion.div>
              ))}
            </Card>
          )}
        </div>

        {/* Trades table */}
        <div>
          <div className={styles.sectionTitle}>
            {appliedFilter ? `Trades — ${appliedFilter}` : 'Recent Disclosures'}
          </div>
          {tradesLoading ? (
            <div className={styles.loading}><Spinner size="lg" /></div>
          ) : trades.length === 0 ? (
            <Card>
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>◈</div>
                <div className={styles.emptyText}>
                  No trades found for the current filter.<br />
                  Note: Senate disclosures have a 45-day filing lag.
                </div>
              </div>
            </Card>
          ) : (
            <Card className={styles.tableCard}>
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Ticker</th>
                      <th>Senator</th>
                      <th>Type</th>
                      <th>Asset</th>
                      <th>Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.slice(0, 200).map((t, i) => (
                      <motion.tr
                        key={`${t.ticker}-${t.date}-${t.senator}-${i}`}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: Math.min(i * 0.02, 0.5), duration: 0.25 }}
                        className={styles.row}
                      >
                        <td><span className={styles.timestamp}>{formatDate(t.date)}</span></td>
                        <td><span className={styles.ticker}>{t.ticker}</span></td>
                        <td><span className={styles.senator}>{t.senator}</span></td>
                        <td>
                          <Badge variant={txVariant(t)}>{txLabel(t)}</Badge>
                        </td>
                        <td><span className={styles.assetType}>{t.asset_type || '—'}</span></td>
                        <td>
                          <span className={styles.mono}>
                            {t.amount_mid > 0 ? fmtAmount(t.amount_mid) : '—'}
                          </span>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      </div>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
