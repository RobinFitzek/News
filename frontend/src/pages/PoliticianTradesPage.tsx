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

const DAYS_OPTIONS = [14, 30, 60, 90]

function fmtAmount(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}K`
  if (v > 0) return `$${v.toFixed(0)}`
  return '—'
}

function formatDate(d: string): string {
  return new Date(d).toLocaleDateString('sv-SE', { year: 'numeric', month: 'short', day: 'numeric' })
}

function txVariant(t: PoliticianTrade): 'success' | 'danger' | 'neutral' {
  if (t.is_buy) return 'success'
  if (t.is_sell) return 'danger'
  return 'neutral'
}

function txLabel(t: PoliticianTrade): string {
  if (t.is_buy) return 'BUY'
  if (t.is_sell) return 'SELL'
  return (t.tx_type || 'OTHER').toUpperCase()
}

function BuySellBar({ buys, sells, total }: { buys: number; sells: number; total: number }) {
  if (total === 0) return null
  const buyPct = (buys / total) * 100
  const sellPct = (sells / total) * 100
  return (
    <div className={styles.miniBar}>
      <div className={styles.miniBarBuy} style={{ width: `${buyPct}%` }} title={`${buys} buys`} />
      <div className={styles.miniBarSell} style={{ width: `${sellPct}%` }} title={`${sells} sells`} />
    </div>
  )
}

export function PoliticianTradesPage() {
  const [days, setDays] = useState(30)
  const [filterTicker, setFilterTicker] = useState('')
  const [activeTicker, setActiveTicker] = useState<string | undefined>(undefined)

  const { data: tradesData, isLoading: tradesLoading } = usePoliticianTrades(activeTicker, days)
  const { data: topData, isLoading: topLoading } = usePoliticianTopTickers(90, 20)

  const trades = tradesData?.trades ?? []
  const topTickers = topData?.tickers ?? []
  const buyCount = trades.filter(t => t.is_buy).length
  const sellCount = trades.filter(t => t.is_sell).length
  const uniqueSenators = new Set(trades.map(t => t.senator)).size

  function applyFilter() {
    const t = filterTicker.trim().toUpperCase()
    setActiveTicker(t || undefined)
  }

  function clearFilter() {
    setFilterTicker('')
    setActiveTicker(undefined)
  }

  function selectTicker(ticker: string) {
    setFilterTicker(ticker)
    setActiveTicker(ticker)
  }

  return (
    <>
      <PageHeader
        title="Senate Trades"
        subtitle="U.S. Senate financial disclosure data — note: 45-day filing lag required by the STOCK Act"
      />

      {/* Controls row */}
      <div className={styles.controls}>
        <div className={styles.filterGroup}>
          <input
            className={styles.tickerInput}
            placeholder="Filter by ticker"
            value={filterTicker}
            onChange={e => setFilterTicker(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && applyFilter()}
            maxLength={10}
          />
          <button className={styles.filterBtn} onClick={applyFilter}>Search</button>
          {activeTicker && (
            <button className={styles.clearBtn} onClick={clearFilter}>✕ Clear</button>
          )}
        </div>
        <div className={styles.daysGroup}>
          {DAYS_OPTIONS.map(d => (
            <button
              key={d}
              className={days === d ? styles.dayActive : styles.dayBtn}
              onClick={() => setDays(d)}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>

      {/* Metrics */}
      <div className={styles.metricsRow}>
        <MetricCard label={activeTicker ? `Trades — ${activeTicker}` : 'Total Trades'} value={tradesData?.count ?? '—'} mono />
        <MetricCard label="Buy Transactions" value={buyCount} delta={buyCount > sellCount ? 'More buys' : undefined} deltaSign="positive" mono />
        <MetricCard label="Sell Transactions" value={sellCount} delta={sellCount > buyCount ? 'More sells' : undefined} deltaSign="negative" mono />
        <MetricCard label="Unique Senators" value={uniqueSenators} mono />
      </div>

      {/* Main layout */}
      <div className={styles.layout}>
        {/* Top tickers sidebar */}
        <Card delay={0} className={styles.topCard}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTitle}>Most Active (90d)</span>
            <Badge variant="ghost">{topTickers.length}</Badge>
          </div>
          {topLoading ? (
            <div className={styles.loading}><Spinner size="md" /></div>
          ) : (
            <div>
              {topTickers.map((t: TopTicker, i: number) => (
                <motion.div
                  key={t.ticker}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.03 }}
                  className={`${styles.topRow} ${activeTicker === t.ticker ? styles.topRowActive : ''}`}
                  onClick={() => selectTicker(t.ticker)}
                >
                  <span className={styles.topRank}>{i + 1}</span>
                  <div className={styles.topInfo}>
                    <div className={styles.topTicker}>{t.ticker}</div>
                    <BuySellBar buys={t.buy_count} sells={t.sell_count} total={t.total_trades} />
                  </div>
                  <div className={styles.topRight}>
                    <span className={styles.topVolume}>{fmtAmount(t.total_volume_mid)}</span>
                    <div className={styles.topCounts}>
                      <span className={styles.buyCount}>{t.buy_count}↑</span>
                      <span className={styles.sellCount}>{t.sell_count}↓</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </Card>

        {/* Trades table */}
        <Card delay={0.05} className={styles.tableCard}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTitle}>
              {activeTicker ? `Disclosures — ${activeTicker}` : 'Recent Disclosures'}
            </span>
            {tradesData && (
              <Badge variant="ghost">{tradesData.count} trades · {days}d window</Badge>
            )}
          </div>

          {tradesLoading ? (
            <div className={styles.loading}><Spinner size="lg" /></div>
          ) : trades.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyTitle}>No disclosures found</div>
              <div className={styles.emptyText}>
                {activeTicker
                  ? `No Senate trades found for ${activeTicker} in the last ${days} days.`
                  : 'No disclosures in the selected window.'
                }<br />
                Remember: the STOCK Act allows a 45-day filing lag.
              </div>
            </div>
          ) : (
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
                  {trades.slice(0, 300).map((t, i) => (
                    <motion.tr
                      key={`${t.ticker}-${t.date}-${t.senator}-${i}`}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: Math.min(i * 0.015, 0.4), duration: 0.2 }}
                      className={styles.row}
                    >
                      <td><span className={styles.timestamp}>{formatDate(t.date)}</span></td>
                      <td>
                        <span
                          className={styles.ticker}
                          onClick={() => selectTicker(t.ticker)}
                          style={{ cursor: 'pointer' }}
                        >
                          {t.ticker}
                        </span>
                      </td>
                      <td><span className={styles.senator}>{t.senator}</span></td>
                      <td><Badge variant={txVariant(t)}>{txLabel(t)}</Badge></td>
                      <td><span className={styles.assetType}>{t.asset_type || '—'}</span></td>
                      <td><span className={styles.num}>{fmtAmount(t.amount_mid)}</span></td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
