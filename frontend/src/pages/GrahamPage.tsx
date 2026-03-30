import { useState } from 'react'
import { motion } from 'framer-motion'
import { useGrahamScreen, useGrahamAAAYield, useGrahamBacktest } from '@/api/endpoints/graham'
import type { GrahamResult } from '@/api/endpoints/graham'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './GrahamPage.module.css'

const DISCOUNT_OPTIONS = [
  { value: 0.0, label: '0% — At IV' },
  { value: 0.2, label: '20% Margin' },
  { value: 0.5, label: '50% Margin' },
  { value: 0.9, label: '90% Margin' },
]

function fmtPrice(v: number | null) {
  if (v == null) return '—'
  return `$${v.toFixed(2)}`
}

function fmtPct(v: number | null, suffix = '%') {
  if (v == null) return '—'
  const sign = v > 0 ? '+' : ''
  return `${sign}${v.toFixed(1)}${suffix}`
}

function UpsideCell({ v }: { v: number | null }) {
  if (v == null) return <span className={styles.muted}>—</span>
  return (
    <span className={v >= 0 ? styles.positive : styles.negative}>
      {fmtPct(v)}
    </span>
  )
}

export function GrahamPage() {
  const [discount, setDiscount] = useState(0.2)
  const [tab, setTab] = useState<'buy' | 'all'>('buy')
  const [backtestRunning, setBacktestRunning] = useState(false)
  const [backtestResult, setBacktestResult] = useState<any>(null)

  const { data, isLoading } = useGrahamScreen(discount)
  const { data: yieldData } = useGrahamAAAYield()
  const backtestMut = useGrahamBacktest()
  const { addToast } = useToastStore()

  const rows: GrahamResult[] = tab === 'buy'
    ? (data?.buy_list ?? [])
    : (data?.results ?? []).filter(r => r.intrinsic_value != null && r.intrinsic_value > 0)

  async function handleBacktest() {
    setBacktestRunning(true)
    try {
      const result = await backtestMut.mutateAsync({ discount, max_positions: 50, holding_days: 252 })
      setBacktestResult(result)
      addToast('Graham backtest complete', 'success')
    } catch {
      addToast('Backtest failed', 'error')
    } finally {
      setBacktestRunning(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Graham Value Screen"
        subtitle="Benjamin Graham intrinsic value formula: V = EPS × (8.5 + 2g) × 4.4 / Y"
        actions={
          <div className={styles.headerActions}>
            <select
              className={styles.select}
              value={discount}
              onChange={e => setDiscount(Number(e.target.value))}
            >
              {DISCOUNT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            <Button
              variant="secondary"
              size="md"
              loading={backtestRunning}
              onClick={handleBacktest}
            >
              Run Backtest
            </Button>
          </div>
        }
      />

      {/* Stats row */}
      <div className={styles.statsRow}>
        <Card delay={0}>
          <MetricCard
            label="AAA Bond Yield"
            value={yieldData ? `${yieldData.aaa_yield_pct.toFixed(2)}%` : '—'}
            mono
          />
        </Card>
        <Card delay={0.05}>
          <MetricCard
            label="IV-Calculable"
            value={data ? `${data.iv_calculable} / ${data.total_screened}` : '—'}
            mono
          />
        </Card>
        <Card delay={0.1} glow="positive">
          <MetricCard
            label="Buy Candidates"
            value={data?.buy_candidates ?? '—'}
            mono
          />
        </Card>
        <Card delay={0.15}>
          <MetricCard
            label="Margin of Safety"
            value={`${(discount * 100).toFixed(0)}%`}
            mono
          />
        </Card>
      </div>

      {/* Backtest result */}
      {backtestResult && (
        <Card delay={0} className={styles.backtestCard}>
          <div className={styles.sectionTitle}>Backtest Results</div>
          <div className={styles.backtestGrid}>
            <div>
              <div className={styles.btLabel}>Trades</div>
              <div className={styles.btValue}>{backtestResult.trades}</div>
            </div>
            <div>
              <div className={styles.btLabel}>IV-Calculable Tickers</div>
              <div className={styles.btValue}>{backtestResult.iv_calculable_tickers}</div>
            </div>
            <div>
              <div className={styles.btLabel}>Avg Return</div>
              <div className={backtestResult.avg_forward_return_pct >= 0 ? styles.positive : styles.negative}>
                {fmtPct(backtestResult.avg_forward_return_pct)}
              </div>
            </div>
            <div>
              <div className={styles.btLabel}>Win Rate</div>
              <div className={styles.btValue}>{fmtPct(backtestResult.win_rate_pct)}</div>
            </div>
            <div>
              <div className={styles.btLabel}>Benchmark (IV-filtered)</div>
              <div className={styles.btValue}>{fmtPct(backtestResult.benchmark_return_pct)}</div>
            </div>
            <div>
              <div className={styles.btLabel}>Alpha vs Benchmark</div>
              <div className={backtestResult.alpha_vs_benchmark >= 0 ? styles.positive : styles.negative}>
                {fmtPct(backtestResult.alpha_vs_benchmark)}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Tab toggle */}
      <div className={styles.tabRow}>
        <button
          className={tab === 'buy' ? styles.tabActive : styles.tab}
          onClick={() => setTab('buy')}
        >
          Buy Candidates ({data?.buy_candidates ?? 0})
        </button>
        <button
          className={tab === 'all' ? styles.tabActive : styles.tab}
          onClick={() => setTab('all')}
        >
          All IV-Calculable
        </button>
      </div>

      {/* Table */}
      <div className={styles.section}>
        {isLoading ? (
          <div className={styles.loading}><Spinner size="lg" /></div>
        ) : rows.length === 0 ? (
          <Card>
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}>◈</div>
              <div className={styles.emptyText}>
                No stocks match the current filter.<br />
                Try reducing the margin of safety or add more tickers to your watchlist.
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
                    <th>Signal</th>
                    <th>Price</th>
                    <th>Intrinsic Value</th>
                    <th>Threshold</th>
                    <th>Upside</th>
                    <th>EPS (TTM)</th>
                    <th>Growth</th>
                    <th>Quarters</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <motion.tr
                      key={r.ticker}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.03, duration: 0.25 }}
                      className={styles.row}
                    >
                      <td><span className={styles.ticker}>{r.ticker}</span></td>
                      <td>
                        <Badge variant={r.buy_signal ? 'success' : 'neutral'}>
                          {r.buy_signal ? 'BUY' : 'HOLD'}
                        </Badge>
                      </td>
                      <td><span className={styles.mono}>{fmtPrice(r.current_price)}</span></td>
                      <td><span className={styles.mono}>{fmtPrice(r.intrinsic_value)}</span></td>
                      <td><span className={styles.mono}>{fmtPrice(r.buy_threshold)}</span></td>
                      <td><UpsideCell v={r.upside_pct} /></td>
                      <td><span className={styles.mono}>{r.ttm_eps != null ? r.ttm_eps.toFixed(2) : '—'}</span></td>
                      <td>
                        <span className={r.growth_rate != null && r.growth_rate > 0 ? styles.positive : styles.muted}>
                          {r.growth_rate != null ? `${r.growth_rate.toFixed(1)}%` : '—'}
                        </span>
                      </td>
                      <td><span className={styles.muted}>{r.eps_history_quarters}Q</span></td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
