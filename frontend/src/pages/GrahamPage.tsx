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
  { value: 0.0, label: '0%' },
  { value: 0.2, label: '20%' },
  { value: 0.5, label: '50%' },
  { value: 0.9, label: '90%' },
]

function fmtPrice(v: number | null) {
  if (v == null) return '—'
  return `$${v.toFixed(2)}`
}

function fmtPct(v: number | null) {
  if (v == null) return '—'
  return `${v > 0 ? '+' : ''}${v.toFixed(1)}%`
}

function UpsideCell({ v }: { v: number | null }) {
  if (v == null) return <span className={styles.muted}>—</span>
  return <span className={v >= 0 ? styles.positive : styles.negative}>{fmtPct(v)}</span>
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
      const r = await backtestMut.mutateAsync({ discount, max_positions: 50, holding_days: 252 })
      setBacktestResult(r)
      addToast('Backtest complete', 'success')
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
        subtitle="V = EPS × (8.5 + 2g) × 4.4 / Y — AAA bond yield sourced live from FRED"
        actions={
          <div className={styles.headerActions}>
            <div className={styles.mosSelector}>
              <span className={styles.mosLabel}>Margin of Safety</span>
              <div className={styles.mosGroup}>
                {DISCOUNT_OPTIONS.map(o => (
                  <button
                    key={o.value}
                    className={discount === o.value ? styles.mosActive : styles.mosBtn}
                    onClick={() => setDiscount(o.value)}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            </div>
            <Button variant="secondary" size="md" loading={backtestRunning} onClick={handleBacktest}>
              Backtest
            </Button>
          </div>
        }
      />

      {/* Metrics row */}
      <div className={styles.metricsRow}>
        <MetricCard
          label="AAA Bond Yield"
          value={yieldData ? `${yieldData.aaa_yield_pct.toFixed(2)}%` : '—'}
          mono
        />
        <MetricCard
          label="Screened"
          value={data ? `${data.total_screened}` : '—'}
          mono
        />
        <MetricCard
          label="IV Calculable"
          value={data ? `${data.iv_calculable}` : '—'}
          mono
        />
        <MetricCard
          label="Buy Candidates"
          value={data?.buy_candidates ?? '—'}
          delta={discount > 0 ? `${(discount * 100).toFixed(0)}% margin of safety` : undefined}
          deltaSign="neutral"
          mono
        />
      </div>

      {/* Backtest results */}
      {backtestResult && (
        <Card delay={0} className={styles.backtestCard}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTitle}>Backtest Results</span>
            <Badge variant="ghost">
              {backtestResult.iv_calculable_tickers} IV-calculable tickers · fair benchmark
            </Badge>
          </div>
          <div className={styles.backtestGrid}>
            {[
              { label: 'Trades', value: backtestResult.trades, neutral: true },
              { label: 'Avg Return', value: fmtPct(backtestResult.avg_forward_return_pct), sign: backtestResult.avg_forward_return_pct },
              { label: 'Win Rate', value: fmtPct(backtestResult.win_rate_pct), neutral: true },
              { label: 'Benchmark', value: fmtPct(backtestResult.benchmark_return_pct), sign: backtestResult.benchmark_return_pct },
              { label: 'Alpha', value: fmtPct(backtestResult.alpha_vs_benchmark), sign: backtestResult.alpha_vs_benchmark },
              { label: 'AAA Yield', value: `${backtestResult.aaa_yield?.toFixed(2)}%`, neutral: true },
            ].map(m => (
              <div key={m.label} className={styles.btMetric}>
                <div className={styles.btLabel}>{m.label}</div>
                <div className={
                  m.neutral ? styles.btValue
                  : m.sign >= 0 ? styles.positive
                  : styles.negative
                }>{m.value}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Tab + table */}
      <Card className={styles.tableCard} delay={0.1}>
        <div className={styles.cardHeader}>
          <div className={styles.tabRow}>
            <button
              className={tab === 'buy' ? styles.tabActive : styles.tab}
              onClick={() => setTab('buy')}
            >
              Buy Candidates
              {data && <span className={styles.tabCount}>{data.buy_candidates}</span>}
            </button>
            <button
              className={tab === 'all' ? styles.tabActive : styles.tab}
              onClick={() => setTab('all')}
            >
              All IV-Calculable
              {data && <span className={styles.tabCount}>{data.iv_calculable}</span>}
            </button>
          </div>
          {data && (
            <span className={styles.muted}>
              sourced {new Date(data.screened_at).toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
        </div>

        {isLoading ? (
          <div className={styles.loading}><Spinner size="lg" /></div>
        ) : rows.length === 0 ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyTitle}>No stocks match this filter</div>
            <div className={styles.emptyText}>
              {tab === 'buy'
                ? 'No stocks are trading below the Graham threshold. Reduce the margin of safety or add more tickers to your watchlist.'
                : 'Could not calculate intrinsic value for any watchlist ticker. Ensure tickers have EPS data available.'}
            </div>
          </div>
        ) : (
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
                  <th>EPS TTM</th>
                  <th>Growth Rate</th>
                  <th>Data</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <motion.tr
                    key={r.ticker}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.025, duration: 0.25 }}
                    className={styles.row}
                  >
                    <td><span className={styles.ticker}>{r.ticker}</span></td>
                    <td>
                      <Badge variant={r.buy_signal ? 'success' : 'neutral'}>
                        {r.buy_signal ? 'BUY' : 'HOLD'}
                      </Badge>
                    </td>
                    <td><span className={styles.num}>{fmtPrice(r.current_price)}</span></td>
                    <td><span className={styles.num}>{fmtPrice(r.intrinsic_value)}</span></td>
                    <td>
                      <span className={styles.num} style={{ color: 'var(--signal-warning)' }}>
                        {fmtPrice(r.buy_threshold)}
                      </span>
                    </td>
                    <td><UpsideCell v={r.upside_pct} /></td>
                    <td><span className={styles.num}>{r.ttm_eps != null ? r.ttm_eps.toFixed(2) : '—'}</span></td>
                    <td>
                      <span className={r.growth_rate != null && r.growth_rate > 0 ? styles.positive : styles.muted}>
                        {r.growth_rate != null ? `${r.growth_rate.toFixed(1)}%` : '—'}
                      </span>
                    </td>
                    <td>
                      <span className={styles.muted}>{r.eps_history_quarters}Q</span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
