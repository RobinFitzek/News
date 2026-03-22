import { useState, useMemo, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Chart, registerables } from 'chart.js'
import api from '@/api/client'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import type { KeyStats, ChartData } from '@/api/endpoints/stock'
import styles from './CompareStocksPage.module.css'

Chart.register(...registerables)

const CHART_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

function useMultiStockData(tickers: string[]) {
  return useQuery<{ stats: Record<string, KeyStats>; charts: Record<string, ChartData> }>({
    queryKey: ['compare-stocks', ...tickers],
    queryFn: async () => {
      const [statsResults, chartResults] = await Promise.all([
        Promise.all(tickers.map(t => api.get(`/api/key-stats/${t}`).then(r => ({ ticker: t, data: r.data })).catch(() => null))),
        Promise.all(tickers.map(t => api.get(`/api/chart-data/${t}`).then(r => ({ ticker: t, data: r.data })).catch(() => null))),
      ])
      const stats: Record<string, KeyStats> = {}
      const charts: Record<string, ChartData> = {}
      for (const r of statsResults) if (r) stats[r.ticker] = r.data
      for (const r of chartResults) if (r) charts[r.ticker] = r.data
      return { stats, charts }
    },
    enabled: tickers.length >= 2,
    staleTime: 120_000,
  })
}

function formatNum(n: number | null | undefined, decimals = 2): string {
  if (n == null) return '—'
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(1)}T`
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  return n.toFixed(decimals)
}

// ── Chart component ────────────────────────────────────────────────────────

function ComparisonChart({
  tickers,
  charts,
}: {
  tickers: string[]
  charts: Record<string, ChartData>
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    chartRef.current?.destroy()

    // Normalize all series to % change from first data point
    const datasets = tickers.map((ticker, i) => {
      const data = charts[ticker]
      if (!data?.prices?.length) return null
      const base = data.prices[0]
      const normalized = data.prices.map(p => ((p - base) / base) * 100)
      return {
        label: ticker,
        data: normalized,
        borderColor: CHART_COLORS[i % CHART_COLORS.length],
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
      }
    }).filter(Boolean)

    // Use the longest labels array
    const longestLabels = tickers.reduce((best, t) => {
      const labels = charts[t]?.labels ?? []
      return labels.length > best.length ? labels : best
    }, [] as string[])

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: { labels: longestLabels, datasets: datasets as any },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { usePointStyle: true, font: { size: 11 } } },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(1)}%` } },
        },
        scales: {
          x: { display: true, ticks: { maxTicksLimit: 8, font: { size: 10 } }, grid: { display: false } },
          y: { display: true, ticks: { callback: v => `${v}%`, font: { size: 10 } }, grid: { color: 'rgba(128,128,128,0.1)' } },
        },
      },
    })

    return () => { chartRef.current?.destroy() }
  }, [tickers, charts])

  return (
    <div className={styles.chartWrapper}>
      <canvas ref={canvasRef} />
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────

export function CompareStocksPage() {
  const [tickers, setTickers] = useState<string[]>([])
  const [input, setInput] = useState('')

  const { data, isLoading } = useMultiStockData(tickers)

  const addTicker = () => {
    const t = input.trim().toUpperCase()
    if (!t || tickers.includes(t) || tickers.length >= 5) return
    setTickers([...tickers, t])
    setInput('')
  }

  const removeTicker = (t: string) => {
    setTickers(tickers.filter(x => x !== t))
  }

  const METRICS: { label: string; key: keyof KeyStats; format?: (v: any) => string }[] = useMemo(() => [
    { label: 'Sector', key: 'sector', format: (v: string) => v || '—' },
    { label: 'Price', key: 'current_price', format: (v: number) => v != null ? `$${v.toFixed(2)}` : '—' },
    { label: 'P/E', key: 'pe_ratio', format: (v: number) => formatNum(v) },
    { label: 'Market Cap', key: 'market_cap', format: (v: number) => formatNum(v) },
    { label: 'Revenue', key: 'revenue', format: (v: number) => formatNum(v) },
    { label: 'EPS', key: 'eps', format: (v: number) => formatNum(v) },
    { label: 'Beta', key: 'beta', format: (v: number) => formatNum(v) },
    { label: '52W High', key: 'week_52_high', format: (v: number) => v != null ? `$${v.toFixed(2)}` : '—' },
    { label: '52W Low', key: 'week_52_low', format: (v: number) => v != null ? `$${v.toFixed(2)}` : '—' },
    { label: 'Risk Score', key: 'risk_score', format: (v: number) => formatNum(v, 0) },
    { label: 'Geo Risk', key: 'geo_risk_score', format: (v: number) => formatNum(v, 0) },
  ], [])

  return (
    <>
      <PageHeader
        title="Compare Stocks"
        subtitle="Side-by-side comparison of up to 5 tickers"
      />

      {/* Ticker input */}
      <Card className={styles.inputCard} delay={0}>
        <div className={styles.inputRow}>
          <input
            className={styles.tickerInput}
            placeholder="Enter ticker (e.g. AAPL)"
            value={input}
            onChange={e => setInput(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && addTicker()}
            maxLength={10}
          />
          <Button onClick={addTicker} disabled={tickers.length >= 5 || !input.trim()}>
            Add
          </Button>
        </div>

        {tickers.length > 0 && (
          <div className={styles.tickerChips}>
            {tickers.map((t, i) => (
              <span
                key={t}
                className={styles.chip}
                style={{ borderColor: CHART_COLORS[i % CHART_COLORS.length] }}
              >
                <span style={{ color: CHART_COLORS[i % CHART_COLORS.length] }}>{t}</span>
                <button
                  className={styles.chipRemove}
                  onClick={() => removeTicker(t)}
                  aria-label={`Remove ${t}`}
                >
                  &times;
                </button>
              </span>
            ))}
          </div>
        )}
      </Card>

      {tickers.length < 2 && (
        <Card className={styles.hintCard}>
          <p className={styles.hintText}>Add at least 2 tickers to compare.</p>
        </Card>
      )}

      {isLoading && (
        <div className={styles.loading}><Spinner size="lg" /></div>
      )}

      {data && tickers.length >= 2 && (
        <>
          {/* Price chart */}
          <Card className={styles.chartCard} delay={0.1}>
            <div className={styles.chartTitle}>Price Performance (% Change)</div>
            <ComparisonChart tickers={tickers} charts={data.charts} />
          </Card>

          {/* Metrics table */}
          <Card className={styles.tableCard} delay={0.2}>
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Metric</th>
                    {tickers.map((t, i) => (
                      <th key={t} style={{ color: CHART_COLORS[i % CHART_COLORS.length] }}>{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map(m => (
                    <tr key={m.key} className={styles.row}>
                      <td className={styles.metricLabel}>{m.label}</td>
                      {tickers.map(t => {
                        const stats = data.stats[t]
                        const val = stats ? (stats as any)[m.key] : null
                        return (
                          <td key={t} className={styles.metricValue}>
                            {m.format ? m.format(val) : (val ?? '—')}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
