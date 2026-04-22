import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Chart, registerables } from 'chart.js'
import { useFearGreedCurrent, useFearGreedHistory, useFGSensitivity } from '@/api/endpoints/fearGreed'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { MetricCard } from '@/components/ui/MetricCard'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import styles from './FearGreedPage.module.css'

Chart.register(...registerables)

// ── F&G history chart ──────────────────────────────────────────────────────

function FGChart({ data }: { data: { date: string; fg_value: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !data.length) return
    chartRef.current?.destroy()

    const last180 = data.slice(-180)
    const labels = last180.map(d => d.date)
    const values = last180.map(d => d.fg_value)

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Fear & Greed',
            data: values,
            borderColor: '#6BB8FF',
            backgroundColor: (ctx) => {
              const chart = ctx.chart
              const { ctx: c, chartArea } = chart
              if (!chartArea) return 'transparent'
              const g = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom)
              g.addColorStop(0, 'rgba(107, 184, 255, 0.15)')
              g.addColorStop(1, 'rgba(107, 184, 255, 0)')
              return g
            },
            fill: true,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.4,
          },
          // Neutral line at 50
          {
            label: 'Neutral',
            data: new Array(labels.length).fill(50),
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ctx.datasetIndex === 0 ? `F&G: ${ctx.parsed.y.toFixed(0)}` : '',
            },
          },
        },
        scales: {
          x: {
            ticks: { maxTicksLimit: 8, font: { size: 10, family: 'JetBrains Mono' }, color: '#6b6560' },
            grid: { display: false },
            border: { color: 'rgba(255,255,255,0.07)' },
          },
          y: {
            min: 0, max: 100,
            ticks: { stepSize: 25, font: { size: 10, family: 'JetBrains Mono' }, color: '#6b6560' },
            grid: { color: 'rgba(255,255,255,0.04)' },
            border: { display: false },
          },
        },
      },
    })

    return () => { chartRef.current?.destroy() }
  }, [data])

  return <div className={styles.chartWrapper}><canvas ref={canvasRef} /></div>
}

// ── Helpers ────────────────────────────────────────────────────────────────

function fgBadgeVariant(v: number | null): 'danger' | 'warning' | 'neutral' | 'success' | 'ghost' {
  if (v == null) return 'ghost'
  if (v <= 20) return 'danger'
  if (v <= 40) return 'warning'
  if (v <= 60) return 'neutral'
  if (v <= 80) return 'success'
  return 'success'
}

function fgColor(v: number | null): string {
  if (v == null) return 'var(--text-muted)'
  if (v <= 20) return 'var(--signal-negative)'
  if (v <= 40) return '#f97316'
  if (v <= 60) return 'var(--signal-warning)'
  if (v <= 80) return '#84cc16'
  return 'var(--signal-positive)'
}

function fgLabel(v: number | null): string {
  if (v == null) return '—'
  if (v <= 20) return 'Extreme Fear'
  if (v <= 40) return 'Fear'
  if (v <= 60) return 'Neutral'
  if (v <= 80) return 'Greed'
  return 'Extreme Greed'
}

function sensitivityColor(v: number | null): string {
  if (v == null) return 'var(--text-muted)'
  if (v > 0.3) return 'var(--signal-positive)'
  if (v < -0.3) return 'var(--signal-negative)'
  return 'var(--signal-warning)'
}

// ── Page ───────────────────────────────────────────────────────────────────

export function FearGreedPage() {
  const [tickerInput, setTickerInput] = useState('')
  const [lookupTicker, setLookupTicker] = useState('')

  const { data: current, isLoading } = useFearGreedCurrent()
  const { data: history } = useFearGreedHistory()
  const { data: sensitivity } = useFGSensitivity(lookupTicker, 60)

  const fg = current?.fg_value ?? null
  const historyData = history?.data ?? []

  const vixAboveMA20 = current?.vix != null && current?.vix_ma20 != null && current.vix > current.vix_ma20

  return (
    <>
      <PageHeader
        title="Fear & Greed"
        subtitle="CNN Fear & Greed Index · VIX rolling averages · per-stock sensitivity factor"
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Top strip: F&G score + VIX metrics */}
          <div className={styles.topStrip}>
            {/* F&G score panel */}
            <Card delay={0} className={styles.fgPanel}>
              <div className={styles.fgScoreWrap}>
                <div className={styles.fgScore} style={{ color: fgColor(fg) }}>
                  {fg != null ? fg.toFixed(0) : '—'}
                </div>
                <div>
                  <div className={styles.fgScoreLabel}>Fear &amp; Greed Index</div>
                  <Badge variant={fgBadgeVariant(fg)}>{fgLabel(fg)}</Badge>
                </div>
              </div>
              <div className={styles.fgTrack}>
                <div
                  className={styles.fgFill}
                  style={{ width: `${fg ?? 0}%`, background: fgColor(fg) }}
                />
              </div>
              <div className={styles.fgScale}>
                <span>Extreme Fear</span>
                <span>Neutral</span>
                <span>Extreme Greed</span>
              </div>
            </Card>

            {/* VIX metrics */}
            <div className={styles.vixMetrics}>
              <MetricCard
                label="VIX"
                value={current?.vix != null ? current.vix.toFixed(2) : '—'}
                delta={vixAboveMA20 ? 'Above 20-day MA ↑' : 'Below 20-day MA ↓'}
                deltaSign={vixAboveMA20 ? 'negative' : 'positive'}
                mono
              />
              <MetricCard label="VIX 10-Day MA" value={current?.vix_ma10 != null ? current.vix_ma10.toFixed(2) : '—'} mono />
              <MetricCard label="VIX 20-Day MA" value={current?.vix_ma20 != null ? current.vix_ma20.toFixed(2) : '—'} mono />
              <MetricCard label="VIX 30-Day MA" value={current?.vix_ma30 != null ? current.vix_ma30.toFixed(2) : '—'} mono />
            </div>
          </div>

          {/* F&G history chart */}
          {historyData.length > 0 && (
            <Card delay={0.1} className={styles.chartCard}>
              <div className={styles.chartHeader}>
                <span className={styles.chartTitle}>Fear &amp; Greed — 180-Day History</span>
                <Badge variant="ghost">{historyData.length} data points (from 2011)</Badge>
              </div>
              <FGChart data={historyData} />
            </Card>
          )}

          {/* Sensitivity lookup */}
          <Card delay={0.15} className={styles.sensitivityCard}>
            <div className={styles.cardHeader}>
              <span className={styles.cardTitle}>Per-Stock Sensitivity Factor</span>
            </div>
            <div className={styles.sensitivityBody}>
              <p className={styles.sensitivityDesc}>
                60-day rolling correlation between a stock's price and the F&G index.
                <br />
                <strong style={{ color: 'var(--signal-positive)' }}>Positive</strong> = risk-on (e.g. Tesla — drops when fear rises).&nbsp;
                <strong style={{ color: 'var(--signal-negative)' }}>Negative</strong> = defensive (e.g. Walmart — rises when fear rises).
              </p>
              <div className={styles.lookupRow}>
                <input
                  className={styles.tickerInput}
                  placeholder="e.g. AAPL"
                  value={tickerInput}
                  onChange={e => setTickerInput(e.target.value.toUpperCase())}
                  onKeyDown={e => e.key === 'Enter' && setLookupTicker(tickerInput.trim())}
                  maxLength={10}
                />
                <Button
                  variant="primary"
                  size="md"
                  onClick={() => setLookupTicker(tickerInput.trim())}
                >
                  Look up
                </Button>
              </div>

              {lookupTicker && sensitivity && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={styles.sensitivityResult}
                >
                  <div className={styles.sensitivityRow}>
                    <span className={styles.sensitivityTicker}>{lookupTicker}</span>
                    <div className={styles.barWrap}>
                      {/* Left label */}
                      <span className={styles.barEdge} style={{ color: 'var(--signal-negative)' }}>← Fear</span>
                      {/* Track */}
                      <div className={styles.barTrack}>
                        <div className={styles.barMidline} />
                        {sensitivity.fg_sensitivity != null && (
                          <div
                            className={styles.barMarker}
                            style={{
                              left: `${((sensitivity.fg_sensitivity + 1) / 2) * 100}%`,
                              background: sensitivityColor(sensitivity.fg_sensitivity),
                            }}
                          />
                        )}
                      </div>
                      {/* Right label */}
                      <span className={styles.barEdge} style={{ color: 'var(--signal-positive)' }}>Greed →</span>
                    </div>
                    <div className={styles.sensitivityStats}>
                      <span
                        className={styles.sensitivityVal}
                        style={{ color: sensitivityColor(sensitivity.fg_sensitivity) }}
                      >
                        {sensitivity.fg_sensitivity != null
                          ? `${sensitivity.fg_sensitivity > 0 ? '+' : ''}${sensitivity.fg_sensitivity.toFixed(3)}`
                          : '—'}
                      </span>
                      <span className={styles.sensitivityInterp}>{sensitivity.interpretation}</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </Card>
        </>
      )}

      <div className="pageEnd" />
    </>
  )
}
