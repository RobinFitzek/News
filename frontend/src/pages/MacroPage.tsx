import { useRef, useEffect } from 'react'
import { Chart, registerables } from 'chart.js'
import { useMacroSnapshot, useMacroEvents } from '@/api/endpoints/macro'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import styles from './MacroPage.module.css'

Chart.register(...registerables)

function importanceColor(imp: string): 'warning' | 'neutral' | 'ghost' {
  switch (imp) {
    case 'high': return 'warning'
    case 'medium': return 'neutral'
    default: return 'ghost'
  }
}

// ── Yield Curve Chart ──────────────────────────────────────────────────────

function YieldSpreadChart({ history }: { history: Array<{ date: string; spread_2_10: number | null; vix: number | null }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !history.length) return
    chartRef.current?.destroy()

    const labels = history.map(h => h.date)
    const spreads = history.map(h => h.spread_2_10 ?? 0)
    const vixData = history.map(h => h.vix ?? 0)

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: '2Y-10Y Spread',
            data: spreads,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99,102,241,0.1)',
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            yAxisID: 'y',
          },
          {
            label: 'VIX',
            data: vixData,
            borderColor: '#ef4444',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            borderDash: [4, 3],
            yAxisID: 'y1',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { usePointStyle: true, font: { size: 11 } } },
        },
        scales: {
          x: { ticks: { maxTicksLimit: 8, font: { size: 10 } }, grid: { display: false } },
          y: { position: 'left', ticks: { callback: v => `${v}%`, font: { size: 10 } }, grid: { color: 'rgba(128,128,128,0.1)' } },
          y1: { position: 'right', ticks: { font: { size: 10 } }, grid: { display: false } },
        },
      },
    })

    return () => { chartRef.current?.destroy() }
  }, [history])

  return (
    <div className={styles.chartWrapper}>
      <canvas ref={canvasRef} />
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────

export function MacroPage() {
  const { data: snapshot, isLoading: snapshotLoading } = useMacroSnapshot()
  const { data: eventsData, isLoading: eventsLoading } = useMacroEvents(30)

  const latest = snapshot?.latest
  const history = snapshot?.history ?? []
  const events = eventsData?.events ?? snapshot?.events ?? []
  const isLoading = snapshotLoading || eventsLoading

  return (
    <>
      <PageHeader
        title="Macro Dashboard"
        subtitle="Yield curve, volatility, and central bank events"
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Key metrics */}
          {latest && (
            <div className={styles.metricsRow}>
              <MetricCard
                label="2Y-10Y Spread"
                value={latest.spread_2_10 != null ? `${latest.spread_2_10.toFixed(2)}%` : '—'}
              />
              <MetricCard
                label="VIX"
                value={latest.vix != null ? latest.vix.toFixed(1) : '—'}
              />
              <MetricCard
                label="10Y Yield"
                value={latest.yield_10y != null ? `${latest.yield_10y.toFixed(2)}%` : '—'}
              />
              <MetricCard
                label="DXY"
                value={latest.dxy != null ? latest.dxy.toFixed(1) : '—'}
              />
              <MetricCard
                label="Regime"
                value={latest.regime || '—'}
              />
            </div>
          )}

          {/* Chart */}
          {history.length > 0 && (
            <Card className={styles.chartCard} delay={0.1}>
              <div className={styles.chartTitle}>90-Day Yield Spread & VIX</div>
              <YieldSpreadChart history={history} />
            </Card>
          )}

          {/* Upcoming events */}
          <Card className={styles.tableCard} delay={0.2}>
            <div className={styles.tableHeader}>
              <span className={styles.tableTitle}>Upcoming Events</span>
              <Badge variant="ghost">{events.length} events</Badge>
            </div>
            {events.length === 0 ? (
              <div className={styles.emptyState}>No upcoming macro events.</div>
            ) : (
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Time</th>
                      <th>Event</th>
                      <th>Country</th>
                      <th>Importance</th>
                      <th>Forecast</th>
                      <th>Previous</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((evt, i) => (
                      <tr key={i} className={styles.row}>
                        <td className={styles.muted}>{evt.date}</td>
                        <td className={styles.muted}>{evt.time || '—'}</td>
                        <td className={styles.eventName}>{evt.event}</td>
                        <td>
                          <Badge variant="ghost">{evt.country}</Badge>
                        </td>
                        <td>
                          <Badge variant={importanceColor(evt.importance)}>
                            {evt.importance}
                          </Badge>
                        </td>
                        <td className={styles.num}>{evt.forecast ?? '—'}</td>
                        <td className={styles.num}>{evt.previous ?? '—'}</td>
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
