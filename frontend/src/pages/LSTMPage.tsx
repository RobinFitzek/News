import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Chart, registerables } from 'chart.js'
import {
  useLSTMSignals, useLSTMPerformance, useLSTMTradeHistory, useLSTMTrain,
} from '@/api/endpoints/lstm'
import type { LSTMPrediction, LSTMTradeEntry } from '@/api/endpoints/lstm'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './LSTMPage.module.css'

Chart.register(...registerables)

// ── Equity curve chart ─────────────────────────────────────────────────────

function EquityChart({ trades }: { trades: LSTMTradeEntry[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)
  const completed = trades.filter(t => t.verified && t.actual_return_pct != null)

  useEffect(() => {
    if (!canvasRef.current || completed.length < 2) return
    chartRef.current?.destroy()

    let equity = 100
    const labels: string[] = []
    const values: number[] = [equity]

    ;[...completed].reverse().forEach(t => {
      equity *= (1 + (t.actual_return_pct ?? 0) / 100)
      labels.push(new Date(t.entered_at).toLocaleDateString('sv-SE', { month: 'short', day: 'numeric' }))
      values.push(Math.round(equity * 100) / 100)
    })

    const isPositive = values[values.length - 1] >= values[0]
    const lineColor = isPositive ? '#4EE88A' : '#E86060'

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data: values,
          borderColor: lineColor,
          backgroundColor: (ctx) => {
            const { ctx: c, chartArea } = ctx.chart
            if (!chartArea) return 'transparent'
            const g = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom)
            g.addColorStop(0, isPositive ? 'rgba(78,232,138,0.12)' : 'rgba(232,96,96,0.12)')
            g.addColorStop(1, 'transparent')
            return g
          },
          fill: true,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { maxTicksLimit: 6, font: { size: 10, family: 'JetBrains Mono' }, color: '#6b6560' },
            grid: { display: false },
            border: { color: 'rgba(255,255,255,0.07)' },
          },
          y: {
            ticks: { font: { size: 10, family: 'JetBrains Mono' }, color: '#6b6560', callback: v => `$${v}` },
            grid: { color: 'rgba(255,255,255,0.04)' },
            border: { display: false },
          },
        },
      },
    })
    return () => { chartRef.current?.destroy() }
  }, [completed])

  if (completed.length < 2) return null
  return (
    <Card delay={0.1} className={styles.chartCard}>
      <div className={styles.cardHeader}>
        <span className={styles.cardTitle}>Equity Curve ($100 starting capital)</span>
        <Badge variant="ghost">{completed.length} verified trades</Badge>
      </div>
      <div className={styles.chartWrapper}><canvas ref={canvasRef} /></div>
    </Card>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────

function fmtPct(v: number | null): string {
  if (v == null) return '—'
  return `${v > 0 ? '+' : ''}${v.toFixed(1)}%`
}

function formatDate(s: string): string {
  return new Date(s).toLocaleDateString('sv-SE', { year: 'numeric', month: 'short', day: 'numeric' })
}

function ConfidenceCell({ value }: { value: number | null }) {
  if (value == null) return <span className={styles.muted}>—</span>
  const pct = value * 100
  const color = value >= 0.7 ? 'var(--signal-positive)' : value >= 0.5 ? '#84cc16' : 'var(--signal-warning)'
  return (
    <div className={styles.confCell}>
      <div className={styles.confTrack}>
        <div className={styles.confFill} style={{ width: `${pct}%`, background: color }} />
        <div className={styles.confLine} /> {/* 50% threshold */}
      </div>
      <span className={styles.confNum} style={{ color }}>{pct.toFixed(1)}%</span>
    </div>
  )
}

// ── Page ───────────────────────────────────────────────────────────────────

export function LSTMPage() {
  const [epochs, setEpochs] = useState(20)
  const [yearsBack, setYearsBack] = useState(3)
  const [trainOpen, setTrainOpen] = useState(false)

  const { data: signalsData, isLoading: signalsLoading } = useLSTMSignals()
  const { data: perf } = useLSTMPerformance()
  const { data: histData } = useLSTMTradeHistory(100)
  const trainMut = useLSTMTrain()
  const { addToast } = useToastStore()

  const signals = signalsData?.signals ?? []
  const trades = histData?.trades ?? []

  async function handleTrain() {
    try {
      const result = await trainMut.mutateAsync({ epochs, years_back: yearsBack })
      if (result.error) {
        addToast(result.error, 'error')
      } else {
        addToast(`Training complete — best val_loss: ${result.best_val_loss}`, 'success')
        setTrainOpen(false)
      }
    } catch {
      addToast('Training failed', 'error')
    }
  }

  const noModel = signals.length === 0 && !signalsLoading && (perf?.completed_trades ?? 0) === 0

  return (
    <>
      <PageHeader
        title="LSTM Model"
        subtitle="2-layer LSTM · 28 features · VIX + Fear & Greed + Senate trades + fundamentals"
        actions={
          <Button
            variant={trainOpen ? 'secondary' : 'primary'}
            size="md"
            onClick={() => setTrainOpen(v => !v)}
          >
            {trainOpen ? 'Cancel' : 'Train Model'}
          </Button>
        }
      />

      {/* Train panel */}
      {trainOpen && (
        <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}>
          <Card delay={0} className={styles.trainCard}>
            <div className={styles.cardHeader}>
              <span className={styles.cardTitle}>Training Configuration</span>
              <Badge variant="warning">Requires PyTorch</Badge>
            </div>
            <div className={styles.trainBody}>
              <p className={styles.trainNote}>
                Trains on all tickers in your watchlist. Fetches historical price, fundamentals,
                VIX rolling averages, CNN Fear & Greed, and Senate trade features automatically.
                <br />
                Install PyTorch first:{' '}
                <code className={styles.code}>
                  pip install torch --index-url https://download.pytorch.org/whl/cpu
                </code>
              </p>
              <div className={styles.trainParams}>
                <div className={styles.paramGroup}>
                  <label className={styles.paramLabel}>Epochs</label>
                  <input
                    type="number"
                    className={styles.paramInput}
                    value={epochs}
                    min={5} max={100}
                    onChange={e => setEpochs(Number(e.target.value))}
                  />
                </div>
                <div className={styles.paramGroup}>
                  <label className={styles.paramLabel}>Years of history</label>
                  <input
                    type="number"
                    className={styles.paramInput}
                    value={yearsBack}
                    min={1} max={10}
                    onChange={e => setYearsBack(Number(e.target.value))}
                  />
                </div>
                <Button variant="primary" size="md" loading={trainMut.isPending} onClick={handleTrain}>
                  Start Training
                </Button>
              </div>
              {trainMut.isPending && (
                <div className={styles.trainStatus}>
                  <Spinner size="sm" />
                  <span>Training in progress — this may take several minutes…</span>
                </div>
              )}
              {trainMut.data?.history && trainMut.data.history.length > 0 && (
                <div className={styles.trainLog}>
                  {trainMut.data.history.map(h => (
                    <div key={h.epoch} className={styles.logLine}>
                      <span className={styles.logEpoch}>ep {h.epoch.toString().padStart(2, '0')}</span>
                      <span>loss {h.train_loss.toFixed(4)}</span>
                      <span className={styles.logSep}>·</span>
                      <span>val {h.val_loss.toFixed(4)}</span>
                      <span className={styles.logSep}>·</span>
                      <span className={h.val_acc >= 0.55 ? styles.positive : styles.muted}>
                        acc {(h.val_acc * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </motion.div>
      )}

      {/* Performance metrics */}
      <div className={styles.metricsRow}>
        <MetricCard label="Completed Trades" value={perf?.completed_trades ?? '—'} mono />
        <MetricCard
          label="CAGR"
          value={fmtPct(perf?.cagr_pct ?? null)}
          deltaSign={perf?.cagr_pct != null ? (perf.cagr_pct >= 0 ? 'positive' : 'negative') : 'neutral'}
          mono
        />
        <MetricCard
          label="Win Rate"
          value={fmtPct(perf?.win_rate_pct ?? null)}
          deltaSign={perf?.win_rate_pct != null ? (perf.win_rate_pct >= 50 ? 'positive' : 'negative') : 'neutral'}
          mono
        />
        <MetricCard
          label="Max Drawdown"
          value={fmtPct(perf?.max_drawdown_pct ?? null)}
          deltaSign={perf?.max_drawdown_pct != null ? (perf.max_drawdown_pct > -15 ? 'neutral' : 'negative') : 'neutral'}
          mono
        />
      </div>

      {/* Equity curve */}
      <EquityChart trades={trades} />

      {/* Buy signals */}
      <Card className={styles.tableCard} delay={0.15}>
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle}>Buy Signals — Watchlist</span>
          <Badge variant={signals.length > 0 ? 'success' : 'ghost'}>
            {signalsLoading ? '…' : `${signals.length} signals ≥ 50% confidence`}
          </Badge>
        </div>

        {signalsLoading ? (
          <div className={styles.loading}><Spinner size="lg" /></div>
        ) : noModel ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyTitle}>No trained model found</div>
            <div className={styles.emptyText}>
              Use the "Train Model" button above to build the LSTM on your watchlist data.
              Training requires PyTorch and takes a few minutes depending on watchlist size.
            </div>
          </div>
        ) : signals.length === 0 ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyTitle}>No buy signals today</div>
            <div className={styles.emptyText}>
              No watchlist ticker currently meets the ≥ 50% confidence threshold.
              This is normal — the model is selective by design.
            </div>
          </div>
        ) : (
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Signal</th>
                  <th>Confidence</th>
                  <th>Predicted At</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((s: LSTMPrediction, i: number) => (
                  <motion.tr
                    key={s.ticker}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.04, duration: 0.25 }}
                    className={styles.row}
                  >
                    <td><span className={styles.ticker}>{s.ticker}</span></td>
                    <td>
                      <Badge variant={s.buy_signal ? 'success' : 'neutral'}>
                        {s.buy_signal ? 'BUY' : 'HOLD'}
                      </Badge>
                    </td>
                    <td><ConfidenceCell value={s.confidence} /></td>
                    <td><span className={styles.timestamp}>{s.predicted_at ? formatDate(s.predicted_at) : '—'}</span></td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Trade history */}
      {trades.length > 0 && (
        <Card className={styles.tableCard} delay={0.2}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTitle}>Trade History Log</span>
            <Badge variant="ghost">
              {trades.filter(t => t.verified).length} verified · {trades.filter(t => !t.verified).length} pending
            </Badge>
          </div>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Entered</th>
                  <th>Confidence</th>
                  <th>Expected</th>
                  <th>Actual</th>
                  <th>Hold</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t: LSTMTradeEntry, i: number) => (
                  <motion.tr
                    key={t.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: Math.min(i * 0.02, 0.5) }}
                    className={styles.row}
                  >
                    <td><span className={styles.ticker}>{t.ticker}</span></td>
                    <td><span className={styles.timestamp}>{formatDate(t.entered_at)}</span></td>
                    <td><span className={styles.num}>{(t.confidence * 100).toFixed(1)}%</span></td>
                    <td>
                      <span className={t.expected_return_pct >= 0 ? styles.positive : styles.negative}>
                        {fmtPct(t.expected_return_pct)}
                      </span>
                    </td>
                    <td>
                      {t.actual_return_pct != null
                        ? <span className={t.actual_return_pct >= 0 ? styles.positive : styles.negative}>
                            {fmtPct(t.actual_return_pct)}
                          </span>
                        : <span className={styles.muted}>—</span>}
                    </td>
                    <td><span className={styles.num}>{t.hold_days}d</span></td>
                    <td>
                      <Badge variant={t.verified ? 'success' : 'ghost'}>
                        {t.verified ? 'Verified' : 'Pending'}
                      </Badge>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
