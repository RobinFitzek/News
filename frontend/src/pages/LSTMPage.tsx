import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  useLSTMSignals, useLSTMPerformance, useLSTMTradeHistory, useLSTMTrain
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

function fmtPct(v: number | null, dp = 1): string {
  if (v == null) return '—'
  return `${v > 0 ? '+' : ''}${v.toFixed(dp)}%`
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = value * 100
  const color = value >= 0.7 ? '#22c55e' : value >= 0.5 ? '#84cc16' : '#eab308'
  return (
    <div className={styles.confWrapper}>
      <div className={styles.confTrack}>
        <div className={styles.confFill} style={{ width: `${pct}%`, background: color }} />
        <div className={styles.confThreshold} />
      </div>
      <span className={styles.confVal} style={{ color }}>
        {pct.toFixed(1)}%
      </span>
    </div>
  )
}

function formatDate(s: string): string {
  return new Date(s).toLocaleDateString('sv-SE', { year: 'numeric', month: 'short', day: 'numeric' })
}

export function LSTMPage() {
  const [trainEpochs, setTrainEpochs] = useState(20)
  const [trainYears, setTrainYears] = useState(3)
  const [showTrainPanel, setShowTrainPanel] = useState(false)

  const { data: signalsData, isLoading: signalsLoading } = useLSTMSignals()
  const { data: perf } = useLSTMPerformance()
  const { data: histData } = useLSTMTradeHistory(50)
  const trainMut = useLSTMTrain()
  const { addToast } = useToastStore()

  const signals = signalsData?.signals ?? []
  const trades = histData?.trades ?? []

  async function handleTrain() {
    try {
      const result = await trainMut.mutateAsync({ epochs: trainEpochs, years_back: trainYears })
      if (result.error) {
        addToast(result.error, 'error')
      } else {
        addToast(`Training complete — val_loss: ${result.best_val_loss}`, 'success')
        setShowTrainPanel(false)
      }
    } catch {
      addToast('Training failed', 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="LSTM Predictor"
        subtitle="Deep learning stock return predictor — 28 features · 50% confidence threshold"
        actions={
          <Button variant="primary" size="md" onClick={() => setShowTrainPanel(v => !v)}>
            {showTrainPanel ? 'Cancel' : 'Train Model'}
          </Button>
        }
      />

      {/* Train panel */}
      {showTrainPanel && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card delay={0} className={styles.trainCard}>
            <div className={styles.trainTitle}>Model Training</div>
            <div className={styles.trainNote}>
              Trains on all tickers in your watchlist using historical price, fundamentals,
              VIX, Fear & Greed index, and Senate trade features.<br />
              <strong>Requires:</strong> <code>pip install torch --index-url https://download.pytorch.org/whl/cpu</code>
            </div>
            <div className={styles.trainParams}>
              <label className={styles.paramLabel}>
                Epochs
                <input
                  type="number"
                  className={styles.paramInput}
                  value={trainEpochs}
                  min={5} max={100}
                  onChange={e => setTrainEpochs(Number(e.target.value))}
                />
              </label>
              <label className={styles.paramLabel}>
                Years of history
                <input
                  type="number"
                  className={styles.paramInput}
                  value={trainYears}
                  min={1} max={10}
                  onChange={e => setTrainYears(Number(e.target.value))}
                />
              </label>
            </div>
            <Button
              variant="primary"
              size="md"
              loading={trainMut.isPending}
              onClick={handleTrain}
            >
              Start Training
            </Button>
            {trainMut.isPending && (
              <div className={styles.trainProgress}>
                Training in progress — this may take several minutes…
              </div>
            )}
            {trainMut.data?.history && (
              <div className={styles.trainLog}>
                {trainMut.data.history.map(h => (
                  <div key={h.epoch} className={styles.trainLogLine}>
                    Epoch {h.epoch}: loss={h.train_loss} · val_loss={h.val_loss} · acc={h.val_acc}
                  </div>
                ))}
              </div>
            )}
          </Card>
        </motion.div>
      )}

      {/* Performance metrics */}
      <div className={styles.statsRow}>
        <Card delay={0}>
          <MetricCard label="Completed Trades" value={perf?.completed_trades ?? '—'} mono />
        </Card>
        <Card delay={0.05} glow={perf?.cagr_pct != null && perf.cagr_pct > 0 ? 'positive' : undefined}>
          <MetricCard label="CAGR" value={fmtPct(perf?.cagr_pct ?? null)} mono />
        </Card>
        <Card delay={0.1}>
          <MetricCard label="Win Rate" value={fmtPct(perf?.win_rate_pct ?? null)} mono />
        </Card>
        <Card delay={0.15} glow={perf?.max_drawdown_pct != null && perf.max_drawdown_pct < -15 ? 'negative' : undefined}>
          <MetricCard label="Max Drawdown" value={fmtPct(perf?.max_drawdown_pct ?? null)} mono />
        </Card>
      </div>

      {/* Buy signals */}
      <div className={styles.section}>
        <div className={styles.sectionTitle}>
          Buy Signals — Watchlist Scan
          {signalsData && (
            <span className={styles.sectionMeta}>
              {signals.length} signal{signals.length !== 1 ? 's' : ''} ≥ {(signalsData.threshold * 100).toFixed(0)}% confidence
            </span>
          )}
        </div>

        {signalsLoading ? (
          <div className={styles.loading}><Spinner size="lg" /></div>
        ) : signals.length === 0 ? (
          <Card>
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}>◈</div>
              <div className={styles.emptyText}>
                No buy signals found, or model not yet trained.<br />
                Use "Train Model" above to build the LSTM on your watchlist.
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
                      <td>
                        {s.confidence != null
                          ? <ConfidenceBar value={s.confidence} />
                          : <span className={styles.muted}>—</span>}
                      </td>
                      <td>
                        <span className={styles.timestamp}>
                          {s.predicted_at ? formatDate(s.predicted_at) : '—'}
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

      {/* Trade history */}
      {trades.length > 0 && (
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Trade History Log</div>
          <Card className={styles.tableCard}>
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Entered</th>
                    <th>Confidence</th>
                    <th>Expected Return</th>
                    <th>Actual Return</th>
                    <th>Hold Days</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t: LSTMTradeEntry, i: number) => (
                    <motion.tr
                      key={t.id}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.03 }}
                      className={styles.row}
                    >
                      <td><span className={styles.ticker}>{t.ticker}</span></td>
                      <td><span className={styles.timestamp}>{formatDate(t.entered_at)}</span></td>
                      <td><span className={styles.mono}>{(t.confidence * 100).toFixed(1)}%</span></td>
                      <td>
                        <span className={t.expected_return_pct >= 0 ? styles.positive : styles.negative}>
                          {fmtPct(t.expected_return_pct)}
                        </span>
                      </td>
                      <td>
                        {t.actual_return_pct != null ? (
                          <span className={t.actual_return_pct >= 0 ? styles.positive : styles.negative}>
                            {fmtPct(t.actual_return_pct)}
                          </span>
                        ) : <span className={styles.muted}>—</span>}
                      </td>
                      <td><span className={styles.mono}>{t.hold_days}d</span></td>
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
        </div>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
