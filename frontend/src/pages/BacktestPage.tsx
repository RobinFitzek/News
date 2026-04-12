import { useState } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ProgressBar } from '@/components/ui/ProgressBar'
import { getCsrfToken } from '@/api/csrf'
import {
  useBacktestProgress,
  useBacktestResults,
  useApplyWeights,
  useRandomBaseline,
  type RandomBaseline,
} from '@/api/endpoints/backtest'
import { useToastStore } from '@/stores/toastStore'
import styles from './BacktestPage.module.css'

const TODAY = new Date().toISOString().split('T')[0]

export function BacktestPage() {
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState(TODAY)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [baseline, setBaseline] = useState<RandomBaseline | null>(null)

  const { addToast } = useToastStore()
  const { data: progress } = useBacktestProgress()
  const { data: results } = useBacktestResults(runId)
  const applyWeightsMut = useApplyWeights()
  const baselineMut = useRandomBaseline()

  async function handleRunBaseline() {
    if (!results) return
    try {
      const res = await baselineMut.mutateAsync({ results, n_simulations: 500 })
      setBaseline(res)
    } catch {
      addToast('Failed to run baseline', 'error')
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setIsSubmitting(true)
    try {
      const csrfToken = await getCsrfToken()
      const body = new URLSearchParams({ start_date: startDate, end_date: endDate, csrf_token: csrfToken })
      const res = await fetch('/backtest/run', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString(),
      })
      if (!res.ok) throw new Error('Backtest failed to start')
      const data = (await res.json()) as { run_id?: string }
      if (data.run_id) setRunId(data.run_id)
      addToast('Backtest started', 'success')
    } catch {
      addToast('Failed to start backtest', 'error')
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleApplyWeights() {
    if (!runId) return
    try {
      await applyWeightsMut.mutateAsync(runId)
      addToast('Weights applied successfully', 'success')
    } catch {
      addToast('Failed to apply weights', 'error')
    }
  }

  const isRunning = progress?.is_running ?? false

  return (
    <>
      <PageHeader
        title="Backtest"
        subtitle="Test strategy performance on historical data"
      />

      {/* Run form */}
      <Card className={styles.formCard}>
        <form onSubmit={handleSubmit} className={styles.formGrid}>
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor="start_date">Start Date</label>
            <input
              id="start_date"
              type="date"
              className={styles.input}
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              required
            />
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.label} htmlFor="end_date">End Date</label>
            <input
              id="end_date"
              type="date"
              className={styles.input}
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              required
            />
          </div>
          <Button
            variant="primary"
            size="md"
            type="submit"
            loading={isSubmitting}
            disabled={isSubmitting || isRunning}
          >
            Run Backtest
          </Button>
        </form>
      </Card>

      {/* Progress card */}
      {isRunning && progress && (
        <Card className={styles.progressCard}>
          <div className={styles.progressInfo}>
            <span className={styles.progressLabel}>Running Backtest</span>
            <span className={styles.progressPct}>{progress.percent.toFixed(0)}%</span>
          </div>
          <ProgressBar value={progress.percent} variant="default" height={4} />
          {progress.current_ticker && (
            <div className={styles.currentTicker}>
              Analyzing: {progress.current_ticker}
            </div>
          )}
        </Card>
      )}

      {/* Results card */}
      {runId && results && (
        <Card className={styles.resultsCard}>
          <div className={styles.resultsTitle}>Results</div>
          <div className={styles.metricsGrid}>
            <div className={styles.metricItem}>
              <div className={styles.metricLabel}>Total Return</div>
              <div className={`${styles.metricValue} ${results.total_return_pct >= 0 ? styles.positive : styles.negative}`}>
                {(results.total_return_pct >= 0 ? '+' : '') + results.total_return_pct.toFixed(2)}%
              </div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricLabel}>Sharpe Ratio</div>
              <div className={styles.metricValue}>
                {results.sharpe_ratio != null ? results.sharpe_ratio.toFixed(2) : '—'}
              </div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricLabel}>Max Drawdown</div>
              <div className={`${styles.metricValue} ${styles.negative}`}>
                -{results.max_drawdown_pct.toFixed(2)}%
              </div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricLabel}>Win Rate</div>
              <div className={styles.metricValue}>
                {results.win_rate.toFixed(1)}%
              </div>
            </div>
            <div className={styles.metricItem}>
              <div className={styles.metricLabel}>Total Trades</div>
              <div className={styles.metricValue}>
                {results.total_trades}
              </div>
            </div>
          </div>
          <div className={styles.resultsActions}>
            <Button
              variant="primary"
              size="md"
              loading={applyWeightsMut.isPending}
              onClick={handleApplyWeights}
            >
              Apply Weights
            </Button>
            <Button
              variant="secondary"
              size="md"
              loading={baselineMut.isPending}
              onClick={handleRunBaseline}
              disabled={baselineMut.isPending}
            >
              vs 500 Random
            </Button>
            <a href={`/api/export/backtest/${runId}`} className={styles.exportLink}>
              Export JSON
            </a>
          </div>
        </Card>
      )}

      {/* Random baseline comparison */}
      {baseline && <RandomBaselineCard baseline={baseline} />}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}

// ── Random Baseline Card ──────────────────────────────────────────────────────

function RandomBaselineCard({ baseline }: { baseline: RandomBaseline }) {
  const hist = baseline.histogram
  const z = baseline.z_score_vs_random
  const beatPct = baseline.pct_simulations_beaten

  const zColor = z === null ? 'var(--text-muted)'
    : z >= 1.5 ? 'var(--signal-positive)'
    : z >= 0   ? 'var(--signal-warning)'
    : 'var(--signal-negative)'

  // Build histogram bars
  const maxBin = hist ? Math.max(...hist.bins, 1) : 1
  const stratBinIdx = hist?.strategy_bin_idx ?? null

  return (
    <Card className={styles.baselineCard}>
      <div className={styles.resultsTitle}>Strategy vs 500 Random Portfolios</div>

      {/* Summary metrics */}
      <div className={styles.baselineMetrics}>
        <div className={styles.metricItem}>
          <div className={styles.metricLabel}>Z-Score</div>
          <div className={styles.metricValue} style={{ color: zColor }}>
            {z !== null ? (z >= 0 ? '+' : '') + z.toFixed(2) : '—'}
          </div>
        </div>
        <div className={styles.metricItem}>
          <div className={styles.metricLabel}>Portfolios Beaten</div>
          <div className={styles.metricValue}
            style={{ color: (beatPct ?? 0) >= 75 ? 'var(--signal-positive)' : 'var(--signal-warning)' }}>
            {beatPct !== null ? `${beatPct.toFixed(0)}%` : '—'}
          </div>
        </div>
        <div className={styles.metricItem}>
          <div className={styles.metricLabel}>Random Mean</div>
          <div className={styles.metricValue}>
            {baseline.mean_random_return >= 0 ? '+' : ''}{baseline.mean_random_return.toFixed(1)}%
          </div>
        </div>
        <div className={styles.metricItem}>
          <div className={styles.metricLabel}>Random Std Dev</div>
          <div className={styles.metricValue}>±{baseline.std_random_return.toFixed(1)}%</div>
        </div>
        <div className={styles.metricItem}>
          <div className={styles.metricLabel}>Strategy</div>
          <div className={clsx(styles.metricValue, baseline.strategy_return >= 0 ? styles.positive : styles.negative)}>
            {baseline.strategy_return >= 0 ? '+' : ''}{baseline.strategy_return.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Significance badge */}
      {z !== null && (
        <div className={styles.baselineBadgeRow}>
          <Badge variant={z >= 2 ? 'success' : z >= 1 ? 'warning' : 'neutral'}>
            {z >= 2 ? 'Statistically Significant (p<0.05)' :
             z >= 1 ? 'Marginally Significant (p<0.16)' :
             'Not Significant'}
          </Badge>
          <span className={styles.baselineNote}>
            {baseline.n_simulations} random portfolios · {baseline.portfolio_size} positions each
          </span>
        </div>
      )}

      {/* Histogram */}
      {hist && hist.bins.length > 0 && (
        <div className={styles.histogramWrap}>
          <div className={styles.histogramLabel}>Distribution of random portfolio returns</div>
          <div className={styles.histogram}>
            {hist.bins.map((count, i) => {
              const isStratBin = i === stratBinIdx
              const pct = (count / maxBin) * 100
              const binLabel = hist.bin_edges[i] !== undefined
                ? `${hist.bin_edges[i].toFixed(0)}%`
                : ''
              return (
                <div
                  key={i}
                  className={styles.histBar}
                  title={`${binLabel}: ${count} portfolios${isStratBin ? ' ← strategy' : ''}`}
                >
                  <div
                    className={styles.histFill}
                    style={{
                      height: `${pct}%`,
                      background: isStratBin
                        ? (baseline.strategy_return >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)')
                        : 'rgba(255,255,255,0.15)',
                    }}
                  />
                  {i % 4 === 0 && (
                    <span className={styles.histTick}>{binLabel}</span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </Card>
  )
}

import clsx from 'clsx'
