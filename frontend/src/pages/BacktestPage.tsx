import { useState } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ProgressBar } from '@/components/ui/ProgressBar'
import { getCsrfToken } from '@/api/csrf'
import { useBacktestProgress, useBacktestResults, useApplyWeights } from '@/api/endpoints/backtest'
import { useToastStore } from '@/stores/toastStore'
import styles from './BacktestPage.module.css'

const TODAY = new Date().toISOString().split('T')[0]

export function BacktestPage() {
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState(TODAY)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)

  const { addToast } = useToastStore()
  const { data: progress } = useBacktestProgress()
  const { data: results } = useBacktestResults(runId)
  const applyWeightsMut = useApplyWeights()

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
            <a href={`/api/export/backtest/${runId}`} className={styles.exportLink}>
              Export JSON
            </a>
          </div>
        </Card>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
