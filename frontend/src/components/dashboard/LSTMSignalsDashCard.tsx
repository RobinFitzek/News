import { useNavigate } from 'react-router-dom'
import { useLSTMSignals, useLSTMPerformance } from '@/api/endpoints/lstm'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './LSTMSignalsDashCard.module.css'

export function LSTMSignalsDashCard() {
  const { data: signalsData, isLoading } = useLSTMSignals()
  const { data: perf } = useLSTMPerformance()
  const navigate = useNavigate()

  const signals = signalsData?.signals?.filter(s => s.buy_signal) ?? []
  const threshold = signalsData?.threshold ?? 0.5

  return (
    <Card
      className={styles.card}
      onClick={() => navigate('/lstm')}
      style={{ cursor: 'pointer' }}
    >
      <div className={styles.header}>
        <h2 className={styles.title}>LSTM Signals</h2>
        <Badge variant={signals.length > 0 ? 'success' : 'ghost'}>
          {isLoading ? '…' : `${signals.length} buy`}
        </Badge>
      </div>

      {isLoading && <div className={styles.loading}><Spinner /></div>}

      {!isLoading && (
        <>
          {/* Performance strip */}
          <div className={styles.perfStrip}>
            <div className={styles.perfItem}>
              <span className={styles.perfNum}>
                {perf?.win_rate_pct !== null && perf?.win_rate_pct !== undefined
                  ? `${perf.win_rate_pct.toFixed(0)}%`
                  : '—'}
              </span>
              <span className={styles.perfLabel}>Win Rate</span>
            </div>
            <div className={styles.perfDivider} />
            <div className={styles.perfItem}>
              <span className={styles.perfNum}>
                {perf?.cagr_pct !== null && perf?.cagr_pct !== undefined
                  ? `${perf.cagr_pct >= 0 ? '+' : ''}${perf.cagr_pct.toFixed(1)}%`
                  : '—'}
              </span>
              <span className={styles.perfLabel}>CAGR</span>
            </div>
            <div className={styles.perfDivider} />
            <div className={styles.perfItem}>
              <span className={styles.perfNum}>{perf?.completed_trades ?? 0}</span>
              <span className={styles.perfLabel}>Trades</span>
            </div>
          </div>

          {signals.length > 0 ? (
            <div className={styles.list}>
              {signals.slice(0, 5).map(s => (
                <div key={s.ticker} className={styles.item}>
                  <span className={styles.itemTicker}>{s.ticker}</span>
                  {/* Mini confidence track */}
                  <div className={styles.confTrack}>
                    {/* threshold tick */}
                    <div
                      className={styles.confTick}
                      style={{ left: `${threshold * 100}%` }}
                    />
                    <div
                      className={styles.confFill}
                      style={{ width: `${(s.confidence ?? 0) * 100}%` }}
                    />
                  </div>
                  <span className={styles.confValue}>
                    {s.confidence !== null ? `${(s.confidence * 100).toFixed(0)}%` : '—'}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className={styles.empty}>
              {signalsData?.count === 0
                ? 'Train model first — go to LSTM Model'
                : 'No buy signals above threshold'}
            </div>
          )}
        </>
      )}
    </Card>
  )
}
