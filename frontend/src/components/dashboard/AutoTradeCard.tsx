import { useAutoTradeStatus } from '@/api/endpoints/autoTrade'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './AutoTradeCard.module.css'

function fmtPnl(n: number | null): string {
  if (n === null) return '—'
  return `${n >= 0 ? '+' : ''}$${Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

export function AutoTradeCard() {
  const { data, isLoading } = useAutoTradeStatus()

  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <h2 className={styles.title}>Auto-Trade</h2>
        {data && (
          <Badge variant={data.enabled ? 'success' : 'neutral'} size="xs">
            {data.enabled ? 'Active' : 'Paused'}
          </Badge>
        )}
      </div>

      {isLoading && (
        <div className={styles.loading}><Spinner /></div>
      )}

      {data && (
        <div className={styles.metrics}>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Mode</span>
            <span className={styles.metricValue}>{data.mode}</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Open</span>
            <span className={styles.metricValue}>{data.open_positions}</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Trades</span>
            <span className={styles.metricValue}>{data.total_trades}</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Win Rate</span>
            <span className={styles.metricValue}>
              {data.win_rate !== null ? `${data.win_rate.toFixed(1)}%` : '—'}
            </span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>P&L</span>
            <span className={`${styles.metricValue} ${
              data.total_pnl !== null
                ? data.total_pnl >= 0 ? styles.positive : styles.negative
                : ''
            }`}>
              {fmtPnl(data.total_pnl)}
            </span>
          </div>
          {data.last_trade_date && (
            <div className={styles.metric}>
              <span className={styles.metricLabel}>Last Trade</span>
              <span className={styles.metricMuted}>{data.last_trade_date}</span>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}
