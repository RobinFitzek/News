import { motion } from 'framer-motion'
import { usePaperTrading } from '@/api/endpoints/paperTrading'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './PaperTradingPage.module.css'

interface TradeLog {
  id: number
  date: string
  ticker: string
  action: 'BUY' | 'SELL'
  shares: number
  price: number
  value: number
  reason: string | null
}

function fmtCurrency(n: number): string {
  return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtPct(n: number): string {
  return (n >= 0 ? '+' : '') + n.toFixed(2) + '%'
}

function PnlCell({ value, pct }: { value: number; pct?: number }) {
  const cls = value >= 0 ? styles.positive : styles.negative
  return (
    <span className={cls}>
      {fmtCurrency(value)}
      {pct !== undefined && <> ({fmtPct(pct)})</>}
    </span>
  )
}

export function PaperTradingPage() {
  const { data, isLoading } = usePaperTrading()

  const actions = (
    <div className={styles.headerActions}>
      <a href="/api/export/paper-trades" className={styles.exportLink}>
        Export CSV
      </a>
    </div>
  )

  if (isLoading) {
    return (
      <>
        <PageHeader
          title="Paper Trading"
          subtitle="Simulated portfolio tracking AI signals"
          actions={actions}
        />
        <div className={styles.loading}>
          <Spinner size="lg" />
        </div>
      </>
    )
  }

  const summary = data
  const positions = summary?.positions ?? []
  const tradeLogs: TradeLog[] = (summary as unknown as { trade_log?: TradeLog[] })?.trade_log ?? []

  return (
    <>
      <PageHeader
        title="Paper Trading"
        subtitle="Simulated portfolio tracking AI signals"
        actions={actions}
      />

      {/* Summary metrics */}
      <div className={styles.metricsRow}>
        <Card className={styles.metricCard} delay={0}>
          <div className={styles.metricLabel}>Portfolio Value</div>
          <div className={styles.metricValue}>
            {summary ? fmtCurrency(summary.total_value) : '—'}
          </div>
        </Card>
        <Card className={styles.metricCard} delay={0.05}>
          <div className={styles.metricLabel}>P&amp;L</div>
          <div className={`${styles.metricValue} ${summary && summary.total_pnl >= 0 ? styles.positive : styles.negative}`}>
            {summary ? fmtCurrency(summary.total_pnl) : '—'}
          </div>
        </Card>
        <Card className={styles.metricCard} delay={0.1}>
          <div className={styles.metricLabel}>Win Rate</div>
          <div className={styles.metricValue}>
            {summary ? summary.win_rate.toFixed(1) + '%' : '—'}
          </div>
        </Card>
        <Card className={styles.metricCard} delay={0.15}>
          <div className={styles.metricLabel}>Sharpe Ratio</div>
          <div className={styles.metricValue}>
            {summary?.sharpe_ratio != null ? summary.sharpe_ratio.toFixed(2) : '—'}
          </div>
        </Card>
      </div>

      {/* Positions table */}
      <Card className={styles.tableCard} delay={0.2}>
        <div className={styles.sectionTitle}>Positions</div>
        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Shares</th>
                <th>Avg Cost</th>
                <th>Current Price</th>
                <th>Market Value</th>
                <th>P&amp;L</th>
                <th>P&amp;L %</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, i) => (
                <motion.tr
                  key={pos.ticker}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03, duration: 0.3 }}
                  className={styles.row}
                >
                  <td><span className={styles.ticker}>{pos.ticker}</span></td>
                  <td><span className={styles.mono}>{pos.shares}</span></td>
                  <td><span className={styles.mono}>{fmtCurrency(pos.avg_cost)}</span></td>
                  <td><span className={styles.mono}>{fmtCurrency(pos.current_price)}</span></td>
                  <td><span className={styles.mono}>{fmtCurrency(pos.market_value)}</span></td>
                  <td><PnlCell value={pos.pnl} /></td>
                  <td>
                    <span className={pos.pnl_pct >= 0 ? styles.positive : styles.negative}>
                      {fmtPct(pos.pnl_pct)}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
          {positions.length === 0 && (
            <div className={styles.emptyState}>No open positions.</div>
          )}
        </div>
      </Card>

      {/* Trade log table */}
      <Card className={styles.tableCard} delay={0.25}>
        <div className={styles.sectionTitle}>Recent Trades</div>
        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Date</th>
                <th>Ticker</th>
                <th>Action</th>
                <th>Shares</th>
                <th>Price</th>
                <th>Value</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {tradeLogs.map((trade, i) => (
                <motion.tr
                  key={trade.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03, duration: 0.3 }}
                  className={styles.row}
                >
                  <td><span className={styles.muted}>{trade.date}</span></td>
                  <td><span className={styles.ticker}>{trade.ticker}</span></td>
                  <td>
                    <Badge variant={trade.action === 'BUY' ? 'success' : 'danger'}>
                      {trade.action}
                    </Badge>
                  </td>
                  <td><span className={styles.mono}>{trade.shares}</span></td>
                  <td><span className={styles.mono}>{fmtCurrency(trade.price)}</span></td>
                  <td><span className={styles.mono}>{fmtCurrency(trade.value)}</span></td>
                  <td>
                    <span className={styles.reasonText} title={trade.reason ?? ''}>
                      {trade.reason ?? '—'}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
          {tradeLogs.length === 0 && (
            <div className={styles.emptyState}>No trades recorded yet.</div>
          )}
        </div>
      </Card>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
