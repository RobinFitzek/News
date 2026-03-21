import { useState } from 'react'
import { useCorporateActions } from '@/api/endpoints/corporateActions'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './CorporateActionsPage.module.css'

const ACTION_TYPES = [
  { value: '', label: 'All Types' },
  { value: 'dividend', label: 'Dividends' },
  { value: 'split', label: 'Stock Splits' },
  { value: 'merger', label: 'Mergers' },
  { value: 'spinoff', label: 'Spinoffs' },
]

function actionBadgeVariant(type: string): 'success' | 'warning' | 'neutral' | 'ghost' {
  switch (type.toLowerCase()) {
    case 'dividend': return 'success'
    case 'split': return 'warning'
    case 'merger': return 'neutral'
    default: return 'ghost'
  }
}

export function CorporateActionsPage() {
  const [tickerFilter, setTickerFilter] = useState('')
  const [typeFilter, setTypeFilter] = useState('')

  const { data, isLoading } = useCorporateActions(
    tickerFilter || undefined,
    typeFilter || undefined,
  )

  const actions = data?.actions ?? []
  const dividendSummary = data?.dividend_summary ?? {}
  const dividendTickers = Object.entries(dividendSummary).sort(([, a], [, b]) => b - a)

  return (
    <>
      <PageHeader
        title="Corporate Actions"
        subtitle="Dividends, splits, mergers, and other events"
      />

      {/* Filters */}
      <Card className={styles.filterCard} delay={0}>
        <div className={styles.filterRow}>
          <div className={styles.field}>
            <label className={styles.label}>Ticker</label>
            <input
              className={styles.input}
              placeholder="All tickers"
              value={tickerFilter}
              onChange={e => setTickerFilter(e.target.value.toUpperCase())}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label}>Type</label>
            <select
              className={styles.select}
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value)}
            >
              {ACTION_TYPES.map(t => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Dividend summary */}
          {dividendTickers.length > 0 && (
            <Card className={styles.summaryCard} delay={0.1}>
              <div className={styles.summaryTitle}>Dividend Income by Ticker</div>
              <div className={styles.summaryGrid}>
                {dividendTickers.map(([ticker, total]) => (
                  <div key={ticker} className={styles.summaryItem}>
                    <span className={styles.summaryTicker}>{ticker}</span>
                    <span className={styles.summaryAmount}>${total.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Actions table */}
          <Card className={styles.tableCard} delay={0.15}>
            {actions.length === 0 ? (
              <div className={styles.emptyState}>No corporate actions found.</div>
            ) : (
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Ticker</th>
                      <th>Type</th>
                      <th>Details</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {actions.map((a, i) => (
                      <tr key={i} className={styles.row}>
                        <td className={styles.muted}>
                          {a.date ? new Date(a.date).toLocaleDateString() : '—'}
                        </td>
                        <td>
                          <span className={styles.ticker}>{a.ticker}</span>
                        </td>
                        <td>
                          <Badge variant={actionBadgeVariant(a.action_type)}>
                            {a.action_type}
                          </Badge>
                        </td>
                        <td className={styles.details}>{a.details || '—'}</td>
                        <td className={styles.value}>
                          {a.value != null ? `$${a.value.toFixed(2)}` : '—'}
                        </td>
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
