import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useHistory } from '@/api/endpoints/history'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import type { SignalType } from '@/types/api'
import styles from './HistoryPage.module.css'

function SignalBadge({ signal }: { signal: SignalType }) {
  if (!signal) return null
  const v = signal === 'BUY' ? 'success' : signal === 'SELL' ? 'danger' : 'neutral'
  return <Badge variant={v}>{signal}</Badge>
}

function RiskCell({ value }: { value: number | null }) {
  if (value === null) return <span className={styles.muted}>—</span>
  const cls = value >= 7 ? styles.riskHigh : value >= 4 ? styles.riskMid : styles.riskLow
  return <span className={`${styles.riskValue} ${cls}`}>{value}/10</span>
}

export function HistoryPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const urlTicker = searchParams.get('ticker') ?? ''

  const [inputValue, setInputValue] = useState(urlTicker)

  const { data, isLoading } = useHistory(urlTicker || null)
  const analyses = data?.analyses ?? []

  function handleFilter() {
    const t = inputValue.trim().toUpperCase()
    if (t) {
      setSearchParams({ ticker: t })
    } else {
      setSearchParams({})
    }
  }

  function handleClear() {
    setInputValue('')
    setSearchParams({})
  }

  function formatTimestamp(ts: string) {
    return new Date(ts).toLocaleString('sv-SE', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <>
      <PageHeader
        title="Analysis History"
        subtitle="Browse all past analyses"
        actions={
          <div className={styles.headerActions}>
            <a href="/api/analysis/export.csv">
              <Button variant="ghost" size="sm">Export CSV</Button>
            </a>
            <a href="/api/export/analyses?format=json">
              <Button variant="ghost" size="sm">Export JSON</Button>
            </a>
          </div>
        }
      />

      <Card className={styles.filterCard}>
        <div className={styles.filterForm}>
          <input
            className={styles.input}
            placeholder="Filter by ticker..."
            value={inputValue}
            onChange={e => setInputValue(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleFilter()}
          />
          <Button variant="primary" size="md" onClick={handleFilter}>
            Filter
          </Button>
          {urlTicker && (
            <Button variant="ghost" size="md" onClick={handleClear}>
              Clear
            </Button>
          )}
        </div>
      </Card>

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <Card className={styles.tableCard}>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Signal</th>
                  <th>Strength</th>
                  <th>Risk Score</th>
                  <th>Geo Risk</th>
                  <th>Timestamp</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {analyses.map((analysis, i) => (
                  <motion.tr
                    key={analysis.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.03, duration: 0.3 }}
                    className={styles.row}
                  >
                    <td>
                      <span className={styles.ticker}>{analysis.ticker}</span>
                    </td>
                    <td>
                      <SignalBadge signal={analysis.signal} />
                    </td>
                    <td>
                      <span className={styles.mono}>
                        {(analysis.confidence / 100).toFixed(2)}
                      </span>
                    </td>
                    <td>
                      <RiskCell value={analysis.risk_score} />
                    </td>
                    <td>
                      <RiskCell value={analysis.geo_risk_score} />
                    </td>
                    <td>
                      <span className={styles.timestamp}>
                        {formatTimestamp(analysis.timestamp)}
                      </span>
                    </td>
                    <td>
                      <div className={styles.actions}>
                        <a href={`/stock/${analysis.ticker}?analysis_id=${analysis.id}`}>
                          <Button variant="secondary" size="sm">Details</Button>
                        </a>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>

            {analyses.length === 0 && (
              <div className={styles.emptyState}>
                <p>No analyses found.</p>
                <a href="/analyze">
                  <Button variant="primary" size="md">Run Analysis</Button>
                </a>
              </div>
            )}
          </div>
        </Card>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
