import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import api from '@/api/client'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './CrosscheckPage.module.css'

interface CrosscheckEntry {
  id: number
  analysis_id: number
  ticker: string
  timestamp: string
  verdict: 'confirmed' | 'contradicted' | 'uncertain'
  sources_checked: number
  key_findings: string
  confidence_boost: number
}

function verdictVariant(verdict: CrosscheckEntry['verdict']): 'success' | 'danger' | 'neutral' {
  if (verdict === 'confirmed') return 'success'
  if (verdict === 'contradicted') return 'danger'
  return 'neutral'
}

function truncate(text: string, max = 80): string {
  if (text.length <= max) return text
  return text.slice(0, max) + '…'
}

export function CrosscheckPage() {
  const { data, isLoading } = useQuery<CrosscheckEntry[]>({
    queryKey: ['crosscheck-history'],
    queryFn: () => api.get('/api/crosscheck/history').then(r => r.data),
    staleTime: 60_000,
  })

  const entries = data ?? []

  return (
    <>
      <PageHeader
        title="Crosscheck"
        subtitle="Multi-source analysis verification"
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <Card className={styles.tableCard}>
          {entries.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyTitle}>No Crosscheck Data</div>
              <p className={styles.emptyDesc}>
                Crosscheck verifies AI analyses against multiple independent sources.
                Run an analysis to generate crosscheck data.
              </p>
            </div>
          ) : (
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Timestamp</th>
                    <th>Verdict</th>
                    <th>Sources</th>
                    <th>Confidence Boost</th>
                    <th>Key Findings</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map((entry, i) => (
                    <motion.tr
                      key={entry.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.03, duration: 0.3 }}
                      className={styles.row}
                    >
                      <td><span className={styles.ticker}>{entry.ticker}</span></td>
                      <td><span className={styles.muted}>{entry.timestamp}</span></td>
                      <td>
                        <Badge variant={verdictVariant(entry.verdict)}>
                          {entry.verdict}
                        </Badge>
                      </td>
                      <td><span className={styles.mono}>{entry.sources_checked}</span></td>
                      <td>
                        <span className={entry.confidence_boost > 0 ? styles.positiveBoost : styles.neutralBoost}>
                          {entry.confidence_boost > 0 ? '+' : ''}{entry.confidence_boost}%
                        </span>
                      </td>
                      <td>
                        <span className={styles.findingsText} title={entry.key_findings}>
                          {truncate(entry.key_findings)}
                        </span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
