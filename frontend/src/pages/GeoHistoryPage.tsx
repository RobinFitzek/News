import { useQuery } from '@tanstack/react-query'
import api from '@/api/client'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import type { GeoScan } from '@/types/api'
import styles from './GeoHistoryPage.module.css'

function severityColor(score: number): string {
  if (score >= 7) return '#ef4444'
  if (score >= 4) return '#f59e0b'
  return '#10b981'
}

export function GeoHistoryPage() {
  const { data, isLoading } = useQuery<GeoScan>({
    queryKey: ['geopolitical'],
    queryFn: () => api.get('/api/geopolitical').then(r => r.data),
    staleTime: 120_000,
  })

  const events = (data?.events ?? []).slice().sort((a, b) => b.severity - a.severity)

  return (
    <>
      <PageHeader
        title="Geopolitical History"
        subtitle="Risk events and portfolio impact analysis"
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Severity summary card */}
          {data && (
            <Card className={styles.severityCard} delay={0}>
              <div className={styles.severityHeader}>
                <div className={styles.severityTitle}>Current Severity</div>
                <div
                  className={styles.severityScore}
                  style={{ color: severityColor(data.overall_severity) }}
                >
                  {data.overall_severity.toFixed(1)}
                  <span className={styles.scoreSuffix}>/10</span>
                </div>
              </div>
              {data.summary && (
                <p className={styles.summaryText}>{data.summary}</p>
              )}
              {data.portfolio_impact && (
                <div>
                  <div className={styles.impactLabel}>Portfolio Impact</div>
                  <p className={styles.impactText}>{data.portfolio_impact}</p>
                </div>
              )}
            </Card>
          )}

          {/* Events table */}
          <Card className={styles.tableCard} delay={0.1}>
            {events.length === 0 ? (
              <div className={styles.emptyState}>No geopolitical events recorded.</div>
            ) : (
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Headline</th>
                      <th>Region</th>
                      <th>Severity</th>
                      <th>Affected Sectors</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((event, i) => (
                      <tr key={i} className={styles.row}>
                        <td>
                          <span className={styles.headlineText}>{event.headline}</span>
                        </td>
                        <td>
                          <Badge variant="ghost" className={styles.regionBadge}>
                            {event.region}
                          </Badge>
                        </td>
                        <td>
                          <span
                            className={styles.severityNum}
                            style={{ color: severityColor(event.severity) }}
                          >
                            {event.severity.toFixed(1)}
                          </span>
                        </td>
                        <td>
                          <span className={styles.sectorsText}>
                            {event.affected_sectors?.join(', ') ?? '—'}
                          </span>
                        </td>
                        <td>
                          <span className={styles.muted}>{event.timestamp}</span>
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
