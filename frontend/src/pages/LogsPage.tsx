import { useState } from 'react'
import { motion } from 'framer-motion'
import clsx from 'clsx'
import { useLogs } from '@/api/endpoints/logs'
import type { DedupAlert } from '@/api/endpoints/logs'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './LogsPage.module.css'

const ALERT_FILTERS = [
  { value: 'active', label: 'Active Alerts' },
  { value: 'all', label: 'All Alerts' },
]

function severityVariant(s: DedupAlert['severity']): 'danger' | 'warning' | 'neutral' {
  if (s === 'critical') return 'danger'
  if (s === 'warning') return 'warning'
  return 'neutral'
}

function formatTs(ts: string): string {
  return new Date(ts).toLocaleString('sv-SE', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function LogsPage() {
  const [alertFilter, setAlertFilter] = useState<'active' | 'all'>('active')
  const { data, isLoading } = useLogs(alertFilter)

  const alerts = data?.dedup_alerts ?? []
  const schedulerLogs = data?.scheduler_logs ?? []
  const alertSummary = data?.alert_summary ?? { critical: 0, warning: 0, info: 0 }
  const loginSummary = data?.login_fail_summary ?? {
    total_failures: 0,
    unique_ips: 0,
    unique_users: 0,
    locked_users: [],
  }
  const loginFailures = data?.recent_login_failures ?? []

  return (
    <>
      <PageHeader
        title="Logs"
        subtitle="System activity and alerts"
        actions={
          <div className={styles.headerMeta}>
            <div className={styles.refreshDot} />
            auto-refresh 30s
          </div>
        }
      />

      {/* Alert filter tabs */}
      <div className={styles.filters}>
        {ALERT_FILTERS.map(f => (
          <button
            key={f.value}
            className={clsx(styles.filterTab, alertFilter === f.value && styles.activeTab)}
            onClick={() => setAlertFilter(f.value as 'active' | 'all')}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Alert summary row */}
      <div className={styles.summaryRow}>
        <Card className={styles.summaryCard} delay={0}>
          <div className={clsx(styles.summaryCount, styles.countCritical)}>
            {alertSummary.critical}
          </div>
          <div className={styles.summaryLabel}>Critical</div>
        </Card>
        <Card className={styles.summaryCard} delay={0.05}>
          <div className={clsx(styles.summaryCount, styles.countWarning)}>
            {alertSummary.warning}
          </div>
          <div className={styles.summaryLabel}>Warning</div>
        </Card>
        <Card className={styles.summaryCard} delay={0.1}>
          <div className={clsx(styles.summaryCount, styles.countInfo)}>
            {alertSummary.info}
          </div>
          <div className={styles.summaryLabel}>Info</div>
        </Card>
      </div>

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          {/* Active Alerts */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Active Alerts</div>
            {alerts.length === 0 ? (
              <div className={styles.emptyState}>No alerts to display.</div>
            ) : (
              <div className={styles.alertList}>
                {alerts.map((alert, i) => (
                  <motion.div
                    key={alert.key}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04, duration: 0.3 }}
                    className={clsx(
                      styles.alertItem,
                      styles[`severity${alert.severity.charAt(0).toUpperCase()}${alert.severity.slice(1)}`],
                      alert.acknowledged && styles.acknowledged
                    )}
                  >
                    <div className={styles.alertBody}>
                      <div className={styles.alertMessage}>{alert.message}</div>
                      <div className={styles.alertMeta}>
                        <span>First: {formatTs(alert.first_seen)}</span>
                        <span>·</span>
                        <span>Last: {formatTs(alert.last_seen)}</span>
                        {alert.acknowledged && (
                          <>
                            <span>·</span>
                            <span>acknowledged</span>
                          </>
                        )}
                      </div>
                    </div>
                    <div className={styles.alertRight}>
                      <Badge variant={severityVariant(alert.severity)}>
                        {alert.severity}
                      </Badge>
                      {alert.count > 1 && (
                        <Badge variant="ghost">×{alert.count}</Badge>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {/* Scheduler Logs */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Scheduler Logs</div>
            <Card className={styles.tableCard}>
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Trigger</th>
                      <th>Status</th>
                      <th>Scanned</th>
                      <th>Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {schedulerLogs.map((log, i) => (
                      <motion.tr
                        key={log.id}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.04, duration: 0.3 }}
                        className={styles.row}
                      >
                        <td><span className={styles.timestamp}>{formatTs(log.run_at)}</span></td>
                        <td><span className={styles.mono}>{log.trigger}</span></td>
                        <td>
                          <Badge variant={log.status === 'success' ? 'success' : 'danger'}>
                            {log.status}
                          </Badge>
                        </td>
                        <td><span className={styles.mono}>{log.tickers_scanned}</span></td>
                        <td>
                          <span className={styles.timestamp}>{log.duration_seconds.toFixed(1)}s</span>
                        </td>
                      </motion.tr>
                    ))}
                    {schedulerLogs.length === 0 && (
                      <tr>
                        <td colSpan={5}>
                          <div className={styles.emptyState}>No scheduler logs.</div>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          {/* Login Security */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Login Security</div>
            <div className={styles.loginStats}>
              <div className={styles.statCard}>
                <div className={styles.statValue}>{loginSummary.total_failures}</div>
                <div className={styles.statLabel}>Total Failures</div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statValue}>{loginSummary.unique_ips}</div>
                <div className={styles.statLabel}>Unique IPs</div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statValue}>{loginSummary.unique_users}</div>
                <div className={styles.statLabel}>Unique Users</div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statValue}>{loginSummary.locked_users.length}</div>
                <div className={styles.statLabel}>Locked Users</div>
                {loginSummary.locked_users.length > 0 && (
                  <div className={styles.lockedUsers}>
                    {loginSummary.locked_users.map(u => (
                      <Badge key={u} variant="danger">{u}</Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <Card className={styles.tableCard}>
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Username</th>
                      <th>IP</th>
                      <th>Timestamp</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {loginFailures.map((f, i) => (
                      <motion.tr
                        key={`${f.username}-${f.timestamp}-${i}`}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.04, duration: 0.3 }}
                        className={styles.row}
                      >
                        <td><span className={styles.mono}>{f.username}</span></td>
                        <td><span className={styles.mono}>{f.ip}</span></td>
                        <td><span className={styles.timestamp}>{formatTs(f.timestamp)}</span></td>
                        <td><span className={styles.muted}>{f.reason}</span></td>
                      </motion.tr>
                    ))}
                    {loginFailures.length === 0 && (
                      <tr>
                        <td colSpan={4}>
                          <div className={styles.emptyState}>No recent login failures.</div>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
