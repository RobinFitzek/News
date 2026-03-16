import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface SchedulerLog {
  id: number
  run_at: string
  trigger: string
  status: 'success' | 'error'
  tickers_scanned: number
  duration_seconds: number
  errors: string[]
}

export interface DedupAlert {
  key: string
  message: string
  severity: 'critical' | 'warning' | 'info'
  count: number
  first_seen: string
  last_seen: string
  acknowledged: boolean
}

export interface AlertSummary {
  critical: number
  warning: number
  info: number
}

export interface LoginFailSummary {
  total_failures: number
  unique_ips: number
  unique_users: number
  locked_users: string[]
}

export interface LoginFailure {
  username: string
  ip: string
  timestamp: string
  reason: string
}

export interface LogsResponse {
  scheduler_logs: SchedulerLog[]
  alerts: DedupAlert[]
  dedup_alerts: DedupAlert[]
  alert_summary: AlertSummary
  alert_filter: string
  login_fail_summary: LoginFailSummary
  recent_login_failures: LoginFailure[]
}

export function useLogs(alertFilter: string) {
  return useQuery<LogsResponse>({
    queryKey: ['logs', alertFilter],
    queryFn: () =>
      api.get('/api/logs', { params: { alert_filter: alertFilter } }).then(r => r.data),
    refetchInterval: 30_000,
  })
}
