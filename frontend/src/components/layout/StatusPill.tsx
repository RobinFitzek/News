import { useApiStatus } from '@/api/endpoints/status'
import { StatusDot } from '@/components/ui/StatusDot'
import type { SchedulerState } from '@/types/api'
import styles from './StatusPill.module.css'

function getState(status: ReturnType<typeof useApiStatus>['data']): {
  state: SchedulerState
  label: string
} {
  if (!status) return { state: 'idle', label: 'Loading' }
  const s = status.scheduler
  if (s.is_scanning) return { state: 'scanning', label: 'Scanning' }
  if (s.is_running)  return { state: 'running',  label: 'Engine Active' }
  return { state: 'stopped', label: 'System Halted' }
}

export function StatusPill() {
  const { data } = useApiStatus()
  const { state, label } = getState(data)

  return (
    <div className={styles.pill}>
      <StatusDot status={state} />
      <span className={styles.label}>{label}</span>
    </div>
  )
}
