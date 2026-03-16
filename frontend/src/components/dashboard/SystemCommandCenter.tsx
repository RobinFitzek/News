import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApiStatus, useScanProgress } from '@/api/endpoints/status'
import { useStartScheduler, useStopScheduler, useScanNow } from '@/api/endpoints/scheduler'
import { StatusDot } from '@/components/ui/StatusDot'
import { Button } from '@/components/ui/Button'
import { ProgressBar } from '@/components/ui/ProgressBar'
import { Badge } from '@/components/ui/Badge'
import type { SchedulerState } from '@/types/api'
import { formatDistanceToNow } from 'date-fns'
import styles from './SystemCommandCenter.module.css'

function Clock() {
  const [time, setTime] = useState(() => new Date())
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])
  return (
    <span className={styles.clock}>
      {time.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
    </span>
  )
}

function relativeTime(ts: string | null): string {
  if (!ts) return 'Never'
  try {
    return formatDistanceToNow(new Date(ts), { addSuffix: true })
  } catch {
    return ts
  }
}

export function SystemCommandCenter() {
  const { data: status } = useApiStatus()
  const scheduler = status?.scheduler
  const isScanning = scheduler?.is_scanning ?? false
  const isRunning = scheduler?.is_running ?? false

  const { data: scanProg } = useScanProgress(isScanning)
  const startMut = useStartScheduler()
  const stopMut = useStopScheduler()
  const scanNowMut = useScanNow()

  const state: SchedulerState = isScanning ? 'scanning' : isRunning ? 'running' : 'stopped'

  const stateLabel =
    state === 'scanning' ? 'Scanning' :
    state === 'running'  ? 'Engine Active' :
    'System Halted'

  return (
    <motion.div
      className={styles.bar}
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.34, 1.2, 0.64, 1] }}
    >
      {/* Main command row */}
      <div className={styles.commandRow}>
        {/* Left — engine status */}
        <div className={styles.leftGroup}>
          <StatusDot status={state} size="md" />
          <span className={styles.stateLabel}>{stateLabel}</span>
          {status?.scheduler.state && (
            <Badge variant="ghost">{status.scheduler.state}</Badge>
          )}
        </div>

        {/* Center — scan now */}
        <div className={styles.centerGroup}>
          <Button
            variant="primary"
            size="sm"
            loading={scanNowMut.isPending || isScanning}
            onClick={() => scanNowMut.mutate()}
            disabled={isScanning}
          >
            {isScanning ? 'Scanning…' : 'Scan Now'}
          </Button>
        </div>

        {/* Right — clock + start/stop */}
        <div className={styles.rightGroup}>
          <Clock />
          <div className={styles.sep} />
          {isRunning ? (
            <Button
              variant="danger"
              size="sm"
              loading={stopMut.isPending}
              onClick={() => stopMut.mutate()}
            >
              Stop Engine
            </Button>
          ) : (
            <Button
              variant="success"
              size="sm"
              loading={startMut.isPending}
              onClick={() => startMut.mutate()}
            >
              Start Engine
            </Button>
          )}
        </div>
      </div>

      {/* Scan progress bar */}
      {isScanning && scanProg && (
        <motion.div
          className={styles.scanProgress}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <ProgressBar
            value={scanProg.percent}
            variant="warning"
            height={3}
            animated={false}
          />
          <div className={styles.scanLabel}>
            <span className={styles.scanTicker}>{scanProg.current_ticker ?? '—'}</span>
            <span className={styles.scanCount}>
              {scanProg.completed}/{scanProg.total} — {scanProg.stage ?? ''}
            </span>
          </div>
        </motion.div>
      )}

      {/* Diagnostic row */}
      <div className={styles.diagnosticRow}>
        <DiagItem label="Last Run"   value={relativeTime(scheduler?.last_run ?? null)} />
        <div className={styles.sep} />
        <DiagItem label="Queue"      value={String(scheduler?.queue_count ?? 0)} mono />
        <div className={styles.sep} />
        <DiagItem label="Pending"    value={String(scheduler?.pending_count ?? 0)} mono />
        <div className={styles.sep} />
        <DiagItem label="Accuracy"   value={status?.accuracy ? `${status.accuracy.toFixed(1)}%` : '—'} mono />
        <div className={styles.sep} />
        <DiagItem label="Watchlist"  value={String(status?.watchlist_count ?? 0)} mono />
        <div className={styles.sep} />
        <DiagItem label="Stale"      value={String(status?.stale_analyses ?? 0)} mono />
        {status?.ml_status && (
          <>
            <div className={styles.sep} />
            <DiagItem label="ML" value={status.ml_status} />
          </>
        )}
      </div>
    </motion.div>
  )
}

function DiagItem({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className={styles.diagItem}>
      <span className={styles.diagLabel}>{label}</span>
      <span className={mono ? `${styles.diagValue} ${styles.mono}` : styles.diagValue}>
        {value}
      </span>
    </div>
  )
}
