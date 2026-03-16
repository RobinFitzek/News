import { useState, useEffect } from 'react'
import { useApiStatus } from '@/api/endpoints/status'
import { useSaveSettings } from '@/api/endpoints/settings'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

export function PanelScheduler() {
  const { data: status } = useApiStatus()
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()
  const sched = status?.scheduler

  const [scanInterval, setScanInterval] = useState(6)
  const [geoInterval, setGeoInterval] = useState(6)
  const [dailyLimit, setDailyLimit] = useState(10)

  useEffect(() => {
    // Pre-populate from status if available
  }, [sched])

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'scheduler',
        scan_interval_hours: scanInterval,
        geo_scan_interval_hours: geoInterval,
        daily_limit: dailyLimit,
      })
      addToast('Scheduler settings saved', 'success')
    } catch {
      addToast('Failed to save settings', 'error')
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Scheduler</h2>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Scan Intervals</h3>

        <div className={styles.form}>
          <FieldRow
            label="Scan Interval (hours)"
            sub="How often to run the main analysis scan"
          >
            <input
              className={styles.input}
              type="number"
              min={1}
              max={24}
              value={scanInterval}
              onChange={e => setScanInterval(Number(e.target.value))}
            />
          </FieldRow>

          <FieldRow
            label="Geo Scan Interval (hours)"
            sub="How often to scan geopolitical news"
          >
            <input
              className={styles.input}
              type="number"
              min={1}
              max={24}
              value={geoInterval}
              onChange={e => setGeoInterval(Number(e.target.value))}
            />
          </FieldRow>

          <FieldRow
            label="Daily Analysis Limit"
            sub="Maximum analyses per day to control API costs"
          >
            <input
              className={styles.input}
              type="number"
              min={1}
              max={100}
              value={dailyLimit}
              onChange={e => setDailyLimit(Number(e.target.value))}
            />
          </FieldRow>

          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={saveMut.isPending} onClick={handleSave}>
              Save Scheduler Settings
            </Button>
          </div>
        </div>
      </Card>

      {/* Current state info */}
      {sched && (
        <Card className={styles.section}>
          <h3 className={styles.sectionTitle}>Current Status</h3>
          <div className={styles.list}>
            <StatusInfo label="State" value={sched.is_scanning ? 'Scanning' : sched.is_running ? 'Running' : 'Stopped'} />
            <StatusInfo label="Queue" value={String(sched.queue_count)} />
            <StatusInfo label="Pending" value={String(sched.pending_count)} />
            <StatusInfo label="Last Run" value={sched.last_run ?? 'Never'} />
          </div>
        </Card>
      )}
    </div>
  )
}

function FieldRow({ label, sub, children }: { label: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className={styles.field}>
      <div>
        <div className={styles.fieldLabel}>{label}</div>
        {sub && <div className={styles.toggleSub}>{sub}</div>}
      </div>
      {children}
    </div>
  )
}

function StatusInfo({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.toggleRow}>
      <span className={styles.toggleLabel}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>{value}</span>
    </div>
  )
}

import type React from 'react'
