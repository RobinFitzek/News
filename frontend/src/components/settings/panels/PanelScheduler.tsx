import { useState, useEffect } from 'react'
import { useApiStatus } from '@/api/endpoints/status'
import { useSettingsData, useSaveSettings } from '@/api/endpoints/settings'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

export function PanelScheduler() {
  const { data: status } = useApiStatus()
  const { data: settingsData } = useSettingsData()
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()
  const sched = status?.scheduler

  const [scanInterval,       setScanInterval]       = useState(2)
  const [activeStart,        setActiveStart]        = useState('08:00')
  const [activeEnd,          setActiveEnd]          = useState('22:00')
  const [deepSleepEnabled,   setDeepSleepEnabled]   = useState(false)
  const [deepSleepStart,     setDeepSleepStart]     = useState('22:00')
  const [deepSleepEnd,       setDeepSleepEnd]       = useState('07:00')
  const [deepSleepIntensity, setDeepSleepIntensity] = useState<'light'|'deep'|'hibernate'>('deep')
  const [fullWeekends,       setFullWeekends]       = useState(false)
  const [cstateMode,         setCstateMode]         = useState(true)

  // Populate from API once loaded
  useEffect(() => {
    const s = settingsData?.scheduler
    if (!s) return
    if (s.scan_interval_hours != null)  setScanInterval(s.scan_interval_hours)
    if (s.active_hours_start)           setActiveStart(s.active_hours_start)
    if (s.active_hours_end)             setActiveEnd(s.active_hours_end)
    if (s.deep_sleep_enabled != null)   setDeepSleepEnabled(s.deep_sleep_enabled)
    if (s.deep_sleep_start)             setDeepSleepStart(s.deep_sleep_start)
    if (s.deep_sleep_end)               setDeepSleepEnd(s.deep_sleep_end)
    if (s.deep_sleep_intensity)         setDeepSleepIntensity(s.deep_sleep_intensity as typeof deepSleepIntensity)
    if (s.deep_sleep_full_weekends != null) setFullWeekends(s.deep_sleep_full_weekends)
    if (s.cstate_overnight_mode != null)    setCstateMode(s.cstate_overnight_mode)
  }, [settingsData])

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'scheduler',
        scan_interval_hours: scanInterval,
        active_hours_start: activeStart,
        active_hours_end: activeEnd,
        deep_sleep_enabled: deepSleepEnabled,
        deep_sleep_start: deepSleepStart,
        deep_sleep_end: deepSleepEnd,
        deep_sleep_intensity: deepSleepIntensity,
        deep_sleep_full_weekends: fullWeekends,
        cstate_overnight_mode: cstateMode,
      })
      addToast('Scheduler settings saved', 'success')
    } catch {
      addToast('Failed to save settings', 'error')
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Scheduler</h2>

      {/* ── Scan intervals ──────────────────────────────────────────────── */}
      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Scan Intervals</h3>
        <div className={styles.form}>
          <FieldRow
            label="Scan Interval (hours)"
            sub="How often the main analysis pipeline runs"
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
            label="Active Hours Start"
            sub="Scans only run inside this window (24h HH:MM)"
          >
            <input
              className={styles.input}
              type="text"
              pattern="\d{2}:\d{2}"
              placeholder="08:00"
              value={activeStart}
              onChange={e => setActiveStart(e.target.value)}
            />
          </FieldRow>

          <FieldRow
            label="Active Hours End"
            sub="Scans stop after this time (24h HH:MM)"
          >
            <input
              className={styles.input}
              type="text"
              pattern="\d{2}:\d{2}"
              placeholder="22:00"
              value={activeEnd}
              onChange={e => setActiveEnd(e.target.value)}
            />
          </FieldRow>
        </div>
      </Card>

      {/* ── CPU C-state / overnight mode ────────────────────────────────── */}
      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>CPU C-State Overnight Mode</h3>
        <p className={styles.sectionSub}>
          Converts continuous market-data polling jobs to strict market-hours
          CronTriggers and suspends remaining interval jobs during the deep
          sleep window — allowing the CPU to reach C6/C7/C8 overnight.
        </p>

        <div className={styles.form}>
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>C-State Overnight Mode</div>
              <div className={styles.toggleSub}>
                Broker sync, P&amp;L snapshots, price alerts, and RSS geo
                triggers are already converted to market-hours-only CronTriggers
                (never fire between 16:05 and 09:00). This toggle additionally
                pauses the main scan and geo scan jobs during the deep sleep
                window below.
              </div>
            </div>
            <button
              className={`${styles.toggle} ${cstateMode ? styles.toggleOn : ''}`}
              onClick={() => setCstateMode(v => !v)}
              role="switch"
              aria-checked={cstateMode}
            />
          </div>
        </div>
      </Card>

      {/* ── Deep sleep window ───────────────────────────────────────────── */}
      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Deep Sleep Window</h3>
        <p className={styles.sectionSub}>
          During the sleep window the scheduler enters reduced-activity mode.
          With C-State Overnight Mode enabled, interval jobs are fully paused
          so the CPU can sustain C8 until the wake time.
        </p>

        <div className={styles.form}>
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Enable Deep Sleep</div>
              <div className={styles.toggleSub}>Activate the overnight sleep window</div>
            </div>
            <button
              className={`${styles.toggle} ${deepSleepEnabled ? styles.toggleOn : ''}`}
              onClick={() => setDeepSleepEnabled(v => !v)}
              role="switch"
              aria-checked={deepSleepEnabled}
            />
          </div>

          <FieldRow label="Sleep Start" sub="Jobs pause at this time (HH:MM)">
            <input
              className={styles.input}
              type="text"
              pattern="\d{2}:\d{2}"
              placeholder="22:00"
              value={deepSleepStart}
              disabled={!deepSleepEnabled}
              onChange={e => setDeepSleepStart(e.target.value)}
            />
          </FieldRow>

          <FieldRow label="Wake Time" sub="Jobs resume at this time (HH:MM)">
            <input
              className={styles.input}
              type="text"
              pattern="\d{2}:\d{2}"
              placeholder="07:00"
              value={deepSleepEnd}
              disabled={!deepSleepEnabled}
              onChange={e => setDeepSleepEnd(e.target.value)}
            />
          </FieldRow>

          <FieldRow label="Intensity" sub="How aggressively to reduce activity">
            <select
              className={styles.input}
              value={deepSleepIntensity}
              disabled={!deepSleepEnabled}
              onChange={e => setDeepSleepIntensity(e.target.value as typeof deepSleepIntensity)}
            >
              <option value="light">Light — skip most scans</option>
              <option value="deep">Deep — skip all scans (default)</option>
              <option value="hibernate">Hibernate — skip everything</option>
            </select>
          </FieldRow>

          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Full Weekend Sleep</div>
              <div className={styles.toggleSub}>Extend deep sleep all day Saturday &amp; Sunday</div>
            </div>
            <button
              className={`${styles.toggle} ${fullWeekends ? styles.toggleOn : ''}`}
              onClick={() => setFullWeekends(v => !v)}
              disabled={!deepSleepEnabled}
              role="switch"
              aria-checked={fullWeekends}
            />
          </div>
        </div>

        {/* Visual: what fires overnight */}
        <div className={styles.overnightInfo}>
          <div className={styles.overnightTitle}>What fires between {deepSleepStart} and {deepSleepEnd}</div>
          <div className={styles.overnightGrid}>
            <OvernightRow job="Broker Sync (5 min)" fires={false} reason="CronTrigger Mon–Fri 09–15 only" />
            <OvernightRow job="P&L Snapshot (15 min)" fires={false} reason="CronTrigger Mon–Fri 09–15 only" />
            <OvernightRow job="Price Alerts (15 min)" fires={false} reason="CronTrigger Mon–Fri 09–15 only" />
            <OvernightRow job="RSS Geo Trigger (15 min)" fires={false} reason="CronTrigger 07–22 only" />
            <OvernightRow job="Geo Scan" fires={false} reason="Fixed 08:00, 14:00, 20:00 only" />
            <OvernightRow job="Main Scan" fires={!(deepSleepEnabled && cstateMode)} reason={deepSleepEnabled && cstateMode ? 'Paused by deep sleep' : 'Fires every ' + scanInterval + 'h (enable sleep to pause)'} />
            <OvernightRow job="Health Check" fires={true} reason="03:00 daily (lightweight)" />
            <OvernightRow job="DB Backup" fires={true} reason="03:30 daily (lightweight)" />
          </div>
        </div>
      </Card>

      <div className={styles.saveRow}>
        <Button variant="primary" size="md" loading={saveMut.isPending} onClick={handleSave}>
          Save Scheduler Settings
        </Button>
      </div>

      {/* ── Current status ───────────────────────────────────────────────── */}
      {sched && (
        <Card className={styles.section}>
          <h3 className={styles.sectionTitle}>Current Status</h3>
          <div className={styles.list}>
            <StatusInfo label="State" value={sched.is_scanning ? 'Scanning' : sched.is_running ? 'Running' : 'Stopped'} />
            <StatusInfo label="Deep Sleep" value={sched.is_sleeping ? 'Active' : 'Inactive'} />
            <StatusInfo label="Market" value={sched.is_market_open ? 'Open' : 'Closed'} />
          </div>
        </Card>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function FieldRow({
  label, sub, children,
}: {
  label: string; sub?: string; children: React.ReactNode
}) {
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
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
        {value}
      </span>
    </div>
  )
}

function OvernightRow({
  job, fires, reason,
}: {
  job: string; fires: boolean; reason: string
}) {
  return (
    <div className={styles.overnightRow}>
      <span
        className={styles.overnightDot}
        style={{ background: fires ? 'var(--signal-warning)' : 'var(--signal-positive)' }}
      />
      <span className={styles.overnightJob}>{job}</span>
      <span className={styles.overnightReason}>{reason}</span>
    </div>
  )
}

import type React from 'react'
