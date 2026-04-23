import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useChangePassword, useSettingsData, useSaveSettings, useSessions, useLogoutSession, useLogoutOtherSessions } from '@/api/endpoints/settings'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

export function PanelSecurity() {
  const changePwd = useChangePassword()
  const { data: settings } = useSettingsData()
  const saveMut = useSaveSettings()
  const { data: sessionsData, refetch: refetchSessions } = useSessions()
  const logoutSession = useLogoutSession()
  const logoutOthers = useLogoutOtherSessions()
  const { addToast } = useToastStore()

  const [current, setCurrent] = useState('')
  const [newPwd, setNewPwd] = useState('')
  const [confirm, setConfirm] = useState('')

  const [maxFailed, setMaxFailed] = useState(5)
  const [attemptWindow, setAttemptWindow] = useState(15)
  const [lockoutMinutes, setLockoutMinutes] = useState(30)

  useEffect(() => {
    if (!settings) return
    const s = settings as Record<string, unknown>
    setMaxFailed(Number(s.auth_max_failed_attempts ?? 5))
    setAttemptWindow(Number(s.auth_attempt_window_minutes ?? 15))
    setLockoutMinutes(Number(s.auth_lockout_minutes ?? 30))
  }, [settings])

  async function handleChangePassword() {
    if (newPwd !== confirm) {
      addToast('Passwords do not match', 'error')
      return
    }
    try {
      await changePwd.mutateAsync({ current_password: current, new_password: newPwd, confirm_password: confirm })
      addToast('Password changed successfully', 'success')
      setCurrent(''); setNewPwd(''); setConfirm('')
    } catch {
      addToast('Failed to change password', 'error')
    }
  }

  async function handleLogoutSession(id: string) {
    try {
      await logoutSession.mutateAsync(id)
      refetchSessions()
      addToast('Session terminated', 'success')
    } catch {
      addToast('Failed to terminate session', 'error')
    }
  }

  async function handleLogoutOthers() {
    try {
      await logoutOthers.mutateAsync()
      refetchSessions()
      addToast('All other sessions terminated', 'success')
    } catch {
      addToast('Failed', 'error')
    }
  }

  async function saveAuthSettings() {
    try {
      await saveMut.mutateAsync({
        section: 'security',
        auth_max_failed_attempts: maxFailed,
        auth_attempt_window_minutes: attemptWindow,
        auth_lockout_minutes: lockoutMinutes,
      })
      addToast('Auth settings saved', 'success')
    } catch {
      addToast('Failed to save', 'error')
    }
  }

  const sessions = (sessionsData as { sessions?: Array<{ id: string; ip_address: string; last_active: string; user_agent: string; is_current: boolean }> })?.sessions ?? []

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Security</h2>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Change Password</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Current Password</div>
            <input className={styles.input} type="password" value={current} onChange={e => setCurrent(e.target.value)} autoComplete="current-password" />
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>New Password</div>
            <input className={styles.input} type="password" value={newPwd} onChange={e => setNewPwd(e.target.value)} autoComplete="new-password" />
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Confirm New Password</div>
            <input className={styles.input} type="password" value={confirm} onChange={e => setConfirm(e.target.value)} autoComplete="new-password" />
          </div>
          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={changePwd.isPending} onClick={handleChangePassword} disabled={!current || !newPwd || !confirm}>
              Change Password
            </Button>
          </div>
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Two-Factor Authentication</h3>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', margin: 'var(--space-3) 0' }}>
          Enable TOTP two-factor authentication for additional security.
        </p>
        <Link to="/settings/2fa/setup">
          <Button variant="secondary" size="sm">Manage 2FA</Button>
        </Link>
      </Card>

      <Card className={styles.section}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-3)' }}>
          <h3 className={styles.sectionTitle} style={{ margin: 0 }}>Active Sessions</h3>
          {sessions.length > 1 && (
            <Button variant="ghost" size="sm" loading={logoutOthers.isPending} onClick={handleLogoutOthers}>
              Terminate others
            </Button>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
          {sessions.map(s => (
            <div key={s.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 'var(--space-3)', background: 'var(--bg-tertiary)', border: '1px solid var(--border-primary)' }}>
              <div>
                <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                  {s.ip_address}
                  {s.is_current && (
                    <span style={{ fontSize: 'var(--text-xs)', color: 'var(--signal-positive)', fontFamily: 'var(--font-mono)' }}>current</span>
                  )}
                </div>
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                  {s.user_agent ? s.user_agent.slice(0, 60) : '—'} · {s.last_active}
                </div>
              </div>
              {!s.is_current && (
                <Button variant="ghost" size="sm" loading={logoutSession.isPending} onClick={() => handleLogoutSession(s.id)}>
                  Terminate
                </Button>
              )}
            </div>
          ))}
          {sessions.length === 0 && (
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>No active sessions found.</p>
          )}
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Auth Lockout</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Max failed attempts</div>
            <div className={styles.sliderRow}>
              <input type="range" min={1} max={20} value={maxFailed} onChange={e => setMaxFailed(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{maxFailed}</span>
            </div>
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Attempt window</div>
            <div className={styles.sliderRow}>
              <input type="range" min={1} max={60} value={attemptWindow} onChange={e => setAttemptWindow(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{attemptWindow}m</span>
            </div>
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Lockout duration</div>
            <div className={styles.sliderRow}>
              <input type="range" min={1} max={1440} step={5} value={lockoutMinutes} onChange={e => setLockoutMinutes(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{lockoutMinutes}m</span>
            </div>
          </div>
          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={saveMut.isPending} onClick={saveAuthSettings}>
              Save Auth Settings
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
