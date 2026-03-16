import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useChangePassword } from '@/api/endpoints/settings'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

export function PanelSecurity() {
  const changePwd = useChangePassword()
  const { addToast } = useToastStore()
  const [current, setCurrent] = useState('')
  const [newPwd, setNewPwd] = useState('')
  const [confirm, setConfirm] = useState('')

  async function handleChangePassword() {
    if (newPwd !== confirm) {
      addToast('Passwords do not match', 'error')
      return
    }
    try {
      await changePwd.mutateAsync({
        current_password: current,
        new_password: newPwd,
        confirm_password: confirm,
      })
      addToast('Password changed successfully', 'success')
      setCurrent(''); setNewPwd(''); setConfirm('')
    } catch {
      addToast('Failed to change password', 'error')
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Security</h2>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Change Password</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Current Password</div>
            <input
              className={styles.input}
              type="password"
              value={current}
              onChange={e => setCurrent(e.target.value)}
              autoComplete="current-password"
            />
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>New Password</div>
            <input
              className={styles.input}
              type="password"
              value={newPwd}
              onChange={e => setNewPwd(e.target.value)}
              autoComplete="new-password"
            />
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Confirm New Password</div>
            <input
              className={styles.input}
              type="password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              autoComplete="new-password"
            />
          </div>
          <div className={styles.saveRow}>
            <Button
              variant="primary"
              size="md"
              loading={changePwd.isPending}
              onClick={handleChangePassword}
              disabled={!current || !newPwd || !confirm}
            >
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
    </div>
  )
}
