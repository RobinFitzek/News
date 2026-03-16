import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import api from '@/api/client'
import { PageHeader } from '@/components/layout/PageHeader'
import { Spinner } from '@/components/ui/Spinner'
import styles from './TwoFactorSetupPage.module.css'

interface TwoFAStatus {
  enabled: boolean
  backup_codes_remaining: number
}

interface SetupData {
  qr_code: string | null
  manual_key: string
  backup_codes: string[]
}

type Step = 'idle' | 'setup' | 'verify' | 'done'

function useTwoFAStatus() {
  return useQuery<TwoFAStatus>({
    queryKey: ['2fa-status'],
    queryFn: () => api.get('/api/auth/2fa/status').then(r => r.data),
  })
}

export function TwoFactorSetupPage() {
  const qc = useQueryClient()
  const { data: status, isLoading: statusLoading } = useTwoFAStatus()

  const [step, setStep] = useState<Step>('idle')
  const [setupData, setSetupData] = useState<SetupData | null>(null)
  const [verifyCode, setVerifyCode] = useState('')
  const [disablePassword, setDisablePassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const initMutation = useMutation({
    mutationFn: () => api.post('/api/auth/2fa/setup-init').then(r => r.data as SetupData),
    onSuccess: data => {
      setSetupData(data)
      setStep('setup')
      setError(null)
    },
    onError: () => setError('Failed to start setup. Try again.'),
  })

  const enableMutation = useMutation({
    mutationFn: (code: string) => api.post('/api/auth/2fa/enable', { code }).then(r => r.data),
    onSuccess: (data: { success: boolean; error?: string }) => {
      if (data.success) {
        qc.invalidateQueries({ queryKey: ['2fa-status'] })
        setStep('done')
        setError(null)
      } else {
        setError(data.error === 'invalid_code' ? 'Invalid code. Check your app and try again.' : 'Setup expired. Start again.')
        if (data.error === 'expired') { setStep('idle'); setSetupData(null) }
      }
    },
    onError: () => setError('Verification failed. Try again.'),
  })

  const disableMutation = useMutation({
    mutationFn: (password: string) => api.post('/api/auth/2fa/disable', { password }).then(r => r.data),
    onSuccess: (data: { success: boolean; error?: string }) => {
      if (data.success) {
        qc.invalidateQueries({ queryKey: ['2fa-status'] })
        setDisablePassword('')
        setError(null)
      } else {
        setError('Wrong password.')
      }
    },
    onError: () => setError('Failed to disable 2FA.'),
  })

  async function copyKey(key: string) {
    await navigator.clipboard.writeText(key).catch(() => {})
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (statusLoading) {
    return (
      <div className={styles.loading}>
        <Spinner size="md" />
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <PageHeader
        title="Two-Factor Authentication"
        subtitle="Add an extra layer of security to your account."
      />

      <div className={styles.content}>
        {status?.enabled ? (
          /* ── 2FA is ON ── */
          <div className={styles.section}>
            <div className={styles.statusBadge} data-enabled="true">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 2L4 6v6c0 5.25 3.5 10.15 8 11.35C16.5 22.15 20 17.25 20 12V6L12 2Z"
                  stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Enabled
            </div>

            <p className={styles.statusNote}>
              {status.backup_codes_remaining > 0
                ? `${status.backup_codes_remaining} backup code${status.backup_codes_remaining !== 1 ? 's' : ''} remaining.`
                : 'No backup codes remaining. Generate new ones in settings.'}
            </p>

            <div className={styles.disableSection}>
              <h3 className={styles.sectionTitle}>Disable two-factor auth</h3>
              <p className={styles.sectionDesc}>Enter your password to confirm.</p>

              <form
                className={styles.disableForm}
                onSubmit={e => { e.preventDefault(); disableMutation.mutate(disablePassword) }}
              >
                <input
                  className={styles.input}
                  type="password"
                  value={disablePassword}
                  onChange={e => setDisablePassword(e.target.value)}
                  placeholder="Your password"
                  required
                  disabled={disableMutation.isPending}
                />
                <AnimatePresence>
                  {error && (
                    <motion.p className={styles.error} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      {error}
                    </motion.p>
                  )}
                </AnimatePresence>
                <button
                  type="submit"
                  className={styles.dangerBtn}
                  disabled={disableMutation.isPending || !disablePassword}
                >
                  {disableMutation.isPending ? <Spinner size="sm" /> : 'Disable 2FA'}
                </button>
              </form>
            </div>
          </div>
        ) : step === 'idle' || step === 'done' ? (
          /* ── 2FA is OFF ── */
          <div className={styles.section}>
            {step === 'done' && (
              <motion.div
                className={styles.successBanner}
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
              >
                Two-factor authentication enabled successfully.
              </motion.div>
            )}
            <div className={styles.statusBadge} data-enabled="false">
              Disabled
            </div>
            <p className={styles.statusNote}>
              Use an authenticator app (Google Authenticator, Authy, 1Password) to secure your account.
            </p>
            <button
              className={styles.primaryBtn}
              onClick={() => { setStep('idle'); initMutation.mutate() }}
              disabled={initMutation.isPending}
            >
              {initMutation.isPending ? <Spinner size="sm" /> : 'Set up two-factor auth'}
            </button>
            {error && <p className={styles.error}>{error}</p>}
          </div>
        ) : step === 'setup' && setupData ? (
          /* ── Step 1+2: QR + backup codes ── */
          <div className={styles.setupGrid}>
            <div className={styles.setupCol}>
              <h3 className={styles.sectionTitle}>1. Scan QR code</h3>
              <p className={styles.sectionDesc}>
                Open your authenticator app and scan this code.
              </p>

              {setupData.qr_code ? (
                <div className={styles.qrBox}>
                  <img
                    src={`data:image/png;base64,${setupData.qr_code}`}
                    alt="TOTP QR code"
                    width={180}
                    height={180}
                  />
                </div>
              ) : (
                <div className={styles.qrBox} style={{ opacity: 0.5, fontSize: '0.75rem', textAlign: 'center' }}>
                  QR unavailable
                </div>
              )}

              <div className={styles.manualKey}>
                <span className={styles.manualLabel}>Manual entry key</span>
                <div className={styles.manualKeyRow}>
                  <code className={styles.keyCode}>{setupData.manual_key}</code>
                  <button
                    type="button"
                    className={styles.copyBtn}
                    onClick={() => copyKey(setupData.manual_key)}
                  >
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
            </div>

            <div className={styles.setupCol}>
              <h3 className={styles.sectionTitle}>2. Save backup codes</h3>
              <p className={styles.sectionDesc}>
                Store these somewhere safe. Each code can be used once if you lose access to your app.
              </p>
              <div className={styles.backupCodesGrid}>
                {setupData.backup_codes.map(c => (
                  <code key={c} className={styles.backupCode}>{c}</code>
                ))}
              </div>
            </div>

            <div className={styles.setupFooter}>
              <button
                className={styles.primaryBtn}
                onClick={() => { setStep('verify'); setError(null) }}
              >
                I've saved my codes — continue
              </button>
              <button
                className={styles.ghostBtn}
                onClick={() => { setStep('idle'); setSetupData(null) }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : step === 'verify' && setupData ? (
          /* ── Step 3: Verify ── */
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>3. Verify your setup</h3>
            <p className={styles.sectionDesc}>
              Enter the 6-digit code from your authenticator to confirm it's working.
            </p>

            <form
              className={styles.verifyForm}
              onSubmit={e => { e.preventDefault(); enableMutation.mutate(verifyCode) }}
            >
              <input
                className={styles.codeInput}
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                value={verifyCode}
                onChange={e => setVerifyCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                placeholder="000000"
                maxLength={6}
                autoFocus
                disabled={enableMutation.isPending}
              />

              <AnimatePresence>
                {error && (
                  <motion.p className={styles.error} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    {error}
                  </motion.p>
                )}
              </AnimatePresence>

              <div className={styles.verifyActions}>
                <button
                  type="submit"
                  className={styles.primaryBtn}
                  disabled={enableMutation.isPending || verifyCode.length < 6}
                >
                  {enableMutation.isPending ? <Spinner size="sm" /> : 'Enable 2FA'}
                </button>
                <button
                  type="button"
                  className={styles.ghostBtn}
                  onClick={() => { setStep('setup'); setError(null) }}
                >
                  Back
                </button>
              </div>
            </form>
          </div>
        ) : null}
      </div>
    </div>
  )
}
