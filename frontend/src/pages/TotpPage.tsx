import { useState, useEffect, useRef, type ChangeEvent } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { queryClient } from '@/api/queryClient'
import { Spinner } from '@/components/ui/Spinner'
import styles from './TotpPage.module.css'

export function TotpPage() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token') ?? ''
  const navigate = useNavigate()

  const [code, setCode] = useState('')
  const [useBackup, setUseBackup] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [csrfToken, setCsrfToken] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!token) { navigate('/login', { replace: true }); return }
    fetch('/api/auth/csrf', { credentials: 'include' })
      .then(r => r.json())
      .then((d: { token: string }) => setCsrfToken(d.token))
      .catch(() => {})
  }, [token, navigate])

  async function submit(codeVal: string) {
    if (!csrfToken || loading) return
    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/api/auth/totp', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        body: JSON.stringify({ token, code: codeVal, use_backup: useBackup }),
      })
      const data = await res.json() as { success: boolean; error?: string; redirect?: string }

      if (data.success) {
        queryClient.removeQueries({ queryKey: ['auth'] })
        navigate(data.redirect ?? '/', { replace: true })
        return
      }

      setError(data.error === 'expired' ? 'Session expired. Please log in again.' : 'Invalid code. Try again.')
      setCode('')
      inputRef.current?.focus()
    } catch {
      setError('Something went wrong. Try again.')
    } finally {
      setLoading(false)
    }
  }

  function handleCodeChange(e: ChangeEvent<HTMLInputElement>) {
    const val = e.target.value.replace(/\D/g, '').slice(0, 6)
    setCode(val)
    if (val.length === 6 && !useBackup) {
      submit(val)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.grain} aria-hidden="true" />

      <motion.div
        className={styles.card}
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.34, 1.2, 0.64, 1] }}
      >
        {/* Shield icon */}
        <div className={styles.iconWrap}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 2L4 6v6c0 5.25 3.5 10.15 8 11.35C16.5 22.15 20 17.25 20 12V6L12 2Z"
              stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>

        <div className={styles.header}>
          <h1 className={styles.title}>Two-factor auth</h1>
          <p className={styles.sub}>
            {useBackup
              ? 'Enter one of your backup codes'
              : 'Enter the 6-digit code from your authenticator app'}
          </p>
        </div>

        <form className={styles.form} onSubmit={e => { e.preventDefault(); submit(code) }}>
          {useBackup ? (
            <input
              ref={inputRef}
              className={styles.backupInput}
              type="text"
              value={code}
              onChange={e => setCode(e.target.value.toUpperCase())}
              placeholder="XXXX-XXXX"
              autoFocus
              autoComplete="one-time-code"
              disabled={loading}
            />
          ) : (
            <input
              ref={inputRef}
              className={styles.codeInput}
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              value={code}
              onChange={handleCodeChange}
              placeholder="000000"
              maxLength={6}
              autoFocus
              autoComplete="one-time-code"
              disabled={loading}
            />
          )}

          <AnimatePresence>
            {error && (
              <motion.p
                className={styles.error}
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                {error}
              </motion.p>
            )}
          </AnimatePresence>

          {useBackup && (
            <button
              type="submit"
              className={styles.submitBtn}
              disabled={loading || !code || !csrfToken}
            >
              {loading ? <Spinner size="sm" /> : 'Verify backup code'}
            </button>
          )}

          {loading && !useBackup && (
            <div className={styles.verifying}>
              <Spinner size="sm" />
              <span>Verifying…</span>
            </div>
          )}
        </form>

        <div className={styles.footer}>
          <button
            type="button"
            className={styles.toggleBtn}
            onClick={() => { setUseBackup(v => !v); setCode(''); setError(null) }}
          >
            {useBackup ? 'Use authenticator app instead' : 'Use a backup code instead'}
          </button>
          <button
            type="button"
            className={styles.backBtn}
            onClick={() => navigate('/login')}
          >
            Back to login
          </button>
        </div>
      </motion.div>
    </div>
  )
}
