import { useState, useEffect, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { queryClient } from '@/api/queryClient'
import { Spinner } from '@/components/ui/Spinner'
import styles from './LoginPage.module.css'

type LoginError = 'invalid' | 'locked' | null

interface MarketIndex {
  label: string
  value: string
  change: string
}

const FALLBACK_INDICES: MarketIndex[] = [
  { label: 'S&P 500', value: '—', change: '' },
  { label: 'NASDAQ', value: '—', change: '' },
  { label: 'DOW', value: '—', change: '' },
  { label: 'VIX', value: '—', change: '' },
]

export function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<LoginError>(null)
  const [lockMinutes, setLockMinutes] = useState(0)
  const [indices, setIndices] = useState<MarketIndex[]>(FALLBACK_INDICES)
  const navigate = useNavigate()

  // Kick off CSRF fetch early
  const [csrfToken, setCsrfToken] = useState<string | null>(null)
  useEffect(() => {
    fetch('/api/auth/csrf', { credentials: 'include' })
      .then(r => r.json())
      .then((d: { token: string }) => setCsrfToken(d.token))
      .catch(() => {})
  }, [])

  // Try to load live market data (works if session cookie still valid)
  useEffect(() => {
    fetch('/api/market-regime', { credentials: 'include' })
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then((d: { spy_price?: number; spy_change_pct?: number; vix?: number; vix_change_pct?: number; nasdaq_price?: number; nasdaq_change_pct?: number; dow_price?: number; dow_change_pct?: number }) => {
        const fmt = (v: number | undefined) => v !== undefined ? v.toLocaleString('en-US', { maximumFractionDigits: 2 }) : '—'
        const pct = (v: number | undefined) => v !== undefined ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : ''
        setIndices([
          { label: 'S&P 500', value: fmt(d.spy_price), change: pct(d.spy_change_pct) },
          { label: 'NASDAQ', value: fmt(d.nasdaq_price), change: pct(d.nasdaq_change_pct) },
          { label: 'DOW', value: fmt(d.dow_price), change: pct(d.dow_change_pct) },
          { label: 'VIX', value: fmt(d.vix), change: pct(d.vix_change_pct) },
        ])
      })
      .catch(() => { /* Not authenticated — keep fallback dashes */ })
  }, [])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!csrfToken) return
    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        body: JSON.stringify({ username, password }),
      })

      const data = await res.json() as {
        success: boolean
        error?: string
        minutes?: number
        requires_totp?: boolean
        token?: string
        redirect?: string
      }

      if (data.success) {
        queryClient.removeQueries({ queryKey: ['auth'] })
        navigate(data.redirect ?? '/', { replace: true })
        return
      }

      if (data.requires_totp && data.token) {
        navigate(`/login/totp?token=${data.token}`)
        return
      }

      if (data.error === 'locked') {
        setError('locked')
        setLockMinutes(data.minutes ?? 5)
      } else {
        setError('invalid')
      }
    } catch {
      setError('invalid')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      {/* Left panel — branding */}
      <div className={styles.left}>
        {/* Orb */}
        <motion.div
          className={styles.orb}
          animate={{
            x: [0, 30, -20, 0],
            y: [0, -40, 20, 0],
            scale: [1, 1.05, 0.98, 1],
          }}
          transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
          aria-hidden="true"
        />
        <div className={styles.grain} aria-hidden="true" />

        <motion.div
          className={styles.leftContent}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className={styles.brand}>
            <div className={styles.logoMark}>S</div>
            <div>
              <h1 className={styles.brandName}>Stockholm</h1>
              <p className={styles.brandSub}>Investment Intelligence</p>
            </div>
          </div>

          <p className={styles.quote}>
            "The market is a device for transferring money from the impatient
            to the patient."
          </p>
          <p className={styles.quoteAuthor}>— Warren Buffett</p>
        </motion.div>

        {/* Market strip at bottom */}
        <div className={styles.marketStrip}>
          {indices.map(idx => (
            <div key={idx.label} className={styles.marketItem}>
              <span className={styles.marketLabel}>{idx.label}</span>
              <span className={styles.marketValue}>{idx.value}</span>
              {idx.change && (
                <span className={idx.change.startsWith('+') ? styles.marketUp : styles.marketDown}>
                  {idx.change}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Right panel — form */}
      <motion.div
        className={styles.right}
        initial={{ opacity: 0, x: 40 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className={styles.formWrap}>
          <div className={styles.formHeader}>
            <h2 className={styles.welcome}>Welcome back</h2>
            <p className={styles.welcomeSub}>Sign in to your account</p>
          </div>

          <form className={styles.form} onSubmit={handleSubmit} noValidate>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="username">Username</label>
              <input
                id="username"
                className={styles.input}
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                autoComplete="username"
                autoFocus
                required
                disabled={loading}
                placeholder="Enter your username"
              />
            </div>

            <div className={styles.field}>
              <label className={styles.label} htmlFor="password">Password</label>
              <input
                id="password"
                className={styles.input}
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                autoComplete="current-password"
                required
                disabled={loading}
                placeholder="••••••••"
              />
            </div>

            <AnimatePresence>
              {error === 'invalid' && (
                <motion.p
                  className={styles.error}
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  Invalid username or password.
                </motion.p>
              )}
              {error === 'locked' && (
                <motion.p
                  className={styles.error}
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  Account locked. Try again in {lockMinutes} minute{lockMinutes !== 1 ? 's' : ''}.
                </motion.p>
              )}
            </AnimatePresence>

            <button
              type="submit"
              className={styles.submitBtn}
              disabled={loading || !csrfToken}
            >
              {loading ? <Spinner size="sm" /> : 'Sign in'}
            </button>
          </form>
        </div>
      </motion.div>
    </div>
  )
}
