import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import axios from 'axios'
import { queryClient } from '@/api/queryClient'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import styles from './LoginPage.module.css'

type LoginError = 'invalid' | 'locked' | null

export function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<LoginError>(null)
  const [lockMinutes, setLockMinutes] = useState(0)
  const navigate = useNavigate()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const form = new URLSearchParams()
      form.set('username', username)
      form.set('password', password)

      await axios.post('/login', form, {
        withCredentials: true,
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        maxRedirects: 0,
        validateStatus: s => s < 400,
      })

      // Clear any cached auth state and invalidate to force re-fetch
      queryClient.removeQueries({ queryKey: ['auth'] })
      navigate('/', { replace: true })
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response) {
        const { status, data } = err.response
        if (status === 429 || (data as Record<string, unknown>)?.locked) {
          setError('locked')
          setLockMinutes((data as Record<string, unknown>)?.minutes as number ?? 5)
        } else {
          setError('invalid')
        }
      } else {
        setError('invalid')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      {/* Atmospheric light cone */}
      <div className={styles.cone} aria-hidden="true" />

      <motion.div
        className={styles.card}
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.34, 1.2, 0.64, 1] }}
      >
        {/* Accent line */}
        <div className={styles.accentLine} />

        {/* Logo */}
        <div className={styles.header}>
          <h1 className={styles.logo}>Stockholm</h1>
          <p className={styles.tagline}>Investment Intelligence</p>
        </div>

        {/* Error messages */}
        {error === 'invalid' && (
          <motion.div
            className={styles.error}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Invalid credentials. Please try again.
          </motion.div>
        )}
        {error === 'locked' && (
          <motion.div
            className={styles.error}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Account locked. Try again in {lockMinutes} minute{lockMinutes !== 1 ? 's' : ''}.
          </motion.div>
        )}

        {/* Form */}
        <form className={styles.form} onSubmit={handleSubmit}>
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
            />
          </div>

          <Button
            type="submit"
            variant="primary"
            size="lg"
            loading={loading}
            className={styles.submitBtn}
          >
            {loading ? <Spinner size="sm" /> : null}
            Sign In
          </Button>
        </form>
      </motion.div>
    </div>
  )
}
