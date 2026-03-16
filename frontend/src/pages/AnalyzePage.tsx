import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getCsrfToken } from '@/api/csrf'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import styles from './AnalyzePage.module.css'

export function AnalyzePage() {
  const [searchParams] = useSearchParams()
  const urlTicker = searchParams.get('ticker') ?? ''

  const [ticker, setTicker] = useState(urlTicker.toUpperCase())
  const [csrfToken, setCsrfToken] = useState('')
  const [isPending, setIsPending] = useState(false)

  useEffect(() => {
    getCsrfToken().then(setCsrfToken).catch(() => {})
  }, [])

  function handleSubmit() {
    setIsPending(true)
    setTimeout(() => setIsPending(false), 3000)
  }

  return (
    <>
      <PageHeader
        title="Analyze"
        subtitle="Run AI-powered stock analysis"
      />

      <div className={styles.formWrapper}>
        <Card className={styles.formCard}>
          <form method="POST" action="/analyze" onSubmit={handleSubmit}>
            <div className={styles.formInner}>
              <input
                className={styles.tickerInput}
                name="ticker"
                type="text"
                placeholder="AAPL, MSFT, TSLA..."
                value={ticker}
                onChange={e => setTicker(e.target.value.toUpperCase())}
                required
                autoFocus
                autoComplete="off"
                spellCheck={false}
              />
              <input type="hidden" name="csrf_token" value={csrfToken} />
              <div className={styles.submitBtn}>
                <Button
                  variant="primary"
                  size="lg"
                  type="submit"
                  loading={isPending}
                  disabled={!ticker.trim() || !csrfToken}
                >
                  Run Analysis
                </Button>
              </div>
            </div>
          </form>
        </Card>
      </div>

      <div className={styles.infoWrapper}>
        <Card className={styles.infoCard}>
          <p className={styles.infoText}>
            Analysis uses Perplexity + Gemini AI to evaluate fundamentals, news
            sentiment, geopolitical risk, and technical signals.
          </p>
          <p className={styles.infoSmall}>
            Results are stored and accessible from Analysis History.
          </p>
        </Card>
      </div>
    </>
  )
}
