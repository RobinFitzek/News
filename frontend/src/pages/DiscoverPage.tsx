import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useRunDiscover } from '@/api/endpoints/discovery'
import type { DiscoveredStock } from '@/api/endpoints/discovery'
import { useAddToWatchlist } from '@/api/endpoints/watchlist'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './DiscoverPage.module.css'

const SECTORS = [
  { value: '', label: 'All Sectors' },
  { value: 'technology', label: 'Technology' },
  { value: 'healthcare', label: 'Healthcare' },
  { value: 'finance', label: 'Finance' },
  { value: 'energy', label: 'Energy' },
  { value: 'consumer', label: 'Consumer' },
  { value: 'industrial', label: 'Industrial' },
]

const FOCUSES = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'growth', label: 'Growth' },
  { value: 'value', label: 'Value' },
  { value: 'dividend', label: 'Dividend' },
]

function scoreColor(score: number): string {
  if (score >= 70) return '#10b981'
  if (score >= 45) return '#f59e0b'
  return '#ef4444'
}

function scoreLabel(score: number): string {
  if (score >= 70) return 'High'
  if (score >= 45) return 'Medium'
  return 'Low'
}

// ── Stock result card ──────────────────────────────────────────────────────

function StockCard({
  stock,
  onAddToWatchlist,
  isAdding,
}: {
  stock: DiscoveredStock
  onAddToWatchlist: (ticker: string) => void
  isAdding: boolean
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={styles.stockCard}>
        <div className={styles.stockHeader}>
          <span className={styles.ticker}>{stock.ticker}</span>
          <Badge
            variant={stock.score >= 70 ? 'success' : stock.score >= 45 ? 'warning' : 'neutral'}
          >
            {stock.score} — {scoreLabel(stock.score)}
          </Badge>
        </div>

        {stock.name && <div className={styles.stockName}>{stock.name}</div>}

        <div className={styles.scoreBar}>
          <div
            className={styles.scoreFill}
            style={{
              width: `${Math.min(100, Math.max(0, stock.score))}%`,
              background: scoreColor(stock.score),
            }}
          />
        </div>

        <div className={styles.fieldGroup}>
          <div className={styles.fieldLabel}>Reason</div>
          <p className={styles.fieldText}>{stock.reason}</p>
        </div>

        {stock.catalyst && (
          <div className={styles.fieldGroup}>
            <div className={styles.fieldLabel}>Catalyst</div>
            <p className={styles.fieldText}>{stock.catalyst}</p>
          </div>
        )}

        <div className={styles.stockActions}>
          <Button
            size="sm"
            onClick={() => onAddToWatchlist(stock.ticker)}
            disabled={isAdding}
          >
            {isAdding ? 'Adding...' : 'Add to Watchlist'}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => window.location.href = `/analyze?ticker=${stock.ticker}`}
          >
            Analyze
          </Button>
        </div>
      </Card>
    </motion.div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────

export function DiscoverPage() {
  const [sector, setSector] = useState('')
  const [focus, setFocus] = useState('balanced')
  const [limit, setLimit] = useState(5)
  const addToast = useToastStore(s => s.addToast)

  const discover = useRunDiscover()
  const addToWatchlist = useAddToWatchlist()

  const handleDiscover = () => {
    discover.mutate(
      { sector: sector || undefined, focus, limit },
      {
        onError: () => addToast('Discovery failed. Check your Perplexity API key.', 'error'),
      }
    )
  }

  const handleAdd = (ticker: string) => {
    addToWatchlist.mutate(
      { ticker },
      {
        onSuccess: () => addToast(`${ticker} added to watchlist`, 'success'),
        onError: () => addToast(`Failed to add ${ticker}`, 'error'),
      }
    )
  }

  return (
    <>
      <PageHeader
        title="AI Stock Discovery"
        subtitle="Use AI to find new investment opportunities"
      />

      {/* Discovery form */}
      <Card className={styles.formCard} delay={0}>
        <div className={styles.formGrid}>
          <div className={styles.field}>
            <label className={styles.label}>Sector</label>
            <select
              className={styles.select}
              value={sector}
              onChange={e => setSector(e.target.value)}
            >
              {SECTORS.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Focus</label>
            <select
              className={styles.select}
              value={focus}
              onChange={e => setFocus(e.target.value)}
            >
              {FOCUSES.map(f => (
                <option key={f.value} value={f.value}>{f.label}</option>
              ))}
            </select>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Stocks</label>
            <input
              type="number"
              className={styles.input}
              min={1}
              max={10}
              value={limit}
              onChange={e => setLimit(Math.min(10, Math.max(1, Number(e.target.value))))}
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>&nbsp;</label>
            <Button
              onClick={handleDiscover}
              disabled={discover.isPending}
              className={styles.discoverBtn}
            >
              {discover.isPending ? 'Discovering...' : 'Discover Stocks'}
            </Button>
          </div>
        </div>

        <p className={styles.hint}>
          Uses 1 Perplexity API call. Results in 20-30 seconds.
        </p>
      </Card>

      {/* Loading state */}
      {discover.isPending && (
        <Card className={styles.loadingCard}>
          <Spinner size="lg" />
          <p className={styles.loadingText}>Analyzing market trends...</p>
        </Card>
      )}

      {/* Error state */}
      {discover.isError && (
        <Card className={styles.errorCard}>
          <p className={styles.errorText}>
            Discovery failed. Ensure your Perplexity API key is configured in Settings.
          </p>
        </Card>
      )}

      {/* Results */}
      <AnimatePresence>
        {discover.data && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            {discover.data.stocks.length === 0 ? (
              <Card className={styles.emptyCard}>
                <p className={styles.emptyText}>
                  No new stocks found matching your criteria. Try a different sector or focus.
                </p>
              </Card>
            ) : (
              <div className={styles.resultsGrid}>
                {discover.data.stocks.map(stock => (
                  <StockCard
                    key={stock.ticker}
                    stock={stock}
                    onAddToWatchlist={handleAdd}
                    isAdding={addToWatchlist.isPending}
                  />
                ))}
              </div>
            )}

            {discover.data.raw_analysis && (
              <Card className={styles.rawCard} delay={0.2}>
                <details>
                  <summary className={styles.rawSummary}>Full AI Analysis</summary>
                  <pre className={styles.rawText}>{discover.data.raw_analysis}</pre>
                </details>
              </Card>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
