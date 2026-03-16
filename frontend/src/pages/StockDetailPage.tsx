import { useState } from 'react'
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Filler,
  type ChartOptions,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import clsx from 'clsx'
import {
  useKeyStats,
  useChartData,
  useEarnings,
  usePeers,
  useSentiment,
  usePatterns,
  useAnalysisDetail,
} from '@/api/endpoints/stock'
import type { SentimentHeadline } from '@/api/endpoints/stock'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import type { SignalType } from '@/types/api'
import styles from './StockDetailPage.module.css'

// Register Chart.js modules
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler)

// ── Types ─────────────────────────────────────────────────────────────────────

type TabKey = 'overview' | 'chart' | 'earnings' | 'peers' | 'sentiment'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'overview',  label: 'Overview' },
  { key: 'chart',     label: 'Chart' },
  { key: 'earnings',  label: 'Earnings' },
  { key: 'peers',     label: 'Peers' },
  { key: 'sentiment', label: 'Sentiment' },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function signalBadgeVariant(
  signal: SignalType | string | null
): 'success' | 'danger' | 'neutral' | 'ghost' {
  if (!signal) return 'ghost'
  const s = String(signal).toUpperCase()
  if (s === 'BUY')  return 'success'
  if (s === 'SELL') return 'danger'
  return 'neutral'
}

function riskClass(score: number | null): string {
  if (score === null) return ''
  if (score >= 7) return styles.riskHigh
  if (score >= 4) return styles.riskMedium
  return styles.riskLow
}

function confidenceClass(pct: number): string {
  if (pct >= 70) return styles.high
  if (pct >= 40) return styles.medium
  return styles.low
}

function formatLargeNumber(n: number | null): string {
  if (n === null || n === undefined) return '—'
  if (Math.abs(n) >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
  if (Math.abs(n) >= 1e9)  return `$${(n / 1e9).toFixed(2)}B`
  if (Math.abs(n) >= 1e6)  return `$${(n / 1e6).toFixed(2)}M`
  return `$${n.toLocaleString()}`
}

function formatNum(n: number | null, decimals = 2): string {
  if (n === null || n === undefined) return '—'
  return n.toFixed(decimals)
}

// ── Tab panel animation variants ──────────────────────────────────────────────

function panelVariants(dir: number) {
  return {
    initial: { opacity: 0, x: dir * 16 },
    animate: { opacity: 1, x: 0, transition: { duration: 0.28, ease: [0.34, 1.2, 0.64, 1] as number[] } },
    exit:    { opacity: 0, x: -dir * 16, transition: { duration: 0.18, ease: 'easeIn' } },
  }
}

// ── Chart.js config ───────────────────────────────────────────────────────────

const chartOptions: ChartOptions<'line'> = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(16,16,24,0.95)',
      borderColor: 'rgba(255,255,255,0.08)',
      borderWidth: 1,
      titleColor: '#a0a0b8',
      bodyColor: '#e4e4f0',
      titleFont: { family: 'var(--font-mono)', size: 11 },
      bodyFont: { family: 'var(--font-mono)', size: 12 },
    },
  },
  scales: {
    x: {
      grid: { color: 'rgba(255,255,255,0.05)' },
      ticks: { color: '#6b6b8a', font: { family: 'var(--font-mono)', size: 10 }, maxTicksLimit: 8 },
    },
    y: {
      position: 'right',
      grid: { color: 'rgba(255,255,255,0.05)' },
      ticks: { color: '#6b6b8a', font: { family: 'var(--font-mono)', size: 10 } },
    },
  },
}

// ── Subcomponents ─────────────────────────────────────────────────────────────

function LoadingPanel() {
  return <div className={styles.loading}><Spinner size="lg" /></div>
}

function ErrorPanel({ message }: { message: string }) {
  return <div className={styles.error}>{message}</div>
}

function EmptyPanel({ message }: { message: string }) {
  return <div className={styles.emptyState}>{message}</div>
}

function ExpandSection({
  label,
  content,
  type,
}: {
  label: string
  content: string
  type: 'bull' | 'bear'
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className={styles.expandSection}>
      <button
        className={clsx(
          styles.expandToggle,
          type === 'bull' ? styles.expandToggleBull : styles.expandToggleBear
        )}
        onClick={() => setOpen(o => !o)}
      >
        <span>{label}</span>
        <span className={clsx(styles.expandChevron, open && styles.open)}>▼</span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeInOut' }}
            style={{ overflow: 'hidden' }}
          >
            <div className={styles.expandContent}>{content}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Overview tab ──────────────────────────────────────────────────────────────

function OverviewTab({ ticker, analysisId }: { ticker: string; analysisId: string | null }) {
  const { data: keyStats, isLoading: statsLoading, isError: statsError } = useKeyStats(ticker)
  const { data: analysis, isLoading: analysisLoading } = useAnalysisDetail(analysisId)

  return (
    <div>
      {/* Signal card (only when analysis_id provided) */}
      {analysisId && (
        <>
          {analysisLoading ? (
            <LoadingPanel />
          ) : analysis ? (
            <Card className={styles.signalCard} animate>
              <div className={styles.signalRow}>
                <span className={styles.signalLabel}>Signal</span>
                <Badge variant={signalBadgeVariant(analysis.signal)} size="sm">
                  {analysis.signal ?? 'N/A'}
                </Badge>
              </div>

              <div className={styles.confidenceRow}>
                <span className={styles.confidenceLabel}>Confidence</span>
                <div className={styles.confidenceBarTrack}>
                  <div
                    className={clsx(styles.confidenceBarFill, confidenceClass(analysis.confidence))}
                    style={{ width: `${analysis.confidence}%` }}
                  />
                </div>
                <span className={styles.confidenceValue}>{analysis.confidence}%</span>
              </div>

              {analysis.synthesis && (
                <p className={styles.synthesis}>{analysis.synthesis}</p>
              )}

              {analysis.bull_case && (
                <ExpandSection label="Bull Case" content={analysis.bull_case} type="bull" />
              )}
              {analysis.bear_case && (
                <ExpandSection label="Bear Case" content={analysis.bear_case} type="bear" />
              )}
            </Card>
          ) : null}
        </>
      )}

      {/* Key stats */}
      <p className={styles.sectionTitle}>Key Statistics</p>
      {statsLoading ? (
        <LoadingPanel />
      ) : statsError ? (
        <ErrorPanel message="Failed to load key stats." />
      ) : keyStats ? (
        <>
          <div className={styles.statsGrid}>
            <Card animate delay={0.05}>
              <MetricCard label="PE Ratio"    value={formatNum(keyStats.pe_ratio, 1)} mono />
            </Card>
            <Card animate delay={0.08}>
              <MetricCard label="Market Cap"  value={formatLargeNumber(keyStats.market_cap)} mono />
            </Card>
            <Card animate delay={0.11}>
              <MetricCard label="Revenue"     value={formatLargeNumber(keyStats.revenue)} mono />
            </Card>
            <Card animate delay={0.14}>
              <MetricCard label="EPS"         value={keyStats.eps !== null ? `$${formatNum(keyStats.eps)}` : '—'} mono />
            </Card>
            <Card animate delay={0.17}>
              <MetricCard label="Beta"        value={formatNum(keyStats.beta)} mono />
            </Card>
            <Card animate delay={0.2}>
              <MetricCard label="Volume"      value={keyStats.volume !== null ? keyStats.volume.toLocaleString() : '—'} mono />
            </Card>
          </div>

          {/* Risk scores */}
          <p className={styles.sectionTitle}>Risk Scores</p>
          <div className={styles.riskRow}>
            <div className={styles.riskItem}>
              <span className={styles.riskItemLabel}>Risk Score</span>
              <span className={clsx(styles.riskItemValue, riskClass(keyStats.risk_score))}>
                {keyStats.risk_score !== null ? `${keyStats.risk_score}/10` : '—'}
              </span>
            </div>
            <div className={styles.riskItem}>
              <span className={styles.riskItemLabel}>Geo Risk</span>
              <span className={clsx(styles.riskItemValue, riskClass(keyStats.geo_risk_score))}>
                {keyStats.geo_risk_score !== null ? `${keyStats.geo_risk_score}/10` : '—'}
              </span>
            </div>
          </div>
        </>
      ) : (
        <EmptyPanel message="No key stats available." />
      )}
    </div>
  )
}

// ── Chart tab ─────────────────────────────────────────────────────────────────

function ChartTab({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useChartData(ticker)

  if (isLoading) return <LoadingPanel />
  if (isError)   return <ErrorPanel message="Failed to load chart data." />
  if (!data || data.prices.length === 0) return <EmptyPanel message="No chart data available." />

  const chartData = {
    labels: data.labels,
    datasets: [
      {
        label: ticker,
        data: data.prices,
        borderColor: 'rgba(107, 184, 255, 0.9)',
        backgroundColor: 'rgba(107, 184, 255, 0.08)',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: true,
        tension: 0.3,
      },
    ],
  }

  return (
    <Card animate>
      <div className={styles.chartWrapper}>
        <Line data={chartData} options={chartOptions} />
      </div>
    </Card>
  )
}

// ── Earnings tab ──────────────────────────────────────────────────────────────

function EarningsTab({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useEarnings(ticker)

  if (isLoading) return <LoadingPanel />
  if (isError)   return <ErrorPanel message="Failed to load earnings data." />
  if (!data || data.earnings.length === 0) return <EmptyPanel message="No earnings data available." />

  return (
    <Card className={styles.tableCard} animate>
      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Date</th>
              <th className={styles.right}>EPS Actual</th>
              <th className={styles.right}>EPS Estimate</th>
              <th className={styles.right}>Revenue Actual</th>
              <th className={styles.right}>Surprise %</th>
            </tr>
          </thead>
          <tbody>
            {data.earnings.map((e, i) => (
              <motion.tr
                key={e.date}
                className={styles.row}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04, duration: 0.3 }}
              >
                <td className={styles.mono}>{e.date}</td>
                <td className={clsx('right', styles.mono)}>
                  {e.eps_actual !== null ? `$${formatNum(e.eps_actual)}` : <span className={styles.muted}>—</span>}
                </td>
                <td className={clsx('right', styles.mono)}>
                  {e.eps_estimate !== null ? `$${formatNum(e.eps_estimate)}` : <span className={styles.muted}>—</span>}
                </td>
                <td className={clsx('right', styles.mono)}>
                  {formatLargeNumber(e.revenue_actual)}
                </td>
                <td className={clsx('right', styles.mono)}>
                  {e.surprise_pct !== null ? (
                    <span className={e.surprise_pct >= 0 ? styles.positive : styles.negative}>
                      {e.surprise_pct >= 0 ? '+' : ''}{formatNum(e.surprise_pct, 1)}%
                    </span>
                  ) : (
                    <span className={styles.muted}>—</span>
                  )}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

// ── Peers tab ─────────────────────────────────────────────────────────────────

function PeersTab({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = usePeers(ticker)

  if (isLoading) return <LoadingPanel />
  if (isError)   return <ErrorPanel message="Failed to load peer data." />
  if (!data || data.peers.length === 0) return <EmptyPanel message="No peer data available." />

  return (
    <Card className={styles.tableCard} animate>
      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Name</th>
              <th className={styles.right}>Market Cap</th>
              <th className={styles.right}>PE Ratio</th>
              <th className={styles.right}>Revenue</th>
              <th>Signal</th>
              <th className={styles.right}>Price</th>
              <th className={styles.right}>Change %</th>
            </tr>
          </thead>
          <tbody>
            {data.peers.map((peer, i) => (
              <motion.tr
                key={peer.ticker}
                className={styles.row}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04, duration: 0.3 }}
              >
                <td>
                  <span className={clsx(styles.mono, peer.ticker === ticker && styles.positive)}>
                    {peer.ticker}
                  </span>
                </td>
                <td>{peer.name}</td>
                <td className={clsx('right', styles.mono)}>{formatLargeNumber(peer.market_cap)}</td>
                <td className={clsx('right', styles.mono)}>{formatNum(peer.pe_ratio, 1)}</td>
                <td className={clsx('right', styles.mono)}>{formatLargeNumber(peer.revenue)}</td>
                <td>
                  {peer.signal ? (
                    <Badge variant={signalBadgeVariant(peer.signal)}>{peer.signal}</Badge>
                  ) : (
                    <span className={styles.muted}>—</span>
                  )}
                </td>
                <td className={clsx('right', styles.mono)}>
                  {peer.price !== null ? `$${formatNum(peer.price)}` : <span className={styles.muted}>—</span>}
                </td>
                <td className={clsx('right', styles.mono)}>
                  {peer.change_pct !== null ? (
                    <span className={peer.change_pct >= 0 ? styles.positive : styles.negative}>
                      {peer.change_pct >= 0 ? '+' : ''}{formatNum(peer.change_pct, 2)}%
                    </span>
                  ) : (
                    <span className={styles.muted}>—</span>
                  )}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

// ── Sentiment tab ─────────────────────────────────────────────────────────────

function sentimentHeadlineBadge(
  s: SentimentHeadline['sentiment']
): 'success' | 'danger' | 'neutral' {
  if (s === 'positive') return 'success'
  if (s === 'negative') return 'danger'
  return 'neutral'
}

function SentimentTab({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useSentiment(ticker)

  if (isLoading) return <LoadingPanel />
  if (isError)   return <ErrorPanel message="Failed to load sentiment data." />
  if (!data)     return <EmptyPanel message="No sentiment data available." />

  const score = data.overall_score

  return (
    <div>
      {/* Score summary */}
      <Card animate className={styles.signalCard}>
        <div className={styles.sentimentScore}>
          <span
            className={clsx(
              styles.sentimentBig,
              score !== null && score >= 0.3
                ? styles.positive
                : score !== null && score <= -0.3
                ? styles.negative
                : styles.muted
            )}
          >
            {score !== null ? score.toFixed(2) : '—'}
          </span>
          <div className={styles.sentimentMeta}>
            <span className={styles.sentimentMetaItem}>
              Positive: {data.positive_count} &nbsp;|&nbsp;
              Negative: {data.negative_count} &nbsp;|&nbsp;
              Neutral: {data.neutral_count}
            </span>
          </div>
        </div>
      </Card>

      {/* Headlines */}
      {data.headlines.length > 0 && (
        <>
          <p className={styles.sectionTitle}>Recent Headlines</p>
          <div className={styles.headlineList}>
            {data.headlines.map((h, i) => (
              <motion.div
                key={`${h.title}-${i}`}
                className={styles.headline}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04, duration: 0.3 }}
              >
                <div className={styles.headlineTitle}>
                  {h.url ? (
                    <a href={h.url} target="_blank" rel="noopener noreferrer">{h.title}</a>
                  ) : (
                    h.title
                  )}
                </div>
                <div className={styles.headlineMeta}>
                  <Badge variant={sentimentHeadlineBadge(h.sentiment)}>{h.sentiment}</Badge>
                  <span className={styles.headlineSource}>{h.source}</span>
                  <span className={styles.headlineSource}>
                    {new Date(h.published_at).toLocaleDateString('en-SE')}
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export function StockDetailPage() {
  const { ticker } = useParams<{ ticker: string }>()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const safeT = ticker ?? ''
  const analysisId = searchParams.get('analysis_id')

  const [activeTab, setActiveTab] = useState<TabKey>('overview')
  const [prevTab, setPrevTab] = useState<TabKey>('overview')

  const { data: keyStats } = useKeyStats(safeT)

  function switchTab(tab: TabKey) {
    setPrevTab(activeTab)
    setActiveTab(tab)
  }

  const tabDirection = TABS.findIndex(t => t.key === activeTab) >= TABS.findIndex(t => t.key === prevTab) ? 1 : -1

  const subtitle = keyStats
    ? [keyStats.sector, keyStats.name].filter(Boolean).join(' · ')
    : undefined

  return (
    <>
      {/* Back button */}
      <div className={styles.backRow}>
        <button className={styles.backButton} onClick={() => navigate(-1)}>
          ← Back
        </button>
      </div>

      <PageHeader
        title={safeT}
        subtitle={subtitle}
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={() => navigate(`/watchlist`)}
          >
            Watchlist
          </Button>
        }
      />

      {/* Tab navigation */}
      <div className={styles.tabs}>
        {TABS.map(t => (
          <button
            key={t.key}
            className={clsx(styles.tab, activeTab === t.key && styles.activeTab)}
            onClick={() => switchTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content with slide animation */}
      <div className={styles.tabPanel}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={activeTab}
            {...panelVariants(tabDirection)}
          >
            {activeTab === 'overview' && (
              <OverviewTab ticker={safeT} analysisId={analysisId} />
            )}
            {activeTab === 'chart' && (
              <ChartTab ticker={safeT} />
            )}
            {activeTab === 'earnings' && (
              <EarningsTab ticker={safeT} />
            )}
            {activeTab === 'peers' && (
              <PeersTab ticker={safeT} />
            )}
            {activeTab === 'sentiment' && (
              <SentimentTab ticker={safeT} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
