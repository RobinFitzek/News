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
  useAnalysisDetail,
} from '@/api/endpoints/stock'
import { useSmartMoney } from '@/api/endpoints/institutional'
import {
  useDCF,
  useMoat,
  useCatalysts,
  useOptionsFlow,
  useSupplyChain,
} from '@/api/endpoints/stockExtras'
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

type TabKey = 'overview' | 'chart' | 'earnings' | 'peers' | 'sentiment' | 'valuation' | 'options' | 'supply'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'overview',   label: 'Overview' },
  { key: 'chart',      label: 'Chart' },
  { key: 'earnings',   label: 'Earnings' },
  { key: 'peers',      label: 'Peers' },
  { key: 'sentiment',  label: 'Sentiment' },
  { key: 'valuation',  label: 'Valuation' },
  { key: 'options',    label: 'Options' },
  { key: 'supply',     label: 'Supply Chain' },
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
  const { data: smartMoney } = useSmartMoney(ticker)

  return (
    <div>
      {/* Smart Money badge */}
      {smartMoney?.smart_money_badge && (
        <Card className={styles.smartMoneyCard} animate>
          <div className={styles.smartMoneyRow}>
            <Badge variant="gold" size="sm">SMART MONEY</Badge>
            <span className={styles.smartMoneyText}>
              {smartMoney.new_positions.length > 0
                ? `New position by ${smartMoney.new_positions[0].filer_name}`
                : smartMoney.increased.length > 0
                ? `Position increased by ${smartMoney.increased[0].filer_name}`
                : `Held by ${smartMoney.total_top_filers_holding} top institutional filers`}
            </span>
          </div>
        </Card>
      )}

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

// ── Valuation tab (DCF + Moat + Catalysts) ───────────────────────────────────

function ValuationTab({ ticker }: { ticker: string }) {
  const { data: dcf, isLoading: dcfLoading } = useDCF(ticker)
  const { data: moat, isLoading: moatLoading } = useMoat(ticker)
  const { data: cats, isLoading: catsLoading } = useCatalysts(ticker)

  return (
    <div>
      {/* DCF */}
      <p className={styles.sectionTitle}>DCF Fair Value</p>
      {dcfLoading ? <LoadingPanel /> : dcf && !dcf.error ? (
        <Card animate className={styles.signalCard}>
          <div className={styles.statsGrid}>
            <div className={styles.riskItem}>
              <span className={styles.riskItemLabel}>Fair Value</span>
              <span className={styles.riskItemValue}>
                {dcf.fair_value !== null ? `$${formatNum(dcf.fair_value)}` : '—'}
              </span>
            </div>
            <div className={styles.riskItem}>
              <span className={styles.riskItemLabel}>Current Price</span>
              <span className={styles.riskItemValue}>
                {dcf.current_price !== null ? `$${formatNum(dcf.current_price)}` : '—'}
              </span>
            </div>
            <div className={styles.riskItem}>
              <span className={styles.riskItemLabel}>Upside</span>
              <span className={clsx(
                styles.riskItemValue,
                dcf.upside_pct !== null && dcf.upside_pct > 0 ? styles.positive :
                dcf.upside_pct !== null && dcf.upside_pct < 0 ? styles.negative : ''
              )}>
                {dcf.upside_pct !== null ? `${dcf.upside_pct >= 0 ? '+' : ''}${formatNum(dcf.upside_pct)}%` : '—'}
              </span>
            </div>
          </div>
          <div className={styles.riskRow} style={{ marginTop: 'var(--space-3)' }}>
            <div className={styles.riskItem}>
              <span className={styles.riskItemLabel}>Growth Rate</span>
              <span className={styles.mono}>{dcf.growth_rate !== null ? `${formatNum(dcf.growth_rate * 100)}%` : '—'}</span>
            </div>
            <div className={styles.riskItem}>
              <span className={styles.riskItemLabel}>Discount Rate</span>
              <span className={styles.mono}>{formatNum(dcf.discount_rate * 100)}%</span>
            </div>
          </div>
        </Card>
      ) : <EmptyPanel message="No DCF data available." />}

      {/* Moat */}
      <p className={styles.sectionTitle}>Economic Moat</p>
      {moatLoading ? <LoadingPanel /> : moat && !moat.error ? (
        <Card animate>
          <div style={{ padding: 'var(--space-4)' }}>
            <div className={styles.signalRow}>
              <span className={styles.signalLabel}>Moat Grade</span>
              <Badge variant={moat.moat_score >= 7 ? 'success' : moat.moat_score >= 4 ? 'neutral' : 'danger'}>
                {moat.moat_grade} ({moat.moat_score}/10)
              </Badge>
            </div>
            {moat.factors.length > 0 && (
              <div className={styles.headlineList}>
                {moat.factors.map(f => (
                  <div key={f.name} className={styles.headline}>
                    <div className={styles.headlineTitle}>{f.name}</div>
                    <div className={styles.confidenceBarTrack}>
                      <div
                        className={clsx(
                          styles.confidenceBarFill,
                          f.score / f.max >= 0.7 ? styles.high :
                          f.score / f.max >= 0.4 ? styles.medium : styles.low
                        )}
                        style={{ width: `${(f.score / f.max) * 100}%` }}
                      />
                    </div>
                    <span className={styles.mono} style={{ fontSize: 'var(--text-xs)' }}>
                      {f.score}/{f.max}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>
      ) : <EmptyPanel message="No moat data available." />}

      {/* Catalysts */}
      <p className={styles.sectionTitle}>Upcoming Catalysts</p>
      {catsLoading ? <LoadingPanel /> : cats && !cats.error && cats.catalysts.length > 0 ? (
        <div className={styles.headlineList}>
          {cats.catalysts.map((c, i) => (
            <Card key={i} animate delay={i * 0.04}>
              <div style={{ padding: 'var(--space-3) var(--space-4)' }}>
                <div className={styles.headlineMeta}>
                  <Badge variant="neutral" size="xs">{c.type}</Badge>
                  <span className={styles.headlineSource}>{c.date}</span>
                </div>
                <div className={styles.headlineTitle} style={{ marginTop: 'var(--space-1)' }}>
                  {c.name}
                </div>
                {c.detail && (
                  <span className={styles.muted} style={{ fontSize: 'var(--text-xs)' }}>{c.detail}</span>
                )}
              </div>
            </Card>
          ))}
        </div>
      ) : <EmptyPanel message="No upcoming catalysts found." />}
    </div>
  )
}

// ── Options Flow tab ─────────────────────────────────────────────────────────

function OptionsTab({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useOptionsFlow(ticker)

  if (isLoading) return <LoadingPanel />
  if (isError) return <ErrorPanel message="Failed to load options data." />
  if (!data || data.error) return <EmptyPanel message={data?.error ?? 'No options data.'} />

  const s = data.summary

  return (
    <div>
      {s && (
        <div className={styles.statsGrid}>
          <Card animate delay={0.05}>
            <MetricCard label="Total Volume" value={s.total_volume?.toLocaleString() ?? '—'} mono />
          </Card>
          <Card animate delay={0.08}>
            <MetricCard label="Put/Call Ratio" value={formatNum(s.put_call_ratio)} mono />
          </Card>
          <Card animate delay={0.11}>
            <MetricCard label="IV" value={s.implied_volatility ? `${formatNum(s.implied_volatility * 100)}%` : '—'} mono />
          </Card>
        </div>
      )}

      {data.unusual_activity.length > 0 && (
        <>
          <p className={styles.sectionTitle}>Unusual Activity</p>
          <Card className={styles.tableCard} animate>
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Type</th>
                    <th className={styles.right}>Strike</th>
                    <th>Expiry</th>
                    <th className={styles.right}>Volume</th>
                    <th className={styles.right}>OI</th>
                    <th className={styles.right}>Premium</th>
                  </tr>
                </thead>
                <tbody>
                  {data.unusual_activity.map((u, i) => (
                    <motion.tr
                      key={i}
                      className={styles.row}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.04, duration: 0.3 }}
                    >
                      <td>
                        <Badge variant={u.type.toLowerCase().includes('call') ? 'success' : 'danger'} size="xs">
                          {u.type}
                        </Badge>
                      </td>
                      <td className={clsx('right', styles.mono)}>${formatNum(u.strike)}</td>
                      <td className={styles.mono}>{u.expiry}</td>
                      <td className={clsx('right', styles.mono)}>{u.volume.toLocaleString()}</td>
                      <td className={clsx('right', styles.mono)}>{u.open_interest.toLocaleString()}</td>
                      <td className={clsx('right', styles.mono)}>{formatLargeNumber(u.premium)}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {data.unusual_activity.length === 0 && (
        <EmptyPanel message="No unusual options activity detected." />
      )}
    </div>
  )
}

// ── Supply Chain tab ─────────────────────────────────────────────────────────

function SupplyChainTab({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useSupplyChain(ticker)

  if (isLoading) return <LoadingPanel />
  if (isError) return <ErrorPanel message="Failed to load supply chain." />
  if (!data || data.error) return <EmptyPanel message={data?.error ?? 'No supply chain data.'} />

  const sections = [
    { label: 'Suppliers', items: data.suppliers, variant: 'neutral' as const },
    { label: 'Customers', items: data.customers, variant: 'success' as const },
    { label: 'Partners', items: data.partners, variant: 'gold' as const },
  ]

  return (
    <div>
      {sections.map(sec => sec.items.length > 0 && (
        <div key={sec.label}>
          <p className={styles.sectionTitle}>{sec.label}</p>
          <div className={styles.headlineList}>
            {sec.items.map((item, i) => (
              <motion.div
                key={`${item.name}-${i}`}
                className={styles.headline}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04, duration: 0.3 }}
              >
                <div className={styles.headlineMeta}>
                  <Badge variant={sec.variant} size="xs">{sec.label.slice(0, -1)}</Badge>
                  {item.ticker && (
                    <span className={clsx(styles.mono, styles.positive)}>{item.ticker}</span>
                  )}
                </div>
                <div className={styles.headlineTitle}>{item.name}</div>
                {item.detail && (
                  <span className={styles.muted} style={{ fontSize: 'var(--text-xs)' }}>{item.detail}</span>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      ))}
      {sections.every(s => s.items.length === 0) && (
        <EmptyPanel message="No supply chain data found." />
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
            {activeTab === 'valuation' && (
              <ValuationTab ticker={safeT} />
            )}
            {activeTab === 'options' && (
              <OptionsTab ticker={safeT} />
            )}
            {activeTab === 'supply' && (
              <SupplyChainTab ticker={safeT} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
