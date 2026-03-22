import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import clsx from 'clsx'
import { usePortfolio, useAddTrade } from '@/api/endpoints/portfolioTracker'
import type { AddTradePayload } from '@/api/endpoints/portfolioTracker'
import { usePortfolioAsk } from '@/api/endpoints/portfolio'
import type { PortfolioQAResponse } from '@/api/endpoints/portfolio'
import {
  usePortfolioVaR,
  usePortfolioCorrelation,
  usePortfolioConcentration,
  useDrawdown,
  useRebalancingPlan,
} from '@/api/endpoints/portfolioRisk'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { Modal } from '@/components/ui/Modal'
import { useToastStore } from '@/stores/toastStore'
import styles from './PortfolioPage.module.css'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function fmtCurrency(n: number): string {
  return `$${fmt(n)}`
}

function pnlClass(n: number): string {
  return n > 0 ? styles.positive : n < 0 ? styles.negative : ''
}

const DEFAULT_FORM: AddTradePayload = {
  ticker: '',
  type: 'BUY',
  amount: 0,
  price: 0,
  date: new Date().toISOString().slice(0, 10),
  fees: 0,
  notes: '',
  currency: 'USD',
}

// ── Risk Analytics ────────────────────────────────────────────────────────────

function PortfolioRiskSection() {
  const { data: varData } = usePortfolioVaR()
  const { data: corrData } = usePortfolioCorrelation()
  const { data: concData } = usePortfolioConcentration()
  const { data: ddData } = useDrawdown()
  const { data: rebalData } = useRebalancingPlan()
  const [showRebal, setShowRebal] = useState(false)

  return (
    <>
      <p className={styles.sectionTitle} style={{ marginTop: 'var(--space-6)' }}>
        Risk Analytics
      </p>

      {/* VaR + Drawdown row */}
      <div className={styles.riskGrid}>
        {varData && !varData.error && (
          <Card delay={0}>
            <div className={styles.riskCard}>
              <span className={styles.riskCardTitle}>Value at Risk</span>
              <div className={styles.riskMetrics}>
                <div className={styles.riskMetric}>
                  <span className={styles.riskMetricLabel}>VaR 95%</span>
                  <span className={clsx(styles.riskMetricValue, styles.negative)}>
                    {fmtCurrency(varData.var_95)}
                  </span>
                </div>
                <div className={styles.riskMetric}>
                  <span className={styles.riskMetricLabel}>VaR 99%</span>
                  <span className={clsx(styles.riskMetricValue, styles.negative)}>
                    {fmtCurrency(varData.var_99)}
                  </span>
                </div>
                <div className={styles.riskMetric}>
                  <span className={styles.riskMetricLabel}>CVaR 95%</span>
                  <span className={clsx(styles.riskMetricValue, styles.negative)}>
                    {fmtCurrency(varData.cvar_95)}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        )}

        {ddData && !ddData.error && (
          <Card delay={0.05}>
            <div className={styles.riskCard}>
              <span className={styles.riskCardTitle}>Drawdown</span>
              <div className={styles.riskMetrics}>
                <div className={styles.riskMetric}>
                  <span className={styles.riskMetricLabel}>Max DD</span>
                  <span className={clsx(styles.riskMetricValue, styles.negative)}>
                    {fmt(ddData.max_drawdown)}%
                  </span>
                </div>
                <div className={styles.riskMetric}>
                  <span className={styles.riskMetricLabel}>Current DD</span>
                  <span className={clsx(styles.riskMetricValue, ddData.current_drawdown < -5 ? styles.negative : '')}>
                    {fmt(ddData.current_drawdown)}%
                  </span>
                </div>
                {ddData.recovery_days !== null && (
                  <div className={styles.riskMetric}>
                    <span className={styles.riskMetricLabel}>Recovery</span>
                    <span className={styles.riskMetricValue}>{ddData.recovery_days}d</span>
                  </div>
                )}
              </div>
            </div>
          </Card>
        )}
      </div>

      {/* Concentration */}
      {concData && !concData.error && (
        <Card className={styles.tableCard} animate={false}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>Concentration</span>
            {concData.warnings.length > 0 && (
              <Badge variant="warning" size="xs">{concData.warnings.length} warnings</Badge>
            )}
          </div>
          <div className={styles.concentrationGrid}>
            <div className={styles.riskMetric}>
              <span className={styles.riskMetricLabel}>Top Position</span>
              <span className={styles.riskMetricValue}>{fmt(concData.top_position_weight)}%</span>
            </div>
            <div className={styles.riskMetric}>
              <span className={styles.riskMetricLabel}>HHI</span>
              <span className={styles.riskMetricValue}>{fmt(concData.herfindahl_index, 4)}</span>
            </div>
          </div>
          {concData.sector_exposure.length > 0 && (
            <div className={styles.sectorBars}>
              {concData.sector_exposure.map(s => (
                <div key={s.sector} className={styles.sectorBar}>
                  <span className={styles.sectorBarLabel}>{s.sector}</span>
                  <div className={styles.sectorBarTrack}>
                    <div
                      className={styles.sectorBarFill}
                      style={{ width: `${Math.min(s.weight, 100)}%` }}
                    />
                  </div>
                  <span className={styles.sectorBarValue}>{fmt(s.weight)}%</span>
                </div>
              ))}
            </div>
          )}
          {concData.warnings.length > 0 && (
            <div className={styles.warningsList}>
              {concData.warnings.map((w, i) => (
                <div key={i} className={styles.warningItem}>{w}</div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Correlation Matrix */}
      {corrData && !corrData.error && corrData.tickers.length >= 2 && (
        <Card className={styles.tableCard} animate={false}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>Correlation Matrix</span>
          </div>
          <div className={styles.tableWrapper}>
            <table className={styles.corrTable}>
              <thead>
                <tr>
                  <th />
                  {corrData.tickers.map(t => (
                    <th key={t} className={styles.corrHeader}>{t}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corrData.tickers.map((rowT, ri) => (
                  <tr key={rowT}>
                    <td className={styles.corrRowLabel}>{rowT}</td>
                    {corrData.matrix[ri].map((val, ci) => {
                      const abs = Math.abs(val)
                      const color = ri === ci
                        ? 'transparent'
                        : val > 0.7
                          ? `rgba(232,96,96,${abs * 0.3})`
                          : val < -0.3
                            ? `rgba(78,232,138,${abs * 0.3})`
                            : `rgba(107,184,255,${abs * 0.15})`
                      return (
                        <td
                          key={ci}
                          className={styles.corrCell}
                          style={{ backgroundColor: color }}
                        >
                          {ri === ci ? '1.00' : val.toFixed(2)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Rebalancing Plan */}
      {rebalData && !rebalData.error && rebalData.count > 0 && (
        <Card className={styles.tableCard} animate={false}>
          <button
            className={styles.tradeLogToggle}
            onClick={() => setShowRebal(v => !v)}
          >
            <span className={clsx(styles.toggleIcon, showRebal && styles.open)}>▶</span>
            Rebalancing Plan
            <span className={styles.tradeCount}>{rebalData.count} actions</span>
          </button>
          <AnimatePresence>
            {showRebal && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
                style={{ overflow: 'hidden' }}
              >
                <div className={styles.tableWrapper}>
                  <table className={styles.tradeTable}>
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Action</th>
                        <th>Current %</th>
                        <th>Target %</th>
                        <th>Shares</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rebalData.plan.map(r => (
                        <tr key={r.ticker} className={styles.tradeRow}>
                          <td><span className={styles.ticker}>{r.ticker}</span></td>
                          <td>
                            <Badge variant={r.action === 'BUY' ? 'success' : r.action === 'SELL' ? 'danger' : 'neutral'}>
                              {r.action}
                            </Badge>
                          </td>
                          <td><span className={styles.mono}>{fmt(r.current_weight)}%</span></td>
                          <td><span className={styles.mono}>{fmt(r.target_weight)}%</span></td>
                          <td><span className={styles.mono}>{r.delta_shares >= 0 ? '+' : ''}{r.delta_shares}</span></td>
                          <td><span className={clsx(styles.mono, r.delta_value >= 0 ? styles.positive : styles.negative)}>
                            {r.delta_value >= 0 ? '+' : ''}{fmtCurrency(r.delta_value)}
                          </span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
      )}
    </>
  )
}

// ── Component ─────────────────────────────────────────────────────────────────

export function PortfolioPage() {
  const { data, isLoading } = usePortfolio()
  const addTradeMut = useAddTrade()
  const askMut = usePortfolioAsk()
  const { addToast } = useToastStore()

  const [showModal, setShowModal] = useState(false)
  const [showTradeLog, setShowTradeLog] = useState(false)
  const [form, setForm] = useState<AddTradePayload>(DEFAULT_FORM)
  const [question, setQuestion] = useState('')
  const [qaResult, setQaResult] = useState<PortfolioQAResponse | null>(null)
  const qaInputRef = useRef<HTMLInputElement>(null)

  const summary = data?.summary
  const holdings = summary?.holdings ?? []
  const trades = data?.trades ?? []

  function setField<K extends keyof AddTradePayload>(key: K, value: AddTradePayload[K]) {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  async function handleAddTrade() {
    if (!form.ticker.trim()) {
      addToast('Ticker is required', 'warning')
      return
    }
    try {
      await addTradeMut.mutateAsync({
        ...form,
        ticker: form.ticker.trim().toUpperCase(),
        amount: Number(form.amount),
        price: Number(form.price),
        fees: Number(form.fees ?? 0),
      })
      addToast(`Trade for ${form.ticker.toUpperCase()} added`, 'success')
      setForm(DEFAULT_FORM)
      setShowModal(false)
    } catch {
      addToast('Failed to add trade', 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="Portfolio"
        subtitle="Track your positions and performance"
        actions={
          <Button variant="primary" size="md" onClick={() => setShowModal(true)}>
            Add Trade
          </Button>
        }
      />

      {/* Summary metrics */}
      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          <div className={styles.summaryGrid}>
            <Card delay={0}>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>Total Value</span>
                <span className={styles.metricValue}>
                  {summary ? fmtCurrency(summary.total_value) : '—'}
                </span>
              </div>
            </Card>
            <Card delay={0.05}>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>Total P&amp;L</span>
                <span className={clsx(styles.metricValue, summary ? pnlClass(summary.total_pnl) : '')}>
                  {summary
                    ? `${summary.total_pnl >= 0 ? '+' : ''}${fmtCurrency(summary.total_pnl)}`
                    : '—'}
                </span>
              </div>
            </Card>
            <Card delay={0.1}>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>P&amp;L %</span>
                <span className={clsx(styles.metricValue, summary ? pnlClass(summary.total_pnl_pct) : '')}>
                  {summary
                    ? `${summary.total_pnl_pct >= 0 ? '+' : ''}${fmt(summary.total_pnl_pct)}%`
                    : '—'}
                </span>
              </div>
            </Card>
            <Card delay={0.15}>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>Holdings</span>
                <span className={styles.metricValue}>{holdings.length}</span>
              </div>
            </Card>
          </div>

          {/* Holdings table */}
          <Card className={styles.tableCard} animate={false}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionTitle}>Holdings</span>
              <a className={styles.exportLink} href="/portfolio/export">
                Export CSV
              </a>
            </div>
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th className={styles.right}>Shares</th>
                    <th className={styles.right}>Avg Cost</th>
                    <th className={styles.right}>Current</th>
                    <th className={styles.right}>Market Value</th>
                    <th className={styles.right}>P&amp;L</th>
                    <th className={styles.right}>P&amp;L %</th>
                    <th className={styles.right}>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((h, i) => (
                    <motion.tr
                      key={h.ticker}
                      className={styles.row}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.03, duration: 0.3 }}
                    >
                      <td><span className={styles.ticker}>{h.ticker}</span></td>
                      <td className={styles.right}>
                        <span className={styles.mono}>{fmt(h.shares, 4)}</span>
                      </td>
                      <td className={styles.right}>
                        <span className={styles.mono}>{fmtCurrency(h.avg_cost)}</span>
                      </td>
                      <td className={styles.right}>
                        <span className={styles.mono}>{fmtCurrency(h.current_price)}</span>
                      </td>
                      <td className={styles.right}>
                        <span className={styles.mono}>{fmtCurrency(h.market_value)}</span>
                      </td>
                      <td className={styles.right}>
                        <span className={clsx(styles.mono, pnlClass(h.pnl))}>
                          {h.pnl >= 0 ? '+' : ''}{fmtCurrency(h.pnl)}
                        </span>
                      </td>
                      <td className={styles.right}>
                        <span className={clsx(styles.mono, pnlClass(h.pnl_pct))}>
                          {h.pnl_pct >= 0 ? '+' : ''}{fmt(h.pnl_pct)}%
                        </span>
                      </td>
                      <td className={styles.right}>
                        <span className={styles.weight}>{fmt(h.weight)}%</span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
              {holdings.length === 0 && (
                <div className={styles.emptyState}>No holdings yet.</div>
              )}
            </div>
          </Card>

          {/* Trade Log */}
          <Card className={styles.tableCard} animate={false}>
            <button
              className={styles.tradeLogToggle}
              onClick={() => setShowTradeLog(v => !v)}
            >
              <span className={clsx(styles.toggleIcon, showTradeLog && styles.open)}>▶</span>
              Trade Log
              <span className={styles.tradeCount}>{trades.length} trades</span>
            </button>
            <AnimatePresence>
              {showTradeLog && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
                  style={{ overflow: 'hidden' }}
                >
                  <div className={styles.tableWrapper}>
                    <table className={styles.tradeTable}>
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Ticker</th>
                          <th>Type</th>
                          <th>Shares</th>
                          <th>Price</th>
                          <th>Fees</th>
                          <th>Currency</th>
                          <th>Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trades.map((t, i) => (
                          <motion.tr
                            key={t.id}
                            className={styles.tradeRow}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: i * 0.02 }}
                          >
                            <td>{t.date}</td>
                            <td><span className={styles.ticker}>{t.ticker}</span></td>
                            <td>
                              <Badge variant={t.type === 'BUY' ? 'success' : 'danger'}>
                                {t.type}
                              </Badge>
                            </td>
                            <td>{fmt(t.amount, 4)}</td>
                            <td>{fmtCurrency(t.price)}</td>
                            <td>{t.fees ? fmtCurrency(t.fees) : '—'}</td>
                            <td>{t.currency}</td>
                            <td>{t.notes || '—'}</td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                    {trades.length === 0 && (
                      <div className={styles.emptyState}>No trades recorded.</div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>
        </>
      )}

      {/* Risk Analytics Section */}
      <PortfolioRiskSection />

      {/* Add Trade Modal */}
      <Modal
        open={showModal}
        onClose={() => setShowModal(false)}
        title="Add Trade"
        size="md"
      >
        <div className={styles.formGrid}>
          <div className={styles.formField}>
            <label className={styles.label}>Ticker</label>
            <input
              className={styles.input}
              placeholder="AAPL"
              value={form.ticker}
              onChange={e => setField('ticker', e.target.value.toUpperCase())}
            />
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Type</label>
            <select
              className={styles.select}
              value={form.type}
              onChange={e => setField('type', e.target.value as 'BUY' | 'SELL')}
            >
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Shares / Amount</label>
            <input
              className={styles.input}
              type="number"
              min="0"
              step="any"
              placeholder="100"
              value={form.amount || ''}
              onChange={e => setField('amount', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Price</label>
            <input
              className={styles.input}
              type="number"
              min="0"
              step="any"
              placeholder="150.00"
              value={form.price || ''}
              onChange={e => setField('price', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Date</label>
            <input
              className={styles.input}
              type="date"
              value={form.date}
              onChange={e => setField('date', e.target.value)}
            />
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Fees</label>
            <input
              className={styles.input}
              type="number"
              min="0"
              step="any"
              placeholder="0.00"
              value={form.fees || ''}
              onChange={e => setField('fees', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Currency</label>
            <select
              className={styles.select}
              value={form.currency}
              onChange={e => setField('currency', e.target.value)}
            >
              <option value="USD">USD</option>
              <option value="EUR">EUR</option>
              <option value="SEK">SEK</option>
              <option value="GBP">GBP</option>
            </select>
          </div>
          <div className={clsx(styles.formField, styles.fullWidth)}>
            <label className={styles.label}>Notes</label>
            <textarea
              className={styles.textarea}
              placeholder="Optional notes about this trade..."
              value={form.notes ?? ''}
              onChange={e => setField('notes', e.target.value)}
            />
          </div>
        </div>
        <div className={styles.modalActions}>
          <Button variant="ghost" size="md" onClick={() => setShowModal(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="md"
            loading={addTradeMut.isPending}
            onClick={handleAddTrade}
            disabled={!form.ticker.trim() || !form.amount || !form.price}
          >
            Add Trade
          </Button>
        </div>
      </Modal>

      {/* Portfolio Q&A Widget (#56) */}
      <Card className={styles.qaCard} glow="neutral">
        <div className={styles.qaHeader}>
          <span className={styles.qaTitle}>Ask about your portfolio</span>
          <Badge variant="neutral" size="xs">AI</Badge>
        </div>
        <div className={styles.qaInputRow}>
          <input
            ref={qaInputRef}
            className={styles.qaInput}
            placeholder="e.g. Which holdings are most exposed to tariff risk?"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && question.trim()) {
                askMut.mutate(question.trim(), {
                  onSuccess: (result) => {
                    setQaResult(result)
                    if (result.rate_limited) {
                      addToast('Rate limited — wait 30s between queries', 'warning')
                    }
                  },
                  onError: () => addToast('Failed to get answer', 'error'),
                })
              }
            }}
            disabled={askMut.isPending}
          />
          <Button
            variant="primary"
            size="md"
            loading={askMut.isPending}
            disabled={!question.trim()}
            onClick={() => {
              if (!question.trim()) return
              askMut.mutate(question.trim(), {
                onSuccess: (result) => {
                  setQaResult(result)
                  if (result.rate_limited) {
                    addToast('Rate limited — wait 30s between queries', 'warning')
                  }
                },
                onError: () => addToast('Failed to get answer', 'error'),
              })
            }}
          >
            Ask
          </Button>
        </div>

        <AnimatePresence>
          {qaResult && (
            <motion.div
              className={styles.qaResult}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            >
              {qaResult.error ? (
                <div className={styles.qaError}>{qaResult.error}</div>
              ) : (
                <>
                  <div className={styles.qaAnswer}>{qaResult.answer}</div>
                  {qaResult.sources && qaResult.sources.length > 0 && (
                    <div className={styles.qaSources}>
                      <span className={styles.qaSourcesLabel}>Sources:</span>
                      {qaResult.sources.map((s, i) => (
                        <Badge key={i} variant="ghost" size="xs">{s}</Badge>
                      ))}
                    </div>
                  )}
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
