import { useState } from 'react'
import { motion } from 'framer-motion'
import { useFearGreedCurrent, useFearGreedHistory, useFGSensitivity } from '@/api/endpoints/fearGreed'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { MetricCard } from '@/components/ui/MetricCard'
import { Spinner } from '@/components/ui/Spinner'
import { Button } from '@/components/ui/Button'
import styles from './FearGreedPage.module.css'

function fgColor(v: number | null): string {
  if (v == null) return 'var(--text-muted)'
  if (v <= 20) return '#ef4444'
  if (v <= 40) return '#f97316'
  if (v <= 60) return '#eab308'
  if (v <= 80) return '#84cc16'
  return '#22c55e'
}

function vixColor(vix: number | null, ma20: number | null): string {
  if (!vix || !ma20) return 'var(--text-muted)'
  if (vix > ma20 * 1.2) return 'var(--signal-negative)'
  if (vix < ma20 * 0.9) return 'var(--signal-positive)'
  return 'var(--text-secondary)'
}

function SensitivityBar({ value }: { value: number | null }) {
  if (value == null) return <span className={styles.muted}>—</span>
  const pct = ((value + 1) / 2) * 100
  const color = value > 0.3 ? '#22c55e' : value < -0.3 ? '#ef4444' : '#eab308'
  return (
    <div className={styles.sensitivityWrapper}>
      <div className={styles.sensitivityBar}>
        <div
          className={styles.sensitivityFill}
          style={{ width: `${pct}%`, background: color }}
        />
        <div className={styles.sensitivityMid} />
      </div>
      <span className={styles.sensitivityVal} style={{ color }}>
        {value > 0 ? '+' : ''}{value.toFixed(3)}
      </span>
    </div>
  )
}

export function FearGreedPage() {
  const [tickerInput, setTickerInput] = useState('')
  const [lookupTicker, setLookupTicker] = useState('')

  const { data: current, isLoading: currentLoading } = useFearGreedCurrent()
  const { data: history, isLoading: historyLoading } = useFearGreedHistory()
  const { data: sensitivity } = useFGSensitivity(lookupTicker, 60)

  const fg = current?.fg_value ?? null
  const recentHistory = history?.data.slice(-30) ?? []

  function handleLookup() {
    if (tickerInput.trim()) setLookupTicker(tickerInput.trim().toUpperCase())
  }

  return (
    <>
      <PageHeader
        title="Fear & Greed"
        subtitle="CNN Fear & Greed Index + VIX rolling averages + per-stock sensitivity factor"
      />

      {/* Main metrics */}
      {currentLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <>
          <div className={styles.topRow}>
            {/* Fear & Greed gauge */}
            <Card delay={0} className={styles.gaugeCard}>
              <div className={styles.gaugeLabel}>Fear & Greed Index</div>
              <div className={styles.gaugeValue} style={{ color: fgColor(fg) }}>
                {fg != null ? fg.toFixed(0) : '—'}
              </div>
              <div className={styles.gaugeSub} style={{ color: fgColor(fg) }}>
                {current?.fg_label ?? '—'}
              </div>
              <div className={styles.gaugeTrack}>
                <div
                  className={styles.gaugeFill}
                  style={{ width: `${fg ?? 0}%`, background: fgColor(fg) }}
                />
              </div>
              <div className={styles.gaugeScale}>
                <span>Extreme Fear</span>
                <span>Neutral</span>
                <span>Extreme Greed</span>
              </div>
            </Card>

            {/* VIX panel */}
            <Card delay={0.05} className={styles.vixCard}>
              <div className={styles.sectionTitle}>VIX &amp; Rolling Averages</div>
              <div className={styles.vixGrid}>
                <div>
                  <div className={styles.metaLabel}>VIX (Current)</div>
                  <div
                    className={styles.vixVal}
                    style={{ color: vixColor(current?.vix ?? null, current?.vix_ma20 ?? null) }}
                  >
                    {current?.vix != null ? current.vix.toFixed(2) : '—'}
                  </div>
                </div>
                <div>
                  <div className={styles.metaLabel}>10-Day MA</div>
                  <div className={styles.vixVal}>{current?.vix_ma10 != null ? current.vix_ma10.toFixed(2) : '—'}</div>
                </div>
                <div>
                  <div className={styles.metaLabel}>20-Day MA</div>
                  <div className={styles.vixVal}>{current?.vix_ma20 != null ? current.vix_ma20.toFixed(2) : '—'}</div>
                </div>
                <div>
                  <div className={styles.metaLabel}>30-Day MA</div>
                  <div className={styles.vixVal}>{current?.vix_ma30 != null ? current.vix_ma30.toFixed(2) : '—'}</div>
                </div>
              </div>
              <div className={styles.vixNote}>
                VIX above 20-day MA → elevated volatility regime
              </div>
            </Card>
          </div>

          {/* Sensitivity lookup */}
          <Card delay={0.1} className={styles.sensitivityCard}>
            <div className={styles.sectionTitle}>Fear &amp; Greed Sensitivity Factor</div>
            <div className={styles.sensitivityNote}>
              60-day rolling correlation between stock price and F&G index.<br />
              Positive = risk-on (drops with fear). Negative = defensive (rises with fear).
            </div>
            <div className={styles.lookupRow}>
              <input
                className={styles.tickerInput}
                placeholder="Enter ticker (e.g. AAPL)"
                value={tickerInput}
                onChange={e => setTickerInput(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === 'Enter' && handleLookup()}
              />
              <Button variant="primary" size="md" onClick={handleLookup}>
                Lookup
              </Button>
            </div>
            {lookupTicker && sensitivity && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={styles.sensitivityResult}
              >
                <div className={styles.sensitivityTicker}>{lookupTicker}</div>
                <SensitivityBar value={sensitivity.fg_sensitivity} />
                <div className={styles.sensitivityInterp}>{sensitivity.interpretation}</div>
              </motion.div>
            )}
          </Card>

          {/* Recent F&G history */}
          {!historyLoading && recentHistory.length > 0 && (
            <Card delay={0.15} className={styles.historyCard}>
              <div className={styles.sectionTitle}>Last 30 Days — Fear &amp; Greed History</div>
              <div className={styles.historyGrid}>
                {recentHistory.map((point, i) => (
                  <motion.div
                    key={point.date}
                    initial={{ opacity: 0, scaleY: 0.5 }}
                    animate={{ opacity: 1, scaleY: 1 }}
                    transition={{ delay: i * 0.015, duration: 0.2 }}
                    className={styles.historyBar}
                    style={{
                      height: `${point.fg_value}%`,
                      background: fgColor(point.fg_value),
                      opacity: 0.7,
                    }}
                    title={`${point.date}: ${point.fg_value.toFixed(0)}`}
                  />
                ))}
              </div>
              <div className={styles.historyLabels}>
                <span>{recentHistory[0]?.date}</span>
                <span>{recentHistory[recentHistory.length - 1]?.date}</span>
              </div>
            </Card>
          )}
        </>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
