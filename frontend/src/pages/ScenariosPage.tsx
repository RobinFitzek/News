import { useState } from 'react'
import { motion } from 'framer-motion'
import { useScenarios, useRunScenario } from '@/api/endpoints/scenarios'
import type { ScenarioResult } from '@/api/endpoints/scenarios'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import styles from './ScenariosPage.module.css'

function impactColor(pct: number): string {
  return pct >= 0 ? '#10b981' : '#ef4444'
}

function formatSector(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// ── Scenario card ──────────────────────────────────────────────────────────

function ScenarioCard({
  scenarioKey,
  name,
  description,
  sectorImpacts,
  historicalAnalog,
}: {
  scenarioKey: string
  name: string
  description: string
  sectorImpacts: Record<string, number>
  historicalAnalog: string
}) {
  const runMut = useRunScenario()
  const [result, setResult] = useState<ScenarioResult | null>(null)

  const sortedSectors = Object.entries(sectorImpacts).sort(([, a], [, b]) => a - b)

  const handleRun = () => {
    runMut.mutate(scenarioKey, {
      onSuccess: data => setResult(data),
    })
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={styles.card}>
        <div className={styles.cardInner}>
          <div>
            <div className={styles.cardTitle}>{name}</div>
            <div className={styles.cardDesc}>{description}</div>
          </div>

          {/* Sector impacts */}
          <div>
            <div className={styles.sectorLabel}>Sector Impacts</div>
            {sortedSectors.map(([sector, impact]) => {
              const pct = impact * 100
              const barWidth = Math.min(Math.abs(pct) * 2, 80)
              return (
                <div key={sector} className={styles.impactRow}>
                  <span className={styles.sectorName}>{formatSector(sector)}</span>
                  <div
                    className={styles.barFill}
                    style={{
                      width: `${barWidth}px`,
                      background: impactColor(pct),
                    }}
                  />
                  <span className={styles.pct} style={{ color: impactColor(pct) }}>
                    {pct >= 0 ? '+' : ''}{pct.toFixed(0)}%
                  </span>
                </div>
              )
            })}
          </div>

          <div className={styles.analog}>Historical analog: {historicalAnalog}</div>

          {/* Run button + result */}
          <div className={styles.runArea}>
            <Button
              onClick={handleRun}
              disabled={runMut.isPending}
              style={{ width: '100%' }}
            >
              {runMut.isPending ? 'Running...' : result ? 'Run Again' : 'Run Scenario'}
            </Button>

            {runMut.isError && (
              <div className={styles.resultBox} style={{ color: '#ef4444' }}>
                Scenario failed. Check your portfolio data.
              </div>
            )}

            {result && (
              <div className={styles.resultBox}>
                <div className={styles.impactLabel}>
                  Estimated Portfolio Impact:{' '}
                  <span
                    className={styles.impactValue}
                    style={{ color: impactColor(result.portfolio_impact_pct) }}
                  >
                    {result.portfolio_impact_pct >= 0 ? '+' : ''}
                    {result.portfolio_impact_pct.toFixed(2)}%
                  </span>
                </div>

                {result.holdings_impact && result.holdings_impact.length > 0 ? (
                  <>
                    <div className={styles.exposureHeader}>Top exposures:</div>
                    {result.holdings_impact.slice(0, 5).map(h => (
                      <div key={h.ticker} className={styles.exposureRow}>
                        <span className={styles.exposureTicker}>{h.ticker}</span>
                        <span
                          className={styles.exposurePnl}
                          style={{ color: impactColor(h.estimated_pnl_pct) }}
                        >
                          {h.estimated_pnl_pct >= 0 ? '+' : ''}{h.estimated_pnl_pct.toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </>
                ) : (
                  <div className={styles.noPositions}>
                    No portfolio positions to cross-reference.
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────

export function ScenariosPage() {
  const { data: scenarios, isLoading } = useScenarios()

  const entries = scenarios ? Object.entries(scenarios) : []

  return (
    <>
      <PageHeader
        title="Geopolitical Scenarios"
        subtitle="Stress-test your portfolio against predefined macro events"
      />

      <Card delay={0}>
        <div className={styles.infoBanner}>
          <svg
            className={styles.infoIcon}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            width="18"
            height="18"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <p>
            Scenarios are estimated impact models based on sector weights. They are educational
            tools, not price predictions. Click <strong>Run Scenario</strong> to estimate impact
            on your current portfolio.
          </p>
        </div>
      </Card>

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <div className={styles.grid}>
          {entries.map(([key, sc]) => (
            <ScenarioCard
              key={key}
              scenarioKey={key}
              name={sc.name}
              description={sc.description}
              sectorImpacts={sc.sector_impacts}
              historicalAnalog={sc.historical_analog}
            />
          ))}
        </div>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
