import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import api from '@/api/client'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './SectorScreenPage.module.css'

interface SectorEntry {
  name: string
  momentum_score: number
  avg_signal: 'BUY' | 'SELL' | 'HOLD'
  ticker_count: number
  top_picks: string[]
  risk_level: 'low' | 'medium' | 'high'
}

interface SectorScreenResponse {
  sectors: SectorEntry[]
}

function momentumColor(score: number): string {
  if (score >= 7) return '#10b981'
  if (score >= 4) return '#f59e0b'
  return '#ef4444'
}

function signalVariant(signal: string): 'success' | 'danger' | 'neutral' {
  if (signal === 'BUY') return 'success'
  if (signal === 'SELL') return 'danger'
  return 'neutral'
}

function riskVariant(risk: string): 'success' | 'warning' | 'danger' {
  if (risk === 'low') return 'success'
  if (risk === 'medium') return 'warning'
  return 'danger'
}

export function SectorScreenPage() {
  const { data, isLoading } = useQuery<SectorScreenResponse>({
    queryKey: ['sector-screen'],
    queryFn: () => api.get('/api/sector-screen').then(r => r.data),
    staleTime: 60_000,
  })

  const sectors = (data?.sectors ?? []).slice().sort((a, b) => b.momentum_score - a.momentum_score)

  return (
    <>
      <PageHeader
        title="Sector Screen"
        subtitle="Sector momentum and opportunity ranking"
      />

      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : (
        <div className={styles.grid}>
          {sectors.map((sector, i) => (
            <motion.div
              key={sector.name}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.35 }}
            >
              <Card className={styles.sectorCard} animate={false}>
                <div className={styles.sectorHeader}>
                  <h3 className={styles.sectorName}>{sector.name}</h3>
                  <div
                    className={styles.momentumScore}
                    style={{ color: momentumColor(sector.momentum_score) }}
                  >
                    {sector.momentum_score.toFixed(1)}
                    <span className={styles.scoreSuffix}>/10</span>
                  </div>
                </div>

                <div className={styles.badgeRow}>
                  <Badge variant={signalVariant(sector.avg_signal)}>
                    {sector.avg_signal}
                  </Badge>
                  <Badge variant={riskVariant(sector.risk_level)}>
                    risk: {sector.risk_level}
                  </Badge>
                </div>

                <div className={styles.tickerCount}>{sector.ticker_count} tickers</div>

                {sector.top_picks.length > 0 && (
                  <div className={styles.topPicksRow}>
                    {sector.top_picks.map(ticker => (
                      <span key={ticker} className={styles.tickerPill}>{ticker}</span>
                    ))}
                  </div>
                )}
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
