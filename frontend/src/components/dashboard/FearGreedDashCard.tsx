import { useFearGreedCurrent } from '@/api/endpoints/fearGreed'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './FearGreedDashCard.module.css'

function fgColor(score: number): string {
  if (score >= 75) return 'var(--signal-positive)'
  if (score >= 55) return '#84cc16'
  if (score >= 45) return 'var(--signal-warning)'
  if (score >= 25) return '#f97316'
  return 'var(--signal-negative)'
}

function fgVariant(label: string | null): 'success' | 'warning' | 'danger' | 'neutral' {
  if (!label) return 'neutral'
  const l = label.toLowerCase()
  if (l.includes('extreme fear')) return 'danger'
  if (l.includes('fear')) return 'warning'
  if (l.includes('extreme greed') || l.includes('greed')) return 'success'
  return 'neutral'
}

export function FearGreedDashCard() {
  const { data, isLoading, isError } = useFearGreedCurrent()

  const score = data?.fg_value ?? null
  const label = data?.fg_label ?? null
  const color = score !== null ? fgColor(score) : 'var(--text-muted)'

  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <h2 className={styles.title}>Fear &amp; Greed</h2>
        {label && <Badge variant={fgVariant(label)}>{label}</Badge>}
      </div>

      {isLoading && <div className={styles.loading}><Spinner /></div>}
      {isError && <div className={styles.error}>Failed to load</div>}

      {data && score !== null && (
        <>
          <div className={styles.scoreRow}>
            <span className={styles.score} style={{ color }}>{Math.round(score)}</span>
            <div className={styles.meta}>
              <div className={styles.metaItem}>
                <span className={styles.metaLabel}>VIX</span>
                <span className={styles.metaValue}>
                  {data.vix !== null ? data.vix.toFixed(1) : '—'}
                </span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaLabel}>VIX MA20</span>
                <span className={styles.metaValue}>
                  {data.vix_ma20 !== null ? data.vix_ma20.toFixed(1) : '—'}
                </span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaLabel}>VIX MA30</span>
                <span className={styles.metaValue}>
                  {data.vix_ma30 !== null ? data.vix_ma30.toFixed(1) : '—'}
                </span>
              </div>
            </div>
          </div>
          <div className={styles.track}>
            <div className={styles.fill} style={{ width: `${score}%`, background: color }} />
          </div>
          <div className={styles.scale}>
            <span>Extreme Fear</span>
            <span>Neutral</span>
            <span>Extreme Greed</span>
          </div>
        </>
      )}
    </Card>
  )
}
