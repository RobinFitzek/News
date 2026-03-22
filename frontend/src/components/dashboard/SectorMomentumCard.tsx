import { useSectorMomentum } from '@/api/endpoints/sectorMomentum'
import type { SectorEntry } from '@/api/endpoints/sectorMomentum'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './SectorMomentumCard.module.css'

function momentumBadge(m: SectorEntry['momentum']): 'success' | 'danger' | 'neutral' {
  if (m === 'hot') return 'success'
  if (m === 'cold') return 'danger'
  return 'neutral'
}

function returnColor(val: number): string {
  if (val > 2) return 'var(--signal-positive)'
  if (val < -2) return 'var(--signal-negative)'
  return 'var(--text-secondary)'
}

export function SectorMomentumCard() {
  const { data, isLoading } = useSectorMomentum()

  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <h2 className={styles.title}>Sector Momentum</h2>
        {data && (
          <span className={styles.spyReturn}>
            SPY {data.spy_return >= 0 ? '+' : ''}{data.spy_return}%
          </span>
        )}
      </div>

      {isLoading && (
        <div className={styles.loading}><Spinner /></div>
      )}

      {data && data.sectors.length > 0 && (
        <div className={styles.grid}>
          {data.sectors.map(s => (
            <div
              key={s.etf}
              className={styles.tile}
              style={{
                borderLeftColor: s.color,
                background: `linear-gradient(135deg, ${s.color}08, transparent)`,
              }}
            >
              <div className={styles.tileTop}>
                <span className={styles.tileEtf}>{s.etf}</span>
                <Badge variant={momentumBadge(s.momentum)} size="xs">
                  {s.momentum}
                </Badge>
              </div>
              <span className={styles.tileName}>{s.name}</span>
              <div className={styles.tileReturns}>
                <span className={styles.returnItem}>
                  <span className={styles.returnLabel}>1W</span>
                  <span style={{ color: returnColor(s.return_1wk) }}>
                    {s.return_1wk >= 0 ? '+' : ''}{s.return_1wk}%
                  </span>
                </span>
                <span className={styles.returnItem}>
                  <span className={styles.returnLabel}>1M</span>
                  <span style={{ color: returnColor(s.return_1mo) }}>
                    {s.return_1mo >= 0 ? '+' : ''}{s.return_1mo}%
                  </span>
                </span>
                <span className={styles.returnItem}>
                  <span className={styles.returnLabel}>RS</span>
                  <span style={{ color: returnColor(s.relative_strength) }}>
                    {s.relative_strength >= 0 ? '+' : ''}{s.relative_strength}
                  </span>
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {data && data.sectors.length === 0 && (
        <p className={styles.empty}>No sector data available.</p>
      )}
    </Card>
  )
}
