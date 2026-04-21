import { memo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useGrahamScreen, useGrahamAAAYield } from '@/api/endpoints/graham'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import styles from './GrahamDashCard.module.css'

const GrahamDashCard = memo(function GrahamDashCard() {
  const { data: screen, isLoading } = useGrahamScreen(0.2, true)
  const { data: yieldData } = useGrahamAAAYield()
  const navigate = useNavigate()

  const candidates = screen?.buy_list ?? []
  const ivCalculable = screen?.iv_calculable ?? 0

  return (
    <Card
      className={styles.card}
      onClick={() => navigate('/graham')}
      style={{ cursor: 'pointer' }}
    >
      <div className={styles.header}>
        <h2 className={styles.title}>Graham Value</h2>
        <Badge variant={candidates.length > 0 ? 'success' : 'ghost'}>
          {isLoading ? '…' : `${candidates.length} buy`}
        </Badge>
      </div>

      {isLoading && <div className={styles.loading}><Spinner /></div>}

      {!isLoading && (
        <>
          <div className={styles.stats}>
            <div className={styles.stat}>
              <span className={styles.statNum}>{candidates.length}</span>
              <span className={styles.statLabel}>Candidates</span>
            </div>
            <div className={styles.divider} />
            <div className={styles.stat}>
              <span className={styles.statNum}>{ivCalculable}</span>
              <span className={styles.statLabel}>IV Calculable</span>
            </div>
            <div className={styles.divider} />
            <div className={styles.stat}>
              <span className={styles.statNum}>
                {yieldData?.aaa_yield_pct ? `${yieldData.aaa_yield_pct.toFixed(1)}%` : '—'}
              </span>
              <span className={styles.statLabel}>AAA Yield</span>
            </div>
          </div>

          {candidates.length > 0 ? (
            <div className={styles.list}>
              {candidates.slice(0, 5).map(c => (
                <div key={c.ticker} className={styles.item}>
                  <span className={styles.ticker}>{c.ticker}</span>
                  <div className={styles.right}>
                    <span
                      className={styles.upside}
                      style={{
                        color: (c.upside_pct ?? 0) >= 0
                          ? 'var(--signal-positive)'
                          : 'var(--signal-negative)',
                      }}
                    >
                      {c.upside_pct !== null
                        ? `${c.upside_pct >= 0 ? '+' : ''}${c.upside_pct.toFixed(0)}%`
                        : '—'}
                    </span>
                    {c.current_price !== null && (
                      <span className={styles.price}>${c.current_price.toFixed(0)}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className={styles.empty}>No candidates at 20% MoS</div>
          )}
        </>
      )}
    </Card>
  )
})

export { GrahamDashCard }
