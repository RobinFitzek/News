import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useGeopolitical, useGeoExposure } from '@/api/endpoints/geopolitical'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatDistanceToNow } from 'date-fns'
import styles from './GeoRadarCard.module.css'

function severityVariant(n: number): 'danger' | 'warning' | 'neutral' | 'ghost' {
  if (n >= 8) return 'danger'
  if (n >= 5) return 'warning'
  if (n >= 3) return 'neutral'
  return 'ghost'
}

export function GeoRadarCard() {
  const { data: geoData, isLoading } = useGeopolitical()
  const { data: exposureData } = useGeoExposure()
  const [expanded, setExpanded] = useState(false)

  const scan = geoData?.scan
  const exposures = exposureData?.exposures ?? []

  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>Geopolitical Radar</h2>
          {scan && (
            <Badge variant={severityVariant(scan.overall_severity)}>
              Severity {scan.overall_severity}/10
            </Badge>
          )}
        </div>
        {scan && (
          <span className={styles.timestamp}>
            {formatDistanceToNow(new Date(scan.timestamp), { addSuffix: true })}
          </span>
        )}
      </div>

      {isLoading && <div className={styles.loading}><Spinner /></div>}

      {scan && (
        <>
          {/* Exposure heatmap */}
          {exposures.length > 0 && (
            <div className={styles.exposureGrid}>
              {exposures.slice(0, 12).map(e => (
                <div
                  key={e.ticker}
                  className={styles.exposureTile}
                  style={{
                    backgroundColor: `rgba(${
                      e.geo_risk_score >= 7 ? '232,96,96' :
                      e.geo_risk_score >= 4 ? '212,168,75' :
                      '78,232,138'
                    }, ${0.04 + e.geo_risk_score * 0.02})`,
                    borderColor: `rgba(${
                      e.geo_risk_score >= 7 ? '232,96,96' :
                      e.geo_risk_score >= 4 ? '212,168,75' :
                      '78,232,138'
                    }, 0.3)`,
                  }}
                  title={`${e.ticker}: ${e.exposure_detail}`}
                >
                  <span className={styles.tileLabel}>{e.ticker}</span>
                  <span className={styles.tileScore}>{e.geo_risk_score}</span>
                </div>
              ))}
            </div>
          )}

          {/* Summary */}
          {scan.summary && (
            <p className={styles.summary}>{scan.summary}</p>
          )}

          {/* Expandable events */}
          {scan.events?.length > 0 && (
            <div>
              <button
                className={styles.expandBtn}
                onClick={() => setExpanded(e => !e)}
              >
                {expanded ? 'Hide' : `Show ${scan.events.length} Events`}
                <svg
                  width="10" height="6" viewBox="0 0 10 6"
                  className={`${styles.chevron} ${expanded ? styles.chevronOpen : ''}`}
                >
                  <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>

              <AnimatePresence>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className={styles.events}
                  >
                    {scan.events.map((ev, i) => (
                      <div key={i} className={styles.event}>
                        <Badge variant={severityVariant(ev.severity)} className={styles.eventBadge}>
                          {ev.severity}
                        </Badge>
                        <div className={styles.eventContent}>
                          <p className={styles.eventHeadline}>{ev.headline}</p>
                          <span className={styles.eventRegion}>{ev.region}</span>
                        </div>
                      </div>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </>
      )}
    </Card>
  )
}
