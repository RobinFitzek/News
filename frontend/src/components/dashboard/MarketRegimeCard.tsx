import { useMarketRegime } from '@/api/endpoints/macro'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Delta } from '@/components/ui/Delta'
import { Spinner } from '@/components/ui/Spinner'
import styles from './MarketRegimeCard.module.css'

export function MarketRegimeCard() {
  const { data, isLoading, isError } = useMarketRegime()

  const regimeBadge = data?.regime === 'bull' ? 'success' :
                      data?.regime === 'bear' ? 'danger' :
                      data?.regime === 'volatile' ? 'warning' : 'neutral'

  return (
    <Card glow={data?.regime === 'bull' ? 'positive' : data?.regime === 'bear' ? 'negative' : 'neutral'} className={styles.card}>
      <div className={styles.header}>
        <h2 className={styles.title}>Market Regime</h2>
        {data && <Badge variant={regimeBadge}>{data.regime_label}</Badge>}
      </div>

      {isLoading && (
        <div className={styles.loading}><Spinner /></div>
      )}

      {isError && (
        <div className={styles.error}>Failed to load market data</div>
      )}

      {data && (
        <div className={styles.metrics}>
          <MetricRow label="SPY" value={`$${data.spy_price.toFixed(2)}`}>
            <Delta value={`${data.spy_change_pct >= 0 ? '+' : ''}${data.spy_change_pct.toFixed(2)}%`}
                   sign={data.spy_change_pct >= 0 ? 'positive' : 'negative'} />
          </MetricRow>
          <MetricRow label="VIX"  value={data.vix.toFixed(2)} />
          <MetricRow label="10Y"  value={`${data.yield_10y.toFixed(2)}%`} />
          <div className={styles.smaRow}>
            <SMAIndicator label="SMA50"  active={data.sma50_above} />
            <SMAIndicator label="SMA200" active={data.sma200_above} />
          </div>
        </div>
      )}
    </Card>
  )
}

function MetricRow({ label, value, children }: { label: string; value: string; children?: React.ReactNode }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}</span>
      <div className={styles.rowValue}>
        <span className={styles.value}>{value}</span>
        {children}
      </div>
    </div>
  )
}

function SMAIndicator({ label, active }: { label: string; active: boolean }) {
  return (
    <div className={`${styles.sma} ${active ? styles.smaAbove : styles.smaBelow}`}>
      <span className={styles.smaDot} />
      <span className={styles.smaLabel}>{label}</span>
      <span className={styles.smaStatus}>{active ? 'Above' : 'Below'}</span>
    </div>
  )
}

import type React from 'react'
