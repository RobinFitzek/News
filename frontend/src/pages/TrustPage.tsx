import { useQuery } from '@tanstack/react-query'
import api from '@/api/client'
import { useSignalAccuracy } from '@/api/endpoints/settings'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import type { SignalAccuracy } from '@/types/api'
import styles from './TrustPage.module.css'

interface TrustGate {
  gate_open: boolean
  reasons: string[]
  overall_score: number
  checks: {
    accuracy_ok: boolean
    kill_switch_off: boolean
    market_regime_ok: boolean
    risk_gate_ok: boolean
  }
}

interface RiskGateStatus {
  status: string
  message?: string
}

const CHECK_LABELS: Record<keyof TrustGate['checks'], string> = {
  accuracy_ok: 'Accuracy Check',
  kill_switch_off: 'Kill Switch',
  market_regime_ok: 'Market Regime',
  risk_gate_ok: 'Risk Gate',
}

export function TrustPage() {
  const { data: trustGate, isLoading: loadingTrust } = useQuery<TrustGate>({
    queryKey: ['trust-gate'],
    queryFn: () => api.get('/api/auto-trade/trust-gate').then(r => r.data),
  })

  const { data: _riskGate } = useQuery<RiskGateStatus>({
    queryKey: ['risk-gate-status'],
    queryFn: () => api.get('/api/auto-trade/risk-gate-status').then(r => r.data),
  })

  const { data: accuracy } = useSignalAccuracy()

  if (loadingTrust) {
    return (
      <>
        <PageHeader
          title="Trust Gate"
          subtitle="AI trading confidence and safety checks"
        />
        <div className={styles.loading}><Spinner size="lg" /></div>
      </>
    )
  }

  return (
    <>
      <PageHeader
        title="Trust Gate"
        subtitle="AI trading confidence and safety checks"
      />

      {/* Gate status */}
      <Card
        className={`${styles.statusCard} ${trustGate?.gate_open ? styles.statusCardOpen : styles.statusCardClosed}`}
        glow={trustGate?.gate_open ? 'positive' : 'negative'}
      >
        <div className={styles.statusHeader}>
          <div>
            <div className={`${styles.statusLabel} ${trustGate?.gate_open ? styles.open : styles.closed}`}>
              {trustGate?.gate_open ? 'GATE OPEN' : 'GATE CLOSED'}
            </div>
            <div style={{ marginTop: 'var(--space-2)' }}>
              <Badge variant={trustGate?.gate_open ? 'success' : 'danger'} size="sm">
                {trustGate?.gate_open ? 'Trading Enabled' : 'Trading Disabled'}
              </Badge>
            </div>
          </div>
          <div className={styles.scoreDisplay}>
            {trustGate?.overall_score ?? 0}
            <span className={styles.scoreSuffix}>/100</span>
          </div>
        </div>
      </Card>

      {/* Checks grid */}
      {trustGate?.checks && (
        <div className={styles.checksGrid}>
          {(Object.keys(trustGate.checks) as Array<keyof TrustGate['checks']>).map((key, i) => {
            const pass = trustGate.checks[key]
            return (
              <Card key={key} className={styles.checkCard} delay={i * 0.05}>
                <div className={`${styles.checkIcon} ${pass ? styles.pass : styles.fail}`}>
                  {pass ? '✓' : '✗'}
                </div>
                <div className={styles.checkName}>{CHECK_LABELS[key]}</div>
                <Badge variant={pass ? 'success' : 'danger'} size="xs">
                  {pass ? 'OK' : 'FAIL'}
                </Badge>
              </Card>
            )
          })}
        </div>
      )}

      {/* Reasons */}
      {(trustGate?.reasons ?? []).length > 0 && (
        <div className={styles.reasonsList}>
          <div className={styles.reasonsTitle}>Gate Reasons</div>
          {(trustGate?.reasons ?? []).map((reason, i) => (
            <Card key={i} className={styles.reasonRow} delay={i * 0.03}>
              <span className={styles.reasonText}>{reason}</span>
            </Card>
          ))}
        </div>
      )}

      {/* Signal accuracy */}
      {accuracy && (
        <Card className={styles.accuracyCard}>
          <div className={styles.accuracyTitle}>Signal Accuracy</div>
          <div className={styles.overallAccuracy}>
            {accuracy.overall_accuracy.toFixed(1)}%
          </div>
          <SignalBreakdown accuracy={accuracy} />
        </Card>
      )}

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}

function SignalBreakdown({ accuracy }: { accuracy: SignalAccuracy }) {
  const signals = [
    { label: 'BUY', data: accuracy.by_signal.buy },
    { label: 'SELL', data: accuracy.by_signal.sell },
    { label: 'HOLD', data: accuracy.by_signal.hold },
  ]
  return (
    <div className={styles.signalBreakdown}>
      {signals.map(({ label, data }) => (
        <div key={label} className={styles.signalItem}>
          <div className={styles.signalName}>{label}</div>
          <div className={styles.signalAccuracy}>{data.accuracy.toFixed(1)}%</div>
          <div className={styles.signalCount}>{data.count} signals</div>
        </div>
      ))}
    </div>
  )
}
