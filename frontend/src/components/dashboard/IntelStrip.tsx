import { Link } from 'react-router-dom'
import { useDiscoveryStats } from '@/api/endpoints/discovery'
import { useSignalAccuracy } from '@/api/endpoints/settings'
import { Button } from '@/components/ui/Button'
import styles from './IntelStrip.module.css'

export function IntelStrip() {
  const { data: discovery } = useDiscoveryStats()
  const { data: accuracy } = useSignalAccuracy()

  return (
    <div className={styles.strip}>
      <IntelMetric
        value={accuracy ? `${accuracy.overall_accuracy.toFixed(0)}%` : '—'}
        label="AI Hit Rate"
        highlight={accuracy && accuracy.overall_accuracy >= 60 ? 'positive' : undefined}
      />
      <div className={styles.sep} />
      <IntelMetric
        value={String(accuracy?.verified_predictions ?? '—')}
        label="Verified"
      />
      <div className={styles.sep} />
      <IntelMetric
        value={String(discovery?.discovered_7d ?? '—')}
        label="Discovered 7d"
      />
      <div className={styles.sep} />
      <IntelMetric
        value={String(discovery?.promoted_7d ?? '—')}
        label="Promoted 7d"
      />
      <div className={styles.sep} />
      <IntelMetric
        value={String(accuracy?.pending_predictions ?? '—')}
        label="Pending"
      />

      <div className={styles.actions}>
        <Link to="/discoveries">
          <Button variant="secondary" size="sm">Discoveries</Button>
        </Link>
        <Link to="/learning">
          <Button variant="secondary" size="sm">Learning</Button>
        </Link>
      </div>
    </div>
  )
}

function IntelMetric({
  value,
  label,
  highlight,
}: {
  value: string
  label: string
  highlight?: 'positive' | 'negative'
}) {
  return (
    <div className={styles.metric}>
      <div
        className={`${styles.metricValue} ${
          highlight === 'positive' ? styles.positive :
          highlight === 'negative' ? styles.negative : ''
        }`}
      >
        {value}
      </div>
      <div className={styles.metricLabel}>{label}</div>
    </div>
  )
}
