import { useBenchmark } from '@/api/endpoints/portfolio'
import { Card } from '@/components/ui/Card'
import { Delta } from '@/components/ui/Delta'
import { Spinner } from '@/components/ui/Spinner'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
} from 'chart.js'
import styles from './BenchmarkCard.module.css'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip)

export function BenchmarkCard() {
  const { data, isLoading, isError } = useBenchmark()

  return (
    <Card glow={data && data.alpha >= 0 ? 'positive' : 'negative'} className={styles.card}>
      <div className={styles.header}>
        <h2 className={styles.title}>Portfolio vs SPY</h2>
        {data && (
          <div className={styles.alphaBadge}>
            <span className={styles.alphaLabel}>Alpha</span>
            <Delta
              value={`${data.alpha >= 0 ? '+' : ''}${data.alpha.toFixed(2)}%`}
              sign={data.alpha >= 0 ? 'positive' : 'negative'}
            />
          </div>
        )}
      </div>

      {isLoading && <div className={styles.loading}><Spinner /></div>}
      {isError  && <div className={styles.error}>Failed to load benchmark data</div>}

      {data && (
        <>
          <div className={styles.metricsRow}>
            <BenchMetric
              label="Portfolio"
              value={`${data.portfolio_return >= 0 ? '+' : ''}${data.portfolio_return.toFixed(2)}%`}
              sign={data.portfolio_return >= 0 ? 'positive' : 'negative'}
            />
            <div className={styles.sep} />
            <BenchMetric
              label="SPY"
              value={`${data.spy_return >= 0 ? '+' : ''}${data.spy_return.toFixed(2)}%`}
              sign={data.spy_return >= 0 ? 'positive' : 'negative'}
            />
          </div>

          {data.labels?.length > 0 && (
            <div className={styles.chart}>
              <Line
                data={{
                  labels: data.labels,
                  datasets: [
                    {
                      label: 'Portfolio',
                      data: data.portfolio_series,
                      borderColor: 'var(--signal-positive)',
                      borderWidth: 1.5,
                      pointRadius: 0,
                      tension: 0.3,
                    },
                    {
                      label: 'SPY',
                      data: data.spy_series,
                      borderColor: 'var(--text-muted)',
                      borderWidth: 1,
                      borderDash: [4, 4],
                      pointRadius: 0,
                      tension: 0.3,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  animation: false,
                  plugins: { legend: { display: false }, tooltip: { enabled: false } },
                  scales: {
                    x: { display: false },
                    y: {
                      display: false,
                      ticks: { color: 'rgba(0,0,0,0)' },
                    },
                  },
                }}
              />
            </div>
          )}
        </>
      )}
    </Card>
  )
}

function BenchMetric({
  label,
  value,
  sign,
}: { label: string; value: string; sign: 'positive' | 'negative' }) {
  return (
    <div className={styles.benchMetric}>
      <span className={styles.benchLabel}>{label}</span>
      <Delta value={value} sign={sign} showArrow={false} className={styles.benchValue} />
    </div>
  )
}
