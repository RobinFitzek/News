import { useEffect, useState } from 'react'
import { useSettingsData, useSaveSettings } from '@/api/endpoints/settings'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className={styles.toggle}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
      <span className={styles.toggleTrack} />
      <span className={styles.toggleThumb} />
    </label>
  )
}

const STRATEGIES = [
  { key: 'volume_spike',      label: 'Volume spike',      sub: 'Unusual volume vs 20-day average' },
  { key: 'breakout',          label: 'Breakout',          sub: '52-week highs, golden cross signals' },
  { key: 'oversold',          label: 'Oversold quality',  sub: 'Strong fundamentals with RSI < 30' },
  { key: 'sector_rotation',   label: 'Sector rotation',   sub: 'Hot sector momentum plays' },
  { key: 'insider_buy',       label: 'Insider buying',    sub: 'Cluster insider purchase activity' },
  { key: 'value_screen',      label: 'Value screen',      sub: 'Low P/E, high quality metrics' },
]

const DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

export function PanelAutoDiscovery() {
  const { data: settings } = useSettingsData()
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()

  const [enabled, setEnabled] = useState(true)
  const [dailyTime, setDailyTime] = useState('06:00')
  const [weeklyDay, setWeeklyDay] = useState('wed')
  const [weeklyTime, setWeeklyTime] = useState('12:00')
  const [promotionThreshold, setPromotionThreshold] = useState(55)
  const [maxPromotePerRun, setMaxPromotePerRun] = useState(5)
  const [maxWatchlistSize, setMaxWatchlistSize] = useState(50)
  const [activeStrategies, setActiveStrategies] = useState<string[]>(STRATEGIES.map(s => s.key))

  useEffect(() => {
    if (!settings) return
    const s = settings as Record<string, unknown>
    setEnabled(s.discovery_enabled !== false)
    setDailyTime(String(s.discovery_daily_time ?? '06:00'))
    setWeeklyDay(String(s.discovery_weekly_day ?? 'wed'))
    setWeeklyTime(String(s.discovery_weekly_time ?? '12:00'))
    setPromotionThreshold(Number(s.discovery_promotion_threshold ?? 55))
    setMaxPromotePerRun(Number(s.discovery_max_promote_per_run ?? 5))
    setMaxWatchlistSize(Number(s.discovery_max_watchlist_size ?? 50))
    const strats = s.discovery_strategies
    if (Array.isArray(strats)) setActiveStrategies(strats as string[])
  }, [settings])

  function toggleStrategy(key: string) {
    setActiveStrategies(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    )
  }

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'auto_discovery',
        discovery_enabled: enabled,
        discovery_daily_time: dailyTime,
        discovery_weekly_day: weeklyDay,
        discovery_weekly_time: weeklyTime,
        discovery_promotion_threshold: promotionThreshold,
        discovery_max_promote_per_run: maxPromotePerRun,
        discovery_max_watchlist_size: maxWatchlistSize,
        discovery_strategies: activeStrategies,
      })
      addToast('Discovery settings saved', 'success')
    } catch { addToast('Failed to save', 'error') }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Auto-Discovery</h2>

      <Card className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Discovery Engine</h3>
          <Toggle checked={enabled} onChange={setEnabled} />
        </div>
        <p className={styles.sectionSub}>
          Automatically scans markets for new opportunities and promotes high-scoring tickers to your watchlist.
        </p>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Schedule</h3>
        <div className={styles.form}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-3)' }}>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Daily scan time</div>
              <input className={styles.input} type="time" value={dailyTime} onChange={e => setDailyTime(e.target.value)} />
            </div>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Weekly scan time</div>
              <input className={styles.input} type="time" value={weeklyTime} onChange={e => setWeeklyTime(e.target.value)} />
            </div>
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Weekly scan day</div>
            <select className={styles.input} value={weeklyDay} onChange={e => setWeeklyDay(e.target.value)}>
              {DAYS.map(d => <option key={d} value={d}>{d.charAt(0).toUpperCase() + d.slice(1)}</option>)}
            </select>
          </div>
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Promotion Rules</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Quant score threshold</div>
            <div className={styles.sliderRow}>
              <input type="range" min={30} max={90} step={5} value={promotionThreshold} onChange={e => setPromotionThreshold(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{promotionThreshold}</span>
            </div>
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Max promotions per run</div>
            <div className={styles.sliderRow}>
              <input type="range" min={1} max={20} value={maxPromotePerRun} onChange={e => setMaxPromotePerRun(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{maxPromotePerRun}</span>
            </div>
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Max watchlist size (auto-stop)</div>
            <div className={styles.sliderRow}>
              <input type="range" min={10} max={200} step={5} value={maxWatchlistSize} onChange={e => setMaxWatchlistSize(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{maxWatchlistSize}</span>
            </div>
          </div>
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Active Strategies</h3>
        <div className={styles.form}>
          {STRATEGIES.map(s => (
            <div key={s.key} className={styles.toggleRow}>
              <div>
                <div className={styles.toggleLabel}>{s.label}</div>
                <div className={styles.toggleSub}>{s.sub}</div>
              </div>
              <Toggle checked={activeStrategies.includes(s.key)} onChange={() => toggleStrategy(s.key)} />
            </div>
          ))}
          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={saveMut.isPending} onClick={handleSave}>Save Discovery Settings</Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
