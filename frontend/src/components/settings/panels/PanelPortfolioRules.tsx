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

function NumField({ label, sub, value, onChange, min, max, step, unit }: {
  label: string; sub?: string; value: number; onChange: (v: number) => void
  min: number; max: number; step?: number; unit?: string
}) {
  return (
    <div className={styles.field}>
      <div className={styles.fieldLabel}>{label}{sub && <span style={{ color: 'var(--text-muted)', marginLeft: 6, textTransform: 'none', letterSpacing: 0 }}>{sub}</span>}</div>
      <div className={styles.sliderRow}>
        <input type="range" min={min} max={max} step={step ?? 1} value={value} onChange={e => onChange(Number(e.target.value))} className={styles.slider} />
        <span className={styles.sliderValue}>{value}{unit ?? ''}</span>
      </div>
    </div>
  )
}

export function PanelPortfolioRules() {
  const { data: settings } = useSettingsData()
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()

  const [maxPositionPct, setMaxPositionPct] = useState(10)
  const [stopLossPct, setStopLossPct] = useState(15)
  const [maxSectorPct, setMaxSectorPct] = useState(30)
  const [rebalanceDrift, setRebalanceDrift] = useState(5)
  const [riskGuard, setRiskGuard] = useState(true)
  const [globalLossLimit, setGlobalLossLimit] = useState(10)
  const [riskCooldown, setRiskCooldown] = useState(24)

  useEffect(() => {
    if (!settings) return
    const s = settings as Record<string, unknown>
    setMaxPositionPct(Number(s.portfolio_max_position_pct ?? 10))
    setStopLossPct(Number(s.portfolio_stop_loss_pct ?? 15))
    setMaxSectorPct(Number(s.portfolio_max_sector_pct ?? 30))
    setRebalanceDrift(Number(s.portfolio_rebalance_drift_pct ?? 5))
    setRiskGuard(s.portfolio_risk_guard_enabled !== false)
    setGlobalLossLimit(Number(s.portfolio_global_loss_limit_pct ?? 10))
    setRiskCooldown(Number(s.portfolio_risk_cooldown_hours ?? 24))
  }, [settings])

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'portfolio_rules',
        portfolio_max_position_pct: maxPositionPct,
        portfolio_stop_loss_pct: stopLossPct,
        portfolio_max_sector_pct: maxSectorPct,
        portfolio_rebalance_drift_pct: rebalanceDrift,
        portfolio_risk_guard_enabled: riskGuard,
        portfolio_global_loss_limit_pct: globalLossLimit,
        portfolio_risk_cooldown_hours: riskCooldown,
      })
      addToast('Portfolio rules saved', 'success')
    } catch { addToast('Failed to save', 'error') }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Portfolio Rules</h2>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Position Limits</h3>
        <div className={styles.form}>
          <NumField label="Max position size" value={maxPositionPct} onChange={setMaxPositionPct} min={1} max={100} unit="%" />
          <NumField label="Stop-loss per position" value={stopLossPct} onChange={setStopLossPct} min={1} max={50} unit="%" />
          <NumField label="Max sector concentration" value={maxSectorPct} onChange={setMaxSectorPct} min={5} max={100} step={5} unit="%" />
          <NumField label="Rebalance drift threshold" value={rebalanceDrift} onChange={setRebalanceDrift} min={1} max={25} unit="%" />
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Risk Guard</h3>
        <div className={styles.form}>
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Risk guard enabled</div>
              <div className={styles.toggleSub}>Pause auto-trading when portfolio drawdown exceeds limit</div>
            </div>
            <Toggle checked={riskGuard} onChange={setRiskGuard} />
          </div>
          {riskGuard && (
            <>
              <NumField label="Global loss limit" value={globalLossLimit} onChange={setGlobalLossLimit} min={1} max={50} unit="%" />
              <NumField label="Cooldown after trigger" value={riskCooldown} onChange={setRiskCooldown} min={0} max={168} unit="h" />
            </>
          )}
          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={saveMut.isPending} onClick={handleSave}>Save Portfolio Rules</Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
