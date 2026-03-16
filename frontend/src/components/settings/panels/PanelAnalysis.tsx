import { useState } from 'react'
import { useSaveSettings } from '@/api/endpoints/settings'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useToastStore } from '@/stores/toastStore'
import styles from './Panel.module.css'

export function PanelAnalysis() {
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()
  const [threshold, setThreshold] = useState(65)
  const [riskTolerance, setRiskTolerance] = useState('moderate')
  const [useML, setUseML] = useState(true)

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'analysis',
        confidence_threshold: threshold,
        risk_tolerance: riskTolerance,
        use_ml_labeler: useML,
      })
      addToast('Analysis settings saved', 'success')
    } catch {
      addToast('Failed to save', 'error')
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Analysis</h2>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Signal Parameters</h3>
        <div className={styles.form}>

          <div className={styles.field}>
            <div className={styles.fieldLabel}>Confidence Threshold (%)</div>
            <div className={styles.sliderRow}>
              <input
                type="range" min={40} max={95} step={5}
                value={threshold}
                onChange={e => setThreshold(Number(e.target.value))}
                className={styles.slider}
              />
              <span className={styles.sliderValue}>{threshold}%</span>
            </div>
          </div>

          <div className={styles.field}>
            <div className={styles.fieldLabel}>Risk Tolerance</div>
            <select
              className={styles.input}
              value={riskTolerance}
              onChange={e => setRiskTolerance(e.target.value)}
            >
              <option value="conservative">Conservative</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>

          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Use ML Meta-Labeler</div>
              <div className={styles.toggleSub}>Blend ML confidence with quant score</div>
            </div>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={useML}
                onChange={e => setUseML(e.target.checked)}
              />
              <span className={styles.toggleTrack} />
              <span className={styles.toggleThumb} />
            </label>
          </div>

          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={saveMut.isPending} onClick={handleSave}>
              Save Analysis Settings
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
