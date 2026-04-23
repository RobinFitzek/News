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

export function PanelAnalysis() {
  const { data: settings } = useSettingsData()
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()

  const [threshold, setThreshold] = useState(65)
  const [riskTolerance, setRiskTolerance] = useState('moderate')
  const [useML, setUseML] = useState(true)
  const [includeNews, setIncludeNews] = useState(true)
  const [includeFundamental, setIncludeFundamental] = useState(true)
  const [includeTechnical, setIncludeTechnical] = useState(true)
  const [verificationDays, setVerificationDays] = useState(30)
  const [perplexityBudget, setPerplexityBudget] = useState(20)
  const [geminiBudget, setGeminiBudget] = useState(10)

  useEffect(() => {
    if (!settings) return
    const s = settings as Record<string, unknown>
    setThreshold(Number(s.confidence_threshold ?? 65))
    setRiskTolerance(String(s.risk_tolerance ?? 'moderate'))
    setUseML(s.use_ml_labeler !== false)
    setIncludeNews(s.include_news !== false)
    setIncludeFundamental(s.include_fundamental !== false)
    setIncludeTechnical(s.include_technical !== false)
    setVerificationDays(Number(s.learning_verification_days ?? 30))
    setPerplexityBudget(Number(s.perplexity_monthly_budget ?? 20))
    setGeminiBudget(Number(s.gemini_monthly_budget ?? 10))
  }, [settings])

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'analysis',
        confidence_threshold: threshold,
        risk_tolerance: riskTolerance,
        use_ml_labeler: useML,
        include_news: includeNews,
        include_fundamental: includeFundamental,
        include_technical: includeTechnical,
        learning_verification_days: verificationDays,
        perplexity_monthly_budget: perplexityBudget,
        gemini_monthly_budget: geminiBudget,
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
            <div className={styles.fieldLabel}>Confidence Threshold</div>
            <div className={styles.sliderRow}>
              <input type="range" min={40} max={95} step={5} value={threshold} onChange={e => setThreshold(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{threshold}%</span>
            </div>
          </div>

          <div className={styles.field}>
            <div className={styles.fieldLabel}>Risk Tolerance</div>
            <select className={styles.input} value={riskTolerance} onChange={e => setRiskTolerance(e.target.value)}>
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
            <Toggle checked={useML} onChange={setUseML} />
          </div>
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Data Sources</h3>
        <div className={styles.form}>
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>News analysis</div>
              <div className={styles.toggleSub}>Include news sentiment in AI analysis</div>
            </div>
            <Toggle checked={includeNews} onChange={setIncludeNews} />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Fundamental data</div>
              <div className={styles.toggleSub}>P/E, EPS, revenue, balance sheet</div>
            </div>
            <Toggle checked={includeFundamental} onChange={setIncludeFundamental} />
          </div>
          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Technical indicators</div>
              <div className={styles.toggleSub}>RSI, MACD, moving averages, volume</div>
            </div>
            <Toggle checked={includeTechnical} onChange={setIncludeTechnical} />
          </div>
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Learning & Budget</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Verification period</div>
            <div className={styles.sliderRow}>
              <input type="range" min={7} max={90} step={1} value={verificationDays} onChange={e => setVerificationDays(Number(e.target.value))} className={styles.slider} />
              <span className={styles.sliderValue}>{verificationDays}d</span>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-3)' }}>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Perplexity budget / mo</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <input className={styles.input} type="number" min={0} max={500} value={perplexityBudget} onChange={e => setPerplexityBudget(Number(e.target.value))} style={{ width: '100%' }} />
                <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', whiteSpace: 'nowrap' }}>EUR</span>
              </div>
            </div>
            <div className={styles.field}>
              <div className={styles.fieldLabel}>Gemini budget / mo</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <input className={styles.input} type="number" min={0} max={500} value={geminiBudget} onChange={e => setGeminiBudget(Number(e.target.value))} style={{ width: '100%' }} />
                <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', whiteSpace: 'nowrap' }}>EUR</span>
              </div>
            </div>
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
