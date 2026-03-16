import { useBudget } from '@/api/endpoints/budget'
import { useSaveSettings } from '@/api/endpoints/settings'
import { useState } from 'react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ProgressBar } from '@/components/ui/ProgressBar'
import { useToastStore } from '@/stores/toastStore'
import { Spinner } from '@/components/ui/Spinner'
import styles from './Panel.module.css'

export function PanelBudget() {
  const { data: budget, isLoading } = useBudget()
  const saveMut = useSaveSettings()
  const { addToast } = useToastStore()
  const [perplexityLimit, setPerplexityLimit] = useState(5)
  const [geminiLimit, setGeminiLimit] = useState(5)

  async function handleSave() {
    try {
      await saveMut.mutateAsync({
        section: 'budget',
        perplexity_monthly_eur: perplexityLimit,
        gemini_monthly_eur: geminiLimit,
      })
      addToast('Budget settings saved', 'success')
    } catch {
      addToast('Failed to save', 'error')
    }
  }

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Budget</h2>

      {isLoading && <div className={styles.loading}><Spinner /></div>}

      {budget && (
        <Card className={styles.section}>
          <h3 className={styles.sectionTitle}>Current Usage</h3>
          <div className={styles.list}>
            <BudgetRow
              label="Perplexity"
              used={budget.perplexity.used_eur}
              limit={budget.perplexity.limit_eur}
              pct={budget.perplexity.pct_used}
            />
            <BudgetRow
              label="Gemini"
              used={budget.gemini.used_eur}
              limit={budget.gemini.limit_eur}
              pct={budget.gemini.pct_used}
            />
          </div>
          <div style={{ marginTop: 'var(--space-3)', color: 'var(--text-muted)', fontSize: 'var(--text-xs)', fontFamily: 'var(--font-mono)' }}>
            Avg cost/analysis: ${budget.avg_cost_per_analysis_usd?.toFixed(4)} · 7d cost: ${budget.cost_7d?.toFixed(2)}
          </div>
        </Card>
      )}

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Monthly Limits (EUR)</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Perplexity Budget (€)</div>
            <input
              className={styles.input}
              type="number"
              min={0}
              step={0.5}
              value={perplexityLimit}
              onChange={e => setPerplexityLimit(Number(e.target.value))}
            />
          </div>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Gemini Budget (€)</div>
            <input
              className={styles.input}
              type="number"
              min={0}
              step={0.5}
              value={geminiLimit}
              onChange={e => setGeminiLimit(Number(e.target.value))}
            />
          </div>
          <div className={styles.saveRow}>
            <Button variant="primary" size="md" loading={saveMut.isPending} onClick={handleSave}>
              Save Budget
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}

function BudgetRow({ label, used, limit, pct }: { label: string; used: number; limit: number; pct: number }) {
  const variant = pct >= 90 ? 'danger' : pct >= 70 ? 'warning' : 'success'
  return (
    <div style={{ marginBottom: 'var(--space-4)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
          €{used.toFixed(2)} / €{limit.toFixed(2)}
        </span>
      </div>
      <ProgressBar value={pct} variant={variant} height={4} />
    </div>
  )
}
