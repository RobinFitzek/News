import { motion } from 'framer-motion'
import clsx from 'clsx'
import {
  useWeightSuggestions,
  useFeatureImportance,
  useApplyWeights,
} from '@/api/endpoints/learning'
import type { WeightSuggestion, FeatureImportance } from '@/api/endpoints/learning'
import { useSignalAccuracy } from '@/api/endpoints/settings'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { useToastStore } from '@/stores/toastStore'
import styles from './LearningPage.module.css'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtPct(n: number, decimals = 1): string {
  return `${(n * 100).toFixed(decimals)}%`
}

function weightDelta(current: number, suggested: number): string {
  const delta = suggested - current
  if (delta === 0) return '±0'
  return delta > 0 ? `+${(delta * 100).toFixed(1)}` : `${(delta * 100).toFixed(1)}`
}

function deltaClass(current: number, suggested: number): string {
  const delta = suggested - current
  if (delta > 0) return styles.deltaPositive
  if (delta < 0) return styles.deltaNegative
  return styles.deltaNeutral
}

// ── Sub-components ────────────────────────────────────────────────────────────

function WeightSuggestionRow({ suggestion }: { suggestion: WeightSuggestion }) {
  return (
    <div className={styles.suggestionRow}>
      <div className={styles.suggestionTop}>
        <span className={styles.featureName}>{suggestion.feature}</span>
        <div className={styles.weightFlow}>
          <span className={styles.currentWeight}>
            {(suggestion.current_weight * 100).toFixed(1)}
          </span>
          <span className={styles.arrow}>→</span>
          <span className={styles.suggestedWeight}>
            {(suggestion.suggested_weight * 100).toFixed(1)}
          </span>
          <span className={deltaClass(suggestion.current_weight, suggestion.suggested_weight)}>
            ({weightDelta(suggestion.current_weight, suggestion.suggested_weight)})
          </span>
        </div>
        <span className={styles.confidence}>
          {Math.round(suggestion.confidence * 100)}% conf.
        </span>
      </div>
      {suggestion.reason && (
        <p className={styles.reason}>{suggestion.reason}</p>
      )}
    </div>
  )
}

function FeatureImportanceRow({
  item,
  maxImportance,
  index,
}: {
  item: FeatureImportance
  maxImportance: number
  index: number
}) {
  const widthPct = maxImportance > 0 ? (item.importance / maxImportance) * 100 : 0

  return (
    <motion.div
      className={styles.barRow}
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.04, duration: 0.3 }}
    >
      <div className={styles.barMeta}>
        <span className={styles.barLabel}>{item.feature}</span>
        <Badge variant={item.direction === 'positive' ? 'success' : 'danger'} size="xs">
          {item.direction}
        </Badge>
        <span className={styles.barPct}>{(item.importance * 100).toFixed(1)}%</span>
      </div>
      <div className={styles.barTrack}>
        <div
          className={clsx(styles.barFill, styles[item.direction])}
          style={{ width: `${widthPct}%` }}
        />
      </div>
    </motion.div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export function LearningPage() {
  const { data: suggestions, isLoading: loadingSuggestions } = useWeightSuggestions()
  const { data: importance, isLoading: loadingImportance } = useFeatureImportance()
  const { data: accuracy, isLoading: loadingAccuracy } = useSignalAccuracy()
  const applyMut = useApplyWeights()
  const { addToast } = useToastStore()

  const maxImportance =
    importance && importance.length > 0
      ? Math.max(...importance.map(f => f.importance))
      : 1

  async function handleApplyWeights() {
    if (!suggestions || suggestions.length === 0) return
    const weights: Record<string, number> = {}
    for (const s of suggestions) {
      weights[s.feature] = s.suggested_weight
    }
    try {
      await applyMut.mutateAsync({ weights })
      addToast('Suggested weights applied', 'success')
    } catch {
      addToast('Failed to apply weights', 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="Learning"
        subtitle="AI weight optimization and feature analysis"
      />

      {/* Two-column: Weight Suggestions + Feature Importance */}
      <div className={styles.twoCol}>
        {/* Weight Suggestions */}
        <Card animate={false}>
          <div className={styles.cardInner}>
            <p className={styles.cardTitle}>Weight Suggestions</p>

            {loadingSuggestions ? (
              <div className={styles.loading}><Spinner size="md" /></div>
            ) : !suggestions || suggestions.length === 0 ? (
              <p className={styles.muted}>No suggestions available.</p>
            ) : (
              <>
                <div className={styles.suggestionList}>
                  {suggestions.map((s, i) => (
                    <motion.div
                      key={s.feature}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.04, duration: 0.3 }}
                    >
                      <WeightSuggestionRow suggestion={s} />
                    </motion.div>
                  ))}
                </div>
                <div className={styles.applyBtn}>
                  <Button
                    variant="primary"
                    size="md"
                    loading={applyMut.isPending}
                    onClick={handleApplyWeights}
                  >
                    Apply Suggested Weights
                  </Button>
                </div>
              </>
            )}
          </div>
        </Card>

        {/* Feature Importance */}
        <Card animate={false}>
          <div className={styles.cardInner}>
            <p className={styles.cardTitle}>Feature Importance</p>

            {loadingImportance ? (
              <div className={styles.loading}><Spinner size="md" /></div>
            ) : !importance || importance.length === 0 ? (
              <p className={styles.muted}>No feature data available.</p>
            ) : (
              <div className={styles.barChart}>
                {importance.map((item, i) => (
                  <FeatureImportanceRow
                    key={item.feature}
                    item={item}
                    maxImportance={maxImportance}
                    index={i}
                  />
                ))}
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Signal Accuracy */}
      <Card className={styles.accuracyCard}>
        <div className={styles.cardInner}>
          <p className={styles.cardTitle}>Signal Accuracy</p>

          {loadingAccuracy ? (
            <div className={styles.loading}><Spinner size="md" /></div>
          ) : !accuracy ? (
            <p className={styles.muted}>No accuracy data available.</p>
          ) : (
            <>
              <div className={styles.accuracyHeader}>
                <div className={styles.overallAccuracy}>
                  <span className={styles.overallValue}>
                    {fmtPct(accuracy.overall_accuracy)}
                  </span>
                  <span className={styles.overallLabel}>Overall Accuracy</span>
                </div>
                <div className={styles.accuracyMeta}>
                  <span className={styles.accuracyMetaItem}>
                    {accuracy.verified_predictions} verified predictions
                  </span>
                  <span className={styles.accuracyMetaItem}>
                    {accuracy.pending_predictions} pending
                  </span>
                  {accuracy.kill_switch_active && (
                    <Badge variant="warning" size="xs">Kill Switch Active</Badge>
                  )}
                </div>
              </div>

              <div className={styles.accuracyGrid}>
                <div className={styles.accuracyItem}>
                  <span className={styles.accuracyItemLabel}>
                    <Badge variant="success" size="xs">BUY</Badge>
                    Accuracy
                  </span>
                  <span className={styles.accuracyItemValue}>
                    {fmtPct(accuracy.by_signal.buy.accuracy)}
                  </span>
                  <span className={styles.accuracyCount}>
                    {accuracy.by_signal.buy.count} signals
                  </span>
                </div>
                <div className={styles.accuracyItem}>
                  <span className={styles.accuracyItemLabel}>
                    <Badge variant="danger" size="xs">SELL</Badge>
                    Accuracy
                  </span>
                  <span className={styles.accuracyItemValue}>
                    {fmtPct(accuracy.by_signal.sell.accuracy)}
                  </span>
                  <span className={styles.accuracyCount}>
                    {accuracy.by_signal.sell.count} signals
                  </span>
                </div>
                <div className={styles.accuracyItem}>
                  <span className={styles.accuracyItemLabel}>
                    <Badge variant="neutral" size="xs">HOLD</Badge>
                    Accuracy
                  </span>
                  <span className={styles.accuracyItemValue}>
                    {fmtPct(accuracy.by_signal.hold.accuracy)}
                  </span>
                  <span className={styles.accuracyCount}>
                    {accuracy.by_signal.hold.count} signals
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      </Card>

      <div style={{ height: 'var(--space-16)' }} />
    </>
  )
}
