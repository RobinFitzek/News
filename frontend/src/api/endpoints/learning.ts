import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'

// ── Types ────────────────────────────────────────────────────────────────────

export interface WeightSuggestion {
  feature: string
  current_weight: number
  suggested_weight: number
  confidence: number
  reason: string
}

export interface FeatureImportance {
  feature: string
  importance: number
  direction: 'positive' | 'negative'
}

export interface ApplyWeightsPayload {
  weights: Record<string, number>
}

// ── Hooks ─────────────────────────────────────────────────────────────────────

export function useWeightSuggestions() {
  return useQuery<WeightSuggestion[]>({
    queryKey: ['learning-weight-suggestions'],
    queryFn: () => api.get('/api/learning/weight-suggestions').then(r => r.data),
    staleTime: 300_000,
  })
}

export function useFeatureImportance() {
  return useQuery<FeatureImportance[]>({
    queryKey: ['learning-feature-importance'],
    queryFn: () => api.get('/api/learning/feature-importance').then(r => r.data),
    staleTime: 300_000,
  })
}

export function useApplyWeights() {
  return useMutation({
    mutationFn: (payload: ApplyWeightsPayload) =>
      api.post('/api/learning/apply-weights', payload).then(r => r.data),
  })
}
