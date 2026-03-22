import { useQuery, useMutation } from '@tanstack/react-query'
import api from '@/api/client'
import { getCsrfToken } from '@/api/csrf'

// ── Types ─────────────────────────────────────────────────────────────────

export interface SectorImpacts {
  [sector: string]: number
}

export interface Scenario {
  name: string
  description: string
  keywords: string[]
  sector_impacts: SectorImpacts
  historical_analog: string
  severity_threshold: number
}

export interface ScenariosMap {
  [key: string]: Scenario
}

export interface HoldingImpact {
  ticker: string
  sector: string
  weight_pct: number
  sector_impact_pct: number
  estimated_pnl_pct: number
}

export interface ScenarioResult {
  scenario: Scenario
  portfolio_impact_pct: number
  holdings_impact: HoldingImpact[]
  ran_at: string
}

// ── Queries ───────────────────────────────────────────────────────────────

export function useScenarios() {
  return useQuery<ScenariosMap>({
    queryKey: ['scenarios'],
    queryFn: () => api.get('/api/scenarios').then(r => r.data.scenarios),
    staleTime: 300_000,
  })
}

// ── Mutations ─────────────────────────────────────────────────────────────

export function useRunScenario() {
  return useMutation<ScenarioResult, Error, string>({
    mutationFn: async (scenarioKey: string) => {
      const csrf = await getCsrfToken()
      const res = await api.post(`/api/scenarios/run?name=${encodeURIComponent(scenarioKey)}`, {}, {
        headers: { 'X-CSRF-Token': csrf },
      })
      return res.data
    },
  })
}
