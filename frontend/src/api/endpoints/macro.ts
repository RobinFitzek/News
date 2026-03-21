import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { MacroEvent, MarketRegime } from '@/types/api'

export function useMacroEvents(days = 14) {
  return useQuery<{ events: MacroEvent[] }>({
    queryKey: ['macro-events', days],
    queryFn: () => api.get(`/api/macro/events?days=${days}`).then(r => r.data),
    staleTime: 300_000,
  })
}

export function useMarketRegime() {
  return useQuery<MarketRegime>({
    queryKey: ['market-regime'],
    queryFn: () => api.get('/api/market-regime').then(r => r.data),
    staleTime: 300_000,
  })
}

// ── Macro Snapshot ─────────────────────────────────────────────────────────

export interface MacroSnapshotEntry {
  date: string
  yield_2y: number | null
  yield_10y: number | null
  spread_2_10: number | null
  vix: number | null
  dxy: number | null
  credit_spread: number | null
  regime: string | null
}

export interface MacroSnapshot {
  latest: MacroSnapshotEntry | null
  history: MacroSnapshotEntry[]
  events: MacroEvent[]
}

export function useMacroSnapshot() {
  return useQuery<MacroSnapshot>({
    queryKey: ['macro-snapshot'],
    queryFn: () => api.get('/api/macro/snapshot').then(r => r.data),
    staleTime: 300_000,
  })
}
