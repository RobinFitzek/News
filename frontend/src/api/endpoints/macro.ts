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
