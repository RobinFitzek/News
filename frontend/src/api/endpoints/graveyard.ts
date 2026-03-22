import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface GraveyardEntry {
  ticker: string
  reason: string
  removed_at: string
  removal_price: number
  current_price: number
  change_pct: number
}

export function useGraveyardPerformance() {
  return useQuery<{ results: GraveyardEntry[] }>({
    queryKey: ['graveyard-performance'],
    queryFn: () => api.get('/api/graveyard/performance').then(r => r.data),
    staleTime: 300_000,
  })
}
