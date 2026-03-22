import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface SectorEntry {
  etf: string
  name: string
  color: string
  return_1wk: number
  return_1mo: number
  return_3mo: number
  relative_strength: number
  momentum: 'hot' | 'cold' | 'neutral'
  rank: number
}

export interface SectorHeatMap {
  sectors: SectorEntry[]
  spy_return: number
  best_sector: SectorEntry | null
  worst_sector: SectorEntry | null
  updated: string
}

export function useSectorMomentum() {
  return useQuery<SectorHeatMap>({
    queryKey: ['sector-momentum'],
    queryFn: () => api.get('/api/sector-momentum').then(r => r.data),
    staleTime: 300_000,
  })
}
