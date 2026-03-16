import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { PaperTradingSummary } from '@/types/api'

export function usePaperTrading() {
  return useQuery<PaperTradingSummary>({
    queryKey: ['paper-trading'],
    queryFn: () => api.get('/api/paper-trading').then(r => r.data),
    staleTime: 30_000,
  })
}
