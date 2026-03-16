import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { HistoryResponse } from '@/types/api'

export function useHistory(ticker: string | null) {
  return useQuery<HistoryResponse>({
    queryKey: ['history', ticker],
    queryFn: () => {
      const params = new URLSearchParams({ limit: '100' })
      if (ticker) params.set('ticker', ticker)
      return api.get(`/api/history?${params.toString()}`).then(r => r.data)
    },
    staleTime: 30_000,
  })
}
