import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'

export interface InsiderSignal {
  ticker: string
  insider_name: string
  role: string
  transaction_type: 'BUY' | 'SELL'
  shares: number
  value: number
  date: string
  significance_score: number
}

export interface InsiderActivityResponse {
  signals: InsiderSignal[]
}

export function useInsiderActivity() {
  return useQuery<InsiderActivityResponse>({
    queryKey: ['insider-activity'],
    queryFn: () => api.get('/api/insider-activity').then(r => r.data),
    staleTime: 300_000,
  })
}

export function useScanInsiderActivity() {
  return useMutation({
    mutationFn: () => api.post('/api/insider-activity/scan').then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['insider-activity'] }),
  })
}
