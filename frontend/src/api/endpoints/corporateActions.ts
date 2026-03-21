import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface CorporateAction {
  ticker: string
  date: string
  action_type: string
  details: string
  value: number | null
}

export interface CorporateActionsResponse {
  actions: CorporateAction[]
  dividend_summary: Record<string, number>
}

export function useCorporateActions(ticker?: string, actionType?: string) {
  const params = new URLSearchParams()
  if (ticker) params.set('ticker', ticker)
  if (actionType) params.set('type', actionType)
  const qs = params.toString()

  return useQuery<CorporateActionsResponse>({
    queryKey: ['corporate-actions', ticker ?? 'all', actionType ?? 'all'],
    queryFn: () => api.get(`/api/corporate-actions${qs ? `?${qs}` : ''}`).then(r => r.data),
    staleTime: 300_000,
  })
}
