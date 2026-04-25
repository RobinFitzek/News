import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'

// ── Types ────────────────────────────────────────────────────────────────────

export interface PortfolioHolding {
  ticker: string
  shares: number
  avg_cost: number
  current_price: number
  market_value: number
  pnl: number
  pnl_pct: number
  weight: number
}

export interface PortfolioSummary {
  total_value: number
  total_cost: number
  total_pnl: number
  total_pnl_pct: number
  holdings: PortfolioHolding[]
}

export interface Trade {
  id: number
  ticker: string
  type: 'BUY' | 'SELL'
  amount: number
  price: number
  date: string
  fees: number
  notes: string
  currency: string
}

export interface PortfolioResponse {
  summary: PortfolioSummary
  trades: Trade[]
}

export interface AddTradePayload {
  ticker: string
  type: 'BUY' | 'SELL'
  amount: number
  price: number
  date: string
  fees?: number
  notes?: string
  currency: string
}

// ── Hooks ─────────────────────────────────────────────────────────────────────

export function usePortfolio() {
  return useQuery<PortfolioResponse>({
    queryKey: ['portfolio'],
    queryFn: () => api.get('/api/portfolio').then(r => r.data),
    staleTime: 30_000,
  })
}

export function useAddTrade() {
  return useMutation({
    mutationFn: (payload: AddTradePayload) =>
      api.post('/api/portfolio/add-trade', payload).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['portfolio'] }),
  })
}

export function useUpdateTrade() {
  return useMutation({
    mutationFn: ({ id, ...payload }: AddTradePayload & { id: number }) =>
      api.patch(`/api/portfolio/trade/${id}`, payload).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['portfolio'] }),
  })
}

export function useDeleteTrade() {
  return useMutation({
    mutationFn: (id: number) =>
      api.delete(`/api/portfolio/trade/${id}`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['portfolio'] }),
  })
}

export function useImportPortfolio() {
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData()
      form.append('file', file)
      return api.post('/api/portfolio/import', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }).then(r => r.data as { imported: number; skipped: number; errors: string[] })
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['portfolio'] }),
  })
}
