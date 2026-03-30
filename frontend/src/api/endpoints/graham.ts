import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'

export interface GrahamResult {
  ticker: string
  buy_signal: boolean
  reason: string
  intrinsic_value: number | null
  buy_threshold: number | null
  current_price: number | null
  upside_pct: number | null
  ttm_eps: number | null
  growth_rate: number | null
  book_value_per_share: number | null
  discount_factor: number
  aaa_yield: number
  eps_history_quarters: number
}

export interface GrahamScreenResponse {
  screened_at: string
  aaa_yield: number
  discount_factor: number
  total_screened: number
  iv_calculable: number
  buy_candidates: number
  max_positions: number
  results: GrahamResult[]
  buy_list: GrahamResult[]
}

export interface GrahamBacktestResponse {
  backtest_date: string
  discount_factor: number
  aaa_yield: number
  trades: number
  iv_calculable_tickers: number
  avg_forward_return_pct: number | null
  win_rate_pct: number | null
  benchmark_return_pct: number | null
  alpha_vs_benchmark: number | null
  trade_results: GrahamResult[]
}

export function useGrahamScreen(discount = 0.2, enabled = true) {
  return useQuery<GrahamScreenResponse>({
    queryKey: ['graham-screen', discount],
    queryFn: () => api.get('/api/graham/screen', { params: { discount } }).then(r => r.data),
    staleTime: 300_000,
    enabled,
  })
}

export function useGrahamTicker(ticker: string, discount = 0.2) {
  return useQuery<GrahamResult>({
    queryKey: ['graham-ticker', ticker, discount],
    queryFn: () => api.get(`/api/graham/ticker/${ticker}`, { params: { discount } }).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}

export function useGrahamAAAYield() {
  return useQuery<{ aaa_yield_pct: number; source: string }>({
    queryKey: ['graham-aaa-yield'],
    queryFn: () => api.get('/api/graham/aaa-yield').then(r => r.data),
    staleTime: 3_600_000,
  })
}

export function useGrahamBacktest() {
  return useMutation({
    mutationFn: (params: { discount?: number; max_positions?: number; holding_days?: number }) =>
      api.get('/api/graham/backtest', { params }).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['graham-screen'] }),
  })
}
