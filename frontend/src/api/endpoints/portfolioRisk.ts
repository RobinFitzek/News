import { useQuery } from '@tanstack/react-query'
import api from '../client'

// ── VaR ──────────────────────────────────────────────────────────────────────

export interface VaRResult {
  var_95: number
  var_99: number
  cvar_95: number
  portfolio_value: number
  method: string
  error?: string
}

export function usePortfolioVaR() {
  return useQuery<VaRResult>({
    queryKey: ['portfolio-var'],
    queryFn: () => api.get('/api/portfolio/var').then(r => r.data),
    staleTime: 120_000,
  })
}

// ── Correlation ──────────────────────────────────────────────────────────────

export interface CorrelationData {
  tickers: string[]
  matrix: number[][]
  error?: string
}

export function usePortfolioCorrelation() {
  return useQuery<CorrelationData>({
    queryKey: ['portfolio-correlation'],
    queryFn: () => api.get('/api/portfolio/correlation').then(r => r.data),
    staleTime: 120_000,
  })
}

// ── Concentration ────────────────────────────────────────────────────────────

export interface ConcentrationData {
  sector_exposure: { sector: string; weight: number }[]
  top_position_weight: number
  herfindahl_index: number
  warnings: string[]
  error?: string
}

export function usePortfolioConcentration() {
  return useQuery<ConcentrationData>({
    queryKey: ['portfolio-concentration'],
    queryFn: () => api.get('/api/portfolio/concentration').then(r => r.data),
    staleTime: 120_000,
  })
}

// ── Drawdown ─────────────────────────────────────────────────────────────────

export interface DrawdownData {
  max_drawdown: number
  current_drawdown: number
  recovery_days: number | null
  peak_date: string | null
  trough_date: string | null
  error?: string
}

export function useDrawdown() {
  return useQuery<DrawdownData>({
    queryKey: ['drawdown'],
    queryFn: () => api.get('/api/drawdown').then(r => r.data),
    staleTime: 120_000,
  })
}

// ── Rebalancing Plan ─────────────────────────────────────────────────────────

export interface RebalanceAction {
  ticker: string
  action: 'BUY' | 'SELL' | 'HOLD'
  current_weight: number
  target_weight: number
  delta_shares: number
  delta_value: number
}

export interface RebalancingPlan {
  plan: RebalanceAction[]
  count: number
  error?: string
}

export function useRebalancingPlan() {
  return useQuery<RebalancingPlan>({
    queryKey: ['rebalancing-plan'],
    queryFn: () => api.get('/api/portfolio/rebalancing-plan').then(r => r.data),
    staleTime: 120_000,
  })
}
