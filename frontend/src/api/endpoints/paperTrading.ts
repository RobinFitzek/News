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

// ── Equity Curve ─────────────────────────────────────────────────────────────

export interface EquityCurvePoint {
  date: string
  value: number
}

export interface EquityCurveData {
  curve: EquityCurvePoint[]
  error?: string
}

export function usePaperEquityCurve(days = 90) {
  return useQuery<EquityCurveData>({
    queryKey: ['paper-equity-curve', days],
    queryFn: () => api.get(`/api/paper-trading/equity-curve?days=${days}`).then(r => r.data),
    staleTime: 60_000,
  })
}

// ── Risk Metrics ─────────────────────────────────────────────────────────────

export interface PaperRiskMetrics {
  sharpe_ratio: number | null
  sortino_ratio: number | null
  calmar_ratio: number | null
  max_drawdown: number | null
  volatility: number | null
  error?: string
}

export function usePaperRiskMetrics() {
  return useQuery<PaperRiskMetrics>({
    queryKey: ['paper-risk-metrics'],
    queryFn: () => api.get('/api/paper-trading/risk-metrics').then(r => r.data),
    staleTime: 60_000,
  })
}

// ── SPY Correlation ──────────────────────────────────────────────────────────

export interface SpyCorrelation {
  beta: number | null
  alpha: number | null
  r_squared: number | null
  correlation: number | null
  error?: string
}

export function useSpyCorrelation() {
  return useQuery<SpyCorrelation>({
    queryKey: ['paper-spy-correlation'],
    queryFn: () => api.get('/api/paper-trading/spy-correlation').then(r => r.data),
    staleTime: 60_000,
  })
}
