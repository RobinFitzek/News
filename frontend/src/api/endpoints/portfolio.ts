import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import type { BenchmarkData, PortfolioAlertsResponse } from '@/types/api'

export function useBenchmark() {
  return useQuery<BenchmarkData>({
    queryKey: ['benchmark'],
    queryFn: () => api.get('/api/portfolio/benchmark').then(r => r.data),
    staleTime: 60_000,
  })
}

export function usePortfolioAlerts() {
  return useQuery<PortfolioAlertsResponse>({
    queryKey: ['portfolio-alerts'],
    queryFn: () => api.get('/api/portfolio/alerts').then(r => r.data),
    staleTime: 60_000,
    refetchInterval: 60_000,
  })
}

// ── Portfolio Q&A (#56) ─────────────────────────────────────────────────────

export interface PortfolioQAResponse {
  answer: string | null
  sources: string[]
  rate_limited: boolean
  error: string | null
}

export function usePortfolioAsk() {
  return useMutation<PortfolioQAResponse, Error, string>({
    mutationFn: (question: string) =>
      api.post('/api/portfolio/ask', { question }).then(r => r.data),
  })
}
