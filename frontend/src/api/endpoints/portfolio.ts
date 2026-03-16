import { useQuery } from '@tanstack/react-query'
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
