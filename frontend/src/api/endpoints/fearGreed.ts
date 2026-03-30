import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface FearGreedCurrent {
  fg_value: number | null
  fg_label: string | null
  vix: number | null
  vix_ma10: number | null
  vix_ma20: number | null
  vix_ma30: number | null
  fetched_at: string
}

export interface FearGreedDataPoint {
  date: string
  fg_value: number
}

export interface FGSensitivity {
  ticker: string
  fg_sensitivity: number | null
  lookback_days: number
  interpretation: string
}

export interface FGFeatures extends FearGreedCurrent {
  ticker: string
  fg_sensitivity_60d: number | null
}

export function useFearGreedCurrent() {
  return useQuery<FearGreedCurrent>({
    queryKey: ['fear-greed-current'],
    queryFn: () => api.get('/api/fear-greed/current').then(r => r.data),
    staleTime: 300_000,
    refetchInterval: 600_000,
  })
}

export function useFearGreedHistory() {
  return useQuery<{ data: FearGreedDataPoint[]; count: number }>({
    queryKey: ['fear-greed-history'],
    queryFn: () => api.get('/api/fear-greed/history').then(r => r.data),
    staleTime: 3_600_000,
  })
}

export function useFGSensitivity(ticker: string, lookback = 60) {
  return useQuery<FGSensitivity>({
    queryKey: ['fg-sensitivity', ticker, lookback],
    queryFn: () =>
      api.get(`/api/fear-greed/sensitivity/${ticker}`, { params: { lookback } }).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}

export function useFGFeatures(ticker: string) {
  return useQuery<FGFeatures>({
    queryKey: ['fg-features', ticker],
    queryFn: () => api.get(`/api/fear-greed/features/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}
