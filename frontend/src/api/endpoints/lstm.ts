import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'

export interface LSTMPrediction {
  ticker: string
  confidence: number | null
  buy_signal: boolean
  threshold: number
  predicted_at: string
  error?: string
}

export interface LSTMPerformance {
  completed_trades: number
  win_rate_pct: number | null
  avg_return_pct: number | null
  cagr_pct: number | null
  max_drawdown_pct: number | null
}

export interface LSTMTradeEntry {
  id: number
  ticker: string
  entered_at: string
  expected_return_pct: number
  hold_days: number
  confidence: number
  actual_return_pct: number | null
  verified: boolean
}

export interface LSTMTrainResult {
  status: string
  tickers?: number
  train_samples?: number
  val_samples?: number
  epochs?: number
  best_val_loss?: number
  history?: { epoch: number; train_loss: number; val_loss: number; val_acc: number }[]
  error?: string
}

export interface LSTMSignalsResponse {
  signals: LSTMPrediction[]
  count: number
  threshold: number
}

export function useLSTMSignals() {
  return useQuery<LSTMSignalsResponse>({
    queryKey: ['lstm-signals'],
    queryFn: () => api.get('/api/lstm/signals').then(r => r.data),
    staleTime: 300_000,
  })
}

export function useLSTMPredict(ticker: string) {
  return useQuery<LSTMPrediction>({
    queryKey: ['lstm-predict', ticker],
    queryFn: () => api.get(`/api/lstm/predict/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}

export function useLSTMPerformance() {
  return useQuery<LSTMPerformance>({
    queryKey: ['lstm-performance'],
    queryFn: () => api.get('/api/lstm/performance').then(r => r.data),
    staleTime: 60_000,
  })
}

export function useLSTMTradeHistory(limit = 100) {
  return useQuery<{ trades: LSTMTradeEntry[]; count: number }>({
    queryKey: ['lstm-trade-history', limit],
    queryFn: () => api.get('/api/lstm/trade-history', { params: { limit } }).then(r => r.data),
    staleTime: 60_000,
  })
}

export function useLSTMTrain() {
  return useMutation<LSTMTrainResult, Error, { epochs?: number; years_back?: number }>({
    mutationFn: (body) => api.post('/api/lstm/train', body).then(r => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['lstm-signals'] })
      queryClient.invalidateQueries({ queryKey: ['lstm-performance'] })
    },
  })
}
