import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../client'
import { queryClient } from '../queryClient'

interface BacktestProgress {
  is_running: boolean
  current_ticker: string | null
  percent: number
  stage: string | null
}

interface BacktestResults {
  run_id: string
  total_return_pct: number
  sharpe_ratio: number | null
  max_drawdown_pct: number
  win_rate: number
  total_trades: number
}

export function useBacktestProgress() {
  return useQuery<BacktestProgress>({
    queryKey: ['backtest-progress'],
    queryFn: () => api.get('/api/backtest/progress').then(r => r.data),
    refetchInterval: 2_000,
  })
}

export function useBacktestResults(runId: string | null) {
  return useQuery<BacktestResults>({
    queryKey: ['backtest-results', runId],
    queryFn: () => api.get(`/api/backtest/results/${runId}`).then(r => r.data),
    enabled: !!runId,
    staleTime: 300_000,
  })
}

export function useApplyWeights() {
  return useMutation({
    mutationFn: (runId: string) =>
      api.post(`/api/backtest/apply-weights/${runId}`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['backtest-results'] }),
  })
}
