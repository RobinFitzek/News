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

export interface RandomBaseline {
  n_simulations: number
  portfolio_size: number
  mean_random_return: number
  std_random_return: number
  z_score_vs_random: number | null
  pct_simulations_beaten: number | null
  histogram: {
    bins: number[]
    bin_edges: number[]
    strategy_return: number
    strategy_bin_idx: number | null
  } | null
  strategy_return: number
}

export function useRandomBaseline() {
  return useMutation<RandomBaseline, Error, { results: BacktestResults; n_simulations?: number }>({
    mutationFn: ({ results, n_simulations = 500 }) =>
      api.post('/api/backtest/random-baseline', {
        total_return_pct: results.total_return_pct,
        total_trades: results.total_trades,
        n_simulations,
      }).then(r => r.data),
  })
}
