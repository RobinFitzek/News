import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface TopPick {
  ticker: string
  predictions_count: number
  accuracy: number
  avg_confidence: number
  last_signal: string
  last_signal_date: string
  win_streak: number
}

export interface RecentSignal {
  ticker: string
  signal: string
  confidence: number
  timestamp: string
}

export interface LearningStats {
  total_predictions: number
  accuracy_rate: number
  last_updated: string
}

export interface TopPicksResponse {
  top_picks: TopPick[]
  recent_signals: RecentSignal[]
  learning_stats: LearningStats
  total_trusted: number
}

export function useTopPicks() {
  return useQuery<TopPicksResponse>({
    queryKey: ['top-picks'],
    queryFn: () => api.get('/api/top-picks').then(r => r.data),
    staleTime: 60_000,
  })
}
