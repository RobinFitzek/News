import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface AutoTradeStatus {
  enabled: boolean
  mode: string
  open_positions: number
  total_trades: number
  win_rate: number | null
  total_pnl: number | null
  last_trade_date: string | null
}

export function useAutoTradeStatus() {
  return useQuery<AutoTradeStatus>({
    queryKey: ['auto-trade-status'],
    queryFn: () => api.get('/api/auto-trade/status').then(r => r.data),
    staleTime: 30_000,
  })
}
