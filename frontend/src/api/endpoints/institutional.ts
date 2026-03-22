import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface SmartMoneyPosition {
  filer_name: string
  shares: number
  value_usd?: number
  filing_date: string
  change_pct?: number
}

export interface SmartMoneyData {
  ticker: string
  new_positions: SmartMoneyPosition[]
  increased: SmartMoneyPosition[]
  decreased: SmartMoneyPosition[]
  smart_money_badge: boolean
  total_top_filers_holding: number
}

export function useSmartMoney(ticker: string) {
  return useQuery<SmartMoneyData>({
    queryKey: ['smart-money', ticker],
    queryFn: () => api.get(`/api/smart-money/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}
