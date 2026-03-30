import { useQuery } from '@tanstack/react-query'
import api from '../client'

export interface PoliticianTrade {
  ticker: string
  date: string
  tx_type: string
  asset_type: string
  senator: string
  amount_mid: number
  is_buy: boolean
  is_sell: boolean
}

export interface TopTicker {
  ticker: string
  total_trades: number
  buy_count: number
  sell_count: number
  unique_senators: number
  total_volume_mid: number
}

export interface PoliticianFeatures {
  ticker: string
  date: string
  pol_total_trades: number
  pol_buy_count: number
  pol_sell_count: number
  pol_exchange_count: number
  pol_options_count: number
  pol_bond_other_count: number
  pol_unique_senators: number
  pol_money_range_mid: number
  pol_log_money_mid: number
}

export function usePoliticianTrades(ticker?: string, days = 30) {
  return useQuery<{ trades: PoliticianTrade[]; count: number }>({
    queryKey: ['politician-trades', ticker, days],
    queryFn: () =>
      api.get('/api/politicians/recent', { params: { ticker, days } }).then(r => r.data),
    staleTime: 300_000,
  })
}

export function usePoliticianTopTickers(days = 90, topN = 20) {
  return useQuery<{ tickers: TopTicker[] }>({
    queryKey: ['politician-top-tickers', days, topN],
    queryFn: () =>
      api.get('/api/politicians/top-tickers', { params: { days, top_n: topN } }).then(r => r.data),
    staleTime: 300_000,
  })
}

export function usePoliticianFeatures(ticker: string) {
  return useQuery<PoliticianFeatures>({
    queryKey: ['politician-features', ticker],
    queryFn: () => api.get(`/api/politicians/features/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}
