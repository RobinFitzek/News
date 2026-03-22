import { useQuery } from '@tanstack/react-query'
import api from '../client'

// ── DCF Valuation ────────────────────────────────────────────────────────────

export interface DCFResult {
  ticker: string
  fair_value: number | null
  current_price: number | null
  upside_pct: number | null
  fcf: number | null
  growth_rate: number | null
  terminal_rate: number
  discount_rate: number
  error?: string
}

export function useDCF(ticker: string) {
  return useQuery<DCFResult>({
    queryKey: ['dcf', ticker],
    queryFn: () => api.get(`/api/dcf/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}

// ── Economic Moat ────────────────────────────────────────────────────────────

export interface MoatResult {
  ticker: string
  moat_score: number
  moat_grade: string
  factors: { name: string; score: number; max: number }[]
  error?: string
}

export function useMoat(ticker: string) {
  return useQuery<MoatResult>({
    queryKey: ['moat', ticker],
    queryFn: () => api.get(`/api/moat/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}

// ── Catalysts ────────────────────────────────────────────────────────────────

export interface Catalyst {
  type: string
  name: string
  date: string
  detail: string
}

export interface CatalystsResult {
  ticker: string
  catalysts: Catalyst[]
  error?: string
}

export function useCatalysts(ticker: string) {
  return useQuery<CatalystsResult>({
    queryKey: ['catalysts', ticker],
    queryFn: () => api.get(`/api/catalysts/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}

// ── Options Flow ─────────────────────────────────────────────────────────────

export interface OptionsFlowResult {
  summary: {
    total_volume: number
    put_call_ratio: number
    implied_volatility: number
    unusual_count: number
  } | null
  unusual_activity: {
    type: string
    strike: number
    expiry: string
    volume: number
    open_interest: number
    premium: number
  }[]
  error?: string
}

export function useOptionsFlow(ticker: string) {
  return useQuery<OptionsFlowResult>({
    queryKey: ['options-flow', ticker],
    queryFn: () => api.get(`/api/options-flow/${ticker}`).then(r => r.data),
    staleTime: 60_000,
    enabled: !!ticker,
  })
}

// ── Supply Chain ─────────────────────────────────────────────────────────────

export interface SupplyChainEntry {
  name: string
  ticker: string | null
  relationship: 'supplier' | 'customer' | 'partner'
  detail: string
}

export interface SupplyChainResult {
  ticker: string
  suppliers: SupplyChainEntry[]
  customers: SupplyChainEntry[]
  partners: SupplyChainEntry[]
  error?: string
}

export function useSupplyChain(ticker: string) {
  return useQuery<SupplyChainResult>({
    queryKey: ['supply-chain', ticker],
    queryFn: () => api.get(`/api/supply-chain/${ticker}`).then(r => r.data),
    staleTime: 300_000,
    enabled: !!ticker,
  })
}
