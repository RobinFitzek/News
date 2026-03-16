/* ================================================
   STOCKHOLM — Stock Detail Endpoint Hooks
   ================================================ */

import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { Analysis } from '@/types/api'

// ── Types ────────────────────────────────────────────────────────────────────

export interface KeyStats {
  ticker: string
  name: string
  sector: string
  pe_ratio: number | null
  market_cap: number | null
  revenue: number | null
  eps: number | null
  beta: number | null
  volume: number | null
  avg_volume: number | null
  current_price: number | null
  risk_score: number | null
  geo_risk_score: number | null
  week_52_high: number | null
  week_52_low: number | null
}

export interface ChartData {
  labels: string[]
  prices: number[]
  volumes: number[]
}

export interface EarningsEntry {
  date: string
  eps_actual: number | null
  eps_estimate: number | null
  revenue_actual: number | null
  revenue_estimate: number | null
  surprise_pct: number | null
}

export interface EarningsData {
  ticker: string
  earnings: EarningsEntry[]
  next_earnings_date: string | null
}

export interface PeerEntry {
  ticker: string
  name: string
  market_cap: number | null
  pe_ratio: number | null
  revenue: number | null
  signal: string | null
  price: number | null
  change_pct: number | null
}

export interface PeersData {
  ticker: string
  peers: PeerEntry[]
}

export interface SentimentHeadline {
  title: string
  source: string
  published_at: string
  sentiment: 'positive' | 'negative' | 'neutral'
  url: string | null
}

export interface SentimentData {
  ticker: string
  overall_score: number | null
  positive_count: number
  negative_count: number
  neutral_count: number
  headlines: SentimentHeadline[]
}

export interface PatternEntry {
  name: string
  type: 'bullish' | 'bearish' | 'neutral'
  confidence: number
  description: string
  detected_at: string
}

export interface PatternsData {
  ticker: string
  patterns: PatternEntry[]
  trend: string | null
  support: number | null
  resistance: number | null
}

export interface AnalysisDetail extends Analysis {
  ticker: string
  // Analysis already has id, signal, confidence, risk_score, geo_risk_score,
  // bull_case, bear_case, synthesis, timestamp, stage
}

// ── Hooks ────────────────────────────────────────────────────────────────────

export function useKeyStats(ticker: string) {
  return useQuery<KeyStats>({
    queryKey: ['key-stats', ticker],
    queryFn: () => api.get(`/api/key-stats/${ticker}`).then(r => r.data),
    staleTime: 60_000,
    enabled: !!ticker,
  })
}

export function useChartData(ticker: string) {
  return useQuery<ChartData>({
    queryKey: ['chart-data', ticker],
    queryFn: () => api.get(`/api/chart-data/${ticker}`).then(r => r.data),
    staleTime: 60_000,
    enabled: !!ticker,
  })
}

export function useEarnings(ticker: string) {
  return useQuery<EarningsData>({
    queryKey: ['earnings', ticker],
    queryFn: () => api.get(`/api/earnings/${ticker}`).then(r => r.data),
    staleTime: 120_000,
    enabled: !!ticker,
  })
}

export function usePeers(ticker: string) {
  return useQuery<PeersData>({
    queryKey: ['peers', ticker],
    queryFn: () => api.get(`/api/peers/${ticker}`).then(r => r.data),
    staleTime: 120_000,
    enabled: !!ticker,
  })
}

export function useSentiment(ticker: string) {
  return useQuery<SentimentData>({
    queryKey: ['sentiment', ticker],
    queryFn: () => api.get(`/api/sentiment/${ticker}`).then(r => r.data),
    staleTime: 60_000,
    enabled: !!ticker,
  })
}

export function usePatterns(ticker: string) {
  return useQuery<PatternsData>({
    queryKey: ['patterns', ticker],
    queryFn: () => api.get(`/api/patterns/${ticker}`).then(r => r.data),
    staleTime: 60_000,
    enabled: !!ticker,
  })
}

export function useAnalysisDetail(analysisId: string | null) {
  return useQuery<AnalysisDetail>({
    queryKey: ['analysis', analysisId],
    queryFn: () => api.get(`/analysis/${analysisId}`).then(r => r.data),
    staleTime: 120_000,
    enabled: !!analysisId,
  })
}
