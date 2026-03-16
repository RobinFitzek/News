/* ================================================
   STOCKHOLM — API Response Types
   ================================================ */

// ── Auth ────────────────────────────────────────

export interface CsrfTokenResponse {
  token: string
}

// ── Scheduler / Status ──────────────────────────

export type SchedulerState = 'running' | 'sleeping' | 'scanning' | 'stopped' | 'idle'

export interface SchedulerStatus {
  is_running: boolean
  is_scanning: boolean
  state: SchedulerState
  last_run: string | null
  next_run: string | null
  queue_count: number
  pending_count: number
  jobs: SchedulerJob[]
}

export interface SchedulerJob {
  id: string
  name: string
  next_run_time: string | null
  trigger: string
}

export interface ApiUsage {
  used: number
  limit: number
  remaining: number
  cost_usd: number
  requests_today: number
}

export interface ApiStatus {
  scheduler: SchedulerStatus
  api_usage: {
    perplexity: ApiUsage
    gemini: ApiUsage
  }
  watchlist_count: number
  stale_analyses: number
  ml_status?: string
  accuracy?: number
}

// ── Health ──────────────────────────────────────

export interface HealthStatus {
  overall_status: 'healthy' | 'degraded' | 'critical'
  disk: { status: string; usage_percent: number }
  memory: { status: string; usage_percent: number }
  database: { status: string; size_mb: number }
  errors: string[]
  uptime: string
}

// ── Scan Progress ───────────────────────────────

export interface ScanProgress {
  is_scanning: boolean
  current_ticker: string | null
  total: number
  completed: number
  stage: string | null
  percent: number
}

// ── Geopolitical ────────────────────────────────

export interface GeoEvent {
  headline: string
  severity: number
  region: string
  timestamp: string
  source?: string
  affected_sectors?: string[]
}

export interface GeoScan {
  timestamp: string
  overall_severity: number
  events: GeoEvent[]
  summary: string
  portfolio_impact: string
}

export interface GeoExposure {
  ticker: string
  name: string
  geo_risk_score: number
  regions: string[]
  exposure_detail: string
}

// ── Market / Macro ──────────────────────────────

export interface MarketRegime {
  regime: 'bull' | 'bear' | 'neutral' | 'volatile'
  spy_price: number
  spy_change_pct: number
  vix: number
  yield_10y: number
  sma50_above: boolean
  sma200_above: boolean
  regime_label: string
  timestamp: string
}

export interface MacroEvent {
  date: string
  time: string
  event: string
  country: string
  importance: 'high' | 'medium' | 'low'
  forecast: string | null
  previous: string | null
  actual: string | null
}

// ── Portfolio ───────────────────────────────────

export interface BenchmarkData {
  portfolio_return: number
  spy_return: number
  alpha: number
  start_date: string
  end_date: string
  labels: string[]
  portfolio_series: number[]
  spy_series: number[]
}

export interface PortfolioAlert {
  id: string
  type: string
  severity: 'high' | 'medium' | 'low'
  message: string
  ticker?: string
  timestamp: string
  acknowledged: boolean
}

export interface PortfolioAlertsResponse {
  alerts: PortfolioAlert[]
  active_alerts: PortfolioAlert[]
  alert_summary: { high: number; medium: number; low: number }
  raw_alert_count: number
  holdings: Holding[]
}

export interface Holding {
  ticker: string
  name: string
  shares: number
  avg_cost: number
  current_price: number
  market_value: number
  pnl: number
  pnl_pct: number
  weight: number
  sector: string
}

// ── Discovery ───────────────────────────────────

export interface DiscoveryStats {
  enabled: boolean
  discovered_7d: number
  promoted_7d: number
  last_run: string | null
  next_run: string | null
  total_discovered: number
}

// ── Budget ──────────────────────────────────────

export interface BudgetStatus {
  perplexity: {
    used_eur: number
    limit_eur: number
    pct_used: number
    requests: number
  }
  gemini: {
    used_eur: number
    limit_eur: number
    pct_used: number
    requests: number
  }
  avg_cost_per_analysis_usd: number
  cost_7d: number
  analyses_this_month: number
}

// ── Signal / Learning ───────────────────────────

export interface SignalAccuracy {
  overall_accuracy: number
  verified_predictions: number
  pending_predictions: number
  kill_switch_active: boolean
  kill_switch_threshold: number
  by_signal: {
    buy: { accuracy: number; count: number }
    sell: { accuracy: number; count: number }
    hold: { accuracy: number; count: number }
  }
}

// ── Watchlist ───────────────────────────────────

export type WatchlistTier = 'core' | 'swing' | 'research' | 'earnings_play'
export type SignalType = 'BUY' | 'SELL' | 'HOLD' | 'WATCH' | null

export interface WatchlistItem {
  id: number
  ticker: string
  name: string
  tier: WatchlistTier
  added_date: string
  geo_risk_score: number | null
  sentiment: string | null
  signal: SignalType
  confidence: number | null
  last_analyzed: string | null
  days_since_analysis: number | null
  is_active: boolean
  note: string | null
}

// ── Stock / Analysis ────────────────────────────

export interface Analysis {
  id: number
  ticker: string
  signal: SignalType
  confidence: number
  risk_score: number
  geo_risk_score: number
  bull_case: string
  bear_case: string
  synthesis: string
  timestamp: string
  stage: number
}

export interface StockDetail {
  ticker: string
  name: string
  sector: string
  market_cap: string
  current_price: number
  pre_market_price: number | null
  post_market_price: number | null
  week_52_high: number
  week_52_low: number
  short_pct_float: number | null
  days_to_cover: number | null
  in_watchlist: boolean
  latest_analysis: Analysis | null
}

// ── Settings ────────────────────────────────────

export interface Provider {
  id: string
  name: string
  type: 'perplexity' | 'gemini' | 'openai' | 'custom'
  is_configured: boolean
  is_active: boolean
  last_tested: string | null
  test_status: 'ok' | 'error' | 'untested'
  model?: string
  base_url?: string
}

export interface PersonalKey {
  id: string
  label: string
  scope: string
  created_at: string
  last_used: string | null
  masked_key: string
}

export interface Plugin {
  id: string
  name: string
  description: string
  is_enabled: boolean
  version: string
  last_run: string | null
  has_settings: boolean
}

export interface SettingsData {
  scheduler: {
    scan_interval_hours: number
    geo_scan_interval_hours: number
    auto_discovery_enabled: boolean
    auto_trading_enabled: boolean
    daily_limit: number
  }
  analysis: {
    confidence_threshold: number
    risk_tolerance: string
    preferred_signals: string[]
    use_ml_labeler: boolean
  }
  notifications: {
    email_enabled: boolean
    email_address: string | null
    geo_alert_threshold: number
    notify_on_signal: boolean
    notify_on_discovery: boolean
  }
  portfolio: {
    max_position_size_pct: number
    stop_loss_pct: number
    max_sector_concentration: number
    benchmark: string
  }
  budget: {
    perplexity_monthly_eur: number
    gemini_monthly_eur: number
  }
  security: {
    two_factor_enabled: boolean
    session_timeout_minutes: number
  }
  appearance: {
    theme: 'dark' | 'light' | 'system'
    glow_intensity: number
    depth_effects: boolean
  }
}

// ── Paper Trading ───────────────────────────────

export interface PaperTradingSummary {
  total_value: number
  total_pnl: number
  total_pnl_pct: number
  win_rate: number
  total_trades: number
  sharpe_ratio: number | null
  positions: PaperPosition[]
}

export interface PaperPosition {
  ticker: string
  shares: number
  avg_cost: number
  current_price: number
  market_value: number
  pnl: number
  pnl_pct: number
}

// ── Misc ─────────────────────────────────────────

export interface ToastMessage {
  id: string
  message: string
  type: 'success' | 'error' | 'warning' | 'info'
  duration?: number
}

export interface ApiError {
  detail: string
  status: number
}
