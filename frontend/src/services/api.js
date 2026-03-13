/**
 * API Service for Stockholm Dashboard
 * Centralized API communication with the Python backend
 */

const API_BASE = '/api'

/**
 * Generic API fetch function
 * @param {string} endpoint - API endpoint
 * @param {object} options - Fetch options
 * @returns {Promise<any>} API response data
 */
export async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`
  
  // Set default headers
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    credentials: 'include'  // Changed to 'include' for cross-origin requests
  }
  
  const mergedOptions = { ...defaultOptions, ...options }
  
  try {
    const response = await fetch(url, mergedOptions)
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `API request failed: ${response.status}`)
    }
    
    // Handle empty responses
    if (response.status === 204) {
      return null
    }
    
    return await response.json()
  } catch (error) {
    console.error(`API Error: ${error.message}`)
    throw error
  }
}

/**
 * Dashboard API Functions
 */
export async function getDashboardData() {
  return apiFetch('/dashboard')
}

/**
 * System Status Functions
 */
export async function getSystemStatus() {
  return apiFetch('/status')
}

export async function getHealthStatus() {
  return apiFetch('/health')
}

/**
 * Portfolio Functions
 */
export async function getPortfolioAlerts() {
  return apiFetch('/portfolio/alerts')
}

/**
 * Market Data Functions
 */
export async function getMarketRegime() {
  return apiFetch('/market-regime')
}

export async function getSectorMomentum() {
  return apiFetch('/sector-momentum')
}

export async function getEconomicCalendar(days = 14) {
  return apiFetch(`/macro/events?days=${days}`)
}

/**
 * Geopolitical Functions
 */
export async function getGeopoliticalData() {
  return apiFetch('/geopolitical')
}

export async function getGeopoliticalExposure() {
  return apiFetch('/geopolitical/exposure')
}

/**
 * Learning & Performance Functions
 */
export async function getSignalAccuracy() {
  return apiFetch('/signal-accuracy')
}

export async function getPaperTradingAuto() {
  return apiFetch('/paper-trading/auto')
}

export async function getAutoTradeStatus() {
  return apiFetch('/auto-trade/status')
}

/**
 * Chart Data Functions
 */
export async function getChartData(ticker) {
  return apiFetch(`/chart-data/${ticker}`)
}

/**
 * Export the API service
 */
export default {
  getDashboardData,
  getSystemStatus,
  getHealthStatus,
  getPortfolioAlerts,
  getMarketRegime,
  getSectorMomentum,
  getEconomicCalendar,
  getGeopoliticalData,
  getGeopoliticalExposure,
  getSignalAccuracy,
  getPaperTradingAuto,
  getAutoTradeStatus,
  getChartData
}