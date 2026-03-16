/* ================================================
   STOCKHOLM — Axios Client
   Session-based auth, CSRF interceptor
   ================================================ */

import axios from 'axios'
import { getCsrfToken, invalidateCsrfToken } from './csrf'

const MUTATION_METHODS = new Set(['post', 'put', 'patch', 'delete'])

const api = axios.create({
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
})

// Request interceptor — inject CSRF token for mutations
api.interceptors.request.use(async (config) => {
  const method = (config.method ?? '').toLowerCase()
  if (MUTATION_METHODS.has(method)) {
    try {
      const token = await getCsrfToken()
      config.headers['X-CSRF-Token'] = token
    } catch {
      // If CSRF fetch fails, proceed anyway — server will reject if needed
    }
  }
  return config
})

// Response interceptor — handle auth & session errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (!error.response) {
      return Promise.reject(error)
    }

    const { status } = error.response

    if (status === 401) {
      // Session expired or not authenticated — redirect to login
      window.location.href = '/login'
      return Promise.reject(error)
    }

    if (status === 403) {
      // CSRF token mismatch — invalidate and retry once
      invalidateCsrfToken()
    }

    return Promise.reject(error)
  }
)

export default api
