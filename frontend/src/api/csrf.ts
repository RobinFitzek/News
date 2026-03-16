/* ================================================
   CSRF Token — Singleton cache for SPA
   ================================================ */

let _token: string | null = null
let _inflight: Promise<string> | null = null

export async function getCsrfToken(): Promise<string> {
  if (_token) return _token
  if (_inflight) return _inflight

  _inflight = fetch('/api/csrf-token', { credentials: 'include' })
    .then(r => {
      if (!r.ok) throw new Error('Failed to fetch CSRF token')
      return r.json()
    })
    .then((data: { token: string }) => {
      _token = data.token
      _inflight = null
      return _token
    })
    .catch(err => {
      _inflight = null
      throw err
    })

  return _inflight
}

export function invalidateCsrfToken(): void {
  _token = null
  _inflight = null
}
