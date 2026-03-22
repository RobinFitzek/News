import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { router } from './router'
import { queryClient } from './api/queryClient'
import './styles/globals.css'
import './styles/glass.css'

// Apply persisted theme + Breathe settings before first render to prevent flash
const storedState = (() => {
  try {
    return JSON.parse(localStorage.getItem('stockholm-theme') ?? '{}').state ?? {}
  } catch {
    return {}
  }
})()

const resolvedTheme = (() => {
  const theme = storedState.theme ?? 'dark'
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return theme
})()

document.documentElement.setAttribute('data-theme', resolvedTheme)
document.documentElement.setAttribute('data-depth', (storedState.depthEffects ?? true) ? 'on' : 'off')
document.documentElement.setAttribute('data-particles', (storedState.particleField ?? true) ? 'on' : 'off')
document.documentElement.style.setProperty('--glow-intensity', String(storedState.glowIntensity ?? 0.6))

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  </StrictMode>
)
