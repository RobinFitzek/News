import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { router } from './router'
import { queryClient } from './api/queryClient'
import './styles/globals.css'
import './styles/glass.css'

// Apply persisted theme before first render
const storedTheme = (() => {
  try {
    const stored = JSON.parse(localStorage.getItem('stockholm-theme') ?? '{}')
    return stored.state?.theme ?? 'dark'
  } catch {
    return 'dark'
  }
})()

const resolved =
  storedTheme === 'system'
    ? window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    : storedTheme

document.documentElement.setAttribute('data-theme', resolved)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  </StrictMode>
)
