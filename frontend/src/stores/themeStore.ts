import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ThemeStore {
  theme: 'dark' | 'light' | 'system'
  glowIntensity: number
  depthEffects: boolean
  sidebarExpanded: boolean
  setTheme: (theme: ThemeStore['theme']) => void
  setGlowIntensity: (v: number) => void
  setDepthEffects: (v: boolean) => void
  setSidebarExpanded: (v: boolean) => void
  toggleSidebar: () => void
  applyTheme: () => void
}

function resolveTheme(theme: ThemeStore['theme']): 'dark' | 'light' {
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return theme
}

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      glowIntensity: 0.6,
      depthEffects: true,
      sidebarExpanded: true,

      setTheme: (theme) => {
        set({ theme })
        get().applyTheme()
      },

      setGlowIntensity: (v) => {
        set({ glowIntensity: v })
        document.documentElement.style.setProperty('--glow-intensity', String(v))
      },

      setDepthEffects: (v) => {
        set({ depthEffects: v })
        document.documentElement.setAttribute('data-depth', String(v))
      },

      setSidebarExpanded: (v) => set({ sidebarExpanded: v }),

      toggleSidebar: () => set(s => ({ sidebarExpanded: !s.sidebarExpanded })),

      applyTheme: () => {
        const { theme, glowIntensity, depthEffects } = get()
        const resolved = resolveTheme(theme)
        document.documentElement.setAttribute('data-theme', resolved)
        document.documentElement.style.setProperty('--glow-intensity', String(glowIntensity))
        document.documentElement.setAttribute('data-depth', String(depthEffects))
        // Update meta theme-color
        const meta = document.querySelector('meta[name="theme-color"]')
        if (meta) {
          meta.setAttribute('content', resolved === 'dark' ? '#080810' : '#F5F0E8')
        }
      },
    }),
    {
      name: 'stockholm-theme',
      partialize: (state) => ({
        theme: state.theme,
        glowIntensity: state.glowIntensity,
        depthEffects: state.depthEffects,
        sidebarExpanded: state.sidebarExpanded,
      }),
    }
  )
)
