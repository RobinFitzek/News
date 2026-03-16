import { useEffect } from 'react'
import { useThemeStore } from '@/stores/themeStore'

export function useThemeInit() {
  const applyTheme = useThemeStore(s => s.applyTheme)
  useEffect(() => { applyTheme() }, [applyTheme])
}
