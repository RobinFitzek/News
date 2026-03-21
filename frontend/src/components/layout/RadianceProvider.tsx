/**
 * RadianceProvider — Context for the proximity light-reflection system.
 *
 * Wraps page content. Cards register themselves via context to participate
 * in the radiance engine (light bleeding between nearby glass surfaces).
 *
 * Also renders shared SVG <defs> for edge-refraction filters.
 */

import { createContext, useContext, type ReactNode } from 'react'
import { useRadiance, type GlowColor } from '@/hooks/useRadiance'

// ── Context ─────────────────────────────────────────────────────────────────

interface RadianceContextValue {
  register: (id: string, element: HTMLElement, glowColor: GlowColor) => void
  unregister: (id: string) => void
}

const RadianceContext = createContext<RadianceContextValue | null>(null)

export function useRadianceContext() {
  return useContext(RadianceContext)
}

// ── Provider ────────────────────────────────────────────────────────────────

interface RadianceProviderProps {
  children: ReactNode
}

export function RadianceProvider({ children }: RadianceProviderProps) {
  const { register, unregister } = useRadiance()

  return (
    <RadianceContext.Provider value={{ register, unregister }}>
      {/* Shared SVG filter definitions for edge refraction */}
      <svg
        width="0"
        height="0"
        style={{ position: 'absolute', overflow: 'hidden' }}
        aria-hidden="true"
      >
        <defs>
          {/* Subtle glass-edge refraction — applied to radiance glow layer */}
          <filter id="radiance-refract" x="-10%" y="-10%" width="120%" height="120%">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.8"
              numOctaves="2"
              seed="42"
              result="noise"
            />
            <feDisplacementMap
              in="SourceGraphic"
              in2="noise"
              scale="2"
              xChannelSelector="R"
              yChannelSelector="G"
            />
          </filter>

          {/* Signal glyph symbols — referenced via <use href="#signal-buy"> */}
          <symbol id="signal-buy" viewBox="0 0 16 16">
            <title>Buy Signal</title>
            <polygon points="8,2 14,12 2,12" fill="currentColor" />
          </symbol>

          <symbol id="signal-sell" viewBox="0 0 16 16">
            <title>Sell Signal</title>
            <polygon points="8,14 14,4 2,4" fill="currentColor" />
          </symbol>

          <symbol id="signal-hold" viewBox="0 0 16 16">
            <title>Hold Signal</title>
            <rect x="2" y="5" width="12" height="2" rx="0" fill="currentColor" />
            <rect x="2" y="9" width="12" height="2" rx="0" fill="currentColor" />
          </symbol>

          <symbol id="signal-watch" viewBox="0 0 16 16">
            <title>Watch Signal</title>
            <path
              d="M1 8 Q8 1 15 8 Q8 15 1 8 Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <circle cx="8" cy="8" r="2.5" fill="currentColor" />
          </symbol>
        </defs>
      </svg>

      {children}
    </RadianceContext.Provider>
  )
}
