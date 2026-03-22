/**
 * useRadiance — Proximity-based light bleeding between glass cards.
 *
 * When two cards with glow colors are near each other, the facing edges
 * of each card receive a soft colored light from the neighbor's glow.
 * This creates the illusion of light reflecting off glass surfaces.
 */

import { useCallback, useEffect, useRef } from 'react'

// ── Types ───────────────────────────────────────────────────────────────────

export type GlowColor = 'positive' | 'negative' | 'neutral' | 'gold'

interface CardEntry {
  element: HTMLElement
  glowColor: GlowColor
}

interface Rect {
  top: number
  bottom: number
  left: number
  right: number
  cx: number
  cy: number
}

// ── Color map ───────────────────────────────────────────────────────────────

const GLOW_RGB: Record<GlowColor, string> = {
  positive: '107, 255, 158',  // #6BFF9E
  negative: '255, 107, 107',  // #FF6B6B
  neutral:  '107, 184, 255',  // #6BB8FF
  gold:     '255, 216, 122',  // #FFD87A
}

// ── Constants ───────────────────────────────────────────────────────────────

const MAX_DISTANCE = 120       // px — beyond this, no radiance
const RECALC_INTERVAL = 200    // ms — position recalculation interval
const BASE_OPACITY = 0.12      // max radiance opacity at distance 0

// ── Edge detection ──────────────────────────────────────────────────────────

function getRect(el: HTMLElement): Rect {
  const r = el.getBoundingClientRect()
  return {
    top: r.top,
    bottom: r.bottom,
    left: r.left,
    right: r.right,
    cx: (r.left + r.right) / 2,
    cy: (r.top + r.bottom) / 2,
  }
}

type Edge = 'top' | 'right' | 'bottom' | 'left'

function edgeDistances(a: Rect, b: Rect): { edge: Edge; distance: number }[] {
  const results: { edge: Edge; distance: number }[] = []

  // Horizontal overlap check (needed for top/bottom edges)
  const hOverlap = a.right > b.left && a.left < b.right

  // Vertical overlap check (needed for left/right edges)
  const vOverlap = a.bottom > b.top && a.top < b.bottom

  // B is below A → A's bottom edge receives light from B
  if (hOverlap && b.top >= a.bottom) {
    results.push({ edge: 'bottom', distance: b.top - a.bottom })
  }
  // B is above A → A's top edge receives light from B
  if (hOverlap && b.bottom <= a.top) {
    results.push({ edge: 'top', distance: a.top - b.bottom })
  }
  // B is right of A → A's right edge receives light from B
  if (vOverlap && b.left >= a.right) {
    results.push({ edge: 'right', distance: b.left - a.right })
  }
  // B is left of A → A's left edge receives light from B
  if (vOverlap && b.right <= a.left) {
    results.push({ edge: 'left', distance: a.left - b.right })
  }

  return results.filter(r => r.distance < MAX_DISTANCE)
}

// ── Main engine ─────────────────────────────────────────────────────────────

export function useRadiance() {
  const cardsRef = useRef<Map<string, CardEntry>>(new Map())
  const rafRef = useRef<number>(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>()
  const activeRef = useRef(true)

  // Current radiance state per card per edge (for lerping)
  const radianceState = useRef<Map<string, Record<Edge, { color: string; intensity: number }>>>(new Map())

  const register = useCallback((id: string, element: HTMLElement, glowColor: GlowColor) => {
    cardsRef.current.set(id, { element, glowColor })
  }, [])

  const unregister = useCallback((id: string) => {
    cardsRef.current.delete(id)
    radianceState.current.delete(id)
    // Clean up CSS properties
    const cards = cardsRef.current
    const entry = cards.get(id)
    if (entry) {
      clearRadiance(entry.element)
    }
  }, [])

  useEffect(() => {
    // Check reduced motion
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    if (motionQuery.matches) {
      activeRef.current = false
      return
    }

    // Check depth effects
    if (document.documentElement.dataset.depth === 'off') {
      activeRef.current = false
      return
    }

    function recalculate() {
      if (!activeRef.current) return
      const cards = cardsRef.current
      if (cards.size < 2) return

      const entries = Array.from(cards.entries())
      const rects = new Map<string, Rect>()

      // Calculate all positions
      for (const [id, entry] of entries) {
        rects.set(id, getRect(entry.element))
      }

      // Reset target intensities
      const targets = new Map<string, Record<Edge, { rgb: string; intensity: number }>>()
      for (const [id] of entries) {
        targets.set(id, {
          top: { rgb: '', intensity: 0 },
          right: { rgb: '', intensity: 0 },
          bottom: { rgb: '', intensity: 0 },
          left: { rgb: '', intensity: 0 },
        })
      }

      // Calculate pairwise radiance
      for (let i = 0; i < entries.length; i++) {
        for (let j = i + 1; j < entries.length; j++) {
          const [idA, entryA] = entries[i]
          const [idB, entryB] = entries[j]
          const rectA = rects.get(idA)!
          const rectB = rects.get(idB)!

          // Edges where A receives light from B
          const edgesA = edgeDistances(rectA, rectB)
          for (const { edge, distance } of edgesA) {
            const intensity = (1 - distance / MAX_DISTANCE) * BASE_OPACITY
            const target = targets.get(idA)!
            if (intensity > target[edge].intensity) {
              target[edge] = { rgb: GLOW_RGB[entryB.glowColor], intensity }
            }
          }

          // Edges where B receives light from A
          const edgesB = edgeDistances(rectB, rectA)
          for (const { edge, distance } of edgesB) {
            const intensity = (1 - distance / MAX_DISTANCE) * BASE_OPACITY
            const target = targets.get(idB)!
            if (intensity > target[edge].intensity) {
              target[edge] = { rgb: GLOW_RGB[entryA.glowColor], intensity }
            }
          }
        }
      }

      // Apply to DOM
      for (const [id, entry] of entries) {
        const target = targets.get(id)!
        const el = entry.element
        const edges: Edge[] = ['top', 'right', 'bottom', 'left']

        for (const edge of edges) {
          const { rgb, intensity } = target[edge]
          if (intensity > 0.005 && rgb) {
            el.style.setProperty(`--radiance-${edge}-color`, `rgba(${rgb}, ${intensity})`)
          } else {
            el.style.setProperty(`--radiance-${edge}-color`, 'transparent')
          }
        }
      }
    }

    // Run recalculation on interval
    timerRef.current = setInterval(recalculate, RECALC_INTERVAL)

    // Also recalculate on scroll and resize
    const handleScroll = () => recalculate()
    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('resize', recalculate, { passive: true })

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('resize', recalculate)
    }
  }, [])

  return { register, unregister }
}

function clearRadiance(el: HTMLElement) {
  el.style.removeProperty('--radiance-top-color')
  el.style.removeProperty('--radiance-right-color')
  el.style.removeProperty('--radiance-bottom-color')
  el.style.removeProperty('--radiance-left-color')
}
