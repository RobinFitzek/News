import { motion, useMotionValue, useSpring } from 'framer-motion'
import { useEffect, useMemo } from 'react'
import { useThemeStore } from '@/stores/themeStore'
import styles from './Luminary.module.css'

/* Deterministic pseudo-random from seed (mulberry32) */
function seededRandom(seed: number) {
  let t = (seed + 0x6d2b79f5) | 0
  t = Math.imul(t ^ (t >>> 15), t | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const PARTICLE_COUNT = 40
const CONSTELLATION_DISTANCE = 18 // % of viewBox — max distance for a line

interface Particle {
  cx: string
  cy: string
  cxNum: number
  cyNum: number
  r: number
  opacity: number
  duration: number
  delay: number
  dx: number
  dy: number
}

interface ConstellationLine {
  x1: string
  y1: string
  x2: string
  y2: string
  delay: number
}

function generateParticles(): Particle[] {
  const particles: Particle[] = []
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const rng = (offset: number) => seededRandom(i * 7 + offset)
    const cxNum = rng(0) * 100
    const cyNum = rng(1) * 100
    particles.push({
      cx: `${cxNum}%`,
      cy: `${cyNum}%`,
      cxNum,
      cyNum,
      r: 1 + rng(2) * 2,
      opacity: 0.06 + rng(3) * 0.09,
      duration: 20 + rng(4) * 30,
      delay: rng(5) * -40,
      dx: (rng(6) - 0.5) * 60,
      dy: (rng(7) - 0.5) * 60,
    })
  }
  return particles
}

function generateConstellations(particles: Particle[]): ConstellationLine[] {
  const lines: ConstellationLine[] = []
  const maxLines = 15 // cap to avoid visual noise
  for (let i = 0; i < particles.length && lines.length < maxLines; i++) {
    for (let j = i + 1; j < particles.length && lines.length < maxLines; j++) {
      const dx = particles[i].cxNum - particles[j].cxNum
      const dy = particles[i].cyNum - particles[j].cyNum
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < CONSTELLATION_DISTANCE && dist > 4) {
        lines.push({
          x1: particles[i].cx,
          y1: particles[i].cy,
          x2: particles[j].cx,
          y2: particles[j].cy,
          delay: seededRandom(i * 13 + j) * -16,
        })
      }
    }
  }
  return lines
}

export function Luminary() {
  const rawX = useMotionValue(0)
  const rawY = useMotionValue(0)
  const x = useSpring(rawX, { stiffness: 40, damping: 20 })
  const y = useSpring(rawY, { stiffness: 40, damping: 20 })
  const particles = useMemo(generateParticles, [])
  const constellations = useMemo(() => generateConstellations(particles), [particles])
  const showParticles = useThemeStore(s => s.particleField)

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduced) return

    const handler = (e: MouseEvent) => {
      rawX.set((e.clientX / window.innerWidth - 0.5) * 60)
      rawY.set((e.clientY / window.innerHeight - 0.5) * 60)
    }
    window.addEventListener('mousemove', handler, { passive: true })
    return () => window.removeEventListener('mousemove', handler)
  }, [rawX, rawY])

  return (
    <motion.div className={styles.luminary} style={{ x, y }} aria-hidden="true">
      {/* Aurora borealis — flowing light bands */}
      <div className={styles.auroraLayer}>
        <div className={styles.auroraA} />
        <div className={styles.auroraB} />
        <div className={styles.auroraC} />
      </div>

      {/* Ambient orbs */}
      <div className={styles.orbA} />
      <div className={styles.orbB} />

      {/* Particle field + constellation lines */}
      {showParticles && (
        <svg className={styles.particleField} viewBox="0 0 100 100" preserveAspectRatio="none">
          {/* Constellation connecting lines */}
          {constellations.map((line, i) => (
            <line
              key={`c-${i}`}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke="var(--text-primary)"
              strokeWidth="0.15"
              className={styles.constellationLine}
              style={{ animationDelay: `${line.delay}s` }}
            />
          ))}
          {/* Particles */}
          {particles.map((p, i) => (
            <circle
              key={i}
              cx={p.cx}
              cy={p.cy}
              r={p.r}
              fill="var(--text-primary)"
              opacity={p.opacity}
              style={{
                '--drift-dx': `${p.dx}px`,
                '--drift-dy': `${p.dy}px`,
                animationDuration: `${p.duration}s`,
                animationDelay: `${p.delay}s`,
              } as React.CSSProperties}
              className={styles.particle}
            />
          ))}
        </svg>
      )}
    </motion.div>
  )
}
