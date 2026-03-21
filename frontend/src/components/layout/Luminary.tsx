import { motion, useMotionValue, useSpring } from 'framer-motion'
import { useEffect, useMemo } from 'react'
import styles from './Luminary.module.css'

/* Deterministic pseudo-random from seed (mulberry32) */
function seededRandom(seed: number) {
  let t = (seed + 0x6d2b79f5) | 0
  t = Math.imul(t ^ (t >>> 15), t | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const PARTICLE_COUNT = 40

interface Particle {
  cx: string
  cy: string
  r: number
  opacity: number
  duration: number
  delay: number
  dx: number
  dy: number
}

function generateParticles(): Particle[] {
  const particles: Particle[] = []
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const rng = (offset: number) => seededRandom(i * 7 + offset)
    particles.push({
      cx: `${rng(0) * 100}%`,
      cy: `${rng(1) * 100}%`,
      r: 1 + rng(2) * 2,            // 1–3px radius
      opacity: 0.06 + rng(3) * 0.09, // 0.06–0.15
      duration: 20 + rng(4) * 30,    // 20–50s drift cycle
      delay: rng(5) * -40,           // stagger start
      dx: (rng(6) - 0.5) * 60,       // drift range ±30px
      dy: (rng(7) - 0.5) * 60,
    })
  }
  return particles
}

export function Luminary() {
  const rawX = useMotionValue(0)
  const rawY = useMotionValue(0)
  const x = useSpring(rawX, { stiffness: 40, damping: 20 })
  const y = useSpring(rawY, { stiffness: 40, damping: 20 })
  const particles = useMemo(generateParticles, [])

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
      <div className={styles.orbA} />
      <div className={styles.orbB} />
      {/* Particle field — 40 SVG dots drifting slowly */}
      <svg className={styles.particleField} viewBox="0 0 100 100" preserveAspectRatio="none">
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
    </motion.div>
  )
}
