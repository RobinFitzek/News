import { motion, useMotionValue, useSpring } from 'framer-motion'
import { useEffect } from 'react'
import styles from './Luminary.module.css'

export function Luminary() {
  const rawX = useMotionValue(0)
  const rawY = useMotionValue(0)
  const x = useSpring(rawX, { stiffness: 40, damping: 20 })
  const y = useSpring(rawY, { stiffness: 40, damping: 20 })

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
    </motion.div>
  )
}
