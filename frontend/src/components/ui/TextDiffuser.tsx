import { useState, useEffect, useCallback, useRef } from 'react'
import styles from './TextDiffuser.module.css'

const NOISE_CHARS = '░▒▓█▄▀■□▪▫●○◆◇◈◊⬡⬢⌬⏣⊡⊞'

interface TextDiffuserProps {
  /** The final text to reveal */
  text: string
  /** Duration of the full diffusion in ms */
  duration?: number
  /** Delay before starting in ms */
  delay?: number
  /** Callback when diffusion is complete */
  onComplete?: () => void
  className?: string
}

/**
 * Mercury diffusion text effect — noise characters resolve into real text.
 * Each character independently transitions from random glyphs to its final form.
 */
export function TextDiffuser({
  text,
  duration = 1200,
  delay = 0,
  onComplete,
  className,
}: TextDiffuserProps) {
  const [displayed, setDisplayed] = useState(() =>
    text.split('').map(() => NOISE_CHARS[Math.floor(Math.random() * NOISE_CHARS.length)])
  )
  const [started, setStarted] = useState(false)
  const [done, setDone] = useState(false)
  const frameRef = useRef<number>(0)
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete

  // Start after delay
  useEffect(() => {
    const timer = setTimeout(() => setStarted(true), delay)
    return () => clearTimeout(timer)
  }, [delay])

  const diffuse = useCallback(() => {
    if (!started) return

    const startTime = performance.now()
    const chars = text.split('')

    const step = (now: number) => {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)

      const next = chars.map((ch, i) => {
        if (ch === ' ') return ' '
        // Each character has its own reveal threshold based on position
        const charThreshold = (i / chars.length) * 0.7
        if (progress > charThreshold + 0.3) return ch
        if (progress > charThreshold) {
          // Flickering zone — mix noise with real char
          return Math.random() > (progress - charThreshold) / 0.3
            ? NOISE_CHARS[Math.floor(Math.random() * NOISE_CHARS.length)]
            : ch
        }
        return NOISE_CHARS[Math.floor(Math.random() * NOISE_CHARS.length)]
      })

      setDisplayed(next)

      if (progress < 1) {
        frameRef.current = requestAnimationFrame(step)
      } else {
        setDisplayed(chars)
        setDone(true)
        onCompleteRef.current?.()
      }
    }

    frameRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frameRef.current)
  }, [started, text, duration])

  useEffect(() => {
    const cleanup = diffuse()
    return cleanup
  }, [diffuse])

  return (
    <span
      className={`${styles.diffuser} ${done ? styles.done : ''} ${className ?? ''}`}
      aria-label={text}
    >
      {displayed.map((ch, i) => (
        <span
          key={i}
          className={ch === text[i] ? styles.resolved : styles.noise}
        >
          {ch}
        </span>
      ))}
    </span>
  )
}
