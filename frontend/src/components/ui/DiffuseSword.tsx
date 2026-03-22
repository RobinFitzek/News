import { useState, useEffect, useRef } from 'react'
import styles from './DiffuseSword.module.css'

/*
 * ASCII sword centerpiece for the Mercury loading screen.
 * Emerges from noise line-by-line with a wave reveal.
 */

const SWORD_ART = [
  '          ◆',
  '         ╱ ╲',
  '        ╱   ╲',
  '       ╱  ◈  ╲',
  '      ╱       ╲',
  '     ╱    ◇    ╲',
  '    ╱           ╲',
  '   ╱      ⬡      ╲',
  '  ╱               ╲',
  '  ▕               ▏',
  '  ▕  ─ ─ ─ ─ ─   ▏',
  '   ╲             ╱',
  '    ╲           ╱',
  '     ║         ║',
  '     ║    │    ║',
  '     ║    │    ║',
  '     ║    │    ║',
  '     ╠════╪════╣',
  '     ║    │    ║',
  '      ╲   │   ╱',
  '       ╲  │  ╱',
  '        ╲ │ ╱',
  '         ╲│╱',
  '          ▼',
]

const NOISE_CHARS = '░▒▓█▄▀■□▪▫'

function noiseLine(len: number): string {
  return Array.from({ length: len }, () =>
    Math.random() > 0.4
      ? NOISE_CHARS[Math.floor(Math.random() * NOISE_CHARS.length)]
      : ' '
  ).join('')
}

interface DiffuseSwordProps {
  /** Delay before the sword begins revealing (ms) */
  delay?: number
  /** Time per line to diffuse (ms) */
  lineSpeed?: number
  className?: string
}

export function DiffuseSword({
  delay = 300,
  lineSpeed = 80,
  className,
}: DiffuseSwordProps) {
  const maxLen = Math.max(...SWORD_ART.map(l => l.length))
  const [lines, setLines] = useState<string[]>(() =>
    SWORD_ART.map(() => noiseLine(maxLen))
  )
  const [revealedCount, setRevealedCount] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>()

  useEffect(() => {
    const startTimer = setTimeout(() => {
      timerRef.current = setInterval(() => {
        setRevealedCount(prev => {
          const next = prev + 1
          if (next >= SWORD_ART.length) {
            clearInterval(timerRef.current)
          }
          return next
        })
      }, lineSpeed)
    }, delay)

    return () => {
      clearTimeout(startTimer)
      clearInterval(timerRef.current)
    }
  }, [delay, lineSpeed])

  // Update noise for unrevealed lines
  useEffect(() => {
    if (revealedCount >= SWORD_ART.length) return

    const noiseInterval = setInterval(() => {
      setLines(prev =>
        prev.map((_, i) =>
          i < revealedCount
            ? SWORD_ART[i].padEnd(maxLen)
            : noiseLine(maxLen)
        )
      )
    }, 60)

    return () => clearInterval(noiseInterval)
  }, [revealedCount, maxLen])

  // Final state — ensure revealed lines show the real art
  const displayed = lines.map((line, i) =>
    i < revealedCount ? SWORD_ART[i].padEnd(maxLen) : line
  )

  return (
    <pre className={`${styles.sword} ${className ?? ''}`} aria-hidden="true">
      {displayed.map((line, i) => (
        <span
          key={i}
          className={i < revealedCount ? styles.revealed : styles.noise}
        >
          {line}
          {'\n'}
        </span>
      ))}
    </pre>
  )
}
