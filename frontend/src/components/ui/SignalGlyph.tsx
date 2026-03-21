/**
 * SignalGlyph — Crisp geometric SVG signal indicators.
 *
 * Four signal types with distinct visual language:
 * - BUY:   Sharp upward chevron (positive)
 * - SELL:  Sharp downward chevron (negative)
 * - HOLD:  Horizontal double-bar (pause/neutral)
 * - WATCH: Geometric eye (observing/gold)
 *
 * Uses <use href="#signal-*"> references to symbols defined
 * in RadianceProvider's shared <defs> block.
 */

import clsx from 'clsx'
import styles from './SignalGlyph.module.css'

export type SignalType = 'BUY' | 'SELL' | 'HOLD' | 'WATCH'

interface SignalGlyphProps {
  signal: SignalType
  size?: number
  className?: string
  label?: string
}

const SIGNAL_CONFIG: Record<SignalType, { href: string; colorVar: string; label: string }> = {
  BUY:   { href: '#signal-buy',   colorVar: 'var(--glow-positive)', label: 'Buy Signal' },
  SELL:  { href: '#signal-sell',  colorVar: 'var(--glow-negative)', label: 'Sell Signal' },
  HOLD:  { href: '#signal-hold',  colorVar: 'var(--glow-neutral)',  label: 'Hold Signal' },
  WATCH: { href: '#signal-watch', colorVar: 'var(--glow-gold)',     label: 'Watch Signal' },
}

export function SignalGlyph({ signal, size = 16, className, label }: SignalGlyphProps) {
  const config = SIGNAL_CONFIG[signal]

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      className={clsx(styles.glyph, styles[signal.toLowerCase()], className)}
      style={{ color: config.colorVar } as React.CSSProperties}
      role="img"
      aria-label={label || config.label}
    >
      <use href={config.href} />
    </svg>
  )
}
