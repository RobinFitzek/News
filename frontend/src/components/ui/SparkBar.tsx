/**
 * SparkBar — Inline 8-bar micro-chart.
 *
 * Pure SVG, no dependencies. Shows recent data trend at a glance.
 * Most recent bar uses full signal color; older bars fade.
 */

interface SparkBarProps {
  data: number[]
  width?: number
  height?: number
  signal?: 'positive' | 'negative' | 'neutral'
  className?: string
}

const SIGNAL_COLORS = {
  positive: 'var(--glow-positive, #6BFF9E)',
  negative: 'var(--glow-negative, #FF6B6B)',
  neutral: 'var(--glow-neutral, #6BB8FF)',
}

export function SparkBar({
  data,
  width = 40,
  height = 16,
  signal = 'neutral',
  className,
}: SparkBarProps) {
  // Take last 8 values (or pad with 0)
  const bars = data.length >= 8
    ? data.slice(-8)
    : [...Array(8 - data.length).fill(0), ...data]

  const max = Math.max(...bars, 1) // avoid divide by zero
  const min = Math.min(...bars, 0)
  const range = max - min || 1

  const barCount = 8
  const gap = 1
  const barWidth = (width - gap * (barCount - 1)) / barCount
  const color = SIGNAL_COLORS[signal]

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      role="img"
      aria-label="Trend chart"
    >
      {bars.map((value, i) => {
        const normalizedHeight = ((value - min) / range) * (height - 2) + 1
        const x = i * (barWidth + gap)
        const y = height - normalizedHeight
        // Opacity fades: oldest (0) = 0.3, newest (7) = 1.0
        const opacity = 0.3 + (i / (barCount - 1)) * 0.7

        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barWidth}
            height={normalizedHeight}
            fill={color}
            opacity={opacity}
          />
        )
      })}
    </svg>
  )
}
