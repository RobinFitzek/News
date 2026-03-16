import { motion } from 'framer-motion'
import { useRef } from 'react'
import { useInView } from 'framer-motion'
import clsx from 'clsx'
import styles from './Card.module.css'
import type { ReactNode, ElementType } from 'react'

interface CardProps {
  children: ReactNode
  className?: string
  glow?: 'positive' | 'negative' | 'neutral' | 'gold'
  glass?: boolean
  as?: ElementType
  animate?: boolean
  delay?: number
  style?: React.CSSProperties
}

const glowColors = {
  positive: 'rgba(107, 255, 158, 0.06)',
  negative: 'rgba(255, 107, 107, 0.06)',
  neutral: 'rgba(107, 184, 255, 0.06)',
  gold: 'rgba(255, 216, 122, 0.06)',
}

export function Card({
  children,
  className,
  glow,
  glass = false,
  as: Tag = 'div',
  animate = true,
  delay = 0,
  style,
}: CardProps) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-40px' })

  const cardStyle = glow
    ? { ...style, '--card-glow-color': glowColors[glow] } as React.CSSProperties
    : style

  return (
    <motion.div
      ref={ref}
      className={clsx(styles.card, glass && styles.glass, className)}
      style={cardStyle}
      initial={animate ? { opacity: 0, y: 16 } : false}
      animate={animate ? (isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 }) : false}
      transition={{
        duration: 0.45,
        ease: [0.34, 1.2, 0.64, 1],
        delay,
      }}
    >
      {/* Glow blob */}
      {glow && <div className={styles.glowBlob} />}
      {/* Specular highlight */}
      <div className={styles.specular} />
      <Tag className={styles.inner}>{children}</Tag>
    </motion.div>
  )
}
