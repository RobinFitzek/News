import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { TextDiffuser } from '@/components/ui/TextDiffuser'
import { DiffuseSword } from '@/components/ui/DiffuseSword'
import styles from './MercuryLoading.module.css'

interface MercuryLoadingProps {
  /** When true, begins the exit animation */
  ready?: boolean
  /** Called after exit animation completes */
  onDone?: () => void
}

/**
 * Mercury Diffusion — full-screen loading experience.
 * ASCII sword materialises from noise, title text diffuses in,
 * then the whole screen dissolves when content is ready.
 */
export function MercuryLoading({ ready = false, onDone }: MercuryLoadingProps) {
  const [titleDone, setTitleDone] = useState(false)

  return (
    <AnimatePresence onExitComplete={onDone}>
      {!ready && (
        <motion.div
          className={styles.overlay}
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, filter: 'blur(8px)' }}
          transition={{ duration: 0.6, ease: [0.34, 1.2, 0.64, 1] }}
          key="mercury"
        >
          {/* Ambient grain */}
          <div className={styles.grain} />

          <div className={styles.content}>
            {/* Sword centerpiece */}
            <DiffuseSword delay={200} lineSpeed={70} className={styles.sword} />

            {/* Title — diffuses in after sword starts */}
            <div className={styles.title}>
              <TextDiffuser
                text="STOCKHOLM"
                duration={1400}
                delay={600}
                onComplete={() => setTitleDone(true)}
              />
            </div>

            {/* Subtitle — appears after title resolves */}
            <div className={`${styles.subtitle} ${titleDone ? styles.visible : ''}`}>
              <TextDiffuser
                text="Investment Intelligence"
                duration={800}
                delay={titleDone ? 0 : 99999}
              />
            </div>

            {/* Breathing dot loader */}
            <div className={styles.dots}>
              <span className={styles.dot} style={{ animationDelay: '0s' }} />
              <span className={styles.dot} style={{ animationDelay: '0.15s' }} />
              <span className={styles.dot} style={{ animationDelay: '0.3s' }} />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
