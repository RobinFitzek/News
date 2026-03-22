import { useThemeStore } from '@/stores/themeStore'
import { Card } from '@/components/ui/Card'
import clsx from 'clsx'
import styles from './Panel.module.css'
import panelStyles from './PanelAppearance.module.css'

export function PanelAppearance() {
  const {
    theme, glowIntensity, depthEffects, showLoadingScreen, particleField,
    setTheme, setGlowIntensity, setDepthEffects, setShowLoadingScreen, setParticleField,
  } = useThemeStore()

  return (
    <div className={styles.panelContent}>
      <h2 className={styles.panelTitle}>Appearance</h2>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Theme</h3>
        <div className={panelStyles.themeOptions}>
          {(['dark', 'light', 'system'] as const).map(t => (
            <button
              key={t}
              className={clsx(panelStyles.themeOption, theme === t && panelStyles.active)}
              onClick={() => setTheme(t)}
            >
              <span className={panelStyles.themePreview} data-theme-preview={t} />
              <span className={panelStyles.themeLabel}>{t.charAt(0).toUpperCase() + t.slice(1)}</span>
            </button>
          ))}
        </div>
      </Card>

      <Card className={styles.section}>
        <h3 className={styles.sectionTitle}>Effects</h3>
        <div className={styles.form}>
          <div className={styles.field}>
            <div className={styles.fieldLabel}>Glow Intensity</div>
            <div className={styles.sliderRow}>
              <input
                type="range"
                min={0} max={1} step={0.05}
                value={glowIntensity}
                onChange={e => setGlowIntensity(Number(e.target.value))}
                className={styles.slider}
              />
              <span className={styles.sliderValue}>{Math.round(glowIntensity * 100)}%</span>
            </div>
          </div>

          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Depth Effects</div>
              <div className={styles.toggleSub}>Parallax, radiance, and shadow depth</div>
            </div>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={depthEffects}
                onChange={e => setDepthEffects(e.target.checked)}
              />
              <span className={styles.toggleTrack} />
              <span className={styles.toggleThumb} />
            </label>
          </div>

          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Particle Field</div>
              <div className={styles.toggleSub}>Ambient drifting particles in background</div>
            </div>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={particleField}
                onChange={e => setParticleField(e.target.checked)}
              />
              <span className={styles.toggleTrack} />
              <span className={styles.toggleThumb} />
            </label>
          </div>

          <div className={styles.toggleRow}>
            <div>
              <div className={styles.toggleLabel}>Loading Screen</div>
              <div className={styles.toggleSub}>Mercury diffusion intro on app start</div>
            </div>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={showLoadingScreen}
                onChange={e => setShowLoadingScreen(e.target.checked)}
              />
              <span className={styles.toggleTrack} />
              <span className={styles.toggleThumb} />
            </label>
          </div>
        </div>
      </Card>
    </div>
  )
}
