import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import styles from './PlaceholderPage.module.css'

interface PlaceholderPageProps {
  title: string
}

export function PlaceholderPage({ title }: PlaceholderPageProps) {
  return (
    <>
      <PageHeader title={title} subtitle="Coming soon" />
      <Card className={styles.card}>
        <div className={styles.empty}>
          <div className={styles.icon}>⬡</div>
          <p className={styles.text}>This page is being built as part of the React redesign.</p>
        </div>
      </Card>
    </>
  )
}
