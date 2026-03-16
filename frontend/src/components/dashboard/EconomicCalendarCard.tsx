import { useMacroEvents } from '@/api/endpoints/macro'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { format } from 'date-fns'
import styles from './EconomicCalendarCard.module.css'

export function EconomicCalendarCard() {
  const { data, isLoading } = useMacroEvents(14)
  const events = data?.events ?? []

  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <h2 className={styles.title}>Economic Calendar</h2>
      </div>

      {isLoading && <div className={styles.loading}><Spinner /></div>}

      {events.length === 0 && !isLoading && (
        <p className={styles.empty}>No upcoming events</p>
      )}

      <div className={styles.list}>
        {events.slice(0, 8).map((ev, i) => (
          <div key={i} className={styles.item}>
            <div className={styles.dateCol}>
              <span className={styles.date}>
                {format(new Date(ev.date), 'MMM d')}
              </span>
              {ev.time && <span className={styles.time}>{ev.time}</span>}
            </div>
            <div className={styles.content}>
              <span className={styles.event}>{ev.event}</span>
              <span className={styles.country}>{ev.country}</span>
            </div>
            <Badge
              variant={ev.importance === 'high' ? 'danger' : ev.importance === 'medium' ? 'warning' : 'ghost'}
            >
              {ev.importance}
            </Badge>
          </div>
        ))}
      </div>
    </Card>
  )
}
