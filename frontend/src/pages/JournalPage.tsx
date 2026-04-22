import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import clsx from 'clsx'
import {
  useJournal,
  useAddJournalEntry,
  useCloseJournalEntry,
  useDeleteJournalEntry,
  useEditJournalEntry,
} from '@/api/endpoints/journal'
import type { JournalEntry, JournalEntryType, AddJournalEntryPayload } from '@/api/endpoints/journal'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { Modal } from '@/components/ui/Modal'
import { EmptyState } from '@/components/ui/EmptyState'
import { useToastStore } from '@/stores/toastStore'
import styles from './JournalPage.module.css'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-SE', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

type BadgeVariant = 'success' | 'danger' | 'neutral' | 'warning'

function typeBadgeVariant(type: JournalEntryType): BadgeVariant {
  switch (type) {
    case 'ENTRY': return 'success'
    case 'EXIT':  return 'danger'
    case 'NOTE':  return 'neutral'
    case 'ALERT': return 'warning'
  }
}

const ENTRY_TYPES: JournalEntryType[] = ['ENTRY', 'EXIT', 'NOTE', 'ALERT']

const DEFAULT_ADD_FORM: AddJournalEntryPayload = {
  ticker: '', type: 'NOTE', notes: '', price: null,
}

interface CloseForm { exit_price: string; notes: string }

// ── Component ─────────────────────────────────────────────────────────────────

export function JournalPage() {
  const { data, isLoading } = useJournal()
  const addMut = useAddJournalEntry()
  const closeMut = useCloseJournalEntry()
  const deleteMut = useDeleteJournalEntry()
  const editMut = useEditJournalEntry()
  const { addToast } = useToastStore()

  const [addForm, setAddForm] = useState<AddJournalEntryPayload>(DEFAULT_ADD_FORM)
  const [closeModal, setCloseModal] = useState<{ open: boolean; entry: JournalEntry | null }>({ open: false, entry: null })
  const [closeForm, setCloseForm] = useState<CloseForm>({ exit_price: '', notes: '' })

  // Edit modal
  const [editModal, setEditModal] = useState<{ open: boolean; entry: JournalEntry | null }>({ open: false, entry: null })
  const [editNotes, setEditNotes] = useState('')

  // Confirm delete dialog
  const [confirmDelete, setConfirmDelete] = useState<{ open: boolean; entry: JournalEntry | null }>({ open: false, entry: null })

  // Ticker / type / date filter
  const [search, setSearch] = useState('')
  const [typeFilter, setTypeFilter] = useState<JournalEntryType | 'ALL'>('ALL')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')

  const allEntries = data?.entries ?? []

  const entries = useMemo(() => {
    let list = [...allEntries]
    if (typeFilter !== 'ALL') list = list.filter(e => e.type === typeFilter)
    if (fromDate) list = list.filter(e => e.created_at.slice(0, 10) >= fromDate)
    if (toDate)   list = list.filter(e => e.created_at.slice(0, 10) <= toDate)
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter(e =>
        e.ticker.toLowerCase().includes(q) ||
        (e.notes ?? '').toLowerCase().includes(q)
      )
    }
    return list
  }, [allEntries, typeFilter, fromDate, toDate, search])

  function setAddField<K extends keyof AddJournalEntryPayload>(key: K, value: AddJournalEntryPayload[K]) {
    setAddForm(prev => ({ ...prev, [key]: value }))
  }

  async function handleAdd() {
    if (!addForm.ticker.trim()) { addToast('Ticker is required', 'warning'); return }
    if (!addForm.notes.trim())  { addToast('Notes are required', 'warning'); return }
    try {
      await addMut.mutateAsync({
        ...addForm,
        ticker: addForm.ticker.trim().toUpperCase(),
        price: addForm.price ? Number(addForm.price) : null,
      })
      addToast('Journal entry added', 'success')
      setAddForm(DEFAULT_ADD_FORM)
    } catch { addToast('Failed to add entry', 'error') }
  }

  function openCloseModal(entry: JournalEntry) {
    setCloseModal({ open: true, entry })
    setCloseForm({ exit_price: '', notes: '' })
  }

  async function handleClose() {
    if (!closeModal.entry) return
    try {
      await closeMut.mutateAsync({
        id: closeModal.entry.id,
        exit_price: closeForm.exit_price ? Number(closeForm.exit_price) : null,
        notes: closeForm.notes || undefined,
      })
      addToast('Entry closed', 'success')
      setCloseModal({ open: false, entry: null })
    } catch { addToast('Failed to close entry', 'error') }
  }

  function openEditModal(entry: JournalEntry) {
    setEditModal({ open: true, entry })
    setEditNotes(entry.notes ?? '')
  }

  async function handleEdit() {
    if (!editModal.entry) return
    try {
      await editMut.mutateAsync({ id: editModal.entry.id, notes: editNotes })
      addToast('Entry updated', 'success')
      setEditModal({ open: false, entry: null })
    } catch { addToast('Failed to update entry', 'error') }
  }

  function openConfirmDelete(entry: JournalEntry) {
    setConfirmDelete({ open: true, entry })
  }

  async function handleDelete() {
    if (!confirmDelete.entry) return
    try {
      await deleteMut.mutateAsync(confirmDelete.entry.id)
      addToast(`Entry for ${confirmDelete.entry.ticker} deleted`, 'info')
      setConfirmDelete({ open: false, entry: null })
    } catch { addToast('Failed to delete entry', 'error') }
  }

  return (
    <>
      <PageHeader title="Trade Journal" subtitle="Document your trading decisions" />

      {/* ── Add Entry ── */}
      <Card className={styles.addCard}>
        <p className={styles.addTitle}>New Entry</p>
        <div className={styles.formRow}>
          <div className={styles.formField}>
            <label className={styles.label}>Ticker</label>
            <input
              className={styles.input}
              placeholder="AAPL"
              value={addForm.ticker}
              onChange={e => setAddField('ticker', e.target.value.toUpperCase())}
              onKeyDown={e => e.key === 'Enter' && handleAdd()}
            />
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Type</label>
            <select className={styles.select} value={addForm.type} onChange={e => setAddField('type', e.target.value as JournalEntryType)}>
              {ENTRY_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className={styles.formField}>
            <label className={styles.label}>Price (optional)</label>
            <input
              className={styles.input}
              type="number" min="0" step="any" placeholder="150.00"
              value={addForm.price ?? ''}
              onChange={e => setAddField('price', e.target.value ? parseFloat(e.target.value) : null)}
            />
          </div>
        </div>
        <div className={styles.formRow}>
          <div className={clsx(styles.formField, styles.wide)}>
            <label className={styles.label}>Notes</label>
            <textarea
              className={styles.textarea}
              placeholder="Document your reasoning, observations, or alerts..."
              value={addForm.notes}
              onChange={e => setAddField('notes', e.target.value)}
            />
          </div>
        </div>
        <div className={styles.addActions}>
          <Button variant="primary" size="md" loading={addMut.isPending} onClick={handleAdd}
            disabled={!addForm.ticker.trim() || !addForm.notes.trim()}>
            Add Entry
          </Button>
        </div>
      </Card>

      {/* ── Filters ── */}
      <div className={styles.filterRow}>
        <input
          className={clsx(styles.input, styles.searchInput)}
          placeholder="Search ticker or notes…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <select className={styles.select} value={typeFilter} onChange={e => setTypeFilter(e.target.value as JournalEntryType | 'ALL')}>
          <option value="ALL">All Types</option>
          {ENTRY_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        <input
          type="date"
          className={clsx(styles.input, styles.dateInput)}
          value={fromDate}
          onChange={e => setFromDate(e.target.value)}
          title="From date"
        />
        <input
          type="date"
          className={clsx(styles.input, styles.dateInput)}
          value={toDate}
          onChange={e => setToDate(e.target.value)}
          title="To date"
        />
        {(search || typeFilter !== 'ALL' || fromDate || toDate) && (
          <Button variant="ghost" size="sm" onClick={() => { setSearch(''); setTypeFilter('ALL'); setFromDate(''); setToDate('') }}>
            Clear
          </Button>
        )}
        <span className={styles.filterCount}>
          {entries.length} / {allEntries.length} entries
        </span>
      </div>

      {/* ── Entries list ── */}
      {isLoading ? (
        <div className={styles.loading}><Spinner size="lg" /></div>
      ) : entries.length === 0 ? (
        <EmptyState
          message={search || typeFilter !== 'ALL' || fromDate || toDate ? 'No entries match your filter.' : 'No journal entries yet.'}
        />
      ) : (
        <div className={styles.entriesList}>
          {entries.map((entry, i) => (
            <motion.div key={entry.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04, duration: 0.3 }}>
              <Card>
                <div className={styles.entryCard}>
                  <div className={styles.entryHeader}>
                    <a href={`/stock/${entry.ticker}`} className={styles.entryTickerLink}>{entry.ticker}</a>
                    <Badge variant={typeBadgeVariant(entry.type)}>{entry.type}</Badge>
                    <span className={styles.entryDate}>{fmtDate(entry.created_at)}</span>
                  </div>

                  {entry.notes && <p className={styles.entryNotes}>{entry.notes}</p>}

                  <div className={styles.entryMeta}>
                    {entry.price !== null && <span className={styles.entryPrice}>Entry: ${entry.price.toFixed(2)}</span>}
                    {entry.exit_price !== null && <span className={styles.entryPrice}>Exit: ${entry.exit_price.toFixed(2)}</span>}
                    {entry.pnl !== null && (
                      <span className={clsx(styles.entryPnl, entry.pnl >= 0 ? styles.positive : styles.negative)}>
                        {entry.pnl >= 0 ? '+' : ''}${Math.abs(entry.pnl).toFixed(2)} P&L
                      </span>
                    )}
                    {entry.closed_at && <span className={styles.entryPrice}>Closed: {fmtDate(entry.closed_at)}</span>}
                  </div>

                  <div className={styles.entryFooter}>
                    {entry.type === 'ENTRY' && !entry.closed_at && (
                      <Button variant="secondary" size="sm" onClick={() => openCloseModal(entry)}>Close</Button>
                    )}
                    <Button variant="ghost" size="sm" onClick={() => openEditModal(entry)}>Edit</Button>
                    <span className={styles.spacer} />
                    <Button variant="ghost" size="sm" onClick={() => openConfirmDelete(entry)}>Delete</Button>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      {/* ── Close Entry Modal ── */}
      <Modal open={closeModal.open} onClose={() => setCloseModal({ open: false, entry: null })}
        title={`Close ${closeModal.entry?.ticker ?? ''} Entry`} size="sm">
        <div className={styles.closeModalFields}>
          <div>
            <label className={styles.label}>Exit Price (optional)</label>
            <input className={styles.input} type="number" min="0" step="any" placeholder="155.00"
              value={closeForm.exit_price}
              onChange={e => setCloseForm(prev => ({ ...prev, exit_price: e.target.value }))} />
          </div>
          <div>
            <label className={styles.label}>Notes (optional)</label>
            <textarea className={styles.textarea} placeholder="Exit reasoning..."
              value={closeForm.notes}
              onChange={e => setCloseForm(prev => ({ ...prev, notes: e.target.value }))} />
          </div>
        </div>
        <div className={styles.modalActions}>
          <Button variant="ghost" size="md" onClick={() => setCloseModal({ open: false, entry: null })}>Cancel</Button>
          <Button variant="primary" size="md" loading={closeMut.isPending} onClick={handleClose}>Close Entry</Button>
        </div>
      </Modal>

      {/* ── Edit Entry Modal ── */}
      <Modal open={editModal.open} onClose={() => setEditModal({ open: false, entry: null })}
        title={`Edit ${editModal.entry?.ticker ?? ''} Entry`} size="sm">
        <div>
          <label className={styles.label}>Notes</label>
          <textarea
            className={styles.textarea}
            placeholder="Update your notes..."
            value={editNotes}
            onChange={e => setEditNotes(e.target.value)}
            autoFocus
            onKeyDown={e => e.key === 'Enter' && e.ctrlKey && handleEdit()}
          />
          <p className={styles.editHint}>Ctrl+Enter to save</p>
        </div>
        <div className={styles.modalActions}>
          <Button variant="ghost" size="md" onClick={() => setEditModal({ open: false, entry: null })}>Cancel</Button>
          <Button variant="primary" size="md" loading={editMut.isPending} onClick={handleEdit}>Save</Button>
        </div>
      </Modal>

      {/* ── Confirm Delete Modal ── */}
      <Modal open={confirmDelete.open} onClose={() => setConfirmDelete({ open: false, entry: null })}
        title="Delete Journal Entry" size="sm">
        <p className={styles.confirmMessage}>
          Delete this {confirmDelete.entry?.type} entry for <strong>{confirmDelete.entry?.ticker}</strong>?
          This cannot be undone.
        </p>
        <div className={styles.modalActions}>
          <Button variant="ghost" size="md" onClick={() => setConfirmDelete({ open: false, entry: null })}>Cancel</Button>
          <Button variant="danger" size="md" loading={deleteMut.isPending} onClick={handleDelete}>Delete</Button>
        </div>
      </Modal>

      <div className="pageEnd" />
    </>
  )
}
