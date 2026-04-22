# UI Improvements — Backlog

> Fokus: Edit-Funktionen, UX-Sicherheit, Suche/Filter, Qualität

---

## ✅ BEREITS IMPLEMENTIERT (diese Session)

- **[UI-01]** Bestätigungs-Dialoge vor destruktiven Aktionen (Remove, Archive, Delete, Dismiss)
- **[UI-02]** Watchlist Suchleiste (Filter nach Ticker + Name, live)
- **[UI-03]** Tier-Auswahl im "Add"-Formular der Watchlist
- **[UI-04]** Inline Tier-Edit per Dropdown in der Tabelle (Backend-Endpoint + Frontend)
- **[UI-05]** Notiz-Vorschau direkt in der Tabellenspalte (statt nur Edit-Button)
- **[UI-06]** Journal Edit-Modal (Notizen editieren, Backend-Endpoint)
- **[UI-07]** Journal Ticker-Filter (Freitext-Suche in Einträgen)
- **[UI-08]** Discovery Bestätigungs-Dialog für Dismiss

## ✅ UI-Konsistenz (Branch: claude/improve-ui-consistency-ByuiX)

- **[UI-C01]** `EmptyState`-Komponente erstellt — einheitlicher leerer Zustand mit `message`, `hint`, `action`-Props
- **[UI-C02]** EmptyState in 8 Seiten eingesetzt: JournalPage, WatchlistPage, HistoryPage, GrahamPage, PoliticianTradesPage, CorporateActionsPage, DiscoveriesPage, TopPicksPage (ersetzt 5 verschiedene Muster)
- **[UI-C03]** `.pageEnd`-Utility-Klasse in globals.css — ersetzt 7x `style={{ height: 'var(--space-16)' }}`
- **[UI-C04]** Inline-Styles → CSS-Module: `signalBadge`-Klassen in HistoryPage, WatchlistPage, TopPicksPage; `grahamCell` + `upsidePositive/Negative` in WatchlistPage; `confirmMessage` + `modalActions` in DiscoveriesPage
- **[UI-C05]** Abschnittstitel semantisch korrigiert: `<span>` und `<p>` → `<h3>` in PortfolioPage (3×), `<div>` → `<h3>` in TopPicksPage (2×)
- **[UI-C06]** WatchlistPage Actions-Spalte: Spacer + `danger`-Variante für Remove-Button (analog zu JournalPage)
- **[UI-C07]** Animation-Token-Fixes: `0.2s`/`0.15s` → `var(--duration-fast)`/`var(--duration-instant)` in `Panel.module.css` (4 Stellen) und `PanelAppearance.module.css`

---

## 🔲 OFFEN — Mittel-Priorität

### [UI-09] Watchlist CSV-Export
**Was**: Button "Export CSV" der die aktuell gefilterte Watchlist als CSV herunterlädt  
**Aufwand**: ~1h, rein frontend (client-side CSV-Generierung aus geladenen Daten)  
**Datei**: `WatchlistPage.tsx`

### [UI-10] Portfolio Trade bearbeiten
**Was**: Edit-Button in der Trades-Tabelle öffnet Modal mit vorausgefüllten Feldern  
**Aufwand**: ~3h (Backend PATCH /api/portfolio/trade/{id} + Frontend Modal)  
**Dateien**: `PortfolioPage.tsx`, `app.py`

### [UI-11] Journal Datumsfilter
**Was**: Von/Bis Datumsauswahl um Einträge zeitlich einzugrenzen  
**Aufwand**: ~2h, Backend `?from=&to=` Query-Params + Frontend Datepicker  
**Dateien**: `JournalPage.tsx`, `app.py` (`/api/journal`)

### [UI-12] Skeleton Loader statt Spinner
**Was**: Statt eines mittig platzierten Spinners beim Laden: Platzhalter-Zeilen in Tabellenform  
**Aufwand**: ~3h, neue `SkeletonRow`-Komponente, in WatchlistPage + JournalPage + HistoryPage  
**Dateien**: Neue `src/components/ui/Skeleton.tsx`, je Page

### [UI-13] Watchlist Bulk-Tier-Zuweisung
**Was**: Checkboxes in der Tabelle → "Ausgewählte zu Core verschieben"-Dropdown  
**Aufwand**: ~3h, rein frontend + bestehender Tier-Endpoint  
**Datei**: `WatchlistPage.tsx`

---

## 🔲 OFFEN — Niedrig-Priorität / Aufwändig

### [UI-14] Pagination / Infinite Scroll für History + Logs
**Was**: Backend gibt max. 50 Einträge → "Mehr laden"-Button oder automatisches Nachladen  
**Aufwand**: ~1 Tag  
**Dateien**: `HistoryPage.tsx`, `LogsPage.tsx`, `app.py`

### [UI-15] Service Worker (Offline-Cache für Static Assets)
**Was**: `vite-plugin-pwa` cacht JS-Bundles und CSS für 7 Tage im Browser  
**Aufwand**: ~3h  
**Datei**: `vite.config.ts`, `frontend/`

### [UI-16] Mobile-Layout
**Was**: Sidebar kollabiert auf Mobile, Tabellen werden zu Cards, Navigation als Bottom Bar  
**Aufwand**: ~1-2 Tage

### [UI-17] ARIA-Accessibility
**Was**: `aria-label` auf allen Buttons, Focus-Management in Modals, `role="dialog"` auf Overlays  
**Aufwand**: ~1 Tag

### [UI-18] Fehlerseiten verbessern
**Was**: 404/500 Seiten mit Navigations-CTA statt leerem State  
**Aufwand**: ~1h

### [UI-19] Toast-Nachrichten mit Aktion
**Was**: Toast bei "Ticker entfernt" mit "Rückgängig"-Button (optimistic update zurückrollen)  
**Aufwand**: ~4h (optimistic update + undo logic)

### [UI-20] Stock-Link auf Ticker-Symbol
**Was**: Ticker-Spalte in Watchlist/Journal/History ist klickbar und führt zu `/stock/{ticker}`  
**Status**: Teilweise vorhanden in Watchlist (View-Button), aber kein direkter Ticker-Link

---

## 🔲 OFFEN — Konsistenz (nächste Iteration)

### [UI-C08] StockDetailPage Inline-Styles
**Was**: 30+ `style={{ ... }}` Inline-Styles in StockDetailPage.tsx; viele sind dynamisch (berechnet), aber statische sollten in CSS-Module migriert werden  
**Aufwand**: ~2h (groß, vorsichtig vorgehen — größte Datei mit 1200+ Zeilen)  
**Datei**: `StockDetailPage.tsx`, `StockDetailPage.module.css`

### [UI-C09] ScenariosPage + GeoHistoryPage Inline-Styles
**Was**: Farb-Inline-Styles für `impactColor()` und `severityColor()` Funktionen; könnten als CSS-Klassen umgesetzt werden  
**Aufwand**: ~1h  
**Dateien**: `ScenariosPage.tsx`, `GeoHistoryPage.tsx`

### [UI-C10] GrahamPage sectionTitle (buy_threshold)
**Was**: `<span className={styles.num} style={{ color: 'var(--signal-warning)' }}>` — einziger verbleibender Inline-Color-Style in GrahamPage  
**Aufwand**: 15min — neue `.numWarning`-Klasse in GrahamPage.module.css  
**Datei**: `GrahamPage.tsx`, `GrahamPage.module.css`

### [UI-C11] PoliticianTradesPage native Buttons → Button-Komponente
**Was**: PoliticianTradesPage nutzt native `<button className={styles.filterBtn}>` statt der `Button`-Komponente — inkonsistent mit allen anderen Seiten  
**Aufwand**: ~30min  
**Datei**: `PoliticianTradesPage.tsx`, `PoliticianTradesPage.module.css`

---

*Erstellt: 2026-04-21 | Branch: claude/optimize-existing-systems-q4kuS*  
*Konsistenz ergänzt: 2026-04-22 | Branch: claude/improve-ui-consistency-ByuiX*
