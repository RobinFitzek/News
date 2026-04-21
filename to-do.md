# System-Optimierungen & Verbesserungen

> Kein neues Feature. Keine Ablenkung. Nur das, was bereits existiert — aber schneller, robuster, schöner.

---

## 🗄️ DATENBANK

### [DB-01] Index für `analysis_history` hinzufügen — QUICK WIN
**Datei**: `core/database.py` ~Zeile 616
```sql
CREATE INDEX IF NOT EXISTS idx_analysis_latest ON analysis_history(ticker, id DESC);
```
**Warum**: Jeder Watchlist-Load führt ein `ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY id DESC)` aus — ohne Index ist das ein Full Table Scan. Mit 50 Tickern und 10.000+ Einträgen kostet das 1-2s bei jedem Seitenaufruf.

**Unterpunkte**:
- Gleiche Query (Zeile 1455-1462) berechnet `julianday('now') - julianday(ah.timestamp)` auf jeder Row in Python-Land — als gespeichertes Feld oder CTe vorberechnen
- Fehlender Index auf `watchlist(is_active)` — `{active_filter}` WHERE-Clause filtert oft auf `is_active = 1`, kein Index vorhanden
- Fehlender Index auf `analysis_history(timestamp)` — Staleness-Queries (`ORDER BY timestamp DESC`) laufen ebenfalls ohne Index
- Alle 3 Indexes können in einem einzigen Schema-Migration-Block direkt nach Zeile 652 eingefügt werden

---

### [DB-02] N+1 Query-Pattern in Engine-Loops eliminieren
**Dateien**:
- `engine/auto_paper_trader.py` (Zeilen 445, 559, 584, 593, 604, 615, 619, 668)
- `engine/auto_discovery.py` (Zeilen 446-451, 679-691)
- `engine/dark_pool_tracker.py` (Zeile 205)
- `engine/pairs_trader.py` (Zeile 181)
- `engine/discovery_engine.py` (Zeilen 46, 57, 116, 171)

**Pattern**: Watchlist wird geholt, dann für jeden Ticker ein separates `db.query()` aufgerufen — 50 Ticker = 50 einzelne DB-Abfragen statt einer mit JOIN.

**Fix**: Batch-Queries mit `IN`-Klausel oder `LEFT JOIN` verwenden. Daten einmal laden, in Dict gruppieren, dann im Loop aus dem Dict lesen.

**Unterpunkte**:
- `auto_paper_trader.py:445` — `open_trades` wird geladen, dann iteriert der Loop und ruft für jeden Trade einzeln weitere Daten ab; ein einziger `SELECT * FROM auto_paper_trades JOIN watchlist ...` würde reichen
- `discovery_engine.py:46,57,116,171` — 4 separate `db.query()` Calls für dieselbe Pipeline-Session; alle in einer einzigen Funktion `get_discovery_context(tickers)` bündeln
- `dark_pool_tracker.py:205` — Query pro Ticker innerhalb einer `for ticker in tickers` Schleife; `WHERE ticker IN (...)` verwenden
- `pairs_trader.py:181` — Cointegration-Daten werden pro Paar einzeln geladen; gesamte Pairs-Tabelle einmalig laden und als Dict bereitstellen
- Allgemeines Pattern: nach dem Batch-Load `{ticker: row for row in results}` als Lookup-Dict bauen, dann `data.get(ticker, default)` im Loop

---

### [DB-03] Settings-Cache implementieren
**Datei**: `core/database.py` ~Zeile 1071

Aktuell wird bei jedem `get_setting()`-Aufruf `json.loads()` ausgeführt. Settings ändern sich selten — ein In-Memory-Cache mit 5-Minuten-TTL würde Dutzende JSON-Parse-Operationen pro Pipeline-Run sparen.

```python
_settings_cache: dict = {}
_settings_cache_ttl: float = 0

def get_all_settings(self):
    if time.time() < _settings_cache_ttl:
        return _settings_cache
    # ... fetch from DB, update cache
```

**Unterpunkte**:
- `get_all_settings()` wird **24x in `app.py`** und **39x in `engine/`** aufgerufen — bei jedem Request eine neue DB-Connection + JSON-Parse
- `pipeline.py:48` ruft `get_setting('system_paused_accuracy')` als allererstes im Hot-Path auf — bei jedem Scan-Zyklus
- `pipeline.py:107` ruft direkt danach `get_all_settings()` nochmal auf — zwei separate DB-Reads für dieselben Daten
- Cache-Invalidierung einfach: nur bei `set_setting()` den Cache leeren — das passiert selten (nur über Settings-UI)
- Alternativ: Settings einmalig beim Scheduler-Start laden und als Instanzvariable halten, bei `save_settings`-Event refreshen

---

### [DB-04] Pagination für große Tabellen
**Datei**: `app.py` (diverse Endpoints)

Folgende Endpoints liefern aktuell **alle** Einträge ohne Limit zurück:
- `/api/analysis/{ticker}` — vollständige History
- `/api/alerts` — alle Alerts
- `/api/discovery/stats` — alle Discoveries

Mit wachsendem Datenbestand (1.000+ Rows) werden Ladezeiten >2s. Cursor-basierte Pagination mit `?limit=50&offset=0` einbauen.

**Unterpunkte**:
- Alle 24 Page-Komponenten laden Daten ohne Backend-Limit — von `HistoryPage.tsx` bis `InsiderActivityPage.tsx`
- Einfachster Fix: generischen `LIMIT`-Default von 200 direkt in `database.py` als Parameter mit Default-Wert einführen, kein API-Umbau nötig
- `analysis_history` wächst pro Ticker und Scan: bei täglichem Scan und 50 Tickern = 1.750 neue Rows/Woche
- Frontend: `HistoryPage.tsx` und `LogsPage.tsx` sind die kritischsten, da sie vollständige chronologische Listen darstellen
- Für Exports (z.B. CSV-Download) separaten Endpoint ohne Limit behalten — aber UI-Endpoint immer begrenzen

---

### [DB-05] aiosqlite vollständig aktivieren
`aiosqlite` ist in den Requirements, aber Queries laufen synchron. Die async-Infrastruktur ist also schon vorhanden — die Queries müssen nur auf `await db.execute()` umgestellt werden. Gerade für den Scheduler ist das kritisch (siehe SCHED-02).

**Unterpunkte**:
- SQLite-PRAGMAs fehlen komplett für Lese-Performance: `PRAGMA cache_size = -50000` (50MB RAM-Cache), `PRAGMA temp_store = MEMORY`, `PRAGMA mmap_size = 268435456` (256MB memory-mapped I/O) — alle drei können ohne Schema-Änderung zu `_get_conn()` ab Zeile 44 hinzugefügt werden
- WAL-Mode (`PRAGMA journal_mode = WAL`) und `synchronous = NORMAL` sind bereits gesetzt — gute Basis
- Der async-Umbau ist das größte Einzelprojekt: alle 3.750 Zeilen `database.py` müssen auf `async def` umgestellt werden, aber der Gewinn ist dramatisch für den Scheduler

---

## ⏱️ SCHEDULER

### [SCHED-01] Job-Concurrency-Limit einführen
**Datei**: `scheduler.py` Zeilen 310-680

30+ Jobs werden ohne Concurrency-Kontrolle geplant. Bei großer Watchlist können alle gleichzeitig feuern und externe APIs (Perplexity, Gemini, yfinance) gleichzeitig treffen → Rate Limits, fehlerhafte Ergebnisse, gedrosselte Responses.

**Fix**: APScheduler-Executor auf max 3-5 parallele Jobs begrenzen:
```python
executors = {'default': ThreadPoolExecutor(max_workers=4)}
```

**Unterpunkte**:
- Sonntagabend-Problem: `weekly_analysis` (20:00), `weekly_report` (18:00) und `weekly_letter` (19:00) feuern alle innerhalb von 2 Stunden — drei heavy AI-Jobs konkurrieren um dieselben API-Budgets
- NLP Sentiment läuft stündlich 9-17 Uhr = 9 Jobs/Tag, deren Ergebnis aber nur alle paar Stunden nötig ist — auf 3x/Tag reduzieren reicht
- 101 `print()`-Statements in `scheduler.py` statt `self.logger.*` — kein strukturiertes Logging, kein Log-Level-Filtering möglich, Output geht in den Void wenn stdout nicht gecaptured wird
- APScheduler hat bereits `max_instances`-Parameter per Job — diesen für resource-heavy Jobs auf 1 setzen: `max_instances=1`

---

### [SCHED-02] yfinance-Calls asyncifizieren
**Problem**: yfinance ist synchrones I/O. 50 Ticker × ~200ms pro Call = ~10s Blockierzeit pro Scheduler-Zyklus, in dem kein anderer Job laufen kann.

**Fix**: `concurrent.futures.ThreadPoolExecutor` mit `max_workers=5` + `asyncio.gather()` für parallele Fetches. Alternativ `yfinance` durch einen async-fähigen HTTP-Client ersetzen (direkte Yahoo Finance API calls).

**Unterpunkte**:
- 132 yfinance-Calls über **10+ Engine-Module** verteilt: `ai_crosscheck.py`, `pattern_recognition.py`, `portfolio_manager.py`, `dividend_tracker.py`, `fear_greed_tracker.py`, `multi_timeframe.py`, `discovery_hit_rate.py`, `sentiment_analyzer.py`, `auto_discovery.py` u.a.
- Kein zentraler yfinance-Wrapper existiert — jedes Modul ruft `yf.Ticker(ticker).history(...)` direkt auf; ein Wrapper mit integriertem Thread-Pool würde alle 10 Module gleichzeitig fixen
- `portfolio_manager.py:171` macht sogar einen doppelten Import: `import yfinance as yf` steht auf Zeile 6 UND wird nochmal inline in einer Funktion importiert
- Quick Win ohne Async: `yf.download([ticker1, ticker2, ...])` statt einzelner `yf.Ticker(t).history()` Calls — yfinance unterstützt Batch-Downloads nativ

---

### [SCHED-03] Job-Idempotenz sicherstellen
**Datei**: `scheduler.py`

Bei Prozess-Neustart können Jobs doppelt ausgeführt werden → doppelte Alerts, doppelte Trades, kaputte Statistiken. Für jeden Job einen `last_executed_at`-Timestamp in der DB speichern und am Start prüfen, ob der Job in den letzten N Minuten bereits lief.

**Unterpunkte**:
- Alerts haben bereits eine `alert_hash`-Spalte mit `UNIQUE`-Constraint — gutes Vorbild für andere Jobs
- Für `auto_paper_trader`: Trade-Entry-Idempotenz über `(ticker, entry_date, signal)` als UNIQUE-Constraint absichern
- Einfachste Lösung: eine `job_locks`-Tabelle mit `job_id TEXT PRIMARY KEY, locked_at TIMESTAMP` — beim Job-Start `INSERT OR FAIL`, am Ende `DELETE`; Lock mit TTL (z.B. 2h) automatisch als abgelaufen behandeln

---

### [SCHED-04] Scheduler-Metriken sammeln
Aktuell keine Sichtbarkeit, wie lange welcher Job läuft. Eine Job-Execution-Tabelle (`job_name`, `started_at`, `finished_at`, `status`, `error`) würde Debugging massiv vereinfachen und Performance-Regression sichtbar machen.

**Unterpunkte**:
- 101 `print()`-Statements in `scheduler.py` sind der aktuelle "Log" — nicht searchable, kein Timestamp, kein Level
- APScheduler hat Events (`EVENT_JOB_EXECUTED`, `EVENT_JOB_ERROR`) — einen Listener registrieren und in DB schreiben ist 20 Zeilen Code
- `scheduler.py:320` startet den Scheduler mit `self.scheduler.start()` ohne jegliches Event-Listening — der perfekte Ort für einen `add_listener()`-Call
- Die Tabelle aus [SCHED-03] (`job_locks`) und diese Metrics-Tabelle können kombiniert werden: ein Eintrag pro Job-Run mit Status `running` → `completed` / `failed`

---

## 🌐 BACKEND / API

### [API-01] Circuit Breaker für externe APIs
**Dateien**: `clients/perplexity_client.py`, `clients/gemini_client.py`

Wenn Perplexity down ist, schlagen alle 50 Ticker-Analysen nacheinander fehl — jeder wartet auf Timeout. Kein Graceful Degradation.

**Fix**: Circuit Breaker Pattern — nach 3 Fehlschlägen in 60s den Client für 5 Minuten "öffnen" und sofort mit Fallback-Daten antworten statt zu blockieren.

**Unterpunkte**:
- **Perplexity** (`clients/perplexity_client.py`): Hat bereits `Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])` mit Connection-Pooling — das ist der gute Teil; Circuit Breaker darüber legen
- **Gemini** (`clients/gemini_client.py`): **Keine Retry-Logik**, kein Timeout, kein Backoff — bei einem `google.api_core.exceptions` Fehler propagiert die Exception ungebremst durch den ganzen Pipeline-Stack
- Perplexity-Timeout ist 30s (Zeile 131) — für 50 Ticker bei vollem Ausfall = 25 Minuten blockierter Scheduler; mit Circuit Breaker: sofortiger Fail nach 3 Versuchen
- Gemeinsame `BaseAPIClient`-Klasse mit Circuit-Breaker-State würde Code-Duplikation zwischen den Clients eliminieren

---

### [API-02] Cache-Control Headers auf API-Responses
**Datei**: `app.py` Zeilen 84-108

Security Headers sind gut konfiguriert, aber kein `Cache-Control`. Browser und CDN können dann nichts cachen.

```python
# Für stabile Daten:
response.headers["Cache-Control"] = "private, max-age=300"
# Für volatile Finanzdaten:
response.headers["Cache-Control"] = "no-store"
```

**Unterpunkte**:
- `app.py:84-108` setzt X-Frame-Options, X-Content-Type-Options, CSP, HSTS — aber kein einziges `Cache-Control`-Header
- Drei Klassen von Endpoints: (1) statisch — `/api/settings`, `/api/watchlist` → `max-age=300`, (2) Finanzdaten — `/api/analysis/{ticker}` → `max-age=60`, (3) Echtzeit — `/api/auto-trade/status` → `no-store`
- FastAPI-Middleware könnte das automatisch nach URL-Pattern setzen statt manuell per Endpoint
- `ETag`-Support würde 304-Responses ermöglichen — Browser sendet Hash, Server antwortet mit "nicht geändert" → Null Payload, nur Header-Round-Trip

---

### [API-03] Pydantic-Validierung für alle Inputs
**Datei**: `app.py` (Form-Endpoints)

Ticker-Symbole, Mengen, Preise kommen als Strings rein ohne Schema-Validierung. Pydantic-Modelle für alle JSON/Form-Payloads definieren — das gibt gratis Fehler-Responses und verhindert Garbage-Input in der DB.

**Unterpunkte**:
- FastAPI hat Pydantic bereits als Dependency — null Mehraufwand bei den Dependencies
- Kritischste Endpoints ohne Validation: `/watchlist/add` (ticker als freies String-Feld), `/api/orders/execute` (Menge + Preis ohne Range-Check), `/settings/save` (beliebige Key-Value-Paare)
- Ein `TickerSymbol`-Validator wäre projektübergreifend nützlich: `r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$'` — Ticker sind immer 1-5 Großbuchstaben, optional mit Exchange-Suffix
- Pydantic-Modelle dienen gleichzeitig als lebende API-Dokumentation (OpenAPI/Swagger wird automatisch generiert)

---

### [API-04] Request-Timeouts konfigurierbar machen
**Datei**: `core/database.py` Zeile 32

Aktuell hardcoded 10s DB-Timeout. Endpoints sollten konfigurierbare Timeouts haben, besonders für schwere Analyse-Queries. In `core/config.py` als Umgebungsvariable exponieren.

**Unterpunkte**:
- `core/database.py:32` — `_get_conn(self, timeout: float = 10.0)` ist der einzige Timeout-Punkt; alle Callers übergeben keinen Custom-Timeout
- Gemini-Client (`clients/gemini_client.py`) hat **gar keinen** `timeout`-Parameter — bei einem hängenden Request blockiert er unbegrenzt
- Perplexity-Client hat 30s Timeout (gut), aber dieser Wert ist ebenfalls hardcoded in Zeile 131
- Lösung: `DB_TIMEOUT=10`, `PERPLEXITY_TIMEOUT=30`, `GEMINI_TIMEOUT=45` in `core/config.py` als `os.getenv()` mit Defaults — dann in den Clients nutzen

---

### [API-05] Response-Kompression aktivieren
Für größere JSON-Responses (Analyse-History, Discovery-Stats) fehlt gzip/brotli-Kompression. FastAPI unterstützt `GZipMiddleware` out-of-the-box:
```python
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

**Unterpunkte**:
- 3 Zeilen Code, null Dependencies, sofort aktiv — der kleinste Aufwand mit dem größten sichtbaren Effekt
- `minimum_size=1000` bedeutet: Responses unter 1KB werden nicht komprimiert (sinnvoll, da Overhead sonst größer als Ersparnis)
- Analyse-Responses mit vollständiger History können 100KB+ JSON sein — gzip bringt typischerweise 60-80% Reduktion auf JSON
- Browser senden automatisch `Accept-Encoding: gzip` — kein Frontend-Code nötig

---

## 🧠 CACHING

### [CACHE-01] In-Memory-Caches persistieren
**Problem**: Alle Caches (API-Fallback, Discovery-Universe, Sektor-Daten) sind In-Memory. Nach jedem Restart sind sie leer → die ersten 100 Requests sind langsam.

**Fix Option A (einfach)**: Caches beim Shutdown in JSON-Datei serialisieren, beim Start einlesen.  
**Fix Option B (robust)**: Redis mit 24h TTL für Universe-Daten, 4h für Analyse-Ergebnisse, 15min für yfinance-Daten.

**Unterpunkte**:
- `engine/api_fallback.py:23-60` — yfinance-Fallback-Cache mit 15min TTL, reiner Dict im RAM; nach Restart: leer
- `engine/auto_discovery.py:53` — S&P 500 Universe-Cache, 24h TTL, ebenfalls nur RAM
- `engine/backtest_engine.py:32` — Ticker-Sektor-Cache als globales `dict`, wird **nie geleert** (Memory Leak bei langer Laufzeit mit vielen verschiedenen Tickern)
- Option A ist pragmatisch: `pickle.dump(cache, open('cache.pkl', 'wb'))` beim `SIGTERM`-Signal-Handler, `pickle.load()` beim Start — 10 Zeilen Code
- Der Backtest-Cache in Zeile 32 braucht zusätzlich eine maximale Größe (z.B. `maxsize=500` via `functools.lru_cache`) um den Memory Leak zu stoppen

---

### [CACHE-02] Analyse-Ergebnisse cachen
**Datei**: `engine/pipeline.py`

Wenn dasselbe Ticker in 5 Minuten zweimal analysiert wird, läuft die gesamte Pipeline zweimal durch (API-Calls, Berechnungen, DB-Writes). Ein Cache-Check am Pipeline-Eingang mit 4h TTL würde API-Kosten um 20-30% senken.

**Unterpunkte**:
- `pipeline.py:21-35` — `should_scan_ticker()` prüft bereits `last_scanned_at` gegen Tier-Frequenz (Tier1: immer, Tier2: >24h, Tier3: >72h) — die Logik existiert, wird aber nur im Scheduler-Kontext aufgerufen, nicht wenn Analyse manuell über UI getriggert wird
- Manueller Trigger über `/analyze` Endpoint bypassed diese Frequenz-Logik komplett — ein User kann denselben Ticker unbegrenzt oft analysieren und API-Kosten erzeugen
- Fix: `should_scan_ticker()` auch im `/analyze` Endpoint aufrufen; bei "zu frisch" Warnung anzeigen statt blindlings zu analysieren (mit Override-Option)
- `staleness_tracker.py` existiert bereits — dessen Daten für Cache-Entscheidung nutzen statt separaten Check

---

### [CACHE-03] S&P 500 Universe-Cache robuster machen
**Datei**: `engine/auto_discovery.py` Zeilen 53-652

24h-Cache existiert bereits — aber bei Fehler beim Laden wird kein Fallback verwendet. Wenn Wikipedia/yfinance für den Universe-Fetch nicht erreichbar ist, bricht Discovery komplett ab. Fallback auf zuletzt gecachten Stand einbauen.

**Unterpunkte**:
- `engine/auto_discovery.py:53-652` — Cache-Dict und Timestamp sind da, aber kein `try/except` um den Fetch mit Fallback auf den alten Cache-Stand
- S&P 500 Liste ändert sich ~4x pro Jahr — ein 7-Tage-Fallback-Cache wäre völlig ausreichend und würde Netzausfälle unsichtbar machen
- Fallback-Datei: einfach die zuletzt erfolgreiche Universe-Liste als `universe_fallback.json` abspeichern; bei Fetch-Fehler diese Datei laden
- Gleiches Muster für `discovery_engine.py` — wenn externe Datenquellen nicht erreichbar sind, gibt es keine defensive Antwort

---

## ⚛️ FRONTEND

### [FE-01] React Query staleTime anpassen
**Datei**: `frontend/src/api/queryClient.ts` Zeilen 3-14

30 Sekunden staleTime für **alle** Queries ist zu undifferenziert:
- Finanzdaten / Signale: 5-10s
- Watchlist-Metadata: 60s
- Statische Konfiguration (Settings): 300s

Endpoint-spezifische staleTime statt globaler Default.

**Unterpunkte**:
- `frontend/src/api/queryClient.ts:5` — `staleTime: 30_000` als einzige globale Einstellung; keine Query überschreibt das aktuell
- React Query erlaubt `staleTime` pro `useQuery()`-Call — alle 30 Endpoint-Dateien in `frontend/src/api/endpoints/` können individuell angepasst werden ohne Umbau
- Kritischste Kandidaten: Auto-Trade Status sollte `staleTime: 5_000` (Trades ändern sich live), Settings `staleTime: 300_000` (ändert User selten), Discovery-Stats `staleTime: 3_600_000` (einmal täglich)
- `refetchOnWindowFocus: false` ist sinnvoll für Finanzdaten — aber für den Auto-Trade-Status könnte ein `refetchInterval: 10_000` sinnvoller sein als nur beim manuellen Refresh

---

### [FE-02] Teure Komponenten mit `React.memo` wrappen
**Verzeichnis**: `frontend/src/components/dashboard/`

Dashboard-Cards (SectorMomentumCard, Charts, PortfolioSummary) werden bei jedem Parent-Re-render neu gerendert, obwohl sich ihre Props nicht ändern. `React.memo()` + `useMemo()` für Chart-Daten-Berechnungen würden das eliminieren.

**Unterpunkte**:
- **0 von 11** Dashboard-Komponenten nutzen `React.memo` oder `useMemo` — `AutoTradeCard`, `BenchmarkCard`, `EconomicCalendarCard`, `FearGreedDashCard`, `GeoRadarCard`, `GrahamDashCard`, `IntelStrip`, `LSTMSignalsDashCard`, `MarketRegimeCard`, `SectorMomentumCard`, `SystemCommandCenter` — alle re-rendern auf jeden Parent-Update
- framer-motion wird **161x in Pages** und **8x in Dashboard-Cards** eingesetzt — jede Animation triggert Re-renders in Sibling-Komponenten wenn State nicht isoliert ist
- `SectorMomentumCard` und `LSTMSignalsDashCard` dürften Chart-Berechnungen direkt im Render-Body haben — diese in `useMemo([data])` wrappen würde Chart.js-Recalculation auf tatsächliche Datenänderungen begrenzen
- Pragmatische Reihenfolge: erst `React.memo` auf alle 11 Cards, dann `useMemo` für Chart-Data-Transforms, dann gezielt `useCallback` für Event-Handler

---

### [FE-03] Pagination / Infinite Scroll in History-Views
**Datei**: `frontend/src/api/endpoints/history.ts`

Gesamte Analyse-History wird ohne Paginierung geladen. Mit wachsender DB wird das zur Bremse. `useInfiniteQuery` von React Query mit Backend-Pagination kombinieren.

**Unterpunkte**:
- 24 Page-Komponenten, alle ohne Pagination — `HistoryPage`, `LogsPage`, `InsiderActivityPage`, `PoliticianTradesPage`, `JournalPage` sind am kritischsten da sie tabellarische Listen darstellen
- `analysis_history` wächst mit ~1.750 Rows/Woche bei 50 Tickern täglichem Scan — nach 3 Monaten: 21.000 Rows, alle auf einmal geladen
- React Query `useInfiniteQuery` + `fetchNextPage` beim Scroll-ans-Ende ist das sauberste Pattern; alternativ einfache Seiten-Navigation mit `?page=1`
- `LogsPage` ist besonders kritisch — Logs wachsen am schnellsten und werden oft beim Debugging geöffnet genau dann wenn viel passiert ist

---

### [FE-04] Service Worker für Static Assets
Kein Service Worker vorhanden. Workbox würde JavaScript-Bundles, CSS und Icons für 7 Tage im Browser-Cache halten. Bei erneutem Aufruf: sofortiger Load, keine Wartezeit.

```bash
npm install workbox-webpack-plugin
```

**Unterpunkte**:
- Code-Splitting mit `React.lazy()` ist bereits komplett implementiert (`router/index.tsx:14+`) — alle 24 Pages werden lazy geladen; das ist die wichtigste Voraussetzung und ist schon erledigt
- Für Vite gibt es `vite-plugin-pwa` statt Workbox direkt — einfacher zu konfigurieren, generiert Service Worker automatisch
- Was gecacht werden sollte: JS-Chunks (lang, versioniert), CSS, Fonts, Icons (via `CacheFirst`-Strategie), aber NICHT API-Responses (via `NetworkFirst`)
- Für eine Single-User-App (lokale Installation) wäre Offline-Support ein realer Vorteil — Dashboard lesbar auch wenn Backend kurz down ist

---

### [FE-05] `backdrop-filter` Performance
**Datei**: `frontend/src/styles/` (Glass-Morphism-Effekte)

`backdrop-filter: blur(Xpx)` auf mehreren überlagerten Elementen ist GPU-intensiv. Auf schwacher Hardware (oder vielen geöffneten Tabs) kann das die UI ruckeln lassen.

**Fix**: `will-change: transform` nur auf animierten Elementen, `backdrop-filter` auf nicht-kritische Elemente beschränken oder alternative Card-Styles anbieten.

**Unterpunkte**:
- framer-motion macht **161 Animationen** in Page-Komponenten — nicht alle davon brauchen blur; nur Modals und Overlays rechtfertigen `backdrop-filter`
- `@media (prefers-reduced-motion: reduce)` CSS-Query: alle Animationen und blur-Effekte für Nutzer mit aktivierter Accessibility-Option deaktivieren — 1 CSS-Block, alle Komponenten profitieren automatisch
- `DiffuseSword.tsx` (existiert als UI-Komponente) — klingt nach einem dekorativen Effekt; prüfen ob es auf dem Dashboard dauerhaft aktiv ist
- GPU-Layer-Promotion: Elemente mit `backdrop-filter` erstellen automatisch einen neuen Compositor-Layer — zu viele davon überlasten die GPU; Blur-Radius auf maximal 8px begrenzen statt 20px+

---

### [FE-06] Chunk-Size-Budget reduzieren
**Datei**: `frontend/vite.config.ts`

`chunkSizeWarningLimit: 800` ist fast doppelt so hoch wie üblich (400-500KB). Das bedeutet, dass zu große Chunks stillschweigend akzeptiert werden. Limit auf 500KB setzen und schauen, was dann warnt — dann gezielt optimieren.

**Unterpunkte**:
- Aktuell 4 manuelle Chunks: `react-vendor`, `motion`, `query`, `charts` — das ist ein guter Start, aber alle anderen Imports landen im Default-Chunk
- `framer-motion` ist in eigenem Chunk (gut!) — aber `date-fns` (Datums-Utility) und `axios` landen zusammen mit allem anderen im Haupt-Bundle
- `chartjs-plugin-annotation` ist nur auf wenigen Seiten nötig — könnte dynamisch importiert werden: `const annotation = await import('chartjs-plugin-annotation')`
- `@tanstack/react-query-devtools` sollte **nie** im Production-Build sein — `devDependencies` ist richtig, aber prüfen ob es conditional importiert wird oder immer dabei ist

---

## 🔒 SECURITY

### [SEC-01] CSP `unsafe-inline` entfernen
**Datei**: `app.py` Zeilen 101-108

```python
"script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net;"
```

`unsafe-inline` macht das komplette CSP für XSS-Schutz nutzlos. Nonce-basiertes CSP oder Hash-basiertes CSP einbauen. Langfristig: keine CDN-Abhängigkeiten für Scripts, sondern alles lokal bundlen.

**Unterpunkte**:
- `app.py:101` — `"script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net;"` — das CDN ist eine externe Abhängigkeit; wenn jsdelivr kompromittiert wird, ist die App kompromittiert
- Da alle JS-Dateien durch Vite gebundelt werden, gibt es **keinen legitimen Grund** für `unsafe-inline` im Production-Build — Vite erzeugt keine Inline-Scripts
- Einfachster Fix: `unsafe-inline` entfernen und testen ob die App noch funktioniert; wahrscheinlich tut sie es, denn alle Scripts kommen aus `/static/react/`
- CDN-Abhängigkeit eliminieren: `https://cdn.jsdelivr.net` aus der Whitelist entfernen, alle externen Ressourcen lokal bundlen (Fonts, Icons falls vorhanden)

---

### [SEC-02] Logging in allen `except`-Blöcken sicherstellen
**Dateien**: `engine/agents.py`, `engine/pipeline.py`, alle Engine-Module

452 try-Blöcke in der Engine, aber viele `except Exception: pass` ohne Logging. Fehler werden still geschluckt — in Production unmöglich zu debuggen.

**Fix**: Konvention einführen: jedes `except` muss mindestens `logger.exception()` aufrufen. Eine Grep-Rule als Pre-Commit-Hook würde das durchsetzen:
```bash
grep -rn "except.*:\s*$" engine/ | grep -v "logger\."
```

**Unterpunkte**:
- **73 `print()`-Statements in `engine/`** statt `logger.*` — kein Log-Level, kein Timestamp, kein Modul-Context; beim Produktions-Betrieb als Service (systemd) geht stdout in den Void
- **101 `print()`-Statements in `scheduler.py`** — der kritischste Teil der Anwendung (Trades!) loggt komplett unkontrolliert
- **8 `print()`-Statements in `app.py`** — API-Layer loggt Fehler als print statt als strukturierte Logs
- `pipeline.py:29` — `except Exception:` ohne `as e` und ohne Logging — was genau fehlschlägt ist nie rekonstruierbar
- `ErrorBoundary.tsx` hat kein `componentDidCatch` — Frontend-Fehler werden dem User angezeigt aber nie ans Backend gemeldet; ein `componentDidCatch` mit `fetch('/api/client-error', {...})` würde Frontend-Exceptions in den Server-Logs sichtbar machen

---

### [SEC-03] Brute-Force-Schutz für API-Keys
**Datei**: `core/database.py` (api_keys Tabelle)

Login-Failures werden getrackt (gut!), aber Rate-Limiting für API-Key-Validation fehlt. Wenn jemand `/settings/api-keys` mit falschen Keys spammt, keine Gegenwehr.

**Unterpunkte**:
- Das Login-Brute-Force-System (`login_failures`-Tabelle, `idx_login_fail_ip_time`-Index) ist eine fertige Infrastruktur — dieselbe Logik auf alle sensitiven Endpoints übertragen
- `/settings/api-keys` — externe API-Keys werden gespeichert und entschlüsselt; kein Rate-Limit bedeutet, dass ein Angreifer mit Session beliebig oft Keys testen kann
- `/api/orders/execute` — Trade-Execution-Endpoint sollte ebenfalls Rate-Limiting haben: max N Trades pro Minute
- Einfachste Lösung: `slowapi` Library (FastAPI-kompatibel) mit `@limiter.limit("10/minute")` Decorator — minimal invasiv, nutzt die bereits vorhandene IP-Tracking-Logik

---

## 🧪 TESTS

### [TEST-01] Integration Tests für Scheduler-Jobs
Aktuell <20% Testabdeckung bei 100K+ LOC. Der Scheduler ist kritischer Pfad (Trades werden hier ausgelöst) — aber kein einziger Scheduler-Test. Mindestens smoke tests: "Startet der Scheduler, plant er Jobs, laufen sie durch ohne Exception?"

**Unterpunkte**:
- Höchste Priorität: Tests für `engine/pipeline.py` — hier werden Kauf/Verkauf-Signale generiert; ein Bug hier hat echte finanzielle Konsequenzen
- `auto_paper_trader.py` — Trade-Entry- und Exit-Logik ist ungetestet; Edge Cases wie "Trade öffnen wenn bereits offen" oder "Exit bei fehlendem Preis" können silent failures erzeugen
- Test-Infrastruktur: SQLite In-Memory-DB (`sqlite:///:memory:`) für schnelle, isolierte Tests ohne Dateisystem-Side-Effects
- Mock-Pattern: `yfinance` und externe API-Clients mit `unittest.mock.patch` mocken — Tests dürfen keine echten API-Calls machen
- Kritischer Smoke Test: Scheduler starten im Test-Modus, prüfen ob alle `add_job()`-Calls keine Exceptions werfen, sofort wieder stoppen

---

### [TEST-02] DB-Migration Tests
Wenn Schema-Änderungen gemacht werden, gibt es keinen Test, der sicherstellt, dass bestehende DBs migriert werden können ohne Datenverlust. Ein einfacher Test mit Fixture-DB würde Regressions verhindern.

**Unterpunkte**:
- `database.py` nutzt `ALTER TABLE ADD COLUMN IF NOT EXISTS` Pattern für Migrationen — das ist unkontrolliert und akkumuliert sich; nach 50 Features gibt es 50 einzelne ALTER TABLE Calls beim DB-Init
- Kein Versions-Tracking der DB-Schema-Version — es gibt keine Tabelle `schema_version` oder ähnliches; beim Downgrade auf ältere Code-Version ist unklar, welcher Schema-Stand vorliegt
- Fixture-DB: eine SQLite-Datei mit altem Schema in `tests/fixtures/` committen, Test prüft ob `_init_db()` darauf läuft ohne Exception und alle erwarteten Spalten danach vorhanden sind
- Langfristig: Alembic oder eine einfache custom `migrations/`-Ordnerstruktur mit nummerierten SQL-Dateien würde Migrationen nachvollziehbar machen

---

### [TEST-03] Load Tests für kritische Endpoints
`/api/analysis/{ticker}` und `/watchlist` werden am häufigsten aufgerufen. Locust oder k6 Baseline-Tests würden Performance-Regressions beim Deployment erkennen.

**Unterpunkte**:
- Einfachster Start: Python `timeit` oder `pytest-benchmark` für kritische DB-Queries — kein externer Load-Testing-Stack nötig
- Baseline-Metriken heute festhalten: mit `EXPLAIN QUERY PLAN` die aktuellen Query-Kosten in `analysis_history` dokumentieren, dann nach Index-Hinzufügen vergleichen
- `pytest-benchmark` würde Unit-Tests für `get_watchlist()` und `get_all_settings()` mit Millisekunden-Messungen ergänzen — Regression bei nächster Änderung sofort sichtbar
- k6 oder Locust für End-to-End: 10 parallele User laden `/watchlist` → Baseline dokumentieren → nach Optimierungen erneut messen

---

## 📊 MONITORING & OBSERVABILITY

### [OBS-01] Job-Execution-Dashboard
Die Job-Execution-Tabelle aus [SCHED-04] kann ein einfaches Dashboard im Admin-Bereich speisen: welche Jobs laufen wie lange, welche schlagen fehl, Trend über Zeit. Kein externes Tool nötig — reines SQL + bestehende Chart.js-Infrastruktur.

**Unterpunkte**:
- Das SystemCommandCenter (`SystemCommandCenter.tsx`) ist der perfekte Ort — es existiert bereits und zeigt System-Status; Job-Metriken passen dazu
- Visualisierung: Gantt-artige Darstellung (Start/End pro Job als horizontale Bar) mit Chart.js ist mit den vorhandenen Tools machbar
- Alerts: wenn ein Job `status = 'failed'` hat und der User die App öffnet, Toast-Notification zeigen — dafür ist das Zustand-Toast-System bereits vorhanden (`stores/`)
- Daten-Cleanup: Job-Execution-History nach 30 Tagen automatisch purgen — sonst wächst die Tabelle ewig

---

### [OBS-02] API-Cost-Tracking granularer machen
`api_cost_log` existiert — aber wird der tatsächliche Token-Verbrauch pro Analyse-Run gespeichert? Granulares Tracking (Kosten pro Ticker, pro Engine-Modul) würde helfen, die teuersten Operationen zu identifizieren und selektiv zu optimieren.

**Unterpunkte**:
- `api_cost_log` mit Indexes `idx_cost_api_month` und `idx_cost_api_date` ist vorhanden — gute Basis; aber granularer werden: Spalte `ticker TEXT` und `module TEXT` hinzufügen
- Mit Ticker-Granularität lassen sich "teure Ticker" identifizieren (z.B. internationale Aktien brauchen mehr AI-Tokens für Geo-Analyse)
- Mit Modul-Granularität: welche Engine-Module kosten wie viel? `agents.py` Stage 3 Synthesis ist teurer als Stage 1 Flash-Scan
- Budget-Tracker (`core/budget_tracker.py` existiert bereits) könnte diese Granularität direkt liefern wenn `log_cost()` um `ticker` und `module` Parameter erweitert wird

---

### [OBS-03] Slow-Query-Log
SQLite unterstützt keine nativen Slow-Query-Logs, aber ein Decorator um `db.query()` könnte alle Queries >100ms mit Context (calling function, query text) loggen. In Development als Warning, in Production nur bei >500ms.

**Unterpunkte**:
- `core/database.py` hat eine zentrale `execute()`-Methode — ein `@timing_decorator` oder `time.perf_counter()`-Wrapper dort würde alle Queries abfangen ohne jede einzelne Stelle anzufassen
- Python `inspect.stack()[1]` gibt den aufrufenden Funktionsnamen — damit kann der Log-Eintrag sagen "langsame Query aus `get_watchlist()` in `database.py:1442`"
- Threshold konfigurierbar machen: `SLOW_QUERY_MS=100` Env-Variable; in CI auf 50ms für frühzeitige Erkennung
- Die Slow-Query-Logs könnten in dieselbe `job_executions`-Tabelle aus [SCHED-04] geschrieben werden — zentrales Observability-Log statt verstreuter Log-Dateien

---

## 🏎️ PRIORITÄTS-MATRIX

| ID | Aufwand | Impact | Priorität |
|---|---|---|---|
| DB-01 | 15 Min | Hoch (1-2s schneller pro Seitenaufruf) | **P0** |
| API-05 | 30 Min | Mittel | **P0** |
| SCHED-01 | 1h | Hoch (Rate-Limit-Schutz) | **P0** |
| FE-01 | 30 Min | Mittel (frischere Finanzdaten) | **P0** |
| DB-03 | 2h | Mittel | **P1** |
| API-01 | 4h | Hoch (Resilience) | **P1** |
| DB-02 | 4h | Hoch (20+ → 1 DB Query) | **P1** |
| FE-02 | 2h | Mittel | **P1** |
| SCHED-02 | 1 Tag | Hoch (10s → 2s Scheduler-Zyklen) | **P1** |
| CACHE-01 | 1 Tag | Mittel | **P2** |
| CACHE-02 | 1 Tag | Hoch (API-Kosten -30%) | **P2** |
| FE-03 | 1 Tag | Mittel | **P2** |
| SEC-01 | 2h | Hoch (Security) | **P2** |
| DB-05 | 3 Tage | Hoch (async DB) | **P3** |
| CACHE-01 Redis | 3 Tage | Hoch (persistent cache) | **P3** |
| TEST-01-03 | 1 Woche | Mittel (long-term stability) | **P3** |

---

## 🚀 QUICK WINS — Sofort umsetzbar

Diese 4 Änderungen dauern zusammen ~2 Stunden und bringen messbare Verbesserungen:

1. **[DB-01]** Index hinzufügen → Watchlist lädt 1-2s schneller
2. **[API-05]** GZipMiddleware aktivieren → 3 Zeilen Code, 40-60% kleinere Responses
3. **[FE-01]** staleTime auf 10s für Analyse-Endpoints → Nutzer sehen aktuelle Signale
4. **[SCHED-01]** APScheduler max_workers=4 → keine simultanen API-Floods mehr

---

*Erstellt: 2026-04-21 | Branch: claude/optimize-existing-systems-q4kuS*
