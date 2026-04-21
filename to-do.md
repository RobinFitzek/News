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

---

### [DB-04] Pagination für große Tabellen
**Datei**: `app.py` (diverse Endpoints)

Folgende Endpoints liefern aktuell **alle** Einträge ohne Limit zurück:
- `/api/analysis/{ticker}` — vollständige History
- `/api/alerts` — alle Alerts
- `/api/discovery/stats` — alle Discoveries

Mit wachsendem Datenbestand (1.000+ Rows) werden Ladezeiten >2s. Cursor-basierte Pagination mit `?limit=50&offset=0` einbauen.

---

### [DB-05] aiosqlite vollständig aktivieren
`aiosqlite` ist in den Requirements, aber Queries laufen synchron. Die async-Infrastruktur ist also schon vorhanden — die Queries müssen nur auf `await db.execute()` umgestellt werden. Gerade für den Scheduler ist das kritisch (siehe SCHED-02).

---

## ⏱️ SCHEDULER

### [SCHED-01] Job-Concurrency-Limit einführen
**Datei**: `scheduler.py` Zeilen 310-680

30+ Jobs werden ohne Concurrency-Kontrolle geplant. Bei großer Watchlist können alle gleichzeitig feuern und externe APIs (Perplexity, Gemini, yfinance) gleichzeitig treffen → Rate Limits, fehlerhafte Ergebnisse, gedrosselte Responses.

**Fix**: APScheduler-Executor auf max 3-5 parallele Jobs begrenzen:
```python
executors = {'default': ThreadPoolExecutor(max_workers=4)}
```

---

### [SCHED-02] yfinance-Calls asyncifizieren
**Problem**: yfinance ist synchrones I/O. 50 Ticker × ~200ms pro Call = ~10s Blockierzeit pro Scheduler-Zyklus, in dem kein anderer Job laufen kann.

**Fix**: `concurrent.futures.ThreadPoolExecutor` mit `max_workers=5` + `asyncio.gather()` für parallele Fetches. Alternativ `yfinance` durch einen async-fähigen HTTP-Client ersetzen (direkte Yahoo Finance API calls).

---

### [SCHED-03] Job-Idempotenz sicherstellen
**Datei**: `scheduler.py`

Bei Prozess-Neustart können Jobs doppelt ausgeführt werden → doppelte Alerts, doppelte Trades, kaputte Statistiken. Für jeden Job einen `last_executed_at`-Timestamp in der DB speichern und am Start prüfen, ob der Job in den letzten N Minuten bereits lief.

---

### [SCHED-04] Scheduler-Metriken sammeln
Aktuell keine Sichtbarkeit, wie lange welcher Job läuft. Eine Job-Execution-Tabelle (`job_name`, `started_at`, `finished_at`, `status`, `error`) würde Debugging massiv vereinfachen und Performance-Regression sichtbar machen.

---

## 🌐 BACKEND / API

### [API-01] Circuit Breaker für externe APIs
**Dateien**: `clients/perplexity_client.py`, `clients/gemini_client.py`

Wenn Perplexity down ist, schlagen alle 50 Ticker-Analysen nacheinander fehl — jeder wartet auf Timeout. Kein Graceful Degradation.

**Fix**: Circuit Breaker Pattern — nach 3 Fehlschlägen in 60s den Client für 5 Minuten "öffnen" und sofort mit Fallback-Daten antworten statt zu blockieren.

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

---

### [API-03] Pydantic-Validierung für alle Inputs
**Datei**: `app.py` (Form-Endpoints)

Ticker-Symbole, Mengen, Preise kommen als Strings rein ohne Schema-Validierung. Pydantic-Modelle für alle JSON/Form-Payloads definieren — das gibt gratis Fehler-Responses und verhindert Garbage-Input in der DB.

---

### [API-04] Request-Timeouts konfigurierbar machen
**Datei**: `core/database.py` Zeile 32

Aktuell hardcoded 10s DB-Timeout. Endpoints sollten konfigurierbare Timeouts haben, besonders für schwere Analyse-Queries. In `core/config.py` als Umgebungsvariable exponieren.

---

### [API-05] Response-Kompression aktivieren
Für größere JSON-Responses (Analyse-History, Discovery-Stats) fehlt gzip/brotli-Kompression. FastAPI unterstützt `GZipMiddleware` out-of-the-box:
```python
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## 🧠 CACHING

### [CACHE-01] In-Memory-Caches persistieren
**Problem**: Alle Caches (API-Fallback, Discovery-Universe, Sektor-Daten) sind In-Memory. Nach jedem Restart sind sie leer → die ersten 100 Requests sind langsam.

**Fix Option A (einfach)**: Caches beim Shutdown in JSON-Datei serialisieren, beim Start einlesen.  
**Fix Option B (robust)**: Redis mit 24h TTL für Universe-Daten, 4h für Analyse-Ergebnisse, 15min für yfinance-Daten.

---

### [CACHE-02] Analyse-Ergebnisse cachen
**Datei**: `engine/pipeline.py`

Wenn dasselbe Ticker in 5 Minuten zweimal analysiert wird, läuft die gesamte Pipeline zweimal durch (API-Calls, Berechnungen, DB-Writes). Ein Cache-Check am Pipeline-Eingang mit 4h TTL würde API-Kosten um 20-30% senken.

---

### [CACHE-03] S&P 500 Universe-Cache robuster machen
**Datei**: `engine/auto_discovery.py` Zeilen 53-652

24h-Cache existiert bereits — aber bei Fehler beim Laden wird kein Fallback verwendet. Wenn Wikipedia/yfinance für den Universe-Fetch nicht erreichbar ist, bricht Discovery komplett ab. Fallback auf zuletzt gecachten Stand einbauen.

---

## ⚛️ FRONTEND

### [FE-01] React Query staleTime anpassen
**Datei**: `frontend/src/api/queryClient.ts` Zeilen 3-14

30 Sekunden staleTime für **alle** Queries ist zu undifferenziert:
- Finanzdaten / Signale: 5-10s
- Watchlist-Metadata: 60s
- Statische Konfiguration (Settings): 300s

Endpoint-spezifische staleTime statt globaler Default.

---

### [FE-02] Teure Komponenten mit `React.memo` wrappen
**Verzeichnis**: `frontend/src/components/dashboard/`

Dashboard-Cards (SectorMomentumCard, Charts, PortfolioSummary) werden bei jedem Parent-Re-render neu gerendert, obwohl sich ihre Props nicht ändern. `React.memo()` + `useMemo()` für Chart-Daten-Berechnungen würden das eliminieren.

---

### [FE-03] Pagination / Infinite Scroll in History-Views
**Datei**: `frontend/src/api/endpoints/history.ts`

Gesamte Analyse-History wird ohne Paginierung geladen. Mit wachsender DB wird das zur Bremse. `useInfiniteQuery` von React Query mit Backend-Pagination kombinieren.

---

### [FE-04] Service Worker für Static Assets
Kein Service Worker vorhanden. Workbox würde JavaScript-Bundles, CSS und Icons für 7 Tage im Browser-Cache halten. Bei erneutem Aufruf: sofortiger Load, keine Wartezeit.

```bash
npm install workbox-webpack-plugin
```

---

### [FE-05] `backdrop-filter` Performance
**Datei**: `frontend/src/styles/` (Glass-Morphism-Effekte)

`backdrop-filter: blur(Xpx)` auf mehreren überlagerten Elementen ist GPU-intensiv. Auf schwacher Hardware (oder vielen geöffneten Tabs) kann das die UI ruckeln lassen.

**Fix**: `will-change: transform` nur auf animierten Elementen, `backdrop-filter` auf nicht-kritische Elemente beschränken oder alternative Card-Styles anbieten.

---

### [FE-06] Chunk-Size-Budget reduzieren
**Datei**: `frontend/vite.config.ts`

`chunkSizeWarningLimit: 800` ist fast doppelt so hoch wie üblich (400-500KB). Das bedeutet, dass zu große Chunks stillschweigend akzeptiert werden. Limit auf 500KB setzen und schauen, was dann warnt — dann gezielt optimieren.

---

## 🔒 SECURITY

### [SEC-01] CSP `unsafe-inline` entfernen
**Datei**: `app.py` Zeilen 101-108

```python
"script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net;"
```

`unsafe-inline` macht das komplette CSP für XSS-Schutz nutzlos. Nonce-basiertes CSP oder Hash-basiertes CSP einbauen. Langfristig: keine CDN-Abhängigkeiten für Scripts, sondern alles lokal bundlen.

---

### [SEC-02] Logging in allen `except`-Blöcken sicherstellen
**Dateien**: `engine/agents.py`, `engine/pipeline.py`, alle Engine-Module

452 try-Blöcke in der Engine, aber viele `except Exception: pass` ohne Logging. Fehler werden still geschluckt — in Production unmöglich zu debuggen.

**Fix**: Konvention einführen: jedes `except` muss mindestens `logger.exception()` aufrufen. Eine Grep-Rule als Pre-Commit-Hook würde das durchsetzen:
```bash
grep -rn "except.*:\s*$" engine/ | grep -v "logger\."
```

---

### [SEC-03] Brute-Force-Schutz für API-Keys
**Datei**: `core/database.py` (api_keys Tabelle)

Login-Failures werden getrackt (gut!), aber Rate-Limiting für API-Key-Validation fehlt. Wenn jemand `/settings/api-keys` mit falschen Keys spammt, keine Gegenwehr.

---

## 🧪 TESTS

### [TEST-01] Integration Tests für Scheduler-Jobs
Aktuell <20% Testabdeckung bei 100K+ LOC. Der Scheduler ist kritischer Pfad (Trades werden hier ausgelöst) — aber kein einziger Scheduler-Test. Mindestens smoke tests: "Startet der Scheduler, plant er Jobs, laufen sie durch ohne Exception?"

---

### [TEST-02] DB-Migration Tests
Wenn Schema-Änderungen gemacht werden, gibt es keinen Test, der sicherstellt, dass bestehende DBs migriert werden können ohne Datenverlust. Ein einfacher Test mit Fixture-DB würde Regressions verhindern.

---

### [TEST-03] Load Tests für kritische Endpoints
`/api/analysis/{ticker}` und `/watchlist` werden am häufigsten aufgerufen. Locust oder k6 Baseline-Tests würden Performance-Regressions beim Deployment erkennen.

---

## 📊 MONITORING & OBSERVABILITY

### [OBS-01] Job-Execution-Dashboard
Die Job-Execution-Tabelle aus [SCHED-04] kann ein einfaches Dashboard im Admin-Bereich speisen: welche Jobs laufen wie lange, welche schlagen fehl, Trend über Zeit. Kein externes Tool nötig — reines SQL + bestehende Chart.js-Infrastruktur.

---

### [OBS-02] API-Cost-Tracking granularer machen
`api_cost_log` existiert — aber wird der tatsächliche Token-Verbrauch pro Analyse-Run gespeichert? Granulares Tracking (Kosten pro Ticker, pro Engine-Modul) würde helfen, die teuersten Operationen zu identifizieren und selektiv zu optimieren.

---

### [OBS-03] Slow-Query-Log
SQLite unterstützt keine nativen Slow-Query-Logs, aber ein Decorator um `db.query()` könnte alle Queries >100ms mit Context (calling function, query text) loggen. In Development als Warning, in Production nur bei >500ms.

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
