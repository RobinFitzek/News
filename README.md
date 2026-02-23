# 🤖 AI Investment Monitor

Vollautomatisches Investment-Analyse-System für deinen Homeserver.

## ✨ Features

- **🔄 Automatische Scans** - Konfigurierbare Intervalle (z.B. alle 2 Stunden)
- **📊 Multi-Agent Analyse** - 4 KI-Agenten für News, Fundamental, Technical, Synthesis
- **📧 Email Alerts** - Benachrichtigung bei Strong Buy/Sell Signalen
- **🌐 Web Dashboard** - Einstellungen, Watchlist, Historie
- **💾 Persistente Daten** - SQLite Datenbank für alle Einstellungen
- **⚙️ Systemd Service** - Auto-Start beim Booten

---

## 🚀 Quick Start

```bash
# 1. Setup (einmalig)
./setup.sh

# 2. Starten
./start.sh
```

Dashboard öffnen: **http://localhost:8080**

---

## 📁 Projekt-Struktur

```
News/
├── main.py              # Hauptprogramm
├── scheduler.py         # APScheduler für automatische Scans
├── database.py          # SQLite Manager
├── notifications.py     # Email Benachrichtigungen
├── agents.py            # 4 AI-Agenten
├── app.py               # FastAPI Web Dashboard
├── perplexity_client.py # Perplexity API
├── gemini_client.py     # Gemini API (✨ modernisiert mit google.genai)
├── config.py            # Konfiguration
├── migrate_gemini.sh    # Migrations-Script für API-Update
├── templates/           # HTML Templates (Dark Theme)
├── data/                # SQLite Datenbank
├── logs/                # Log-Dateien
└── systemd/             # Service File
```

---

## 🔑 API Keys

### Benötigte Services

| Service | Kosten | Zweck | Link |
|---------|--------|-------|------|
| **Perplexity** | ~$5/Monat | News & Market Research | [perplexity.ai/settings](https://perplexity.ai/settings) |
| **Gemini** | Gratis | Fundamental & Technical Analysis | [ai.google.dev](https://ai.google.dev) |

### API Keys konfigurieren

**Über das Web Dashboard (empfohlen):**

1. **Dashboard öffnen:** `http://localhost:8080`
2. **Zu Settings navigieren:** Klick auf "⚙️ Einstellungen"
3. **API Keys eingeben:**
   - Perplexity API Key: `pplx-xxxxx...`
   - Gemini API Key: `AIzaSy...`
4. **Speichern:** Button "🔑 API Keys speichern" klicken
5. **Status prüfen:** ✅ zeigt erfolgreiche Konfiguration

**Sicherheit:**
- ✅ Keys werden verschlüsselt in lokaler SQLite-Datenbank gespeichert
- ✅ Keine Keys in Logs oder Code-Dateien
- ✅ Nur lokaler Zugriff auf die Datenbank
- ✅ Password-Input-Felder verbergen Keys im Browser

**Alternative: Manuelle Konfiguration (nicht empfohlen):**

API Keys können auch direkt in der Datenbank gesetzt werden, aber die Dashboard-Methode ist sicherer und einfacher.

---

## ⚙️ Einstellungen

Alles konfigurierbar über das Web Dashboard:

| Einstellung | Beschreibung |
|-------------|--------------|
| **Scan-Intervall** | Wie oft scannen (1-24 Stunden) |
| **Aktive Zeit** | Nur während bestimmter Uhrzeiten (z.B. 08:00-22:00) |
| **Email Alerts** | Bei Strong Buy/Sell benachrichtigen |
| **Tägliche Summary** | Zusammenfassung am Abend per Email |
| **Analyse-Tiefe** | News, Fundamental, Technical ein/aus |

---

## 🖥️ Systemd Auto-Start

Für automatischen Start beim Booten des Homeservers:

```bash
# Service installieren
sudo cp systemd/investment-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable investment-monitor
sudo systemctl start investment-monitor

# Status prüfen
sudo systemctl status investment-monitor

# Logs anzeigen
journalctl -u investment-monitor -f
```

---

## 📦 Dependencies

```
google-genai>=1.0.0      # Gemini API (neues Package)
crewai>=0.70.1          # AI Agent Framework
fastapi                  # Web Framework
uvicorn                  # ASGI Server
apscheduler              # Automatische Scans
yfinance                 # Aktien-Daten
python-dotenv            # Umgebungsvariablen
aiosmtplib              # Email Versand
aiosqlite               # Async SQLite
```

> ⚠️ **Wichtig**: Dieses Projekt verwendet das neue `google-genai` Package (v1.0+). 
> Das alte `google-generativeai` ist deprecated und wird nicht mehr unterstützt.

---

## 🔄 Migration von google.generativeai → google.genai

Falls du eine ältere Version dieses Projekts verwendest oder die Warnung siehst:

```
FutureWarning: All support for the `google.generativeai` package has ended.
```

**Option 1: Automatisches Migrations-Script (empfohlen)**

```bash
./migrate_gemini.sh
```

**Option 2: Manuelle Migration**

```bash
# 1. Virtual Environment aktivieren
source venv/bin/activate

# 2. Altes Package entfernen
pip uninstall google-generativeai -y

# 3. Neues Package installieren
pip install google-genai>=1.0.0

# 4. Dependencies aktualisieren
pip install -r requirements.txt --upgrade

# 5. Neustart
./start.sh
```

### Migration Details

**Das neue Package hat folgende Änderungen:**

| Alt (deprecated) | Neu (google.genai) |
|------------------|-------------------|
| `import google.generativeai as genai` | `from google import genai` |
| `genai.configure(api_key=...)` | `client = genai.Client(api_key=...)` |
| `genai.GenerativeModel(...)` | `client.models.generate_content(...)` |

> ✅ Die Migration wurde bereits in `gemini_client.py` implementiert.
> Die neue API ist stabiler und bietet bessere Error-Handling.

---

## 🔧 Manuelle Installation

```bash
# Virtual Environment erstellen
python3 -m venv venv
source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt

# Starten
python main.py
```

---

## � Troubleshooting

### Scheduler startet nicht

**Problem:** "⚠️ Scheduler nicht gestartet - API Keys fehlen"

**Lösung:**
1. Öffne Dashboard: `http://localhost:8080`
2. Gehe zu Settings
3. Prüfe ob beide API Keys konfiguriert sind (✅ Status)
4. Falls ❌ angezeigt wird: Keys neu eingeben und speichern

### API Fehler

**Gemini Fehler:**
```
⚠️ Gemini API nicht konfiguriert
```

**Lösung:**
- Stelle sicher, dass der neue `google-genai` package installiert ist
- Führe `./migrate_gemini.sh` aus
- Prüfe API Key auf [ai.google.dev](https://ai.google.dev)

**Perplexity Fehler:**
```
❌ Perplexity rate limit
```

**Lösung:**
- Warte 60 Sekunden (Rate-Limit-Reset)
- Prüfe dein Kontingent auf [perplexity.ai](https://perplexity.ai)
- Erhöhe Scan-Intervall in Settings

### Dashboard nicht erreichbar

**Problem:** `ERR_CONNECTION_REFUSED` auf Port 8080

**Lösung:**
```bash
# Prüfe ob Server läuft
ps aux | grep "python main.py"

# Neustart
./start.sh

# Falls Port belegt:
sudo lsof -i :8080
```

---

## �📜 Lizenz

Private Nutzung. Nicht für kommerzielle Zwecke.