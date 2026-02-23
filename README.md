# AI Investment Monitor

Ein vollautomatisches Investment-Analyse-System für den lokalen Betrieb auf einem Homeserver. 

## Features

- Automatische Scans: Konfigurierbare Intervalle (z.B. alle 2 Stunden).
- Multi-Agent Analyse: Integration von News, Fundamental- und technischer Analyse.
- Email Alerts: Benachrichtigungen bei Strong Buy/Sell Signalen.
- Web Dashboard: Verwaltung von Einstellungen, Watchlists und Einsicht in historische Daten.
- Persistente Daten: Speicherung aller Einstellungen und Historien in einer lokalen SQLite Datenbank.
- Systemd Service: Automatischer Start beim Booten.

---

## Quick Start

```bash
# 1. Einmaliges Setup
./setup.sh

# 2. System starten
./start.sh
```

Das Dashboard ist anschließend unter **http://localhost:8080** erreichbar.

---

## Projektstruktur

```text
News/
├── main.py              # Hauptprogramm
├── app.py               # FastAPI Web Dashboard
├── scheduler.py         # Automatisierungs-Logik
├── core/                # Kernkomponenten (Datenbank, Konfiguration, Notifications)
├── engine/              # Analyse-Engine (Algorithmen, Agenten)
├── clients/             # API-Clients (Perplexity, Gemini API)
├── templates/           # Web UI HTML-Templates
├── static/              # Web UI Assets (CSS, JS, Fonts)
├── data/                # Lokale SQLite Datenbank
├── logs/                # Log-Dateien
└── systemd/             # Systemd Service-Konfiguration
```

---

## Konfiguration der API Keys

Für die Analyse werden zwei externe Dienste benötigt:
- **Perplexity**: Für News & Market Research.
- **Gemini**: Für die Fundamental- und technische Analyse.

Die Konfiguration erfolgt am sichersten über das Web Dashboard:

1. Dashboard öffnen (`http://localhost:8080`).
2. In die Einstellungen navigieren.
3. Die jeweiligen API Keys in die dafür vorgesehenen Felder eintragen.
4. Speichern. Ein Status-Indikator zeigt den erfolgreichen Verbindungstest an.

Alle Keys werden lokal und verschlüsselt in der SQLite-Datenbank abgelegt. Es erfolgt keine Speicherung in Logdateien oder im Klartext.

---

## Systemeinstellungen

Das Verhalten des Monitors kann über das Web Dashboard angepasst werden:

- Scan-Intervall: Frequenz der automatischen Scans (1-24 Stunden).
- Aktive Zeit: Einschränkung der Scans auf bestimmte Uhrzeiten.
- Email Alerts: Aktivierung der Benachrichtigungen für Handelssignale.
- Tägliche Summary: Zusammenfassender Statusbericht per Email.
- Analyse-Tiefe: Aktivierung/Deaktivierung spezifischer Analyse-Komponenten (News, Fundamental, Technical).

---

## Systemd Auto-Start

Um das System beim Booten des Servers automatisch zu starten, kann der mitgelieferte Systemd-Service genutzt werden:

```bash
# Service installieren
sudo cp systemd/investment-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable investment-monitor
sudo systemctl start investment-monitor

# Status prüfen
sudo systemctl status investment-monitor

# Logs einsehen
journalctl -u investment-monitor -f
```

---

## Migration von google.generativeai zu google.genai

Das Projekt verwendet das aktuelle `google-genai` Package. Die Unterstützung für `google-generativeai` wurde eingestellt. Falls bei einem Update aus einer älteren Version Warnungen diesbezüglich auftreten:

**Migrations-Script (empfohlen):**
```bash
./migrate_gemini.sh
```

**Oder manuelle Migration:**
```bash
source venv/bin/activate
pip uninstall google-generativeai -y
pip install google-genai>=1.0.0
pip install -r requirements.txt --upgrade
./start.sh
```

---

## Manuelle Installation

Falls das Setup-Skript nicht verwendet werden soll:

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

## Troubleshooting

### Scheduler startet nicht
- **Ursache:** API Keys fehlen oder sind invalide.
- **Lösung:** Im Dashboard unter Einstellungen prüfen, ob beide Keys konfiguriert sind und der Status auf erfolgreich steht.

### API Fehler (Gemini)
- **Ursache:** Veraltetes Package oder Key ungültig.
- **Lösung:** Sicherstellen, dass das neue Paket (`google-genai`) installiert ist (siehe Migration).

### API Fehler (Perplexity)
- **Ursache:** Rate Limit erreicht.
- **Lösung:** Einige Sekunden warten. Ggf. das Scan-Intervall in den Einstellungen erhöhen oder das aufgebrauchte Kontingent prüfen.

### Dashboard nicht erreichbar (Port 8080)
- **Ursache:** Der Port ist belegt oder der Prozess ist nicht aktiv.
- **Lösung:** Über `ps aux | grep "python main.py"` prüfen ob der Server läuft. Falls der Port belegt ist, kann `sudo lsof -i :8080` Details liefern. Neustart über `./start.sh` versuchen.

---

## Lizenz

Private Nutzung. Nicht für kommerzielle Zwecke.