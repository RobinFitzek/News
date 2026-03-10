"""
RSS-based real-time geopolitical event scanner.

Fetches RSS feeds from major wire services every 15 minutes and checks
headlines for trigger keywords.  When a match is found (and the 60-minute
cooldown has expired) the scheduler calls run_geopolitical_scan() immediately,
cutting geo-event latency from 6 h to ~15 min.

No extra dependencies — uses stdlib xml.etree.ElementTree + requests.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List

import requests

# Feeds are tried in order; first N successful entries win.
RSS_FEEDS: List[str] = [
    "https://feeds.reuters.com/Reuters/worldNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.ap.org/international-news",
]

# Lower-case trigger keywords — any match in title or description fires a scan.
GEO_KEYWORDS: List[str] = [
    "war", "invasion", "military strike", "sanctions", "coup",
    "blockade", "escalation", "opec", "oil embargo",
    "nuclear", "missile", "attack", "armed conflict", "crisis",
    "trade war", "tariffs imposed", "ceasefire", "peace treaty",
    "geopolit",  # catches geopolitical, geopolitics, etc.
]


class RssGeoScanner:
    """Stateful RSS scanner — remembers seen GUIDs and last trigger time."""

    def __init__(self):
        self._seen_guids: set[str] = set()
        self._last_trigger_at: datetime | None = None

    # ------------------------------------------------------------------
    # Feed fetching
    # ------------------------------------------------------------------

    def _fetch_items(self, url: str, timeout: int = 8) -> List[dict]:
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "StockholmMonitor/1.0 (+bot)"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = []
            for item in root.iter("item"):
                title = item.findtext("title", "") or ""
                desc = item.findtext("description", "") or ""
                guid = item.findtext("guid", "") or title
                items.append({"title": title, "desc": desc, "guid": guid})
            return items
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(self) -> List[str]:
        """
        Fetch all feeds and return a list of *new* matching headlines
        (headlines not seen on a previous call to scan()).
        """
        hits: List[str] = []
        for url in RSS_FEEDS:
            for item in self._fetch_items(url):
                if item["guid"] in self._seen_guids:
                    continue
                self._seen_guids.add(item["guid"])
                text = (item["title"] + " " + item["desc"]).lower()
                if any(kw in text for kw in GEO_KEYWORDS):
                    hits.append(item["title"])

        # Keep guid cache from growing unbounded
        if len(self._seen_guids) > 10_000:
            self._seen_guids = set(list(self._seen_guids)[-5_000:])

        return hits

    def should_trigger(self, hits: List[str], cooldown_minutes: int = 60) -> bool:
        """Return True if hits are non-empty and cooldown has elapsed."""
        if not hits:
            return False
        if self._last_trigger_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_trigger_at).total_seconds() / 60
        return elapsed >= cooldown_minutes

    def mark_triggered(self):
        """Record current time as last trigger (call after firing the scan)."""
        self._last_trigger_at = datetime.now(timezone.utc)


# Singleton used by scheduler
rss_geo_scanner = RssGeoScanner()
