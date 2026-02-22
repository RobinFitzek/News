"""
Institutional Holdings Tracker
Monitors institutional ownership via yfinance data and SEC EDGAR 13F filings.
Tracks major fund position changes and detects new positions by notable investors.
"""
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import db
import logging
import json

logger = logging.getLogger(__name__)

# SEC EDGAR requires a user-agent header
EDGAR_HEADERS = {
    'User-Agent': 'InvestmentMonitor/1.0 (research@example.com)',
    'Accept': 'application/xml, text/xml',
}

EDGAR_SEARCH_URL = 'https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms=13F-HR'


class InstitutionalTracker:
    """Track institutional holdings and SEC 13F filings."""

    def __init__(self):
        self._cache = {}
        self._cache_duration = timedelta(hours=6)
        self._filing_cache = {}
        self._filing_cache_duration = timedelta(hours=12)
        self._init_table()

    def _init_table(self):
        """Create institutional holdings table if it doesn't exist."""
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS institutional_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filer_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    shares INTEGER,
                    value REAL,
                    change_pct REAL,
                    filing_date TEXT,
                    source_url TEXT,
                    fetched_at TEXT,
                    UNIQUE(filer_name, ticker, filing_date)
                )
            """)
        except Exception as e:
            logger.error(f"Failed to create institutional_holdings table: {e}")

    def get_institutional_holders(self, ticker: str) -> Optional[Dict]:
        """
        Get institutional holder data from yfinance.
        Returns major holders summary and top institutional holders.
        """
        cache_key = f"inst_{ticker.upper()}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(ticker.upper())

            # Major holders summary (e.g., % held by institutions)
            major_holders = stock.major_holders
            major_summary = {}
            if major_holders is not None and not major_holders.empty:
                for _, row in major_holders.iterrows():
                    try:
                        value = row.iloc[0]
                        label = row.iloc[1] if len(row) > 1 else str(row.index[0])
                        major_summary[str(label)] = str(value)
                    except Exception:
                        continue

            # Top institutional holders
            inst_holders = stock.institutional_holders
            holders_list = []
            if inst_holders is not None and not inst_holders.empty:
                for _, row in inst_holders.iterrows():
                    try:
                        row_dict = dict(row)
                        holder = {
                            'holder': str(row_dict.get('Holder', row_dict.get('holder', 'Unknown'))),
                            'shares': int(row_dict.get('Shares', row_dict.get('shares', 0))),
                            'date_reported': str(row_dict.get('Date Reported', row_dict.get('dateReported', ''))),
                            'pct_out': float(row_dict.get('% Out', row_dict.get('pctHeld', 0))),
                            'value': float(row_dict.get('Value', row_dict.get('value', 0))),
                        }
                        holders_list.append(holder)
                    except Exception:
                        continue

                # Store top holders in DB for historical tracking
                self._store_holders(ticker.upper(), holders_list)

            data = {
                'ticker': ticker.upper(),
                'major_summary': major_summary,
                'top_holders': holders_list,
                'holder_count': len(holders_list),
                'fetched_at': datetime.now().isoformat(),
            }

            self._cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
            return data

        except Exception as e:
            logger.error(f"Error fetching institutional holders for {ticker}: {e}")
            return None

    def get_recent_filings(self, ticker: str, days: int = 90) -> List[Dict]:
        """Retrieve stored institutional filing records from the database."""
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            rows = db.query("""
                SELECT filer_name, ticker, shares, value, change_pct, filing_date, source_url, fetched_at
                FROM institutional_holdings
                WHERE ticker = ? AND filing_date >= ?
                ORDER BY filing_date DESC
            """, (ticker.upper(), cutoff))
            return [dict(r) for r in rows] if rows else []

        except Exception as e:
            logger.error(f"Error fetching recent filings for {ticker}: {e}")
            return []

    def search_13f_filings(self, ticker: str, days: int = 90) -> Optional[Dict]:
        """
        Search SEC EDGAR for recent 13F-HR filings mentioning the ticker.
        Parses the EDGAR full-text search API response.
        """
        cache_key = f"13f_{ticker.upper()}_{days}"
        if cache_key in self._filing_cache:
            entry = self._filing_cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._filing_cache_duration:
                return entry['data']

        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            url = EDGAR_SEARCH_URL.format(
                ticker=ticker.upper(),
                start=start_date,
                end=end_date,
            )

            logger.info(f"Searching EDGAR 13F filings for {ticker}: {url}")
            response = requests.get(url, headers=EDGAR_HEADERS, timeout=15)

            if response.status_code != 200:
                logger.warning(f"EDGAR search returned status {response.status_code} for {ticker}")
                return {
                    'ticker': ticker.upper(),
                    'filings': [],
                    'total_found': 0,
                    'error': f"HTTP {response.status_code}",
                    'search_url': url,
                }

            filings = self._parse_edgar_response(response.text, ticker.upper())

            # Store parsed filings in DB
            for filing in filings:
                self._store_filing(filing)

            data = {
                'ticker': ticker.upper(),
                'filings': filings,
                'total_found': len(filings),
                'search_period_days': days,
                'search_url': url,
                'fetched_at': datetime.now().isoformat(),
            }

            self._filing_cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
            return data

        except requests.exceptions.Timeout:
            logger.warning(f"EDGAR search timed out for {ticker}")
            return {'ticker': ticker.upper(), 'filings': [], 'total_found': 0, 'error': 'timeout'}

        except requests.exceptions.RequestException as e:
            logger.error(f"EDGAR request error for {ticker}: {e}")
            return {'ticker': ticker.upper(), 'filings': [], 'total_found': 0, 'error': str(e)}

        except Exception as e:
            logger.error(f"Error searching 13F filings for {ticker}: {e}")
            return None

    def _parse_edgar_response(self, response_text: str, ticker: str) -> List[Dict]:
        """Parse EDGAR search response (XML or JSON) into filing records."""
        filings = []

        try:
            # Try JSON parse first (EDGAR EFTS returns JSON)
            data = json.loads(response_text)
            hits = data.get('hits', {}).get('hits', [])

            for hit in hits:
                source = hit.get('_source', {})
                filing = {
                    'filer_name': source.get('display_names', [None])[0] if source.get('display_names') else source.get('entity_name', 'Unknown'),
                    'ticker': ticker,
                    'filing_date': source.get('file_date', source.get('period_of_report', '')),
                    'form_type': source.get('form_type', '13F-HR'),
                    'source_url': f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={source.get('entity_id', '')}&type=13F-HR",
                    'accession_number': source.get('accession_no', ''),
                }
                filings.append(filing)

            return filings

        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try XML parse
        try:
            root = ET.fromstring(response_text)
            # Handle potential namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('.//entry', ns) or root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')

            for entry in entries:
                title_el = entry.find('title', ns) or entry.find('title') or entry.find('{http://www.w3.org/2005/Atom}title')
                link_el = entry.find('link', ns) or entry.find('link') or entry.find('{http://www.w3.org/2005/Atom}link')
                updated_el = entry.find('updated', ns) or entry.find('updated') or entry.find('{http://www.w3.org/2005/Atom}updated')

                title = title_el.text if title_el is not None else 'Unknown'
                link = link_el.get('href', '') if link_el is not None else ''
                date = updated_el.text[:10] if updated_el is not None and updated_el.text else ''

                filing = {
                    'filer_name': title,
                    'ticker': ticker,
                    'filing_date': date,
                    'form_type': '13F-HR',
                    'source_url': link,
                    'accession_number': '',
                }
                filings.append(filing)

        except ET.ParseError as e:
            logger.warning(f"Could not parse EDGAR response as XML: {e}")

        return filings

    def _store_holders(self, ticker: str, holders: List[Dict]):
        """Store institutional holder data in the database."""
        now = datetime.now().isoformat()
        for holder in holders:
            try:
                db.execute("""
                    INSERT INTO institutional_holdings
                        (filer_name, ticker, shares, value, change_pct, filing_date, source_url, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(filer_name, ticker, filing_date) DO UPDATE SET
                        shares = excluded.shares,
                        value = excluded.value,
                        fetched_at = excluded.fetched_at
                """, (
                    holder.get('holder', 'Unknown'),
                    ticker,
                    holder.get('shares'),
                    holder.get('value'),
                    None,  # change_pct not available from yfinance directly
                    holder.get('date_reported', now[:10]),
                    'yfinance',
                    now,
                ))
            except Exception as e:
                logger.warning(f"Could not store holder {holder.get('holder')} for {ticker}: {e}")

    def _store_filing(self, filing: Dict):
        """Store a 13F filing record in the database."""
        try:
            db.execute("""
                INSERT INTO institutional_holdings
                    (filer_name, ticker, shares, value, change_pct, filing_date, source_url, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(filer_name, ticker, filing_date) DO UPDATE SET
                    source_url = excluded.source_url,
                    fetched_at = excluded.fetched_at
            """, (
                filing.get('filer_name', 'Unknown'),
                filing.get('ticker'),
                None,  # Shares not available from search index
                None,  # Value not available from search index
                None,
                filing.get('filing_date', ''),
                filing.get('source_url', ''),
                datetime.now().isoformat(),
            ))
        except Exception as e:
            logger.warning(f"Could not store filing for {filing.get('filer_name')}: {e}")

    def get_ownership_changes(self, ticker: str) -> Optional[Dict]:
        """
        Compare current holders with previously stored data to detect
        new positions, exits, and significant changes.
        """
        try:
            current = self.get_institutional_holders(ticker)
            if not current or not current.get('top_holders'):
                return None

            # Get previously stored holders (from >30 days ago)
            cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            previous = db.query("""
                SELECT filer_name, shares, value, filing_date
                FROM institutional_holdings
                WHERE ticker = ? AND filing_date < ?
                ORDER BY filing_date DESC
            """, (ticker.upper(), cutoff))

            previous_map = {}
            if previous:
                for row in previous:
                    row_dict = dict(row)
                    name = row_dict['filer_name']
                    if name not in previous_map:
                        previous_map[name] = row_dict

            new_positions = []
            increased = []
            decreased = []

            for holder in current['top_holders']:
                name = holder['holder']
                current_shares = holder.get('shares', 0)

                if name in previous_map:
                    prev_shares = previous_map[name].get('shares', 0)
                    if prev_shares and current_shares:
                        change = ((current_shares - prev_shares) / prev_shares) * 100
                        if change > 5:
                            increased.append({
                                'holder': name,
                                'previous_shares': prev_shares,
                                'current_shares': current_shares,
                                'change_pct': round(change, 1),
                            })
                        elif change < -5:
                            decreased.append({
                                'holder': name,
                                'previous_shares': prev_shares,
                                'current_shares': current_shares,
                                'change_pct': round(change, 1),
                            })
                else:
                    new_positions.append({
                        'holder': name,
                        'shares': current_shares,
                        'value': holder.get('value', 0),
                    })

            return {
                'ticker': ticker.upper(),
                'new_positions': new_positions,
                'increased': increased,
                'decreased': decreased,
                'total_changes': len(new_positions) + len(increased) + len(decreased),
                'analyzed_at': datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing ownership changes for {ticker}: {e}")
            return None


# Singleton
institutional_tracker = InstitutionalTracker()
