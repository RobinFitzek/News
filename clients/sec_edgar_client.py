"""
SEC Edgar Client for Insider Trading Tracking
Fetches Form 4 filings (insider transactions) from SEC Edgar database
"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging
import re


class SECEdgarClient:
    """Client for fetching insider trading data from SEC Edgar"""

    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.headers = {
            'User-Agent': 'Investment Monitor System admin@investmentmonitor.local',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 0.15  # SEC requires 10 requests/second max

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a company ticker
        SEC uses CIK as the primary identifier
        """
        try:
            # Use SEC company tickers JSON endpoint
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                companies = response.json()

                # Search for ticker
                for company in companies.values():
                    if company.get('ticker', '').upper() == ticker.upper():
                        cik = str(company['cik_str']).zfill(10)  # Pad to 10 digits
                        self.logger.info(f"Found CIK {cik} for {ticker}")
                        return cik

                self.logger.warning(f"No CIK found for ticker {ticker}")
                return None
            else:
                self.logger.error(f"Failed to fetch company tickers: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting CIK for {ticker}: {e}")
            return None

    def get_insider_transactions(self, ticker: str, days_back: int = 180) -> List[Dict]:
        """
        Get recent insider transactions (Form 4 filings) for a ticker

        Args:
            ticker: Stock ticker symbol
            days_back: How many days of history to fetch (default 180)

        Returns:
            List of insider transaction dictionaries
        """
        self.logger.info(f"Fetching insider transactions for {ticker}")

        # Get CIK first
        cik = self.get_company_cik(ticker)
        if not cik:
            return []

        time.sleep(self.rate_limit_delay)

        try:
            # Fetch recent filings
            filings_url = f"{self.base_url}/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'CIK': cik,
                'type': '4',  # Form 4 = Insider transactions
                'dateb': '',
                'owner': 'include',
                'count': '100',
                'search_text': ''
            }

            response = requests.get(filings_url, params=params, headers=self.headers, timeout=15)

            if response.status_code != 200:
                self.logger.error(f"Failed to fetch filings: {response.status_code}")
                return []

            # Parse filings table
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'class': 'tableFile2'})

            if not table:
                self.logger.warning(f"No filings table found for {ticker}")
                return []

            transactions = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows[:20]:  # Limit to 20 most recent filings
                cols = row.find_all('td')
                if len(cols) < 5:
                    continue

                try:
                    # Extract filing date and document link
                    filing_date_str = cols[3].text.strip()
                    filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')

                    # Skip if too old
                    if filing_date < cutoff_date:
                        continue

                    # Get document link
                    doc_link = cols[1].find('a', {'id': 'documentsbutton'})
                    if not doc_link:
                        continue

                    doc_url = self.base_url + doc_link['href']

                    # Parse the Form 4 document
                    time.sleep(self.rate_limit_delay)
                    transaction = self._parse_form4(ticker, doc_url, filing_date)

                    if transaction:
                        transactions.append(transaction)

                except Exception as e:
                    self.logger.warning(f"Error parsing filing row: {e}")
                    continue

            self.logger.info(f"Found {len(transactions)} insider transactions for {ticker}")
            return transactions

        except Exception as e:
            self.logger.error(f"Error fetching insider transactions for {ticker}: {e}", exc_info=True)
            return []

    def _parse_form4(self, ticker: str, doc_url: str, filing_date: datetime) -> Optional[Dict]:
        """
        Parse a single Form 4 document to extract transaction details

        Form 4 structure varies, so we use heuristics to extract key info
        """
        try:
            # Get document page
            response = requests.get(doc_url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the actual XML file link (usually ends with .xml)
            xml_link = None
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if '.xml' in href and 'form4' in href.lower():
                    xml_link = self.base_url + href if not href.startswith('http') else href
                    break

            if not xml_link:
                # Fallback: Try to find any .xml link
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.endswith('.xml'):
                        xml_link = self.base_url + href if not href.startswith('http') else href
                        break

            if not xml_link:
                self.logger.warning(f"No XML link found in {doc_url}")
                return None

            # Fetch and parse XML
            time.sleep(self.rate_limit_delay)
            xml_response = requests.get(xml_link, headers=self.headers, timeout=15)

            if xml_response.status_code != 200:
                return None

            xml_soup = BeautifulSoup(xml_response.content, 'xml')

            # Extract insider info
            reporting_owner = xml_soup.find('reportingOwner')
            if not reporting_owner:
                return None

            insider_name = self._extract_text(reporting_owner, 'rptOwnerName')
            insider_title = self._extract_text(reporting_owner, 'officerTitle') or \
                           self._extract_text(reporting_owner, 'directorTitle') or \
                           'Insider'

            # Extract transaction details
            non_derivative_txn = xml_soup.find('nonDerivativeTransaction')

            if not non_derivative_txn:
                # Try derivative transaction (options)
                non_derivative_txn = xml_soup.find('derivativeTransaction')

            if not non_derivative_txn:
                return None

            # Transaction type (P = Purchase, S = Sale, A = Award, etc.)
            txn_code = self._extract_text(non_derivative_txn, 'transactionCode')
            txn_type = self._interpret_transaction_code(txn_code)

            # Shares and price
            shares = self._extract_number(non_derivative_txn, 'transactionShares') or \
                    self._extract_number(non_derivative_txn, 'sharesOwnedFollowingTransaction')

            price = self._extract_number(non_derivative_txn, 'transactionPricePerShare')

            # Transaction date
            txn_date_str = self._extract_text(non_derivative_txn, 'transactionDate')
            if txn_date_str:
                try:
                    txn_date = datetime.strptime(txn_date_str.split('T')[0], '%Y-%m-%d')
                except:
                    txn_date = filing_date
            else:
                txn_date = filing_date

            # Calculate value
            value = (shares * price) if shares and price else 0

            # Calculate significance score (0-100)
            significance = self._calculate_significance(
                txn_type=txn_type,
                value=value,
                title=insider_title,
                txn_code=txn_code,
            )

            return {
                'ticker': ticker,
                'insider_name': insider_name or 'Unknown',
                'title': insider_title,
                'transaction_date': txn_date.isoformat(),
                'filing_date': filing_date.isoformat(),
                'transaction_type': txn_type,
                'transaction_code': txn_code,
                'shares': shares or 0,
                'price': price or 0,
                'value': value,
                'significance_score': significance,
                'form4_url': xml_link
            }

        except Exception as e:
            self.logger.warning(f"Error parsing Form 4 from {doc_url}: {e}")
            return None

    def _extract_text(self, soup, tag_name: str) -> Optional[str]:
        """Extract text from XML tag"""
        tag = soup.find(tag_name)
        return tag.text.strip() if tag and tag.text else None

    def _extract_number(self, soup, tag_name: str) -> Optional[float]:
        """Extract numeric value from XML tag"""
        text = self._extract_text(soup, tag_name)
        if text:
            try:
                # Remove commas and convert
                return float(text.replace(',', ''))
            except ValueError:
                return None
        return None

    def _interpret_transaction_code(self, code: str) -> str:
        """
        Interpret SEC transaction code
        https://www.sec.gov/files/form4data.pdf
        """
        codes = {
            'P': 'Purchase',
            'S': 'Sale',
            'A': 'Award',
            'D': 'Disposition',
            'F': 'Payment',
            'I': 'Discretionary',
            'M': 'Exercise',
            'C': 'Conversion',
            'E': 'Expiration',
            'H': 'Held',
            'G': 'Gift',
            'L': 'Small Acquisition',
            'W': 'Will',
            'Z': 'Trust'
        }
        return codes.get(code, f'Other ({code})')

    def _calculate_significance(self, txn_type: str, value: float, title: str,
                                txn_code: str = None) -> int:
        """
        Calculate significance score (0-100) for an insider transaction.
        Voluntary open-market purchases (code 'P') are the strongest signal.
        Awards, exercises, and gifts are noise — scored low.
        """
        score = 30  # Base score

        # Transaction type weight — voluntary purchases are the real signal
        if txn_type == 'Purchase':
            score += 40  # Open market purchase = strongest bullish signal
        elif txn_type == 'Sale':
            score += 10  # Sales are common for compensation, weaker signal
        elif txn_type in ('Award', 'Gift'):
            score -= 15  # Compensation/gifts = noise, not conviction
        elif txn_type == 'Exercise':
            score -= 10  # Option exercises = noise, often automatic

        # Transaction value weight
        if value > 5_000_000:
            score += 20
        elif value > 1_000_000:
            score += 15
        elif value > 500_000:
            score += 10
        elif value > 100_000:
            score += 5

        # Title weight (CEO/CFO more significant than regular insiders)
        title_lower = title.lower()
        if any(x in title_lower for x in ['ceo', 'chief executive', 'president']):
            score += 15
        elif any(x in title_lower for x in ['cfo', 'chief financial']):
            score += 12
        elif any(x in title_lower for x in ['coo', 'chief operating', 'cto', 'chief technology']):
            score += 10
        elif 'director' in title_lower:
            score += 8
        elif any(x in title_lower for x in ['vp', 'vice president', 'svp', 'evp']):
            score += 5
        # 10% owners get lower weight — they may have different motivations
        elif '10%' in title_lower or 'owner' in title_lower:
            score += 2

        # Clamp to 0-100
        return max(0, min(100, score))

    def get_insider_summary(self, ticker: str, days_back: int = 90) -> Dict:
        """
        Get a summary of insider activity for a ticker

        Returns:
            Summary with net buying/selling, key transactions, and overall signal
        """
        transactions = self.get_insider_transactions(ticker, days_back)

        if not transactions:
            return {
                'ticker': ticker,
                'transactions_count': 0,
                'net_signal': 'NEUTRAL',
                'signal_score': 0,
                'summary': 'No recent insider activity'
            }

        # Calculate metrics
        purchases = [t for t in transactions if t['transaction_type'] == 'Purchase']
        sales = [t for t in transactions if t['transaction_type'] == 'Sale']

        total_buy_value = sum(t['value'] for t in purchases)
        total_sell_value = sum(t['value'] for t in sales)
        net_value = total_buy_value - total_sell_value

        # Calculate signal score (-100 to +100)
        if total_buy_value + total_sell_value > 0:
            signal_score = int((net_value / (total_buy_value + total_sell_value)) * 100)
        else:
            signal_score = 0

        # Determine signal
        if signal_score > 30:
            net_signal = 'BULLISH'
        elif signal_score > 10:
            net_signal = 'SLIGHTLY_BULLISH'
        elif signal_score < -30:
            net_signal = 'BEARISH'
        elif signal_score < -10:
            net_signal = 'SLIGHTLY_BEARISH'
        else:
            net_signal = 'NEUTRAL'

        # Find most significant transaction
        significant_txn = max(transactions, key=lambda x: x['significance_score'])

        return {
            'ticker': ticker,
            'days_analyzed': days_back,
            'transactions_count': len(transactions),
            'purchases_count': len(purchases),
            'sales_count': len(sales),
            'total_buy_value': total_buy_value,
            'total_sell_value': total_sell_value,
            'net_value': net_value,
            'net_signal': net_signal,
            'signal_score': signal_score,
            'most_significant': significant_txn,
            'recent_transactions': transactions[:5],  # Top 5 most recent
            'timestamp': datetime.now().isoformat()
        }


# Singleton
sec_client = SECEdgarClient()
