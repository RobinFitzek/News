"""
Corporate Actions Ledger (#43)
Fetches splits and dividends from yfinance, adjusts paper trading cost basis,
and credits dividends to the paper portfolio cash balance.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import yfinance as yf

from core.database import db

logger = logging.getLogger(__name__)


class CorporateActionsTracker:
    """Fetch and apply corporate actions to the paper portfolio."""

    def fetch_and_save(self, ticker: str) -> Dict:
        """
        Fetch splits and dividends for a ticker from yfinance and persist to DB.
        Returns summary dict with counts of new records saved.
        """
        ticker = ticker.upper()
        splits_saved = 0
        divs_saved = 0

        try:
            t = yf.Ticker(ticker)

            # --- Stock splits ---
            splits = t.splits
            if splits is not None and not splits.empty:
                for idx, ratio in splits.items():
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
                    if db.save_corporate_action(
                        ticker=ticker,
                        action_type='split',
                        action_date=date_str,
                        factor=float(ratio),
                        notes=f'{ratio}:1 split'
                    ):
                        splits_saved += 1

            # --- Cash dividends ---
            dividends = t.dividends
            if dividends is not None and not dividends.empty:
                for idx, amount in dividends.items():
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
                    if db.save_corporate_action(
                        ticker=ticker,
                        action_type='dividend',
                        action_date=date_str,
                        amount=float(amount),
                        notes=f'${amount:.4f}/share dividend'
                    ):
                        divs_saved += 1

        except Exception as e:
            logger.warning(f"Error fetching corporate actions for {ticker}: {e}")

        return {'ticker': ticker, 'splits': splits_saved, 'dividends': divs_saved}

    def refresh_all(self) -> Dict:
        """Refresh corporate actions for all watchlist tickers."""
        watchlist = db.get_watchlist(active_only=True)
        total = {'tickers': 0, 'splits': 0, 'dividends': 0}
        for item in watchlist:
            ticker = item['ticker']
            result = self.fetch_and_save(ticker)
            total['tickers'] += 1
            total['splits'] += result['splits']
            total['dividends'] += result['dividends']
            logger.debug(f"Corp actions {ticker}: +{result['splits']} splits, +{result['dividends']} divs")
        logger.info(f"Corporate actions refresh: {total}")
        return total

    def apply_splits_to_portfolio(self) -> int:
        """
        Retroactively adjust portfolio_trades cost basis for pre-split entries.
        For each split in corporate_actions, find any portfolio trades entered
        BEFORE the split date for that ticker and multiply shares / divide price
        by the split factor.
        Returns the number of trades adjusted.
        """
        adjusted = 0
        splits = db.query(
            "SELECT * FROM corporate_actions WHERE action_type = 'split' AND factor IS NOT NULL ORDER BY action_date"
        )
        for split in splits:
            ticker = split['ticker']
            split_date = split['action_date']
            factor = split['factor']
            if not factor or factor <= 0:
                continue
            # Find trades that pre-date the split and haven't been adjusted yet
            trades = db.query(
                "SELECT * FROM portfolio_trades WHERE ticker = ? AND date < ? AND (split_adjusted IS NULL OR split_adjusted = 0)",
                (ticker, split_date)
            )
            for trade in trades:
                new_shares = round(float(trade['amount']) * factor, 6)
                new_price = round(float(trade['price']) / factor, 6)
                try:
                    db.execute(
                        "UPDATE portfolio_trades SET amount = ?, price = ?, split_adjusted = 1 WHERE id = ?",
                        (new_shares, new_price, trade['id'])
                    )
                    adjusted += 1
                except Exception as e:
                    logger.warning(f"Could not adjust trade {trade['id']} for {ticker} split: {e}")
        return adjusted

    def credit_dividends_to_paper_portfolio(self) -> int:
        """
        For each dividend ex-date that has passed since the last credit run,
        find paper trading positions held on that date and add dividend income to cash.
        Returns the number of dividend credits applied.
        """
        credited = 0
        today = datetime.now().strftime('%Y-%m-%d')
        # Dividends whose ex-date has passed
        recent_divs = db.query(
            "SELECT * FROM corporate_actions WHERE action_type = 'dividend' AND action_date <= ? ORDER BY action_date",
            (today,)
        )
        for div in recent_divs:
            ticker = div['ticker']
            ex_date = div['action_date']
            amount_per_share = div.get('amount') or 0
            if amount_per_share <= 0:
                continue
            # Find open paper positions on the ex-date
            positions = db.query(
                """SELECT pt.*, SUM(
                    CASE WHEN pt2.type='BUY' THEN pt2.amount
                         WHEN pt2.type='SELL' THEN -pt2.amount ELSE 0 END
                ) as net_shares
                FROM portfolio_trades pt
                JOIN portfolio_trades pt2 ON pt2.ticker = pt.ticker AND pt2.date <= ?
                WHERE pt.ticker = ?
                GROUP BY pt.ticker
                HAVING net_shares > 0""",
                (ex_date, ticker)
            )
            for pos in positions:
                net_shares = pos.get('net_shares') or 0
                if net_shares <= 0:
                    continue
                dividend_income = round(net_shares * amount_per_share, 4)
                # Credit to paper portfolio as a special SELL entry (cash inflow)
                try:
                    db.execute(
                        """INSERT OR IGNORE INTO portfolio_trades
                           (ticker, type, amount, price, date, notes)
                           VALUES (?, 'DIVIDEND', ?, ?, ?, ?)""",
                        (ticker, net_shares, amount_per_share, ex_date, f'Dividend credit: ${dividend_income:.4f}')
                    )
                    credited += 1
                    logger.debug(f"Dividend credit: {ticker} {ex_date} ${dividend_income:.4f}")
                except Exception as e:
                    logger.warning(f"Dividend credit failed for {ticker} {ex_date}: {e}")
        return credited


corporate_actions_tracker = CorporateActionsTracker()
