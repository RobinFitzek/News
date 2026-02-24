"""
Auto-Discovery Engine
Automated stock discovery — runs on schedule, zero to minimal API cost.
Discovers new stock opportunities using 7 free yfinance-based strategies
and 1 weekly AI (Perplexity) strategy. Promotes winners to watchlist.
"""
import yfinance as yf
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AutoDiscovery:
    """Automated stock discovery — runs on schedule, zero to minimal API cost."""

    SCAN_UNIVERSE_SIZE = 200
    PROMOTION_THRESHOLD = 55
    MAX_DISCOVERIES_PER_RUN = 20
    MAX_AUTO_PROMOTE_PER_RUN = 5

    # S&P 500 representative tickers (top ~200 by market cap)
    # This is a static fallback; the system also pulls from SECTOR_PEERS
    SP500_CORE = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'LLY',
        'AVGO', 'JPM', 'TSLA', 'UNH', 'V', 'XOM', 'MA', 'JNJ', 'PG', 'HD',
        'COST', 'ABBV', 'MRK', 'CRM', 'BAC', 'AMD', 'NFLX', 'CVX', 'KO',
        'PEP', 'LIN', 'WMT', 'TMO', 'ADBE', 'ACN', 'MCD', 'CSCO', 'ABT',
        'DHR', 'WFC', 'GE', 'PM', 'TXN', 'QCOM', 'INTU', 'CMCSA', 'VZ',
        'IBM', 'INTC', 'AMGN', 'AMAT', 'HON', 'UNP', 'CAT', 'LOW', 'PFE',
        'BA', 'GS', 'BLK', 'DE', 'NOW', 'RTX', 'NEE', 'T', 'MS', 'ISRG',
        'SPGI', 'AXP', 'ELV', 'SYK', 'LMT', 'BKNG', 'MDLZ', 'PLD', 'SBUX',
        'GILD', 'ADP', 'ADI', 'CB', 'TJX', 'MMC', 'VRTX', 'SCHW', 'BMY',
        'LRCX', 'CI', 'CME', 'SO', 'DUK', 'MO', 'ZTS', 'CL', 'SLB', 'EOG',
        'NOC', 'BDX', 'ICE', 'COP', 'REGN', 'APD', 'SHW', 'FCX', 'NEM',
        'WELL', 'AMT', 'EQIX', 'SPG', 'O', 'PSA', 'DLR', 'NUE', 'DOW',
        'MPC', 'PSX', 'VLO', 'GIS', 'NKE', 'TMUS', 'DIS', 'ORCL', 'FI',
        'TGT', 'USB', 'PNC', 'COF', 'AIG', 'ALL', 'MET', 'PRU', 'TRV',
        'EMR', 'ETN', 'ITW', 'SRE', 'AEP', 'EXC', 'XEL', 'D', 'ES',
        'FDX', 'ROP', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'PANW',
        'ABNB', 'DXCM', 'IDXX', 'MNST', 'A', 'BIIB', 'MRNA', 'GEHC',
        'HUM', 'CNC', 'MCK', 'CAH', 'ABC', 'WAT', 'IQV', 'EW', 'BSX',
        'MDT', 'STZ', 'CPRT', 'ODFL', 'FAST', 'PAYX', 'CTAS', 'WM', 'RSG',
        'MSCI', 'NDAQ', 'CBOE', 'CMG', 'YUM', 'DPZ', 'ROST', 'ORLY', 'AZO',
        'KMB', 'HRL', 'SJM', 'GPC', 'CHRW', 'EXPD', 'JBHT', 'CSX', 'NSC',
    ]

    def __init__(self):
        self._universe_cache = None
        self._universe_cache_time = None
        self._universe_cache_duration = timedelta(hours=24)
        self._scan_offset = 0  # Rotates through universe

    def run_daily_discovery(self) -> Dict:
        """Run all free discovery strategies. Called by scheduler daily."""
        from core.database import db

        settings = db.get_all_settings()
        if not settings.get('discovery_enabled', True):
            logger.info("Auto-discovery is disabled in settings")
            return {'status': 'disabled'}

        start_time = time.time()
        enabled_strategies = settings.get('discovery_strategies', [
            'volume_spike', 'breakout', 'oversold',
            'sector_rotation', 'insider_buy', 'value_screen', 'mean_reversion'
        ])

        logger.info(f"Starting daily auto-discovery with strategies: {enabled_strategies}")
        print(f"\n{'='*50}")
        print(f"  AUTO-DISCOVERY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*50}")

        # 1. Build universe
        universe = self._build_universe()
        print(f"  Universe: {len(universe)} tickers")

        # 2. Get exclusion set
        excluded = self._get_exclusion_set()
        print(f"  Excluded: {len(excluded)} tickers (watchlist/graveyard/recent)")

        # 3. Run each strategy
        all_candidates = []
        strategies_run = []
        errors = []

        strategy_methods = {
            'volume_spike': self._scan_volume_spikes,
            'breakout': self._scan_breakouts,
            'oversold': self._scan_oversold,
            'sector_rotation': self._scan_sector_rotation,
            'insider_buy': self._scan_insider_buying,
            'value_screen': self._scan_value,
            'mean_reversion': self._scan_mean_reversion,
        }

        for strategy_name in enabled_strategies:
            method = strategy_methods.get(strategy_name)
            if not method:
                continue
            try:
                if strategy_name == 'sector_rotation':
                    candidates = method(excluded)
                elif strategy_name == 'insider_buy':
                    candidates = method(excluded)
                else:
                    candidates = method(universe, excluded)
                all_candidates.extend(candidates)
                strategies_run.append(strategy_name)
                print(f"  [{strategy_name}] Found {len(candidates)} candidates")
            except Exception as e:
                error_msg = f"{strategy_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Strategy {strategy_name} failed: {e}", exc_info=True)
                print(f"  [{strategy_name}] ERROR: {e}")

        # 4. Deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            ticker = c['ticker']
            if ticker not in seen:
                seen.add(ticker)
                unique_candidates.append(c)

        # Cap at max discoveries per run
        unique_candidates = unique_candidates[:self.MAX_DISCOVERIES_PER_RUN]

        # 5. Save to database
        saved_count = 0
        for c in unique_candidates:
            discovery_id = db.save_discovery(
                ticker=c['ticker'],
                signal_type=c['signal_type'],
                confidence=c.get('confidence', 50),
                price=c.get('price', 0),
                strategy=c.get('strategy', ''),
                sector=c.get('sector', ''),
                market_cap=c.get('market_cap'),
                source='auto'
            )
            if discovery_id:
                saved_count += 1

        print(f"  Saved {saved_count} new discoveries")

        # 6. Auto-screen and promote
        promoted = self._auto_promote(unique_candidates, settings)
        print(f"  Auto-promoted {len(promoted)} to watchlist")

        # 7. Log the run
        duration = time.time() - start_time
        db.log_discovery_run(
            run_type='daily_free',
            strategies=json.dumps(strategies_run),
            scanned=len(universe),
            found=saved_count,
            promoted=len(promoted),
            duration=duration,
            errors='; '.join(errors) if errors else ''
        )

        result = {
            'status': 'completed',
            'universe_size': len(universe),
            'excluded': len(excluded),
            'strategies_run': strategies_run,
            'discoveries': saved_count,
            'promoted': promoted,
            'duration_seconds': round(duration, 1),
            'errors': errors,
        }

        print(f"  Completed in {duration:.1f}s")
        print(f"{'='*50}\n")
        logger.info(f"Auto-discovery completed: {saved_count} found, {len(promoted)} promoted in {duration:.1f}s")

        return result

    def run_weekly_ai_discovery(self) -> Dict:
        """One Perplexity call per week for trending stocks."""
        from core.database import db
        from clients.custom_provider_client import custom_provider_client

        settings = db.get_all_settings()
        if not settings.get('discovery_enabled', True):
            return {'status': 'disabled'}

        start_time = time.time()
        logger.info("Starting weekly AI discovery")

        try:
            discovery_provider = db.get_api_provider_for_role('discovery')
            if discovery_provider:
                logger.info(f"Using custom discovery provider: {discovery_provider.get('name')}")
                custom_text = custom_provider_client.generate(
                    discovery_provider,
                    system_prompt=(
                        "You are a discovery engine for public stocks. Return only JSON array with objects: "
                        "ticker, confidence (0-100), sector, note."
                    ),
                    user_prompt=(
                        "Find up to 10 global large/mid-cap stocks with unusual momentum/catalyst potential this week. "
                        "Output strict JSON array only."
                    ),
                    temperature=0.2,
                    max_tokens=900,
                )

                stocks = []
                if custom_text:
                    raw = custom_text.strip()
                    if '```' in raw:
                        raw = raw.replace('```json', '').replace('```', '').strip()
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            stocks = parsed
                    except Exception:
                        stocks = []

                if stocks:
                    excluded = self._get_exclusion_set()
                    saved_count = 0
                    for stock in stocks:
                        ticker = (stock.get('ticker') or '').upper().strip()
                        if not ticker or ticker in excluded:
                            continue
                        if db.is_recently_discovered(ticker, days=14):
                            continue
                        db.save_discovery(
                            ticker=ticker,
                            signal_type='AI_TRENDING',
                            confidence=int(stock.get('confidence', 60) or 60),
                            price=stock.get('price', 0) or 0,
                            strategy='ai_custom_provider',
                            sector=stock.get('sector', ''),
                            source='ai'
                        )
                        saved_count += 1

                    duration = time.time() - start_time
                    db.log_discovery_run(
                        run_type='weekly_ai',
                        strategies=json.dumps(['ai_custom_provider']),
                        scanned=0,
                        found=saved_count,
                        promoted=0,
                        duration=duration
                    )

                    return {
                        'status': 'completed',
                        'discoveries': saved_count,
                        'duration_seconds': round(duration, 1),
                    }

            from clients.perplexity_client import pplx_client
            if not pplx_client.is_configured():
                logger.warning("Perplexity not configured, skipping AI discovery")
                return {'status': 'skipped', 'reason': 'perplexity_not_configured'}

            from engine.discovery_engine import discovery_engine
            result = discovery_engine.discover_with_perplexity(
                sector=None, focus='balanced', limit=10
            )

            if not result.get('success'):
                return {'status': 'error', 'error': result.get('error')}

            excluded = self._get_exclusion_set()
            saved_count = 0

            for stock in result.get('stocks', []):
                ticker = stock.get('ticker', '').upper()
                if not ticker or ticker in excluded:
                    continue
                if db.is_recently_discovered(ticker, days=14):
                    continue

                db.save_discovery(
                    ticker=ticker,
                    signal_type='AI_TRENDING',
                    confidence=stock.get('confidence', 60),
                    price=stock.get('price', 0),
                    strategy='ai_perplexity',
                    sector=stock.get('sector', ''),
                    source='ai'
                )
                saved_count += 1

            duration = time.time() - start_time
            db.log_discovery_run(
                run_type='weekly_ai',
                strategies=json.dumps(['ai_perplexity']),
                scanned=0,
                found=saved_count,
                promoted=0,
                duration=duration
            )

            logger.info(f"Weekly AI discovery: {saved_count} stocks found")
            return {
                'status': 'completed',
                'discoveries': saved_count,
                'duration_seconds': round(duration, 1),
            }

        except Exception as e:
            logger.error(f"Weekly AI discovery failed: {e}", exc_info=True)
            duration = time.time() - start_time
            db.log_discovery_run(
                run_type='weekly_ai',
                strategies=json.dumps(['ai_perplexity']),
                scanned=0, found=0, promoted=0,
                duration=duration,
                errors=str(e)
            )
            return {'status': 'error', 'error': str(e)}

    # --- Individual Strategies ---

    def _scan_volume_spikes(self, universe: List[str], excluded: Set[str]) -> List[Dict]:
        """Find stocks with >2x average volume in last session."""
        candidates = []
        subset = self._get_rotation_subset(universe, excluded, size=50)

        for ticker in subset:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1mo")
                if hist.empty or len(hist) < 5:
                    continue

                avg_vol = hist['Volume'].iloc[:-1].mean()
                latest_vol = hist['Volume'].iloc[-1]

                if avg_vol > 0 and latest_vol > 2 * avg_vol:
                    ratio = latest_vol / avg_vol
                    price = float(hist['Close'].iloc[-1])
                    info = stock.info
                    candidates.append({
                        'ticker': ticker,
                        'signal_type': 'VOLUME_SPIKE',
                        'confidence': min(90, int(50 + ratio * 10)),
                        'price': price,
                        'strategy': 'volume_spike',
                        'sector': info.get('sector', ''),
                        'market_cap': info.get('marketCap'),
                        'detail': f'{ratio:.1f}x avg volume',
                    })
            except Exception:
                continue

        return candidates

    def _scan_breakouts(self, universe: List[str], excluded: Set[str]) -> List[Dict]:
        """Find stocks breaking 52-week highs or golden cross."""
        candidates = []
        subset = self._get_rotation_subset(universe, excluded, size=50)

        for ticker in subset:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if hist.empty or len(hist) < 50:
                    continue

                close = hist['Close']
                current = float(close.iloc[-1])
                high_52w = float(close.max())

                # Near 52-week high (within 2%)
                near_high = current >= high_52w * 0.98

                # Golden cross check
                golden_cross = False
                if len(close) >= 200:
                    sma50 = close.rolling(50).mean()
                    sma200 = close.rolling(200).mean()
                    if len(sma50.dropna()) > 0 and len(sma200.dropna()) > 0:
                        current_diff = float(sma50.iloc[-1] - sma200.iloc[-1])
                        prev_diff = float(sma50.iloc[-5] - sma200.iloc[-5]) if len(sma50) > 5 else current_diff
                        golden_cross = current_diff > 0 and prev_diff <= 0

                if near_high or golden_cross:
                    info = stock.info
                    signal = 'BREAKOUT_HIGH' if near_high else 'GOLDEN_CROSS'
                    candidates.append({
                        'ticker': ticker,
                        'signal_type': signal,
                        'confidence': 70 if golden_cross else 65,
                        'price': current,
                        'strategy': 'breakout',
                        'sector': info.get('sector', ''),
                        'market_cap': info.get('marketCap'),
                    })
            except Exception:
                continue

        return candidates

    def _scan_oversold(self, universe: List[str], excluded: Set[str]) -> List[Dict]:
        """Find quality stocks with RSI < 30."""
        candidates = []
        subset = self._get_rotation_subset(universe, excluded, size=50)

        for ticker in subset:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="3mo")
                if hist.empty or len(hist) < 20:
                    continue

                close = hist['Close']
                rsi = self._compute_rsi(close, 14)
                if rsi is None or rsi >= 30:
                    continue

                # Quality filter: only include if fundamentals are decent
                info = stock.info
                de = info.get('debtToEquity')
                roe = info.get('returnOnEquity')

                if de is not None and de > 200:
                    continue  # Too much debt
                if roe is not None and roe < 0.05:
                    continue  # Poor returns

                current = float(close.iloc[-1])
                candidates.append({
                    'ticker': ticker,
                    'signal_type': 'RSI_OVERSOLD',
                    'confidence': min(85, int(80 - rsi)),
                    'price': current,
                    'strategy': 'oversold',
                    'sector': info.get('sector', ''),
                    'market_cap': info.get('marketCap'),
                    'detail': f'RSI={rsi:.1f}',
                })
            except Exception:
                continue

        return candidates

    def _scan_sector_rotation(self, excluded: Set[str]) -> List[Dict]:
        """Find stocks in sectors gaining momentum."""
        candidates = []

        try:
            from engine.sector_momentum import sector_momentum, SECTOR_ETFS, TICKER_SECTOR_MAP

            rankings = sector_momentum.get_sector_rankings()
            hot_sectors = [r['etf'] for r in rankings if r.get('momentum') == 'hot']

            if not hot_sectors:
                return []

            # Find tickers in hot sectors
            for ticker, etf in TICKER_SECTOR_MAP.items():
                if etf in hot_sectors and ticker not in excluded:
                    try:
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period="1mo")
                        if hist.empty:
                            continue

                        close = hist['Close']
                        if len(close) < 5:
                            continue

                        # Positive recent momentum
                        ret_1m = (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) * 100
                        if ret_1m <= 0:
                            continue

                        info = stock.info
                        candidates.append({
                            'ticker': ticker,
                            'signal_type': 'SECTOR_ROTATION',
                            'confidence': min(75, int(55 + ret_1m)),
                            'price': float(close.iloc[-1]),
                            'strategy': 'sector_rotation',
                            'sector': info.get('sector', SECTOR_ETFS.get(etf, {}).get('name', '')),
                            'market_cap': info.get('marketCap'),
                        })

                        if len(candidates) >= 10:
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Sector rotation scan error: {e}")

        return candidates

    def _scan_insider_buying(self, excluded: Set[str]) -> List[Dict]:
        """Find stocks with recent insider buying clusters."""
        candidates = []

        try:
            from core.database import db

            # Query insider_transactions for recent cluster buys
            conn = db._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ticker,
                       COUNT(*) as buy_count,
                       SUM(value) as total_value,
                       MAX(transaction_date) as latest_date
                FROM insider_transactions
                WHERE transaction_type = 'Purchase'
                  AND transaction_date >= date('now', '-30 days')
                  AND ticker NOT IN ({})
                GROUP BY ticker
                HAVING buy_count >= 2
                ORDER BY total_value DESC
                LIMIT 10
            """.format(','.join('?' * len(excluded)) if excluded else "'__none__'"),
                tuple(excluded) if excluded else ())
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                ticker = row['ticker']
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period="5d")
                    price = float(hist['Close'].iloc[-1]) if not hist.empty else 0

                    candidates.append({
                        'ticker': ticker,
                        'signal_type': 'INSIDER_BUY',
                        'confidence': min(80, int(55 + row['buy_count'] * 5)),
                        'price': price,
                        'strategy': 'insider_buy',
                        'sector': info.get('sector', ''),
                        'market_cap': info.get('marketCap'),
                        'detail': f"{row['buy_count']} buys, ${row['total_value']:,.0f}" if row['total_value'] else '',
                    })
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Insider buying scan error: {e}")

        return candidates

    def _scan_value(self, universe: List[str], excluded: Set[str]) -> List[Dict]:
        """Find undervalued stocks (low P/E, good quality)."""
        candidates = []
        subset = self._get_rotation_subset(universe, excluded, size=50)

        for ticker in subset:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                pe = info.get('trailingPE')
                roe = info.get('returnOnEquity')
                de = info.get('debtToEquity')

                # Value criteria
                if pe is None or pe <= 0 or pe > 20:
                    continue
                if roe is not None and roe < 0.12:
                    continue
                if de is not None and de > 150:
                    continue

                hist = stock.history(period="5d")
                price = float(hist['Close'].iloc[-1]) if not hist.empty else 0

                candidates.append({
                    'ticker': ticker,
                    'signal_type': 'VALUE_SCREEN',
                    'confidence': min(75, int(65 - pe + (roe or 0) * 50)),
                    'price': price,
                    'strategy': 'value_screen',
                    'sector': info.get('sector', ''),
                    'market_cap': info.get('marketCap'),
                    'detail': f'P/E={pe:.1f}, ROE={(roe or 0)*100:.0f}%',
                })
            except Exception:
                continue

        return candidates

    def _scan_mean_reversion(self, universe: List[str], excluded: Set[str]) -> List[Dict]:
        """Find stocks exhibiting mean reversion setups (extended oversold with no catalyst)."""
        candidates = []
        subset = self._get_rotation_subset(universe, excluded, size=50)

        for ticker in subset:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="3mo")
                if hist.empty or len(hist) < 30:
                    continue

                close = hist['Close']
                volume = hist['Volume']
                current = float(close.iloc[-1])

                # Basic filters: price > $5
                if current < 5:
                    continue

                # Compute RSI
                rsi = self._compute_rsi(close, 14)
                if rsi is None:
                    continue

                # Compute SMA20
                sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None

                # Compute Bollinger Band position
                bb_pos = None
                if len(close) >= 20:
                    roll_mean = close.rolling(20).mean()
                    roll_std = close.rolling(20).std()
                    upper = roll_mean + 2 * roll_std
                    lower = roll_mean - 2 * roll_std
                    low_val = float(lower.iloc[-1])
                    up_val = float(upper.iloc[-1])
                    band_range = up_val - low_val
                    if band_range > 0:
                        bb_pos = (current - low_val) / band_range

                # Weekly return
                weekly_ret = None
                if len(close) >= 5:
                    weekly_ret = (current - float(close.iloc[-5])) / float(close.iloc[-5]) * 100

                # Must satisfy at least 2 of the 3 criteria
                signals_met = 0
                signal_details = []

                # Criterion 1: RSI < 35 AND price > 10% below SMA20
                if rsi < 35 and sma20 is not None and current < sma20 * 0.90:
                    signals_met += 1
                    signal_details.append(f'RSI={rsi:.1f} + {((sma20 - current)/sma20*100):.1f}% below SMA20')

                # Criterion 2: Bollinger Band position < 0.10 with positive volume
                if bb_pos is not None and bb_pos < 0.10:
                    avg_vol = float(volume.iloc[:-1].mean())
                    latest_vol = float(volume.iloc[-1])
                    if avg_vol > 0 and latest_vol > 0:
                        signals_met += 1
                        signal_details.append(f'BB position={bb_pos:.2f}')

                # Criterion 3: Weekly return < -12% (no earnings check — if earnings data unavailable, skip)
                if weekly_ret is not None and weekly_ret < -12:
                    signals_met += 1
                    signal_details.append(f'Weekly={weekly_ret:.1f}%')

                if signals_met < 2:
                    continue

                # Fundamental filters
                try:
                    info = stock.info
                except Exception:
                    continue

                de = info.get('debtToEquity')
                market_cap = info.get('marketCap')

                if de is not None and de > 200:
                    continue
                if market_cap is not None and market_cap < 500_000_000:
                    continue

                confidence = min(85, int(55 + signals_met * 10 + max(0, 35 - rsi)))
                candidates.append({
                    'ticker': ticker,
                    'signal_type': 'MEAN_REVERSION',
                    'confidence': confidence,
                    'price': current,
                    'strategy': 'mean_reversion',
                    'sector': info.get('sector', ''),
                    'market_cap': market_cap,
                    'detail': ' | '.join(signal_details),
                })

            except Exception:
                continue

        return candidates

    # --- Utility Methods ---

    def _build_universe(self) -> List[str]:
        """Build scanning universe from S&P 500 + sector peers. Cached 24h."""
        if (self._universe_cache is not None and
                self._universe_cache_time is not None and
                datetime.now() - self._universe_cache_time < self._universe_cache_duration):
            return self._universe_cache

        universe = set(self.SP500_CORE)

        # Add sector peers from quant_screener
        try:
            from engine.quant_screener import QuantScreener
            for peers in QuantScreener.SectorCache.SECTOR_PEERS.values():
                universe.update(peers)
        except Exception:
            pass

        universe_list = sorted(universe)[:self.SCAN_UNIVERSE_SIZE]
        self._universe_cache = universe_list
        self._universe_cache_time = datetime.now()

        return universe_list

    def _get_exclusion_set(self) -> Set[str]:
        """Get set of tickers to exclude from discovery."""
        from core.database import db

        excluded = set()

        # Current watchlist
        try:
            watchlist = db.get_watchlist(active_only=True)
            excluded.update(w['ticker'] for w in watchlist)
        except Exception:
            pass

        # Graveyard
        try:
            graveyard = db.query("SELECT ticker FROM ticker_graveyard")
            excluded.update(g['ticker'] for g in graveyard)
        except Exception:
            pass

        # Recently dismissed (30 days)
        try:
            dismissed = db.query("""
                SELECT DISTINCT ticker FROM discovered_stocks
                WHERE status = 'dismissed'
                  AND dismissed_at >= datetime('now', '-30 days')
            """)
            excluded.update(d['ticker'] for d in dismissed)
        except Exception:
            pass

        # Recently discovered (avoid duplicates)
        try:
            recent = db.query("""
                SELECT DISTINCT ticker FROM discovered_stocks
                WHERE found_at >= datetime('now', '-7 days')
            """)
            excluded.update(r['ticker'] for r in recent)
        except Exception:
            pass

        return excluded

    def _get_rotation_subset(self, universe: List[str], excluded: Set[str],
                             size: int = 50) -> List[str]:
        """Get a rotating subset of the universe, excluding already-processed tickers."""
        available = [t for t in universe if t not in excluded]
        if not available:
            return []

        # Rotate through the universe across runs
        start = self._scan_offset % len(available)
        subset = available[start:start + size]
        if len(subset) < size:
            subset += available[:size - len(subset)]

        self._scan_offset = (self._scan_offset + size) % max(len(available), 1)
        return subset

    def _auto_promote(self, discoveries: List[Dict], settings: Dict) -> List[str]:
        """Screen discoveries with quant_screener, promote winners to watchlist."""
        from core.database import db

        threshold = settings.get('discovery_promotion_threshold', self.PROMOTION_THRESHOLD)
        max_promote = settings.get('discovery_max_promote_per_run', self.MAX_AUTO_PROMOTE_PER_RUN)
        max_watchlist = settings.get('discovery_max_watchlist_size', 50)

        # Check current watchlist size
        current_watchlist = db.get_watchlist(active_only=True)
        if len(current_watchlist) >= max_watchlist:
            logger.info(f"Watchlist at max size ({len(current_watchlist)}/{max_watchlist}), skipping promotion")
            return []

        promote_budget = min(max_promote, max_watchlist - len(current_watchlist))
        if promote_budget <= 0:
            return []

        promoted = []

        try:
            from engine.quant_screener import quant_screener

            # Screen top candidates
            tickers_to_screen = [d['ticker'] for d in discoveries[:15]]
            if not tickers_to_screen:
                return []

            benchmark_hist = quant_screener._get_benchmark_history()

            for ticker in tickers_to_screen:
                if len(promoted) >= promote_budget:
                    break

                try:
                    result = quant_screener.screen_ticker(ticker, benchmark_hist)
                    if not result or 'error' in result:
                        continue

                    score = result.get('composite_score', 0)
                    db.update_discovery_score(ticker, score)

                    if score >= threshold:
                        # Add to watchlist
                        name = result.get('data', {}).get('name', '')
                        db.add_to_watchlist(ticker, name)
                        db.promote_discovery(ticker)
                        promoted.append(ticker)
                        logger.info(f"Auto-promoted {ticker} to watchlist (score: {score})")
                except Exception as e:
                    logger.warning(f"Error screening {ticker}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Auto-promote error: {e}", exc_info=True)

        return promoted

    def get_discovery_stats(self) -> Dict:
        """Stats for dashboard: total discovered, promoted, dismissed, by strategy."""
        from core.database import db
        return db.get_discovery_stats()

    @staticmethod
    def _compute_rsi(series, period: int = 14) -> Optional[float]:
        """Compute RSI."""
        if len(series) < period + 1:
            return None
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])


# Singleton
auto_discovery = AutoDiscovery()
