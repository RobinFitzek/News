"""
Backtesting Engine
Validates the quant screener's scoring logic against historical data.

Key constraint: only technical (RSI, SMA, Bollinger, 52w range) and momentum
(1M/3M/6M returns vs SPY) can be computed historically — yfinance only provides
current-day fundamentals. The backtest scores a 50/50 blend of technicals +
momentum instead of the full 4-factor composite.
"""
import yfinance as yf
import numpy as np
import json
import threading
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional
from core.database import db
from engine.quant_screener import SectorCache
import logging

logger = logging.getLogger(__name__)


SECTOR_ETFS = {
    'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF',
    'Consumer Cyclical': 'XLY', 'Communication Services': 'XLC',
    'Industrials': 'XLI', 'Consumer Defensive': 'XLP', 'Energy': 'XLE',
    'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Basic Materials': 'XLB',
}

# Map tickers to sectors (populated at runtime via yfinance info)
_TICKER_SECTOR_CACHE: Dict[str, str] = {}


class BacktestEngine:
    """Replay historical signals and measure forward returns."""

    # Class-level progress for API polling
    _progress = {
        'run_id': None,
        'pct': 0,
        'msg': '',
        'running': False,
    }
    _lock = threading.Lock()

    COVERAGE_WARNING_2FACTOR = (
        "Backtest validates 50% of live model (tech+mom only). "
        "Valuation (30%) and Quality (20%) are NOT backtested."
    )
    COVERAGE_WARNING_4FACTOR = (
        "Full 4-factor backtest using fundamental snapshots. "
        "All factors (valuation, technical, momentum, quality) are validated."
    )
    MODEL_ALIGNMENT_PCT_2FACTOR = 50
    MODEL_ALIGNMENT_PCT_4FACTOR = 100

    # Dynamic — set during run
    COVERAGE_WARNING = COVERAGE_WARNING_2FACTOR
    MODEL_ALIGNMENT_PCT = MODEL_ALIGNMENT_PCT_2FACTOR

    TRANSACTION_COST_BPS = 10  # 10 bps round-trip (5 entry + 5 exit)

    DEFAULT_SCORE_CONFIG = {
        'signal_buy_threshold': 65,
        'signal_sell_threshold': 35,
        'rsi_ranges': [
            (0, 30, 85), (30, 40, 80), (40, 60, 70), (60, 70, 55), (70, 100, 25)
        ],
        'bb_ranges': [
            (-999, -0.8, 85), (-0.8, -0.3, 70), (-0.3, 0.3, 55), (0.3, 0.8, 35), (0.8, 999, 15)
        ],
        'momentum_excess_ranges': [
            (10, 999, 90), (5, 10, 75), (0, 5, 60), (-5, 0, 40), (-10, -5, 25), (-999, -10, 10)
        ],
    }

    def __init__(self):
        self._hist_cache: Dict[str, object] = {}
        self.score_config = dict(self.DEFAULT_SCORE_CONFIG)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, tickers: List[str] = None, months: int = 24) -> int:
        """
        Main entry point. Downloads history, replays monthly signals,
        measures forward returns, computes accuracy, optimises weights.
        Returns the run_id.
        """
        if tickers is None:
            # Use all sector peer tickers (~88 unique)
            seen = set()
            tickers = []
            for peers in SectorCache.SECTOR_PEERS.values():
                for t in peers:
                    if t not in seen:
                        seen.add(t)
                        tickers.append(t)

        run_id = db.create_backtest_run(len(tickers), months)

        with self._lock:
            self._progress = {
                'run_id': run_id,
                'pct': 0,
                'msg': 'Downloading history...',
                'running': True,
            }

        try:
            self._run_inner(run_id, tickers, months)
        except Exception as e:
            logger.error(f"Backtest run {run_id} failed: {e}", exc_info=True)
            db.update_backtest_run(run_id, status='failed', error=str(e)[:500])
            with self._lock:
                self._progress['running'] = False
                self._progress['msg'] = f'Failed: {e}'
        return run_id

    def get_progress(self) -> Dict:
        with self._lock:
            return dict(self._progress)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_inner(self, run_id: int, tickers: List[str], months: int):
        # 1. Download all history
        self._update_progress(run_id, 2, 'Downloading price history...')
        ticker_hists = self._download_history(tickers, months)
        benchmark_hist = self._download_single('SPY', months)

        if benchmark_hist is None or benchmark_hist.empty:
            raise RuntimeError('Could not download SPY benchmark data')

        valid_tickers = [t for t in tickers if t in ticker_hists]
        if not valid_tickers:
            raise RuntimeError('No valid ticker data downloaded')

        # 1b. Survivorship bias detection
        survivorship = self._detect_survivorship_bias(tickers, ticker_hists, months)

        # 1c. Download sector ETF history
        self._update_progress(run_id, 4, 'Downloading sector ETF benchmarks...')
        sector_etf_hists = self._download_sector_etfs(months)

        # 1d. Resolve ticker -> sector mapping
        self._resolve_ticker_sectors(valid_tickers)

        # 2. Build date list (first trading day of each month going back `months`)
        test_dates = self._build_test_dates(benchmark_hist, months)
        total_work = len(valid_tickers) * len(test_dates)
        done = 0

        self._update_progress(run_id, 5, f'Replaying {len(test_dates)} months across {len(valid_tickers)} tickers...')

        results = []

        for ticker in valid_tickers:
            hist = ticker_hists[ticker]
            sector = _TICKER_SECTOR_CACHE.get(ticker)
            sector_etf_ticker = SECTOR_ETFS.get(sector) if sector else None
            sector_hist = sector_etf_hists.get(sector_etf_ticker) if sector_etf_ticker else None

            for date in test_dates:
                try:
                    r = self._replay_date(ticker, hist, benchmark_hist, date,
                                          sector_hist=sector_hist, sector_etf_ticker=sector_etf_ticker)
                    if r:
                        results.append(r)
                        db.save_backtest_result(run_id, r)
                except Exception as e:
                    logger.debug(f"Skip {ticker} @ {date}: {e}")

                done += 1
                if done % 20 == 0 or done == total_work:
                    pct = 5 + int(75 * done / total_work)
                    self._update_progress(run_id, pct, f'{ticker} — {done}/{total_work}')

        # 3. Calculate accuracy + risk metrics
        self._update_progress(run_id, 82, 'Calculating accuracy & risk metrics...')
        stats = self._calculate_accuracy(results)
        risk = self._calculate_risk_metrics(results)

        # 4. Optimise weights (walk-forward if enough data, else single-split)
        self._update_progress(run_id, 90, 'Optimising weight split...')
        if months >= 18:
            best_weights = self._optimize_weights_walkforward(results)
        else:
            best_weights = self._optimize_weights(results)
            best_weights['walk_forward'] = False

        # Merge walk-forward accuracy into risk metrics
        risk['in_sample_accuracy'] = best_weights.get('in_sample_accuracy')
        risk['out_of_sample_accuracy'] = best_weights.get('out_of_sample_accuracy')

        # 5. Expected value per trade
        risk['expected_value_per_trade'] = stats.get('expected_value_per_trade')
        risk['ev_buy'] = stats.get('ev_buy')
        risk['ev_sell'] = stats.get('ev_sell')

        # 6. Portfolio simulation
        self._update_progress(run_id, 94, 'Simulating portfolio...')
        portfolio_sim = self._simulate_portfolio(results)
        risk['portfolio_simulation'] = portfolio_sim

        # 7. Signal independence
        self._update_progress(run_id, 96, 'Calculating signal independence...')
        independence = self._calculate_signal_independence(results)
        risk['signal_independence'] = independence

        # 8. Multi-window return summary + optimal holding period
        risk['multi_window_returns'] = self._summarize_multi_window_returns(results)
        risk['optimal_holding_days'] = self._find_optimal_holding_period(results)

        # 9. Accuracy by regime
        risk['accuracy_by_regime'] = stats.get('accuracy_by_regime', {})

        # 10. Determine coverage warning based on whether snapshots were used
        # Check if any fundamental snapshots exist for these tickers
        has_snapshots = False
        try:
            snap_count = db.query_one("SELECT COUNT(*) as cnt FROM fundamental_snapshots")
            has_snapshots = snap_count and snap_count['cnt'] > 0
        except Exception:
            pass

        coverage_warning = self.COVERAGE_WARNING_4FACTOR if has_snapshots else self.COVERAGE_WARNING_2FACTOR
        model_alignment = self.MODEL_ALIGNMENT_PCT_4FACTOR if has_snapshots else self.MODEL_ALIGNMENT_PCT_2FACTOR

        wf_windows = json.dumps(best_weights.get('windows', [])) if best_weights.get('walk_forward') else None
        wf_stability = best_weights.get('weight_stability') if best_weights.get('walk_forward') else None

        db.update_backtest_run(
            run_id,
            status='completed',
            overall_accuracy=stats.get('overall_accuracy'),
            avg_return_buy=stats.get('avg_return_buy'),
            avg_return_sell=stats.get('avg_return_sell'),
            best_weights=json.dumps(best_weights),
            progress_pct=100,
            progress_msg=f"Done — {stats.get('overall_accuracy', 0):.1f}% accuracy on {len(results)} signals",
            sharpe_ratio=risk.get('sharpe_ratio'),
            max_drawdown=risk.get('max_drawdown'),
            profit_factor=risk.get('profit_factor'),
            volatility=risk.get('volatility'),
            win_loss_ratio=risk.get('win_loss_ratio'),
            risk_metrics=json.dumps(risk),
            survivorship_warning=json.dumps(survivorship),
            coverage_warning=coverage_warning,
            model_alignment_pct=model_alignment,
            walk_forward_windows=wf_windows,
            weight_stability=wf_stability,
        )
        self._update_progress(run_id, 100, 'Complete')
        with self._lock:
            self._progress['running'] = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _download_history(self, tickers: List[str], months: int) -> Dict:
        """Bulk download — 3 years of daily data for each ticker."""
        period = '3y' if months <= 36 else '5y'
        hists = {}
        for t in tickers:
            h = self._download_single(t, months, period)
            if h is not None and not h.empty and len(h) >= 60:
                hists[t] = h
        return hists

    def _download_single(self, ticker: str, months: int = 36, period: str = '3y'):
        if ticker in self._hist_cache:
            return self._hist_cache[ticker]
        try:
            h = yf.Ticker(ticker).history(period=period)
            if h is not None and not h.empty:
                self._hist_cache[ticker] = h
                return h
        except Exception as e:
            logger.warning(f"Download failed for {ticker}: {e}")
        return None

    def _build_test_dates(self, benchmark_hist, months: int) -> List[datetime]:
        """Return first-of-month dates present in the benchmark history."""
        dates = []
        now = datetime.now()
        # Start from (months) months ago, step forward by 1 month
        # Stop 1 month before now so we can measure 20-day forward returns
        start = now - relativedelta(months=months)
        cursor = start.replace(day=1)
        end = now - relativedelta(months=1)

        idx = benchmark_hist.index
        while cursor <= end:
            # Find the first trading day on or after cursor
            mask = idx >= cursor.strftime('%Y-%m-%d')
            if mask.any():
                first_day = idx[mask][0]
                dates.append(first_day.to_pydatetime() if hasattr(first_day, 'to_pydatetime') else first_day)
            cursor += relativedelta(months=1)

        return dates

    def _download_sector_etfs(self, months: int) -> Dict:
        """Download history for all sector ETFs."""
        hists = {}
        for etf in set(SECTOR_ETFS.values()):
            h = self._download_single(etf, months)
            if h is not None and not h.empty:
                hists[etf] = h
        return hists

    def _resolve_ticker_sectors(self, tickers: List[str]):
        """Populate _TICKER_SECTOR_CACHE for tickers not yet mapped."""
        for t in tickers:
            if t in _TICKER_SECTOR_CACHE:
                continue
            try:
                info = yf.Ticker(t).info
                sector = info.get('sector')
                if sector:
                    _TICKER_SECTOR_CACHE[t] = sector
            except Exception:
                pass

    def _detect_survivorship_bias(self, original_tickers: List[str],
                                  downloaded: Dict, months: int) -> Dict:
        """Detect survivorship bias risk based on download failures and data gaps."""
        total = len(original_tickers)
        failed = [t for t in original_tickers if t not in downloaded]
        failed_pct = (len(failed) / total * 100) if total > 0 else 0

        # Check for partial data (tickers with < 70% of expected trading days)
        expected_days = months * 21  # ~21 trading days per month
        partial = []
        for t, hist in downloaded.items():
            if len(hist) < expected_days * 0.7:
                partial.append(t)
        partial_pct = (len(partial) / total * 100) if total > 0 else 0

        combined_pct = failed_pct + partial_pct
        if combined_pct > 10:
            risk_level = 'High'
        elif combined_pct > 5:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        # Feature 7: Count graveyard entries for additional context
        graveyard_count = 0
        try:
            graveyard_rows = db.query("SELECT COUNT(*) as cnt FROM ticker_graveyard")
            graveyard_count = graveyard_rows[0]['cnt'] if graveyard_rows else 0
        except Exception:
            pass

        return {
            'risk_level': risk_level,
            'total_tickers': total,
            'failed_downloads': len(failed),
            'failed_tickers': failed[:20],  # cap for storage
            'failed_pct': round(failed_pct, 1),
            'partial_data': len(partial),
            'partial_tickers': partial[:20],
            'partial_pct': round(partial_pct, 1),
            'graveyard_count': graveyard_count,
            'message': (
                f"{risk_level} survivorship bias risk: {len(failed)}/{total} tickers "
                f"failed to download ({failed_pct:.0f}%), {len(partial)} have partial data"
                f"{', ' + str(graveyard_count) + ' in graveyard' if graveyard_count else ''}."
            ),
        }

    # ------------------------------------------------------------------
    # Market Regime Detection
    # ------------------------------------------------------------------

    def _detect_regime(self, benchmark_hist, date) -> str:
        """
        Detect market regime at a given date.
        Bull: SPY price > 200 SMA AND 50 SMA > 200 SMA
        Bear: SPY price < 200 SMA AND 50 SMA < 200 SMA
        Choppy: everything else
        """
        try:
            bench_window = benchmark_hist.loc[:date]
            close = bench_window['Close']
            if len(close) < 200:
                return 'choppy'
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            sma_200 = float(close.rolling(200).mean().iloc[-1])
            price = float(close.iloc[-1])
            if price > sma_200 and sma_50 > sma_200:
                return 'bull'
            elif price < sma_200 and sma_50 < sma_200:
                return 'bear'
            else:
                return 'choppy'
        except Exception:
            return 'choppy'

    # ------------------------------------------------------------------
    # Scoring at a historical date
    # ------------------------------------------------------------------

    def _replay_date(self, ticker: str, hist, benchmark_hist, date,
                     sector_hist=None, sector_etf_ticker: str = None) -> Optional[Dict]:
        """Score a ticker using only data up to `date`, then measure forward returns."""
        # Slice history up to the test date
        hist_window = hist.loc[:date]
        bench_window = benchmark_hist.loc[:date]

        if len(hist_window) < 60 or len(bench_window) < 60:
            return None

        tech_score = self._score_technicals_at(hist_window)
        mom_score = self._score_momentum_at(hist_window, bench_window)

        if tech_score is None or mom_score is None:
            return None

        # Check for fundamental snapshot near this date (Feature 5)
        date_str_lookup = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        val_score = None
        qual_score = None
        try:
            snap = db.query_one("""
                SELECT * FROM fundamental_snapshots
                WHERE ticker = ? AND snapshot_date <= ?
                ORDER BY snapshot_date DESC LIMIT 1
            """, (ticker, date_str_lookup))
            if snap:
                val_score = self._score_valuation_snapshot(snap)
                qual_score = self._score_quality_snapshot(snap)
        except Exception:
            pass

        if val_score is not None and qual_score is not None:
            # Full 4-factor composite using same weights as live model
            composite = (
                val_score * 0.30 +
                tech_score * 0.25 +
                mom_score * 0.25 +
                qual_score * 0.20
            )
        else:
            # 50/50 blend (no fundamentals available historically)
            composite = (tech_score * 0.5) + (mom_score * 0.5)
        composite = max(0, min(100, composite))

        signal = self._generate_signal(composite)

        # Forward returns (multi-window)
        fwd_returns = self._measure_forward_returns(hist, date)

        fwd_5d = fwd_returns.get('fwd_5d')
        fwd_10d = fwd_returns.get('fwd_10d')
        fwd_20d = fwd_returns.get('fwd_20d')
        fwd_40d = fwd_returns.get('fwd_40d')
        fwd_60d = fwd_returns.get('fwd_60d')

        # Sector benchmark return (20d forward)
        bench_ticker_used = sector_etf_ticker or 'SPY'
        bench_source = sector_hist if sector_hist is not None else benchmark_hist
        bench_fwd_20d = self._measure_benchmark_forward(bench_source, date)

        # Alpha
        alpha = None
        if fwd_20d is not None and bench_fwd_20d is not None:
            alpha = round(fwd_20d - bench_fwd_20d, 2)

        # Hit: Buy signal beats its sector benchmark (or positive if no benchmark)
        hit = 0
        if fwd_20d is not None:
            if signal == 'Buy':
                if alpha is not None:
                    hit = 1 if alpha > 0 else 0
                else:
                    hit = 1 if fwd_20d > 0 else 0
            elif signal == 'Sell' and fwd_20d < 0:
                hit = 1
            elif signal == 'Hold':
                hit = 1 if abs(fwd_20d) < 3 else 0

        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]

        # Market regime
        regime = self._detect_regime(benchmark_hist, date)

        return {
            'ticker': ticker,
            'test_date': date_str,
            'signal': signal,
            'tech_score': round(tech_score, 1),
            'momentum_score': round(mom_score, 1),
            'composite_score': round(composite, 1),
            'forward_5d_return': round(fwd_5d, 2) if fwd_5d is not None else None,
            'forward_10d_return': round(fwd_10d, 2) if fwd_10d is not None else None,
            'forward_20d_return': round(fwd_20d, 2) if fwd_20d is not None else None,
            'forward_40d_return': round(fwd_40d, 2) if fwd_40d is not None else None,
            'forward_60d_return': round(fwd_60d, 2) if fwd_60d is not None else None,
            'hit': hit,
            'benchmark_ticker': bench_ticker_used,
            'benchmark_return': round(bench_fwd_20d, 2) if bench_fwd_20d is not None else None,
            'alpha': alpha,
            'regime': regime,
        }

    def _score_technicals_at(self, hist) -> Optional[float]:
        """Compute technical score using the same logic as QuantScreener."""
        close = hist['Close']
        if len(close) < 20:
            return None

        scores = []

        # RSI
        rsi = self._compute_rsi(close, 14)
        if rsi is not None:
            rsi_score = self._score_from_ranges(rsi, self.score_config['rsi_ranges'])
            scores.append(rsi_score)

        # SMA cross signal
        current_price = float(close.iloc[-1])
        if len(close) >= 200:
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            sma_200 = float(close.rolling(200).mean().iloc[-1])
            if sma_50 > sma_200:
                scores.append(70)  # bullish
            else:
                scores.append(30)  # bearish
        elif len(close) >= 50:
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            scores.append(60 if current_price > sma_50 else 40)

        # 52-week range position
        high_52w = float(close[-252:].max()) if len(close) >= 252 else float(close.max())
        low_52w = float(close[-252:].min()) if len(close) >= 252 else float(close.min())
        if high_52w != low_52w:
            pos = (current_price - low_52w) / (high_52w - low_52w)
            if pos <= 0.2:
                scores.append(85)
            elif pos <= 0.4:
                scores.append(70)
            elif pos <= 0.6:
                scores.append(55)
            elif pos <= 0.8:
                scores.append(40)
            else:
                scores.append(20)

        # Bollinger position
        if len(close) >= 20:
            bb_sma = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            upper = float(bb_sma.iloc[-1] + 2 * bb_std.iloc[-1])
            lower = float(bb_sma.iloc[-1] - 2 * bb_std.iloc[-1])
            bb_mid = float(bb_sma.iloc[-1])
            if upper != bb_mid:
                bb = (current_price - bb_mid) / (upper - bb_mid)
            else:
                bb = 0.0
            bb_score = self._score_from_ranges(bb, self.score_config['bb_ranges'])
            scores.append(bb_score)

        return sum(scores) / len(scores) if scores else None

    def _score_momentum_at(self, hist, benchmark_hist) -> Optional[float]:
        """Compute momentum score at a historical point."""
        close = hist['Close']
        bench_close = benchmark_hist['Close']
        current = float(close.iloc[-1])
        bench_current = float(bench_close.iloc[-1])

        scores = []

        for days in [21, 63, 126]:
            if len(close) < days or len(bench_close) < days:
                continue
            past = float(close.iloc[-days])
            b_past = float(bench_close.iloc[-days])
            if past == 0 or b_past == 0:
                continue
            ret = ((current - past) / past) * 100
            b_ret = ((bench_current - b_past) / b_past) * 100
            excess = ret - b_ret

            excess_score = self._score_from_ranges(excess, self.score_config['momentum_excess_ranges'])
            scores.append(excess_score)

        # Acceleration
        if len(close) >= 63:
            past_21 = float(close.iloc[-21]) if len(close) >= 21 else 0
            past_63 = float(close.iloc[-63])
            ret_1m_val = ((current - past_21) / past_21) * 100 if past_21 != 0 else None
            ret_3m_val = ((current - past_63) / past_63) * 100 if past_63 != 0 else None
            if ret_1m_val is not None and ret_3m_val is not None:
                monthly_avg = ret_3m_val / 3
                if ret_1m_val > monthly_avg and ret_1m_val > 0:
                    scores.append(75)
                elif ret_1m_val < monthly_avg and ret_1m_val < 0:
                    scores.append(25)
                else:
                    scores.append(50)

        return sum(scores) / len(scores) if scores else None

    @staticmethod
    def _score_from_ranges(value: float, ranges: List[tuple]) -> float:
        """Look up a score from a list of (low, high, score) tuples."""
        for low, high, score in ranges:
            if low <= value < high:
                return score
        # Fallback: return last range score
        return ranges[-1][2] if ranges else 50

    def _measure_forward_returns(self, hist, date) -> Dict:
        """Calculate 5, 10, 20, 40, 60-day forward returns from `date`."""
        result = {'fwd_5d': None, 'fwd_10d': None, 'fwd_20d': None,
                  'fwd_40d': None, 'fwd_60d': None}
        try:
            future = hist.loc[date:]
            close = future['Close']
            if len(close) < 2:
                return result
            base_price = float(close.iloc[0])
            if base_price == 0:
                return result

            for days, key in [(5, 'fwd_5d'), (10, 'fwd_10d'), (20, 'fwd_20d'),
                              (40, 'fwd_40d'), (60, 'fwd_60d')]:
                if len(close) > days:
                    result[key] = ((float(close.iloc[days]) - base_price) / base_price) * 100

            return result
        except Exception:
            return result

    def _measure_benchmark_forward(self, bench_hist, date) -> Optional[float]:
        """Calculate 20-day forward return for a benchmark from `date`."""
        try:
            future = bench_hist.loc[date:]
            close = future['Close']
            if len(close) <= 20:
                return None
            base = float(close.iloc[0])
            if base == 0:
                return None
            return ((float(close.iloc[20]) - base) / base) * 100
        except Exception:
            return None

    def _generate_signal(self, score: float) -> str:
        if score > self.score_config['signal_buy_threshold']:
            return 'Buy'
        elif score < self.score_config['signal_sell_threshold']:
            return 'Sell'
        return 'Hold'

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _calculate_risk_metrics(self, results: List[Dict]) -> Dict:
        """Calculate Sharpe, max drawdown, profit factor, volatility, win/loss ratio."""
        with_fwd = [r for r in results if r.get('forward_20d_return') is not None]
        if not with_fwd:
            return {'sharpe_ratio': None, 'max_drawdown': None, 'profit_factor': None,
                    'volatility': None, 'win_loss_ratio': None}

        returns = np.array([r['forward_20d_return'] for r in with_fwd])
        cost_pct = self.TRANSACTION_COST_BPS / 100.0  # 0.10%

        # Gross returns
        gross_returns = returns.copy()
        # Net returns: deduct transaction costs
        net_returns = returns - cost_pct

        # Use net returns for all metrics
        mean_ret = float(np.mean(net_returns))
        std_ret = float(np.std(net_returns))

        # Sharpe Ratio (annualized, treating each signal as ~monthly)
        sharpe = (mean_ret / std_ret * np.sqrt(12)) if std_ret > 0 else 0.0

        # Max Drawdown (cumulative returns)
        cum = np.cumsum(net_returns / 100)
        peak = np.maximum.accumulate(cum)
        drawdowns = cum - peak
        max_dd = float(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0

        # Profit Factor
        winners = net_returns[net_returns > 0]
        losers = net_returns[net_returns < 0]
        profit_factor = (float(np.sum(winners)) / abs(float(np.sum(losers)))) if len(losers) > 0 and np.sum(losers) != 0 else None

        # Volatility (std of returns)
        volatility = round(std_ret, 2)

        # Win/Loss Ratio (avg win / avg loss)
        avg_win = float(np.mean(winners)) if len(winners) > 0 else 0
        avg_loss = abs(float(np.mean(losers))) if len(losers) > 0 else 0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else None

        # Per-signal breakdown
        buy_returns = np.array([r['forward_20d_return'] for r in with_fwd if r['signal'] == 'Buy'])
        sell_returns = np.array([r['forward_20d_return'] for r in with_fwd if r['signal'] == 'Sell'])

        buy_vol = round(float(np.std(buy_returns)), 2) if len(buy_returns) > 1 else None
        sell_vol = round(float(np.std(sell_returns)), 2) if len(sell_returns) > 1 else None

        # Gross vs net total returns
        gross_return = round(float(np.sum(gross_returns)), 2)
        net_return = round(float(np.sum(net_returns)), 2)

        return {
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor is not None else None,
            'volatility': volatility,
            'win_loss_ratio': round(win_loss_ratio, 2) if win_loss_ratio is not None else None,
            'buy_volatility': buy_vol,
            'sell_volatility': sell_vol,
            'total_signals': len(with_fwd),
            'winning_signals': int(np.sum(net_returns > 0)),
            'losing_signals': int(np.sum(net_returns < 0)),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'gross_return': gross_return,
            'net_return': net_return,
            'transaction_cost_bps': self.TRANSACTION_COST_BPS,
        }

    def _calculate_accuracy(self, results: List[Dict]) -> Dict:
        """Hit rate and average returns by signal bucket."""
        if not results:
            return {'overall_accuracy': 0, 'avg_return_buy': 0, 'avg_return_sell': 0}

        total_with_fwd = [r for r in results if r.get('forward_20d_return') is not None]
        if not total_with_fwd:
            return {'overall_accuracy': 0, 'avg_return_buy': 0, 'avg_return_sell': 0}

        hits = sum(1 for r in total_with_fwd if r['hit'])
        accuracy = (hits / len(total_with_fwd)) * 100

        buy_returns = [r['forward_20d_return'] for r in total_with_fwd if r['signal'] == 'Buy']
        sell_returns = [r['forward_20d_return'] for r in total_with_fwd if r['signal'] == 'Sell']

        # Expected Value per trade
        ev_per_trade = 0.0
        ev_buy = 0.0
        ev_sell = 0.0

        all_returns = [r['forward_20d_return'] for r in total_with_fwd]
        if all_returns:
            arr = np.array(all_returns)
            wins = arr[arr > 0]
            losses = arr[arr <= 0]
            win_rate = len(wins) / len(arr) if len(arr) > 0 else 0
            loss_rate = 1 - win_rate
            avg_win_val = float(np.mean(wins)) if len(wins) > 0 else 0
            avg_loss_val = abs(float(np.mean(losses))) if len(losses) > 0 else 0
            ev_per_trade = round((win_rate * avg_win_val) - (loss_rate * avg_loss_val), 4)

        if buy_returns:
            barr = np.array(buy_returns)
            bwins = barr[barr > 0]
            blosses = barr[barr <= 0]
            bwr = len(bwins) / len(barr) if len(barr) > 0 else 0
            blr = 1 - bwr
            baw = float(np.mean(bwins)) if len(bwins) > 0 else 0
            bal = abs(float(np.mean(blosses))) if len(blosses) > 0 else 0
            ev_buy = round((bwr * baw) - (blr * bal), 4)

        if sell_returns:
            sarr = np.array(sell_returns)
            swins = sarr[sarr < 0]  # For sell, "winning" = price went down
            slosses = sarr[sarr >= 0]
            swr = len(swins) / len(sarr) if len(sarr) > 0 else 0
            slr = 1 - swr
            saw = abs(float(np.mean(swins))) if len(swins) > 0 else 0
            sal = float(np.mean(slosses)) if len(slosses) > 0 else 0
            ev_sell = round((swr * saw) - (slr * sal), 4)

        # Accuracy by regime
        accuracy_by_regime = {}
        for regime in ['bull', 'bear', 'choppy']:
            regime_results = [r for r in total_with_fwd if r.get('regime') == regime]
            if regime_results:
                regime_hits = sum(1 for r in regime_results if r['hit'])
                accuracy_by_regime[regime] = {
                    'accuracy': round((regime_hits / len(regime_results)) * 100, 1),
                    'count': len(regime_results),
                }

        return {
            'overall_accuracy': round(accuracy, 1),
            'avg_return_buy': round(float(np.mean(buy_returns)), 2) if buy_returns else 0,
            'avg_return_sell': round(float(np.mean(sell_returns)), 2) if sell_returns else 0,
            'total_signals': len(total_with_fwd),
            'buy_count': len(buy_returns),
            'sell_count': len(sell_returns),
            'hold_count': len([r for r in total_with_fwd if r['signal'] == 'Hold']),
            'expected_value_per_trade': ev_per_trade,
            'ev_buy': ev_buy,
            'ev_sell': ev_sell,
            'accuracy_by_regime': accuracy_by_regime,
        }

    def _optimize_weights(self, results: List[Dict]) -> Dict:
        """Walk-forward grid search: train on 70% by date, test on 30%."""
        scored = [r for r in results if r.get('forward_20d_return') is not None
                  and r.get('tech_score') is not None and r.get('momentum_score') is not None]
        if not scored:
            return {'tech_weight': 0.5, 'momentum_weight': 0.5, 'accuracy': 0,
                    'in_sample_accuracy': 0, 'out_of_sample_accuracy': 0}

        # Sort by date for walk-forward split
        scored.sort(key=lambda r: r['test_date'])
        split_idx = int(len(scored) * 0.7)
        train_set = scored[:split_idx]
        test_set = scored[split_idx:]

        if not train_set or not test_set:
            train_set = scored
            test_set = scored

        # Grid search on training set: weights AND signal thresholds
        best_acc = 0
        best_w = 0.5
        best_buy_thresh = self.score_config['signal_buy_threshold']
        best_sell_thresh = self.score_config['signal_sell_threshold']

        threshold_options = [(60, 40), (65, 35), (70, 30)]

        for tech_w_int in range(20, 81, 5):  # 0.20 to 0.80 step 0.05
            tech_w = tech_w_int / 100
            mom_w = 1 - tech_w

            for buy_thresh, sell_thresh in threshold_options:
                hits = 0
                total = 0
                for r in train_set:
                    composite = r['tech_score'] * tech_w + r['momentum_score'] * mom_w
                    if composite > buy_thresh:
                        signal = 'Buy'
                    elif composite < sell_thresh:
                        signal = 'Sell'
                    else:
                        signal = 'Hold'
                    fwd = r['forward_20d_return']
                    if signal == 'Buy' and fwd > 0:
                        hits += 1
                    elif signal == 'Sell' and fwd < 0:
                        hits += 1
                    elif signal == 'Hold' and abs(fwd) < 3:
                        hits += 1
                    total += 1

                acc = (hits / total * 100) if total else 0
                if acc > best_acc:
                    best_acc = acc
                    best_w = tech_w
                    best_buy_thresh = buy_thresh
                    best_sell_thresh = sell_thresh

        in_sample_accuracy = round(best_acc, 1)

        # Evaluate on test set with best params
        hits = 0
        total = 0
        for r in test_set:
            composite = r['tech_score'] * best_w + r['momentum_score'] * (1 - best_w)
            if composite > best_buy_thresh:
                signal = 'Buy'
            elif composite < best_sell_thresh:
                signal = 'Sell'
            else:
                signal = 'Hold'
            fwd = r['forward_20d_return']
            if signal == 'Buy' and fwd > 0:
                hits += 1
            elif signal == 'Sell' and fwd < 0:
                hits += 1
            elif signal == 'Hold' and abs(fwd) < 3:
                hits += 1
            total += 1

        out_of_sample_accuracy = round((hits / total * 100) if total else 0, 1)

        return {
            'tech_weight': round(best_w, 2),
            'momentum_weight': round(1 - best_w, 2),
            'accuracy': in_sample_accuracy,
            'in_sample_accuracy': in_sample_accuracy,
            'out_of_sample_accuracy': out_of_sample_accuracy,
            'optimized_buy_threshold': best_buy_thresh,
            'optimized_sell_threshold': best_sell_thresh,
            'train_size': len(train_set),
            'test_size': len(test_set),
        }

    def _optimize_weights_walkforward(self, results: List[Dict],
                                       train_months: int = 12,
                                       test_months: int = 3) -> Dict:
        """Rolling walk-forward optimization to prevent overfitting.

        Instead of a single 70/30 split, uses rolling windows:
        - Window 1: train months 1-12, test 13-15
        - Window 2: train months 4-15, test 16-18
        - etc.

        Returns average OOS accuracy across all windows plus stability metric.
        """
        scored = [r for r in results if r.get('forward_20d_return') is not None
                  and r.get('tech_score') is not None and r.get('momentum_score') is not None]
        if not scored:
            return self._optimize_weights(results)

        # Sort by date
        scored.sort(key=lambda r: r['test_date'])
        dates = [r['test_date'] for r in scored]

        if not dates:
            return self._optimize_weights(results)

        # Determine date range
        from dateutil.relativedelta import relativedelta
        first_date = datetime.strptime(dates[0][:10], '%Y-%m-%d')
        last_date = datetime.strptime(dates[-1][:10], '%Y-%m-%d')
        total_months = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)

        # Need at least train_months + test_months of data
        if total_months < train_months + test_months:
            # Fall back to single-split method
            fallback = self._optimize_weights(results)
            fallback['walk_forward'] = False
            fallback['reason'] = f'Only {total_months} months of data, need {train_months + test_months} for walk-forward'
            return fallback

        # Build rolling windows
        windows = []
        window_start = first_date

        while True:
            train_end = window_start + relativedelta(months=train_months)
            test_end = train_end + relativedelta(months=test_months)

            if test_end > last_date + timedelta(days=15):
                break

            train_start_str = window_start.strftime('%Y-%m-%d')
            train_end_str = train_end.strftime('%Y-%m-%d')
            test_end_str = test_end.strftime('%Y-%m-%d')

            train_set = [r for r in scored if train_start_str <= r['test_date'][:10] < train_end_str]
            test_set = [r for r in scored if train_end_str <= r['test_date'][:10] < test_end_str]

            if len(train_set) >= 10 and len(test_set) >= 3:
                # Grid search on training set
                best_acc = 0
                best_w = 0.5
                best_buy = 65
                best_sell = 35

                for tech_w_int in range(25, 76, 5):
                    tech_w = tech_w_int / 100
                    mom_w = 1 - tech_w

                    for buy_thresh, sell_thresh in [(60, 40), (65, 35), (70, 30)]:
                        hits = total = 0
                        for r in train_set:
                            composite = r['tech_score'] * tech_w + r['momentum_score'] * mom_w
                            if composite > buy_thresh:
                                sig = 'Buy'
                            elif composite < sell_thresh:
                                sig = 'Sell'
                            else:
                                sig = 'Hold'
                            fwd = r['forward_20d_return']
                            if sig == 'Buy' and fwd > 0:
                                hits += 1
                            elif sig == 'Sell' and fwd < 0:
                                hits += 1
                            elif sig == 'Hold' and abs(fwd) < 3:
                                hits += 1
                            total += 1
                        acc = (hits / total * 100) if total else 0
                        if acc > best_acc:
                            best_acc = acc
                            best_w = tech_w
                            best_buy = buy_thresh
                            best_sell = sell_thresh

                # Evaluate on test set
                hits = total = 0
                for r in test_set:
                    composite = r['tech_score'] * best_w + r['momentum_score'] * (1 - best_w)
                    if composite > best_buy:
                        sig = 'Buy'
                    elif composite < best_sell:
                        sig = 'Sell'
                    else:
                        sig = 'Hold'
                    fwd = r['forward_20d_return']
                    if sig == 'Buy' and fwd > 0:
                        hits += 1
                    elif sig == 'Sell' and fwd < 0:
                        hits += 1
                    elif sig == 'Hold' and abs(fwd) < 3:
                        hits += 1
                    total += 1
                oos_acc = round((hits / total * 100) if total else 0, 1)

                windows.append({
                    'train_period': f'{train_start_str[:7]} to {train_end_str[:7]}',
                    'test_period': f'{train_end_str[:7]} to {test_end_str[:7]}',
                    'train_size': len(train_set),
                    'test_size': len(test_set),
                    'best_tech_weight': round(best_w, 2),
                    'best_momentum_weight': round(1 - best_w, 2),
                    'in_sample_accuracy': round(best_acc, 1),
                    'out_of_sample_accuracy': oos_acc,
                })

            # Shift window forward by test_months
            window_start += relativedelta(months=test_months)

        if not windows:
            fallback = self._optimize_weights(results)
            fallback['walk_forward'] = False
            fallback['reason'] = 'Not enough data per window for walk-forward analysis'
            return fallback

        # Aggregate results
        oos_accuracies = [w['out_of_sample_accuracy'] for w in windows]
        avg_oos = round(sum(oos_accuracies) / len(oos_accuracies), 1)
        std_oos = round((sum((a - avg_oos) ** 2 for a in oos_accuracies) / len(oos_accuracies)) ** 0.5, 1) if len(oos_accuracies) > 1 else 0

        # Use median window's weights as best (most robust)
        windows_sorted = sorted(windows, key=lambda w: w['out_of_sample_accuracy'], reverse=True)
        median_idx = len(windows_sorted) // 2
        best_window = windows_sorted[median_idx]

        # Check for degradation (last window accuracy vs first)
        degrading = False
        if len(windows) >= 3:
            first_half_avg = sum(w['out_of_sample_accuracy'] for w in windows[:len(windows)//2]) / (len(windows)//2)
            second_half_avg = sum(w['out_of_sample_accuracy'] for w in windows[len(windows)//2:]) / (len(windows) - len(windows)//2)
            degrading = second_half_avg < first_half_avg - 5  # >5% degradation

        # Also run single-split for comparison
        single_split = self._optimize_weights(results)

        return {
            'tech_weight': best_window['best_tech_weight'],
            'momentum_weight': best_window['best_momentum_weight'],
            'accuracy': avg_oos,
            'in_sample_accuracy': single_split.get('in_sample_accuracy', 0),
            'out_of_sample_accuracy': avg_oos,
            'walk_forward': True,
            'windows': windows,
            'avg_oos_accuracy': avg_oos,
            'oos_std': std_oos,
            'weight_stability': std_oos,
            'degrading': degrading,
            'single_split_oos': single_split.get('out_of_sample_accuracy', 0),
            'train_size': sum(w['train_size'] for w in windows),
            'test_size': sum(w['test_size'] for w in windows),
        }

    # ------------------------------------------------------------------
    # Portfolio Simulation
    # ------------------------------------------------------------------

    def _simulate_portfolio(self, results: List[Dict]) -> Dict:
        """
        Simulate a portfolio: top 20 Buy signals per month, inverse-vol weighted.
        Deducts transaction costs.
        """
        with_fwd = [r for r in results if r.get('forward_20d_return') is not None
                     and r.get('signal') == 'Buy']
        if not with_fwd:
            return {'total_return_gross': 0, 'total_return_net': 0, 'sharpe': 0,
                    'max_drawdown': 0, 'monthly_returns': []}

        # Group by month (test_date YYYY-MM)
        by_month: Dict[str, List[Dict]] = {}
        for r in with_fwd:
            month_key = r['test_date'][:7]  # YYYY-MM
            by_month.setdefault(month_key, []).append(r)

        cost_pct = self.TRANSACTION_COST_BPS / 100.0
        monthly_returns_gross = []
        monthly_returns_net = []
        monthly_turnover = []
        prev_tickers = set()

        for month_key in sorted(by_month.keys()):
            signals = by_month[month_key]
            # Rank by composite score, take top 20
            signals.sort(key=lambda r: r.get('composite_score', 0), reverse=True)
            top = signals[:20]

            # Inverse-volatility weighting
            # Use forward_5d_return as proxy for short-term vol if available,
            # otherwise equal weight
            vols = []
            for r in top:
                # Approximate 20d vol from available data
                fwd5 = r.get('forward_5d_return')
                fwd20 = r.get('forward_20d_return', 0)
                # Use absolute returns as rough vol proxy
                vol = abs(fwd20) if fwd20 else 1.0
                vol = max(vol, 0.1)  # floor to avoid division by zero
                vols.append(vol)

            inv_vols = [1.0 / v for v in vols]
            total_inv_vol = sum(inv_vols)
            weights = [iv / total_inv_vol for iv in inv_vols] if total_inv_vol > 0 else [1.0 / len(top)] * len(top)

            # Portfolio return for the month
            port_return_gross = sum(w * r['forward_20d_return'] for w, r in zip(weights, top))
            monthly_returns_gross.append(port_return_gross)

            # Turnover: how many positions changed
            current_tickers = set(r['ticker'] for r in top)
            turnover = len(current_tickers - prev_tickers)
            monthly_turnover.append(turnover)
            prev_tickers = current_tickers

            # Net return: deduct costs for each position change
            total_positions = len(top)
            port_return_net = port_return_gross - (cost_pct * total_positions / total_positions)
            monthly_returns_net.append(port_return_net)

        # Cumulative returns
        gross_arr = np.array(monthly_returns_gross)
        net_arr = np.array(monthly_returns_net)

        total_return_gross = round(float(np.sum(gross_arr)), 2)
        total_return_net = round(float(np.sum(net_arr)), 2)

        # Sharpe
        mean_net = float(np.mean(net_arr)) if len(net_arr) > 0 else 0
        std_net = float(np.std(net_arr)) if len(net_arr) > 1 else 1
        sharpe = round((mean_net / std_net * np.sqrt(12)) if std_net > 0 else 0, 2)

        # Max drawdown
        cum = np.cumsum(net_arr / 100)
        peak = np.maximum.accumulate(cum) if len(cum) > 0 else np.array([0])
        dd = cum - peak
        max_dd = round(float(np.min(dd)) * 100, 2) if len(dd) > 0 else 0

        avg_turnover = round(float(np.mean(monthly_turnover)), 1) if monthly_turnover else 0

        return {
            'total_return_gross': total_return_gross,
            'total_return_net': total_return_net,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_months': len(monthly_returns_gross),
            'avg_monthly_return_net': round(mean_net, 2),
            'avg_turnover_per_month': avg_turnover,
            'transaction_cost_bps': self.TRANSACTION_COST_BPS,
        }

    # ------------------------------------------------------------------
    # Signal Independence
    # ------------------------------------------------------------------

    def _calculate_signal_independence(self, results: List[Dict]) -> Dict:
        """
        Measure how independent signals are across tickers on the same date.
        """
        if not results:
            return {'signal_diversity': 0, 'avg_sectors_per_date': 0,
                    'effective_n': 0, 'correlation_warning': 'No results'}

        # Group by test_date
        by_date: Dict[str, List[Dict]] = {}
        for r in results:
            by_date.setdefault(r['test_date'], []).append(r)

        diversities = []
        sectors_per_date = []

        for date_str, signals in by_date.items():
            total = len(signals)
            if total < 2:
                continue

            # Count same-direction signals
            buy_count = sum(1 for s in signals if s['signal'] == 'Buy')
            sell_count = sum(1 for s in signals if s['signal'] == 'Sell')
            hold_count = sum(1 for s in signals if s['signal'] == 'Hold')
            max_same = max(buy_count, sell_count, hold_count)

            diversity = 1 - (max_same / total)
            diversities.append(diversity)

            # Count unique sectors in Buy signals
            buy_signals = [s for s in signals if s['signal'] == 'Buy']
            unique_sectors = set()
            for s in buy_signals:
                sector = _TICKER_SECTOR_CACHE.get(s['ticker'])
                if sector:
                    unique_sectors.add(sector)
            sectors_per_date.append(len(unique_sectors))

        avg_diversity = round(float(np.mean(diversities)), 3) if diversities else 0
        avg_sectors = round(float(np.mean(sectors_per_date)), 1) if sectors_per_date else 0

        # Effective N
        avg_signals_per_date = float(np.mean([len(v) for v in by_date.values()]))
        effective_n = round(avg_signals_per_date * avg_diversity, 1)

        correlation_warning = None
        if avg_diversity < 0.3:
            correlation_warning = (
                f"Low signal diversity ({avg_diversity:.2f}). "
                "Most signals point the same direction — accuracy may be overstated."
            )

        return {
            'signal_diversity': avg_diversity,
            'avg_sectors_per_date': avg_sectors,
            'effective_n': effective_n,
            'correlation_warning': correlation_warning,
        }

    # ------------------------------------------------------------------
    # Multi-Window Returns Summary
    # ------------------------------------------------------------------

    def _summarize_multi_window_returns(self, results: List[Dict]) -> Dict:
        """Summarize forward returns across all windows for storage in risk_metrics."""
        summary = {}
        windows = [
            ('fwd_5d', 'forward_5d_return'),
            ('fwd_10d', 'forward_10d_return'),
            ('fwd_20d', 'forward_20d_return'),
            ('fwd_40d', 'forward_40d_return'),
            ('fwd_60d', 'forward_60d_return'),
        ]
        for key, result_key in windows:
            vals = [r[result_key] for r in results if r.get(result_key) is not None]
            if vals:
                arr = np.array(vals)
                summary[key] = {
                    'mean': round(float(np.mean(arr)), 2),
                    'median': round(float(np.median(arr)), 2),
                    'std': round(float(np.std(arr)), 2),
                    'count': len(vals),
                    'pct_positive': round(float(np.sum(arr > 0)) / len(vals) * 100, 1),
                }
            else:
                summary[key] = {'mean': None, 'median': None, 'std': None, 'count': 0, 'pct_positive': None}
        return summary

    def _find_optimal_holding_period(self, results: List[Dict]) -> Optional[int]:
        """Find which forward window has the best expected value for Buy signals."""
        buy_results = [r for r in results if r.get('signal') == 'Buy']
        if not buy_results:
            return None

        windows = [
            (5, 'forward_5d_return'),
            (10, 'forward_10d_return'),
            (20, 'forward_20d_return'),
            (40, 'forward_40d_return'),
            (60, 'forward_60d_return'),
        ]

        best_ev = -999
        best_days = 20  # default

        for days, key in windows:
            vals = [r[key] for r in buy_results if r.get(key) is not None]
            if not vals:
                continue
            arr = np.array(vals)
            wins = arr[arr > 0]
            losses = arr[arr <= 0]
            wr = len(wins) / len(arr)
            lr = 1 - wr
            aw = float(np.mean(wins)) if len(wins) > 0 else 0
            al = abs(float(np.mean(losses))) if len(losses) > 0 else 0
            ev = (wr * aw) - (lr * al)
            if ev > best_ev:
                best_ev = ev
                best_days = days

        return best_days

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fundamental Snapshot Scoring (Feature 5)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_valuation_snapshot(snap: Dict) -> Optional[float]:
        """Score valuation from a stored fundamental snapshot."""
        scores = []
        pe = snap.get('pe_ratio')
        if pe is not None:
            if pe <= 0:
                scores.append(20)
            elif pe <= 10:
                scores.append(90)
            elif pe <= 20:
                scores.append(70)
            elif pe <= 30:
                scores.append(50)
            elif pe <= 50:
                scores.append(30)
            else:
                scores.append(15)

        pb = snap.get('pb_ratio')
        if pb is not None:
            if pb <= 1.0:
                scores.append(85)
            elif pb <= 2.0:
                scores.append(70)
            elif pb <= 4.0:
                scores.append(50)
            elif pb <= 8.0:
                scores.append(30)
            else:
                scores.append(15)

        return sum(scores) / len(scores) if scores else None

    @staticmethod
    def _score_quality_snapshot(snap: Dict) -> Optional[float]:
        """Score quality from a stored fundamental snapshot."""
        scores = []
        de = snap.get('debt_to_equity')
        if de is not None:
            if de < 30:
                scores.append(90)
            elif de < 80:
                scores.append(70)
            elif de < 150:
                scores.append(50)
            elif de < 300:
                scores.append(30)
            else:
                scores.append(10)

        roe = snap.get('roe')
        if roe is not None:
            if 15 <= roe <= 40:
                scores.append(85)
            elif 10 <= roe < 15:
                scores.append(65)
            elif 5 <= roe < 10:
                scores.append(45)
            elif roe > 40:
                scores.append(60)
            elif roe < 0:
                scores.append(15)
            else:
                scores.append(30)

        cr = snap.get('current_ratio')
        if cr is not None:
            if cr >= 2.0:
                scores.append(85)
            elif cr >= 1.5:
                scores.append(70)
            elif cr >= 1.0:
                scores.append(50)
            elif cr >= 0.5:
                scores.append(30)
            else:
                scores.append(10)

        fcf = snap.get('fcf_yield')
        if fcf is not None:
            if fcf > 8:
                scores.append(90)
            elif fcf > 5:
                scores.append(75)
            elif fcf > 2:
                scores.append(55)
            elif fcf > 0:
                scores.append(40)
            else:
                scores.append(15)

        return sum(scores) / len(scores) if scores else None

    @staticmethod
    def _compute_rsi(series, period: int = 14) -> Optional[float]:
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

    def _update_progress(self, run_id: int, pct: float, msg: str):
        with self._lock:
            self._progress['run_id'] = run_id
            self._progress['pct'] = pct
            self._progress['msg'] = msg
            self._progress['running'] = True
        db.update_backtest_run(run_id, progress_pct=pct, progress_msg=msg)


# Singleton
backtest_engine = BacktestEngine()
