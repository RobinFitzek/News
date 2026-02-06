"""
Quantitative Stock Screener
Pure math-based screening — zero API cost. Replaces AI-generated "interest scores"
with actual financial metrics computed from yfinance data.
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from core.config import QUANT_SCREENER_CONFIG
import logging

logger = logging.getLogger(__name__)


class SectorCache:
    """Caches sector median data to avoid redundant yfinance calls."""

    # Major sector peer tickers for median calculations
    SECTOR_PEERS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AVGO', 'ORCL', 'CRM'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT'],
        'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX'],
        'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'MDLZ', 'GIS'],
        'Industrials': ['CAT', 'DE', 'UNP', 'HON', 'MMM', 'GE', 'BA', 'RTX'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
        'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL'],
        'Real Estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'O', 'PSA', 'DLR', 'WELL'],
        'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'DOW'],
    }

    def __init__(self, cache_duration_minutes: int = 60):
        self._cache = {}
        self._cache_duration = timedelta(minutes=cache_duration_minutes)

    def get_sector_medians(self, sector: str, exclude_ticker: str = None) -> Dict:
        """Get median P/E, P/B for a sector. Cached for cache_duration."""
        cache_key = sector
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        peers = self.SECTOR_PEERS.get(sector, [])
        if not peers:
            return {'pe_median': None, 'pb_median': None, 'count': 0}

        pe_values = []
        pb_values = []

        for peer in peers:
            if peer == exclude_ticker:
                continue
            try:
                info = yf.Ticker(peer).info
                pe = info.get('trailingPE')
                pb = info.get('priceToBook')
                if pe and 0 < pe < 500:
                    pe_values.append(pe)
                if pb and 0 < pb < 100:
                    pb_values.append(pb)
            except Exception:
                continue

        data = {
            'pe_median': float(np.median(pe_values)) if pe_values else None,
            'pb_median': float(np.median(pb_values)) if pb_values else None,
            'pe_values': pe_values,
            'pb_values': pb_values,
            'count': len(pe_values),
        }

        self._cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
        return data


class QuantScreener:
    """Pure math screening — replaces AI Stage 1."""

    def __init__(self):
        self.sector_cache = SectorCache()
        self._benchmark_cache = {}
        self._stock_cache = {}
        self._cache_duration = timedelta(minutes=30)
        self.config = QUANT_SCREENER_CONFIG

    def screen_batch(self, tickers: list, variant: str = "balanced") -> list:
        """Screen multiple tickers, return sorted by composite score."""
        logger.info(f"Quant screening {len(tickers)} tickers (variant: {variant})")
        print(f"\n📊 QUANT SCREEN: Scanning {len(tickers)} tickers (zero API cost)")

        # Pre-fetch benchmark data once
        benchmark_hist = self._get_benchmark_history()

        results = []
        for ticker in tickers:
            try:
                result = self.screen_ticker(ticker, benchmark_hist, variant)
                if result and 'error' not in result:
                    results.append(result)
                    signal_icon = {'Opportunity': '+', 'Caution': '!', 'Neutral': '~'}
                    icon = signal_icon.get(result['signal'], '~')
                    print(f"  [{icon}] {ticker}: {result['composite_score']}/100 ({result['signal']})")
            except Exception as e:
                logger.error(f"Quant screen error for {ticker}: {e}")
                print(f"  [x] {ticker}: Error - {e}")

        results.sort(key=lambda x: x['composite_score'], reverse=True)
        print(f"  Screened {len(results)} tickers successfully")
        return results

    def screen_ticker(self, ticker: str, benchmark_hist=None, variant: str = "balanced") -> Dict:
        """Full quantitative screen for a single ticker."""
        ticker = ticker.upper().strip()

        # Fetch stock data (cached)
        stock_data = self._get_stock_data(ticker)
        if not stock_data:
            return {'ticker': ticker, 'error': 'No data available'}

        info = stock_data['info']
        hist = stock_data['hist']

        if hist.empty or len(hist) < 20:
            return {'ticker': ticker, 'error': 'Insufficient price history'}

        if benchmark_hist is None:
            benchmark_hist = self._get_benchmark_history()

        # Compute all metric groups
        valuation = self._relative_valuation(ticker, info)
        technicals = self._technical_indicators(hist)
        momentum = self._momentum_scores(hist, benchmark_hist)
        quality = self._quality_metrics(info)

        # Compute scores for each group (0-100)
        val_score = self._score_valuation(valuation, variant)
        tech_score = self._score_technicals(technicals)
        mom_score = self._score_momentum(momentum)
        qual_score = self._score_quality(quality)

        # Weighted composite
        weights = self.config['composite_weights']
        composite = (
            val_score * weights['valuation'] +
            tech_score * weights['technical'] +
            mom_score * weights['momentum'] +
            qual_score * weights['quality']
        )
        composite = max(0, min(100, int(round(composite))))

        # Anomaly detection
        anomalies = self._anomaly_detection(valuation, technicals, momentum, quality)

        # Signal determination
        signal = self._determine_signal(composite, anomalies, variant)

        return {
            'ticker': ticker,
            'composite_score': composite,
            'signal': signal,
            'scores': {
                'valuation': val_score,
                'technical': tech_score,
                'momentum': mom_score,
                'quality': qual_score,
            },
            'valuation': valuation,
            'technicals': technicals,
            'momentum': momentum,
            'quality': quality,
            'anomalies': anomalies,
            'data': {
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'market_cap': info.get('marketCap'),
            },
            # Compatibility fields for existing pipeline
            'score': composite,
            'initial_reason': f"Quant Score {composite}/100: Val={val_score}, Tech={tech_score}, Mom={mom_score}, Qual={qual_score}",
        }

    # --- Data Fetching (with caching) ---

    def _get_stock_data(self, ticker: str) -> Optional[Dict]:
        """Fetch and cache stock info + 1yr history."""
        cache_key = ticker
        if cache_key in self._stock_cache:
            entry = self._stock_cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or len(info) < 3:
                return None
            hist = stock.history(period="1y")
            data = {'info': info, 'hist': hist}
            self._stock_cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None

    def _get_benchmark_history(self):
        """Fetch SPY 1yr history, cached."""
        cache_key = 'benchmark'
        if cache_key in self._benchmark_cache:
            entry = self._benchmark_cache[cache_key]
            if datetime.now() - entry['timestamp'] < self._cache_duration:
                return entry['data']

        try:
            hist = yf.Ticker(self.config['benchmark_ticker']).history(period="1y")
            self._benchmark_cache[cache_key] = {'data': hist, 'timestamp': datetime.now()}
            return hist
        except Exception:
            return None

    # --- Metric Calculations ---

    def _relative_valuation(self, ticker: str, info: Dict) -> Dict:
        """P/E vs sector median, PEG ratio, P/B vs sector."""
        sector = info.get('sector', 'Unknown')
        sector_medians = self.sector_cache.get_sector_medians(sector, exclude_ticker=ticker)

        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        peg = info.get('pegRatio')

        # If PEG not available, compute manually
        if peg is None and pe and info.get('earningsGrowth'):
            eg = info['earningsGrowth'] * 100  # Convert to percentage
            if eg > 0:
                peg = pe / eg

        pe_vs_sector = None
        if pe and sector_medians.get('pe_median'):
            pe_vs_sector = round(pe / sector_medians['pe_median'], 2)

        pb_vs_sector = None
        if pb and sector_medians.get('pb_median'):
            pb_vs_sector = round(pb / sector_medians['pb_median'], 2)

        return {
            'pe_ratio': round(pe, 2) if pe else None,
            'pe_vs_sector': pe_vs_sector,
            'pb_ratio': round(pb, 2) if pb else None,
            'pb_vs_sector': pb_vs_sector,
            'peg_ratio': round(peg, 2) if peg else None,
            'sector': sector,
            'sector_pe_median': sector_medians.get('pe_median'),
            'sector_pb_median': sector_medians.get('pb_median'),
        }

    def _technical_indicators(self, hist) -> Dict:
        """RSI(14), SMA crossovers, 52-week position, Bollinger position."""
        close = hist['Close']

        # RSI (14-day)
        rsi = self._compute_rsi(close, 14)

        # SMAs
        sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        current_price = float(close.iloc[-1])

        # SMA cross signal
        sma_cross_signal = 'neutral'
        if sma_50 is not None and sma_200 is not None:
            if sma_50 > sma_200:
                # Check if it recently crossed (within last 10 days)
                sma50_series = close.rolling(50).mean()
                sma200_series = close.rolling(200).mean()
                if len(sma50_series) >= 210:
                    recent_diff = (sma50_series - sma200_series).iloc[-10:]
                    if any(d <= 0 for d in recent_diff.dropna()):
                        sma_cross_signal = 'golden_cross'
                    else:
                        sma_cross_signal = 'bullish'
                else:
                    sma_cross_signal = 'bullish'
            else:
                sma50_series = close.rolling(50).mean()
                sma200_series = close.rolling(200).mean()
                if len(sma50_series) >= 210:
                    recent_diff = (sma50_series - sma200_series).iloc[-10:]
                    if any(d >= 0 for d in recent_diff.dropna()):
                        sma_cross_signal = 'death_cross'
                    else:
                        sma_cross_signal = 'bearish'
                else:
                    sma_cross_signal = 'bearish'

        # Price vs 52-week range
        high_52w = float(close.max())
        low_52w = float(close.min())
        price_vs_52w = round((current_price - low_52w) / (high_52w - low_52w), 3) if high_52w != low_52w else 0.5

        # Bollinger Bands (20-day, 2 std dev)
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        if len(bb_sma.dropna()) > 0:
            upper = float(bb_sma.iloc[-1] + 2 * bb_std.iloc[-1])
            lower = float(bb_sma.iloc[-1] - 2 * bb_std.iloc[-1])
            bb_mid = float(bb_sma.iloc[-1])
            if upper != lower:
                bollinger_position = round((current_price - bb_mid) / (upper - bb_mid), 3)
            else:
                bollinger_position = 0.0
        else:
            bollinger_position = 0.0

        return {
            'rsi_14': round(rsi, 1) if rsi else None,
            'sma_50': round(sma_50, 2) if sma_50 else None,
            'sma_200': round(sma_200, 2) if sma_200 else None,
            'sma_cross_signal': sma_cross_signal,
            'price_vs_52w_range': price_vs_52w,
            'bollinger_position': bollinger_position,
            'current_price': round(current_price, 2),
        }

    def _momentum_scores(self, hist, benchmark_hist) -> Dict:
        """1M/3M/6M returns vs SPY benchmark."""
        close = hist['Close']
        current = float(close.iloc[-1])

        def pct_return(series, days):
            if len(series) < days:
                return None
            past = float(series.iloc[-days])
            if past == 0:
                return None
            return round(((current - past) / past) * 100, 2)

        ret_1m = pct_return(close, 21)
        ret_3m = pct_return(close, 63)
        ret_6m = pct_return(close, 126)

        # Benchmark returns
        excess_1m = excess_3m = excess_6m = None
        if benchmark_hist is not None and not benchmark_hist.empty:
            bench_close = benchmark_hist['Close']
            bench_current = float(bench_close.iloc[-1])

            def bench_return(days):
                if len(bench_close) < days:
                    return None
                past = float(bench_close.iloc[-days])
                if past == 0:
                    return None
                return round(((bench_current - past) / past) * 100, 2)

            b_1m = bench_return(21)
            b_3m = bench_return(63)
            b_6m = bench_return(126)

            if ret_1m is not None and b_1m is not None:
                excess_1m = round(ret_1m - b_1m, 2)
            if ret_3m is not None and b_3m is not None:
                excess_3m = round(ret_3m - b_3m, 2)
            if ret_6m is not None and b_6m is not None:
                excess_6m = round(ret_6m - b_6m, 2)

        return {
            'return_1m': ret_1m,
            'return_3m': ret_3m,
            'return_6m': ret_6m,
            'excess_1m': excess_1m,
            'excess_3m': excess_3m,
            'excess_6m': excess_6m,
        }

    def _quality_metrics(self, info: Dict) -> Dict:
        """Debt/equity, current ratio, ROE, FCF yield."""
        de = info.get('debtToEquity')
        cr = info.get('currentRatio')
        roe = info.get('returnOnEquity')

        fcf = info.get('freeCashflow')
        mcap = info.get('marketCap')
        fcf_yield = None
        if fcf and mcap and mcap > 0:
            fcf_yield = round((fcf / mcap) * 100, 2)

        return {
            'debt_to_equity': round(de, 2) if de else None,
            'current_ratio': round(cr, 2) if cr else None,
            'roe': round(roe * 100, 2) if roe else None,
            'fcf_yield': fcf_yield,
        }

    # --- Scoring Functions (each returns 0-100) ---

    def _score_valuation(self, val: Dict, variant: str) -> float:
        """Score valuation metrics. Lower P/E vs sector = higher score."""
        scores = []

        # P/E vs sector (ratio < 1 = cheap)
        if val['pe_vs_sector'] is not None:
            ratio = val['pe_vs_sector']
            if ratio <= 0.5:
                scores.append(95)
            elif ratio <= 0.8:
                scores.append(80)
            elif ratio <= 1.0:
                scores.append(60)
            elif ratio <= 1.3:
                scores.append(40)
            elif ratio <= 1.8:
                scores.append(25)
            else:
                scores.append(10)

        # PEG ratio (< 1 = undervalued for growth)
        if val['peg_ratio'] is not None:
            peg = val['peg_ratio']
            if peg < 0:
                scores.append(20)  # Negative earnings growth
            elif peg <= 0.5:
                scores.append(95)
            elif peg <= 1.0:
                scores.append(75)
            elif peg <= 1.5:
                scores.append(55)
            elif peg <= 2.0:
                scores.append(35)
            else:
                scores.append(15)

        # P/B vs sector
        if val['pb_vs_sector'] is not None:
            ratio = val['pb_vs_sector']
            if ratio <= 0.5:
                scores.append(90)
            elif ratio <= 0.8:
                scores.append(75)
            elif ratio <= 1.2:
                scores.append(55)
            elif ratio <= 2.0:
                scores.append(30)
            else:
                scores.append(10)

        if not scores:
            return 50.0  # No data, neutral
        return sum(scores) / len(scores)

    def _score_technicals(self, tech: Dict) -> float:
        """Score technical indicators."""
        scores = []

        # RSI scoring (30-70 is healthy)
        if tech['rsi_14'] is not None:
            rsi = tech['rsi_14']
            if 40 <= rsi <= 60:
                scores.append(70)  # Healthy range
            elif 30 <= rsi <= 40:
                scores.append(80)  # Approaching oversold = opportunity
            elif rsi < 30:
                scores.append(85)  # Oversold = potential bounce
            elif 60 <= rsi <= 70:
                scores.append(55)  # Getting warm
            else:
                scores.append(25)  # Overbought

        # SMA cross signal
        cross_scores = {
            'golden_cross': 90,
            'bullish': 70,
            'neutral': 50,
            'bearish': 30,
            'death_cross': 15,
        }
        scores.append(cross_scores.get(tech['sma_cross_signal'], 50))

        # Price vs 52-week range (lower = more upside potential)
        pos = tech['price_vs_52w_range']
        if pos <= 0.2:
            scores.append(85)  # Near 52-week low
        elif pos <= 0.4:
            scores.append(70)
        elif pos <= 0.6:
            scores.append(55)
        elif pos <= 0.8:
            scores.append(40)
        else:
            scores.append(20)  # Near 52-week high

        # Bollinger position (-1 to +1, negative = below middle band)
        bb = tech['bollinger_position']
        if bb <= -0.8:
            scores.append(85)  # Near lower band
        elif bb <= -0.3:
            scores.append(70)
        elif bb <= 0.3:
            scores.append(55)
        elif bb <= 0.8:
            scores.append(35)
        else:
            scores.append(15)  # Near upper band

        return sum(scores) / len(scores) if scores else 50.0

    def _score_momentum(self, mom: Dict) -> float:
        """Score momentum — positive excess returns = higher score."""
        scores = []

        for key in ['excess_1m', 'excess_3m', 'excess_6m']:
            excess = mom.get(key)
            if excess is not None:
                if excess > 10:
                    scores.append(90)
                elif excess > 5:
                    scores.append(75)
                elif excess > 0:
                    scores.append(60)
                elif excess > -5:
                    scores.append(40)
                elif excess > -10:
                    scores.append(25)
                else:
                    scores.append(10)

        # Absolute momentum trend (accelerating or decelerating)
        if mom['return_1m'] is not None and mom['return_3m'] is not None:
            monthly_avg = mom['return_3m'] / 3 if mom['return_3m'] else 0
            if mom['return_1m'] > monthly_avg and mom['return_1m'] > 0:
                scores.append(75)  # Accelerating positive momentum
            elif mom['return_1m'] < monthly_avg and mom['return_1m'] < 0:
                scores.append(25)  # Accelerating negative momentum
            else:
                scores.append(50)

        return sum(scores) / len(scores) if scores else 50.0

    def _score_quality(self, qual: Dict) -> float:
        """Score quality metrics — low debt, high ROE, positive FCF."""
        scores = []

        # Debt to equity (lower is better)
        de = qual['debt_to_equity']
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

        # Current ratio (> 1.5 is healthy)
        cr = qual['current_ratio']
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

        # ROE (higher is better, but not suspiciously high)
        roe = qual['roe']
        if roe is not None:
            if 15 <= roe <= 40:
                scores.append(85)
            elif 10 <= roe < 15:
                scores.append(65)
            elif 5 <= roe < 10:
                scores.append(45)
            elif roe > 40:
                scores.append(60)  # Very high ROE can indicate leverage
            elif roe < 0:
                scores.append(15)
            else:
                scores.append(30)

        # FCF yield (higher is better)
        fcf = qual['fcf_yield']
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

        return sum(scores) / len(scores) if scores else 50.0

    # --- Anomaly Detection ---

    def _anomaly_detection(self, valuation: Dict, technicals: Dict,
                           momentum: Dict, quality: Dict) -> List[Dict]:
        """Flag metrics that are extreme outliers."""
        anomalies = []
        z_threshold = self.config['anomaly_z_threshold']

        # Valuation anomalies
        if valuation['pe_vs_sector'] is not None:
            if valuation['pe_vs_sector'] < 0.5:
                anomalies.append({
                    'metric': 'P/E vs Sector',
                    'value': valuation['pe_vs_sector'],
                    'direction': 'positive',
                    'description': f"Trading at {valuation['pe_vs_sector']}x sector P/E median (significantly undervalued)"
                })
            elif valuation['pe_vs_sector'] > 2.0:
                anomalies.append({
                    'metric': 'P/E vs Sector',
                    'value': valuation['pe_vs_sector'],
                    'direction': 'negative',
                    'description': f"Trading at {valuation['pe_vs_sector']}x sector P/E median (significantly overvalued)"
                })

        # Technical anomalies
        if technicals['rsi_14'] is not None:
            if technicals['rsi_14'] < 25:
                anomalies.append({
                    'metric': 'RSI(14)',
                    'value': technicals['rsi_14'],
                    'direction': 'positive',
                    'description': f"RSI at {technicals['rsi_14']} — deeply oversold"
                })
            elif technicals['rsi_14'] > 80:
                anomalies.append({
                    'metric': 'RSI(14)',
                    'value': technicals['rsi_14'],
                    'direction': 'negative',
                    'description': f"RSI at {technicals['rsi_14']} — heavily overbought"
                })

        if technicals['sma_cross_signal'] == 'golden_cross':
            anomalies.append({
                'metric': 'SMA Crossover',
                'value': 'golden_cross',
                'direction': 'positive',
                'description': "SMA50 just crossed above SMA200 (Golden Cross)"
            })
        elif technicals['sma_cross_signal'] == 'death_cross':
            anomalies.append({
                'metric': 'SMA Crossover',
                'value': 'death_cross',
                'direction': 'negative',
                'description': "SMA50 just crossed below SMA200 (Death Cross)"
            })

        # Momentum anomalies
        for period, key in [('1M', 'excess_1m'), ('3M', 'excess_3m'), ('6M', 'excess_6m')]:
            val = momentum.get(key)
            if val is not None and abs(val) > 20:
                direction = 'positive' if val > 0 else 'negative'
                anomalies.append({
                    'metric': f'Excess Return ({period})',
                    'value': val,
                    'direction': direction,
                    'description': f"{period} return is {val:+.1f}% vs benchmark — extreme {'outperformance' if val > 0 else 'underperformance'}"
                })

        # Quality anomalies
        if quality['debt_to_equity'] is not None and quality['debt_to_equity'] > 300:
            anomalies.append({
                'metric': 'Debt/Equity',
                'value': quality['debt_to_equity'],
                'direction': 'negative',
                'description': f"D/E ratio of {quality['debt_to_equity']} — dangerously high leverage"
            })

        if quality['fcf_yield'] is not None and quality['fcf_yield'] < -5:
            anomalies.append({
                'metric': 'FCF Yield',
                'value': quality['fcf_yield'],
                'direction': 'negative',
                'description': f"Negative FCF yield of {quality['fcf_yield']}% — burning cash"
            })

        return anomalies

    def _determine_signal(self, composite: int, anomalies: List[Dict], variant: str) -> str:
        """Determine signal based on composite score and anomalies."""
        opp_threshold = self.config['opportunity_threshold']
        caut_threshold = self.config['caution_threshold']

        # Adjust thresholds for variant
        if variant == 'aggressive':
            opp_threshold -= 10
            caut_threshold -= 5
        elif variant == 'conservative':
            opp_threshold += 5
            caut_threshold += 5

        positive_anomalies = [a for a in anomalies if a['direction'] == 'positive']
        negative_anomalies = [a for a in anomalies if a['direction'] == 'negative']

        # Critical negative anomalies override everything
        critical_negatives = [a for a in negative_anomalies
                              if a['metric'] in ('Debt/Equity', 'FCF Yield')]
        if critical_negatives:
            return 'Caution'

        if composite >= opp_threshold and len(positive_anomalies) >= 1:
            return 'Opportunity'
        elif composite <= caut_threshold or len(negative_anomalies) >= 2:
            return 'Caution'
        else:
            return 'Neutral'

    # --- Utility ---

    @staticmethod
    def _compute_rsi(series, period: int = 14) -> Optional[float]:
        """Compute RSI using exponential moving average."""
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
quant_screener = QuantScreener()
