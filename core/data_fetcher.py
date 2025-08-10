"""
Optimized data fetching with caching and error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st
from .cache_manager import cache

class DataFetcher:
    """Optimized data fetching with intelligent caching"""
    
    def __init__(self):
        self.sector_etfs = {
            'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF',
            'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Energy': 'XLE',
            'Industrials': 'XLI', 'Real Estate': 'XLRE', 'Utilities': 'XLU',
            'Materials': 'XLB', 'Communication Services': 'XLC'
        }
        
        self.market_indices = {
            'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'DOW': '^DJI', 'VIX': '^VIX'
        }
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get stock data with caching"""
        cache_params = {'symbol': symbol, 'period': period, 'interval': interval}
        
        # Check cache first (15 min TTL for intraday, 60 min for daily)
        ttl = 15 if interval in ['1m', '5m', '15m', '30m', '1h'] else 60
        cached_data = cache.get('stock_data', cache_params, ttl)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Track API call
            if hasattr(st.session_state, 'performance_monitor'):
                st.session_state.performance_monitor.track_api_call()
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if not data.empty:
                # Add technical indicators efficiently
                data = self._add_technical_indicators(data)
                cache.set('stock_data', cache_params, data)
                return data
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return None
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get stock info with caching"""
        cache_params = {'symbol': symbol}
        cached_info = cache.get('stock_info', cache_params, 240)  # 4 hour TTL
        
        if cached_info is not None:
            return cached_info
        
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            cache.set('stock_info', cache_params, info)
            return info
        except Exception as e:
            st.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    def get_market_indices(self) -> Dict[str, Dict]:
        """Get market indices data with caching"""
        cache_params = {'type': 'market_indices'}
        cached_data = cache.get('market_data', cache_params, 30)  # 30 min TTL
        
        if cached_data is not None:
            return cached_data
        
        indices_data = {}
        for name, symbol in self.market_indices.items():
            try:
                data = self.get_stock_data(symbol, period="5d", interval="1h")
                if data is not None and not data.empty:
                    current = data['Close'].iloc[-1]
                    prev_day = data['Close'].iloc[-24] if len(data) >= 24 else current
                    change_1d = (current - prev_day) / prev_day if prev_day != 0 else 0
                    
                    indices_data[name] = {
                        'current': current,
                        'change_1d': change_1d,
                        'volatility': data['Close'].pct_change().std() * np.sqrt(252 * 24),
                        'symbol': symbol
                    }
            except Exception as e:
                continue
        
        cache.set('market_data', cache_params, indices_data)
        return indices_data
    
    def get_sector_data(self, sector: str) -> Optional[Dict]:
        """Get sector ETF data"""
        etf_symbol = self.sector_etfs.get(sector)
        if not etf_symbol:
            return None
        
        cache_params = {'sector': sector, 'etf': etf_symbol}
        cached_data = cache.get('sector_data', cache_params, 60)
        
        if cached_data is not None:
            return cached_data
        
        try:
            data = self.get_stock_data(etf_symbol, period="1mo")
            if data is not None and not data.empty:
                current = data['Close'].iloc[-1]
                week_ago = data['Close'].iloc[-5] if len(data) >= 5 else current
                month_ago = data['Close'].iloc[0]
                
                sector_data = {
                    'symbol': etf_symbol,
                    'current': current,
                    'change_1w': (current - week_ago) / week_ago if week_ago != 0 else 0,
                    'change_1m': (current - month_ago) / month_ago if month_ago != 0 else 0
                }
                
                cache.set('sector_data', cache_params, sector_data)
                return sector_data
        except Exception as e:
            st.error(f"Error fetching sector data for {sector}: {str(e)}")
        
        return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators efficiently (vectorized operations)"""
        # Moving averages
        data['MA_5'] = data['Close'].rolling(5).mean()
        data['MA_20'] = data['Close'].rolling(20).mean()
        data['MA_50'] = data['Close'].rolling(50).mean()
        data['MA_200'] = data['Close'].rolling(200).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['MA_20'] + (2 * bb_std)
        data['BB_Lower'] = data['MA_20'] - (2 * bb_std)
        
        # ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Volatility
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        return data

# Global data fetcher instance
data_fetcher = DataFetcher()