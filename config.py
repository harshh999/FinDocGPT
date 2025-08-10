"""
Configuration file for FinDocGPT
Contains API keys, dataset paths, and system settings
"""

import os
from typing import Dict, List

class Config:
    # API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'HFAMHP0XVRPT5CDA')
    QUANDL_API_KEY = os.getenv('QUANDL_API_KEY', 'demo')
    
    # FinanceBench Dataset Configuration
    FINANCEBENCH_DATA_PATH = os.getenv('FINANCEBENCH_DATA_PATH', './data/financebench/')
    
    # Model Configuration
    MODEL_CACHE_DIR = './models/'
    SCALER_CACHE_DIR = './scalers/'
    
    # Forecasting Parameters
    DEFAULT_FORECAST_DAYS = 30
    MIN_TRAINING_SAMPLES = 50
    TECHNICAL_INDICATORS_WINDOW = 20
    
    # Investment Strategy Parameters
    RISK_FREE_RATE = 0.02  # 2%
    MARKET_RETURN = 0.10   # 10%
    
    # Sentiment Analysis
    SENTIMENT_THRESHOLD_POSITIVE = 0.1
    SENTIMENT_THRESHOLD_NEGATIVE = -0.1
    
    # Risk Assessment Thresholds
    HIGH_VOLATILITY_THRESHOLD = 0.03
    MEDIUM_VOLATILITY_THRESHOLD = 0.02
    
    # Portfolio Optimization
    MAX_POSITION_SIZE = 0.20  # 20% max per position
    MIN_POSITION_SIZE = 0.05  # 5% min per position
    
    # Data Sources
    YAHOO_FINANCE_ENABLED = True
    ALPHA_VANTAGE_ENABLED = True
    QUANDL_ENABLED = True
    
    # Supported Stock Symbols for Demo
    DEMO_SYMBOLS = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        'META', 'NVDA', 'JPM', 'JNJ', 'V'
    ]
    
    # Financial Sectors
    SECTORS = {
        'Technology': ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS'],
        'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG']
    }
    
    # FinanceBench Dataset Structure
    FINANCEBENCH_CATEGORIES = {
        'earnings_reports': 'earnings/',
        'market_data': 'market/',
        'sentiment_data': 'sentiment/',
        'news_articles': 'news/'
    }
    
    # Evaluation Metrics
    EVALUATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'mae',  # Mean Absolute Error
        'mse',  # Mean Squared Error
        'rmse', # Root Mean Squared Error
        'sharpe_ratio',
        'max_drawdown'
    ]
    
    @classmethod
    def get_api_key(cls, service: str) -> str:
        """Get API key for specified service"""
        keys = {
            'alpha_vantage': cls.ALPHA_VANTAGE_API_KEY,
            'quandl': cls.QUANDL_API_KEY
        }
        return keys.get(service.lower(), 'demo')
    
    @classmethod
    def get_sector_symbols(cls, sector: str) -> List[str]:
        """Get stock symbols for a specific sector"""
        return cls.SECTORS.get(sector, [])
    
    @classmethod
    def is_demo_mode(cls) -> bool:
        """Check if running in demo mode (no real API keys)"""
        return (cls.ALPHA_VANTAGE_API_KEY == 'demo' and 
                cls.QUANDL_API_KEY == 'demo')

# Benchmarking Configuration
BENCHMARK_CONFIG = {
    'prediction_accuracy_threshold': 0.70,  # 70% accuracy target
    'qa_accuracy_threshold': 0.80,          # 80% Q&A accuracy target
    'recommendation_success_rate': 0.65,    # 65% successful recommendations
    'response_time_threshold': 5.0,         # 5 seconds max response time
    'user_satisfaction_threshold': 4.0      # 4/5 user satisfaction rating
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'findocgpt.log'
}