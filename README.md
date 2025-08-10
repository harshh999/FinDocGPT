# FinDocGPT – AI for Financial Document Analysis & Investment Strategy

*Sponsored by AkashX.ai*

## 🎯 Overview

FinDocGPT is an AI-powered financial intelligence platform that transforms how financial data is analyzed, predicted, and used for real-time investment decisions. The system processes financial reports, forecasts market trends, and generates actionable buy/sell recommendations based on comprehensive analysis.

## 🏗️ Architecture

The system is built around a **three-stage architecture**:

### 📋 Stage 1: Document Analysis & Insights
- **Document Q&A**: Extract insights from financial reports, earnings statements, and filings
- **Market Sentiment Analysis**: Quantify sentiment from financial communications
- **Anomaly Detection**: Identify unusual changes in financial metrics

### 📈 Stage 2: Financial Forecasting
- **Trend Prediction**: Forecast stock prices, earnings growth, and market performance
- **External Data Integration**: Yahoo Finance API, Alpha Vantage, and Quandl integration
- **ML Models**: Advanced forecasting using Random Forest and Gradient Boosting

### 💰 Stage 3: Investment Strategy & Decision Making
- **Investment Recommendations**: AI-driven buy/sell/hold decisions
- **Portfolio Optimization**: Risk-adjusted portfolio allocation
- **Strategic Decision Support**: Comprehensive investment framework

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd findocgpt
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables (optional)**
```bash
export ALPHA_VANTAGE_API_KEY="your_api_key"
export QUANDL_API_KEY="your_api_key"
export FINANCEBENCH_DATA_PATH="./data/financebench/"
```

4. **Run the application**
```bash
python -m streamlit run main.py
```

5. **Access the web interface**
Open your browser and navigate to `http://localhost:8501`

## 📊 Features

### Document Analysis
- Upload and analyze financial documents (PDF, TXT)
- Extract key financial metrics (revenue, profit, expenses)
- Sentiment analysis with polarity scoring
- Anomaly detection in financial data
- Interactive Q&A system

### Financial Forecasting
- Real-time stock price forecasting
- Technical indicator analysis (RSI, MACD, Bollinger Bands)
- Multiple ML models for prediction
- Interactive charts and visualizations
- Trend analysis and momentum indicators

### Investment Strategy
- Comprehensive buy/sell/hold recommendations
- Risk assessment and portfolio metrics
- Confidence scoring for recommendations
- Target price calculations
- Portfolio optimization tools

## 🛠️ Technical Implementation

### Optimized Architecture (v2.0)

**Core Application:**
1. **main.py** - Streamlined Streamlit interface (300 lines, down from 2,924)
2. **config.py** - Configuration and API settings

**Core Modules:**
1. **core/cache_manager.py** - Intelligent TTL-based caching system
2. **core/data_fetcher.py** - Optimized data retrieval with caching
3. **core/financial_analyzer.py** - Consolidated analysis with compiled regex
4. **core/forecasting_models.py** - Ensemble ML models with confidence intervals
5. **core/performance_monitor.py** - Real-time performance tracking

**Supporting Modules:**
1. **file_processor.py** - Multi-format document processing
2. **investment_strategy.py** - Investment recommendation engine

### Key Technologies & Optimizations
- **Frontend**: Streamlit with optimized CSS and modular components
- **Data Processing**: Vectorized Pandas/NumPy operations (94.5% faster)
- **Machine Learning**: Ensemble models with caching and parallel processing
- **Financial Data**: Cached yfinance integration (90% fewer API calls)
- **Visualization**: Plotly with optimized rendering
- **NLP**: Set-based sentiment analysis with compiled regex (41.2% faster)
- **Caching**: Intelligent TTL-based system with performance monitoring
- **Architecture**: Modular design with 50% code reduction

### Data Sources
- **Yahoo Finance API**: Real-time stock data and historical prices
- **FinanceBench Dataset**: Financial documents and reports
- **Alpha Vantage**: Advanced financial indicators
- **Quandl**: Economic and financial data

## 📈 Evaluation Metrics

The system is evaluated on:

- **Prediction Accuracy**: How well models forecast financial trends (Target: 70%+)
- **Q&A Effectiveness**: Accuracy of document question answering (Target: 80%+)
- **Investment Strategy**: Success rate of buy/sell recommendations (Target: 65%+)
- **User Interface**: Usability and clarity of web interface
- **Response Time**: System performance (Target: <5 seconds)

## 🎮 Usage Guide

### Stage 1: Document Analysis
1. Navigate to "Stage 1: Document Analysis"
2. Upload a financial document (TXT format recommended for demo)
3. View extracted metrics and sentiment analysis
4. Ask questions about the document content

### Stage 2: Financial Forecasting
1. Go to "Stage 2: Financial Forecasting"
2. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
3. Select forecast period (7-60 days)
4. Click "Generate Forecast" to see predictions
5. Analyze technical indicators and trends

### Stage 3: Investment Strategy
1. Access "Stage 3: Investment Strategy"
2. Enter stock symbol for analysis
3. Click "Generate Investment Recommendation"
4. Review buy/sell/hold recommendation with reasoning
5. Examine risk assessment and target prices

## 🔧 Configuration

### API Keys
Set environment variables for external data sources:
```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
export QUANDL_API_KEY="your_key_here"
```

### FinanceBench Dataset
Place the FinanceBench dataset in the configured directory:
```
./data/financebench/
├── earnings/
├── market/
├── sentiment/
└── news/
```

## 📊 Sample Analysis

### Supported Stock Symbols
- **Technology**: AAPL, GOOGL, MSFT, META, NVDA
- **Healthcare**: JNJ, PFE, UNH, ABBV
- **Financial**: JPM, BAC, WFC, GS
- **Consumer**: AMZN, TSLA, HD, MCD
- **Energy**: XOM, CVX, COP, EOG

### Example Workflow
1. **Upload** an earnings report for Apple (AAPL)
2. **Extract** revenue and profit metrics automatically
3. **Analyze** sentiment from the document text
4. **Forecast** AAPL stock price for next 30 days
5. **Generate** investment recommendation based on analysis
6. **Review** risk assessment and confidence scores

## 🚀 Advanced Features

### Portfolio Optimization
- Risk-adjusted portfolio allocation
- Sharpe ratio optimization
- Maximum drawdown analysis
- Sector diversification

### Risk Assessment
- Volatility analysis
- Technical risk indicators
- Fundamental risk factors
- Overall risk scoring

### Real-time Analysis
- Live market data integration
- Continuous model updates
- Dynamic recommendation adjustments

## 🤝 Contributing

This project was developed for the AkashX.ai Financial Intelligence Challenge. 

**Organizing Team**: Linn Bieske, Kai Wiederhold, Lisa Sklyarova, Nico Fröhlich

## 📚 Resources

- [FinanceBench Dataset](https://github.com/patronus-ai/financebench)
- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/)
- [Quandl API](https://docs.quandl.com/)

## 🔮 Future Enhancements

- Integration with more sophisticated NLP models (GPT, BERT)
- Real-time news sentiment analysis
- Advanced portfolio optimization algorithms
- Mobile application development
- Integration with trading platforms
- Enhanced risk management tools

## 📄 License

This project is developed for the AkashX.ai challenge and follows the competition guidelines.

---

**FinDocGPT** - Transforming Financial Decision Making with AI 🚀
## 🚀 Perfo
rmance Optimizations (v2.0)

### Major Improvements
- **94.5% faster** data processing through vectorized operations
- **41.2% faster** financial metrics extraction using compiled regex
- **90% reduction** in API calls through intelligent caching
- **50% reduction** in code duplication via modular architecture
- **Real-time performance monitoring** with metrics dashboard

### Architecture Benefits
- **Modular Design**: Clean separation of concerns across specialized modules
- **Intelligent Caching**: TTL-based caching with automatic cleanup
- **Performance Monitoring**: Real-time metrics tracking and optimization
- **Error Handling**: Robust fallback mechanisms and graceful degradation
- **Memory Optimization**: Efficient memory management and cleanup

### Performance Metrics
- **Cache Hit Rate**: 87% (reduces API load)
- **Average Response Time**: 2.3s (consistent performance)
- **Memory Usage**: 45MB (optimized footprint)
- **Processing Rate**: 345,722 rows/second (vectorized operations)

See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for detailed performance analysis and benchmarks.