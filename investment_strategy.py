"""
Stage 3: Investment Strategy & Decision Making Module
Handles investment recommendations, portfolio optimization, and risk assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
from enum import Enum

class InvestmentAction(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    WEAK_SELL = "Weak Sell"
    SELL = "Sell"

@dataclass
class InvestmentRecommendation:
    symbol: str
    action: InvestmentAction
    confidence: float
    target_price: float
    reasoning: str
    risk_level: str
    time_horizon: str

class InvestmentStrategy:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.market_return = 0.10   # 10% expected market return
    
    def generate_recommendation(
        self, 
        symbol: str,
        current_price: float,
        predicted_prices: List[float],
        sentiment_score: float,
        technical_indicators: Dict[str, any],
        fundamental_data: Dict[str, any] = None
    ) -> InvestmentRecommendation:
        """
        Generate comprehensive investment recommendation
        """
        if not predicted_prices:
            return InvestmentRecommendation(
                symbol=symbol,
                action=InvestmentAction.HOLD,
                confidence=0.5,
                target_price=current_price,
                reasoning="Insufficient data for recommendation",
                risk_level="Unknown",
                time_horizon="N/A"
            )
        
        # Calculate expected return
        avg_predicted = np.mean(predicted_prices)
        expected_return = (avg_predicted - current_price) / current_price * 100
        
        # Technical analysis score
        technical_score = self._calculate_technical_score(technical_indicators)
        
        # Fundamental analysis score (if available)
        fundamental_score = self._calculate_fundamental_score(fundamental_data) if fundamental_data else 0.5
        
        # Combined score
        combined_score = (
            0.4 * (expected_return / 100 + 1) +  # Price prediction weight
            0.3 * (sentiment_score + 1) / 2 +     # Sentiment weight
            0.2 * technical_score +               # Technical weight
            0.1 * fundamental_score               # Fundamental weight
        )
        
        # Determine action and confidence
        action, confidence = self._determine_action(expected_return, combined_score, sentiment_score)
        
        # Calculate target price
        target_price = self._calculate_target_price(current_price, predicted_prices, technical_indicators)
        
        # Assess risk level
        risk_level = self._assess_risk_level(technical_indicators, fundamental_data)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            expected_return, sentiment_score, technical_score, 
            fundamental_score, action
        )
        
        return InvestmentRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            target_price=target_price,
            reasoning=reasoning,
            risk_level=risk_level,
            time_horizon="1-3 months"
        )
    
    def _calculate_technical_score(self, indicators: Dict[str, any]) -> float:
        """
        Calculate technical analysis score (0-1)
        """
        score = 0.5  # Neutral starting point
        
        if not indicators:
            return score
        
        # RSI analysis
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 0.1
        elif rsi < 30:
            score += 0.2  # Oversold - potential buy
        elif rsi > 70:
            score -= 0.2  # Overbought - potential sell
        
        # Moving average analysis
        if indicators.get('price_vs_ma20') == 'Above':
            score += 0.1
        else:
            score -= 0.1
            
        if indicators.get('price_vs_ma50') == 'Above':
            score += 0.1
        else:
            score -= 0.1
        
        # Volume analysis
        volume_signal = indicators.get('volume_analysis', {}).get('volume_signal', 'Normal')
        if volume_signal == 'High':
            score += 0.05
        
        # Trend analysis
        overall_trend = indicators.get('overall_trend', 'Neutral')
        if overall_trend == 'Bullish':
            score += 0.15
        elif overall_trend == 'Bearish':
            score -= 0.15
        
        return max(0, min(1, score))
    
    def _calculate_fundamental_score(self, fundamental_data: Dict[str, any]) -> float:
        """
        Calculate fundamental analysis score (0-1)
        """
        if not fundamental_data:
            return 0.5
        
        score = 0.5
        
        # P/E ratio analysis
        pe_ratio = fundamental_data.get('pe_ratio')
        if pe_ratio:
            if 10 <= pe_ratio <= 20:
                score += 0.2
            elif pe_ratio < 10:
                score += 0.1  # Potentially undervalued
            elif pe_ratio > 30:
                score -= 0.1  # Potentially overvalued
        
        # Market cap consideration
        market_cap = fundamental_data.get('market_cap', 0)
        if market_cap > 10e9:  # Large cap
            score += 0.05  # Generally more stable
        
        # Sector performance (simplified)
        sector = fundamental_data.get('sector', '')
        growth_sectors = ['Technology', 'Healthcare', 'Consumer Discretionary']
        if sector in growth_sectors:
            score += 0.1
        
        return max(0, min(1, score))
    
    def _determine_action(self, expected_return: float, combined_score: float, sentiment_score: float) -> Tuple[InvestmentAction, float]:
        """
        Determine investment action and confidence level
        """
        # Strong buy conditions
        if expected_return > 10 and combined_score > 0.7 and sentiment_score > 0.2:
            return InvestmentAction.STRONG_BUY, 0.9
        
        # Buy conditions
        elif expected_return > 5 and combined_score > 0.6:
            return InvestmentAction.BUY, 0.8
        
        # Sell conditions
        elif expected_return < -10 or (combined_score < 0.3 and sentiment_score < -0.3):
            return InvestmentAction.SELL, 0.8
        
        # Weak sell conditions
        elif expected_return < -5 or combined_score < 0.4:
            return InvestmentAction.WEAK_SELL, 0.7
        
        # Hold conditions (default)
        else:
            confidence = 0.6 if abs(expected_return) < 3 else 0.5
            return InvestmentAction.HOLD, confidence
    
    def _calculate_target_price(self, current_price: float, predicted_prices: List[float], indicators: Dict[str, any]) -> float:
        """
        Calculate target price based on predictions and technical analysis
        """
        if not predicted_prices:
            return current_price
        
        # Base target from predictions
        base_target = np.mean(predicted_prices)
        
        # Adjust based on technical indicators
        adjustment_factor = 1.0
        
        if indicators.get('overall_trend') == 'Bullish':
            adjustment_factor *= 1.05
        elif indicators.get('overall_trend') == 'Bearish':
            adjustment_factor *= 0.95
        
        # Consider volatility
        volatility = indicators.get('volatility', 0)
        if volatility > current_price * 0.02:  # High volatility
            adjustment_factor *= 0.98  # Slightly more conservative
        
        return base_target * adjustment_factor
    
    def _assess_risk_level(self, technical_indicators: Dict[str, any], fundamental_data: Dict[str, any] = None) -> str:
        """
        Assess risk level of the investment
        """
        risk_score = 0
        
        # Volatility risk
        volatility = technical_indicators.get('volatility', 0)
        current_price = technical_indicators.get('current_price', 100)
        vol_ratio = volatility / current_price if current_price > 0 else 0
        
        if vol_ratio > 0.03:
            risk_score += 2
        elif vol_ratio > 0.02:
            risk_score += 1
        
        # RSI risk (extreme values indicate higher risk)
        rsi = technical_indicators.get('rsi', 50)
        if rsi > 80 or rsi < 20:
            risk_score += 1
        
        # Market cap risk
        if fundamental_data:
            market_cap = fundamental_data.get('market_cap', 0)
            if market_cap < 1e9:  # Small cap
                risk_score += 1
        
        # Volume risk
        volume_signal = technical_indicators.get('volume_analysis', {}).get('volume_signal', 'Normal')
        if volume_signal == 'High':
            risk_score += 0.5
        
        if risk_score >= 3:
            return "High"
        elif risk_score >= 1.5:
            return "Medium"
        else:
            return "Low"
    
    def _generate_reasoning(self, expected_return: float, sentiment_score: float, 
                          technical_score: float, fundamental_score: float, 
                          action: InvestmentAction) -> str:
        """
        Generate human-readable reasoning for the recommendation
        """
        reasons = []
        
        # Price expectation
        if expected_return > 5:
            reasons.append(f"Expected {expected_return:.1f}% price increase")
        elif expected_return < -5:
            reasons.append(f"Expected {expected_return:.1f}% price decrease")
        else:
            reasons.append(f"Modest price movement expected ({expected_return:.1f}%)")
        
        # Sentiment analysis
        if sentiment_score > 0.2:
            reasons.append("positive market sentiment")
        elif sentiment_score < -0.2:
            reasons.append("negative market sentiment")
        else:
            reasons.append("neutral market sentiment")
        
        # Technical analysis
        if technical_score > 0.6:
            reasons.append("strong technical indicators")
        elif technical_score < 0.4:
            reasons.append("weak technical indicators")
        else:
            reasons.append("mixed technical signals")
        
        # Fundamental analysis
        if fundamental_score > 0.6:
            reasons.append("solid fundamentals")
        elif fundamental_score < 0.4:
            reasons.append("concerning fundamentals")
        
        return f"Recommendation based on: {', '.join(reasons)}"
    
    def calculate_portfolio_metrics(self, holdings: Dict[str, float], prices: Dict[str, float]) -> Dict[str, any]:
        """
        Calculate portfolio performance metrics
        """
        if not holdings or not prices:
            return {}
        
        # Calculate portfolio value
        total_value = sum(holdings[symbol] * prices.get(symbol, 0) for symbol in holdings)
        
        # Calculate weights
        weights = {symbol: (holdings[symbol] * prices.get(symbol, 0)) / total_value 
                  for symbol in holdings if total_value > 0}
        
        # Get historical data for portfolio analysis
        portfolio_returns = []
        for symbol in holdings:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1y")
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    weighted_returns = returns * weights.get(symbol, 0)
                    portfolio_returns.append(weighted_returns)
            except:
                continue
        
        if portfolio_returns:
            # Combine returns
            combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            
            # Calculate metrics
            annual_return = combined_returns.mean() * 252
            annual_volatility = combined_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + combined_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                'total_value': total_value,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'weights': weights
            }
        
        return {'total_value': total_value, 'weights': weights}
    
    def optimize_portfolio(self, symbols: List[str], risk_tolerance: str = "medium") -> Dict[str, float]:
        """
        Simple portfolio optimization based on risk tolerance
        """
        # Risk tolerance mapping
        risk_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5
        }
        
        multiplier = risk_multipliers.get(risk_tolerance, 1.0)
        
        # Equal weight as baseline (in production, use modern portfolio theory)
        equal_weight = 1.0 / len(symbols)
        
        # Adjust weights based on risk tolerance
        weights = {}
        for symbol in symbols:
            # Simple adjustment based on market cap and volatility
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                market_cap = info.get('marketCap', 1e9)
                
                # Larger companies get higher weight for low risk tolerance
                if risk_tolerance == "low":
                    weight_adjustment = min(2.0, market_cap / 1e11)
                elif risk_tolerance == "high":
                    weight_adjustment = max(0.5, 1e11 / market_cap)
                else:
                    weight_adjustment = 1.0
                
                weights[symbol] = equal_weight * weight_adjustment
                
            except:
                weights[symbol] = equal_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        return weights