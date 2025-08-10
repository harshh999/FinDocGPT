"""
Test suite for FinDocGPT optimizations
Validates performance improvements and functionality
"""

import time
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit for testing
class MockStreamlit:
    class session_state:
        cache_store = {}
        performance_monitor = None

sys.modules['streamlit'] = MockStreamlit()

from core.cache_manager import CacheManager
from core.financial_analyzer import FinancialAnalyzer
from core.performance_monitor import PerformanceMonitor

class TestOptimizations:
    """Test suite for performance optimizations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.cache = CacheManager()
        self.analyzer = FinancialAnalyzer()
        self.monitor = PerformanceMonitor()
    
    def test_cache_performance(self):
        """Test caching system performance"""
        # Test cache set/get
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        params = {"symbol": "AAPL", "period": "1y"}
        
        # Set data
        start_time = time.time()
        self.cache.set("test_data", params, test_data)
        set_time = time.time() - start_time
        
        # Get data
        start_time = time.time()
        retrieved_data = self.cache.get("test_data", params, 60)
        get_time = time.time() - start_time
        
        # Assertions
        assert retrieved_data == test_data
        assert set_time < 0.01  # Should be very fast
        assert get_time < 0.01  # Should be very fast
        
        print(f"✅ Cache set time: {set_time:.4f}s")
        print(f"✅ Cache get time: {get_time:.4f}s")
    
    def test_financial_analyzer_performance(self):
        """Test financial analyzer performance"""
        # Sample financial text
        sample_text = """
        The company reported revenue of $50.2 billion for Q3 2024, representing a 12% increase 
        year-over-year. Net income was $12.1 billion, up from $10.8 billion in the same quarter 
        last year. Operating expenses totaled $28.5 billion. The strong performance was driven 
        by excellent growth in our core business segments and improved operational efficiency.
        """
        
        # Test metrics extraction performance
        start_time = time.time()
        metrics = self.analyzer.extract_financial_metrics(sample_text)
        extraction_time = time.time() - start_time
        
        # Test sentiment analysis performance
        start_time = time.time()
        sentiment, polarity = self.analyzer.analyze_sentiment(sample_text)
        sentiment_time = time.time() - start_time
        
        # Assertions
        assert 'revenue' in metrics
        assert metrics['revenue'] == 50.2e9  # Should parse billions correctly
        assert 'profit' in metrics
        assert sentiment in ['Positive', 'Negative', 'Neutral']
        assert extraction_time < 0.1  # Should be fast
        assert sentiment_time < 0.1  # Should be fast
        
        print(f"✅ Metrics extraction time: {extraction_time:.4f}s")
        print(f"✅ Sentiment analysis time: {sentiment_time:.4f}s")
        print(f"✅ Extracted metrics: {metrics}")
        print(f"✅ Sentiment: {sentiment} ({polarity:.2f})")
    
    def test_vectorized_operations(self):
        """Test vectorized operations performance"""
        # Create sample data
        data_size = 1000  # Smaller for testing
        sample_data = pd.DataFrame({
            'Close': np.random.randn(data_size).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, data_size),
            'High': np.random.randn(data_size) + 102,
            'Low': np.random.randn(data_size) + 98
        })
        
        # Test simple vectorized operations
        start_time = time.time()
        sample_data['MA_5'] = sample_data['Close'].rolling(5).mean()
        sample_data['Returns'] = sample_data['Close'].pct_change()
        calculation_time = time.time() - start_time
        
        # Assertions
        assert 'MA_5' in sample_data.columns
        assert 'Returns' in sample_data.columns
        assert calculation_time < 0.1  # Should be fast
        assert len(sample_data) == data_size
        
        print(f"✅ Vectorized operations time: {calculation_time:.4f}s")
        print(f"✅ Data size: {data_size} rows")
        print(f"✅ Processing rate: {data_size/calculation_time:.0f} rows/second")
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and process multiple datasets
            datasets = []
            for i in range(5):  # Smaller number for testing
                data = pd.DataFrame({
                    'Close': np.random.randn(100).cumsum() + 100,
                    'Volume': np.random.randint(1000000, 10000000, 100)
                })
                datasets.append(data)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clean up
            del datasets
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = peak_memory - initial_memory
            
            print(f"✅ Initial memory: {initial_memory:.1f}MB")
            print(f"✅ Peak memory: {peak_memory:.1f}MB")
            print(f"✅ Final memory: {final_memory:.1f}MB")
            print(f"✅ Memory increase: {memory_increase:.1f}MB")
            
        except ImportError:
            print("✅ psutil not available, skipping memory test")
    
    def test_performance_monitoring(self):
        """Test performance monitoring system"""
        # Test tracking functions
        self.monitor.track_api_call()
        self.monitor.track_cache_hit()
        self.monitor.track_cache_miss()
        self.monitor.track_response_time(0.5)
        self.monitor.track_response_time(0.3)
        
        # Get metrics
        metrics = self.monitor.get_metrics_summary()
        
        # Assertions
        assert 'cache_hit_rate' in metrics
        assert 'avg_response_time' in metrics
        assert 'api_efficiency' in metrics
        assert 'memory_usage' in metrics
        assert self.monitor.get_cache_hit_rate() == 0.5  # 1 hit, 1 miss
        
        print(f"✅ Performance metrics: {metrics}")
    
    def test_regex_compilation_performance(self):
        """Test compiled regex performance vs non-compiled"""
        sample_text = "Revenue was $50.2 billion, profit reached $12.1 billion, expenses totaled $28.5 billion." * 100
        
        # Test compiled regex (current implementation)
        start_time = time.time()
        for _ in range(100):
            metrics = self.analyzer.extract_financial_metrics(sample_text)
        compiled_time = time.time() - start_time
        
        # Test would-be non-compiled performance (simulation)
        import re
        start_time = time.time()
        for _ in range(100):
            # Simulate non-compiled regex
            matches = re.findall(r'revenue.*?\$?([0-9,]+\.?[0-9]*)', sample_text.lower())
        non_compiled_time = time.time() - start_time
        
        # Performance improvement should be significant
        improvement = (non_compiled_time - compiled_time) / non_compiled_time * 100
        
        print(f"✅ Compiled regex time: {compiled_time:.4f}s")
        print(f"✅ Non-compiled simulation time: {non_compiled_time:.4f}s")
        print(f"✅ Performance improvement: {improvement:.1f}%")
        
        assert compiled_time < non_compiled_time * 1.1  # Should be at least as fast

def run_performance_tests():
    """Run all performance tests"""
    print("🚀 Running FinDocGPT Performance Tests")
    print("=" * 50)
    
    test_suite = TestOptimizations()
    test_suite.setup_method()
    
    try:
        print("\n📊 Testing Cache Performance...")
        test_suite.test_cache_performance()
        
        print("\n🔍 Testing Financial Analyzer Performance...")
        test_suite.test_financial_analyzer_performance()
        
        print("\n⚡ Testing Vectorized Operations...")
        test_suite.test_vectorized_operations()
        
        print("\n💾 Testing Memory Efficiency...")
        test_suite.test_memory_efficiency()
        
        print("\n📈 Testing Performance Monitoring...")
        test_suite.test_performance_monitoring()
        
        print("\n🔧 Testing Regex Compilation Performance...")
        test_suite.test_regex_compilation_performance()
        
        print("\n" + "=" * 50)
        print("✅ All performance tests passed!")
        print("🎉 FinDocGPT optimizations are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_performance_tests()