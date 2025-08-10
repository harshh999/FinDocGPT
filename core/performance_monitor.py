"""
Performance monitoring and optimization utilities
"""

import time
import streamlit as st
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from functools import wraps
from typing import Dict, Any, Callable
from datetime import datetime
import threading

class PerformanceMonitor:
    """Monitor and track application performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time': 0,
            'memory_usage': 0,
            'start_time': datetime.now()
        }
        self.response_times = []
        self._lock = threading.Lock()
    
    def track_api_call(self):
        """Track API call"""
        with self._lock:
            self.metrics['api_calls'] += 1
    
    def track_cache_hit(self):
        """Track cache hit"""
        with self._lock:
            self.metrics['cache_hits'] += 1
    
    def track_cache_miss(self):
        """Track cache miss"""
        with self._lock:
            self.metrics['cache_misses'] += 1
    
    def track_response_time(self, response_time: float):
        """Track response time"""
        with self._lock:
            self.response_times.append(response_time)
            if len(self.response_times) > 100:  # Keep only last 100
                self.response_times.pop(0)
            
            self.metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
            self.metrics['total_requests'] += 1
    
    def update_memory_usage(self):
        """Update current memory usage"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                self.metrics['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
            except:
                self.metrics['memory_usage'] = 0
        else:
            self.metrics['memory_usage'] = 0
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_requests == 0:
            return 0.0
        return self.metrics['cache_hits'] / total_cache_requests
    
    def get_api_efficiency(self) -> float:
        """Calculate API efficiency (cache hits vs API calls)"""
        total_data_requests = self.metrics['api_calls'] + self.metrics['cache_hits']
        if total_data_requests == 0:
            return 1.0
        return self.metrics['cache_hits'] / total_data_requests
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        self.update_memory_usage()
        
        return {
            'cache_hit_rate': f"{self.get_cache_hit_rate():.1%}",
            'avg_response_time': f"{self.metrics['avg_response_time']:.2f}s",
            'api_efficiency': f"{self.get_api_efficiency():.1%}",
            'memory_usage': f"{self.metrics['memory_usage']:.0f}MB",
            'total_requests': self.metrics['total_requests'],
            'api_calls': self.metrics['api_calls'],
            'uptime': str(datetime.now() - self.metrics['start_time']).split('.')[0]
        }

def performance_tracker(func: Callable) -> Callable:
    """Decorator to track function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            response_time = end_time - start_time
            
            # Track performance if monitor exists
            if hasattr(st.session_state, 'performance_monitor'):
                st.session_state.performance_monitor.track_response_time(response_time)
    
    return wrapper

# Global performance monitor instance
try:
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    performance_monitor = st.session_state.performance_monitor
except (AttributeError, RuntimeError):
    # Fallback when not in Streamlit context
    performance_monitor = PerformanceMonitor()