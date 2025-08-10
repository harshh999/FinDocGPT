"""
Centralized caching system for performance optimization
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import hashlib
import json

class CacheManager:
    """Centralized cache management with TTL support"""
    
    def __init__(self):
        try:
            if 'cache_store' not in st.session_state:
                st.session_state.cache_store = {}
            self._use_session_state = True
        except (AttributeError, RuntimeError):
            # Fallback to instance-level cache when not in Streamlit context
            self._cache_store = {}
            self._use_session_state = False
    
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True, default=str)
        hash_obj = hashlib.md5(param_str.encode())
        return f"{prefix}_{hash_obj.hexdigest()}"
    
    def get(self, prefix: str, params: Dict[str, Any], ttl_minutes: int = 60) -> Optional[Any]:
        """Get cached data if not expired"""
        key = self._generate_key(prefix, params)
        
        # Get cache store based on context
        cache_store = st.session_state.cache_store if self._use_session_state else self._cache_store
        
        if key in cache_store:
            cached_item = cache_store[key]
            
            # Check if expired
            if datetime.now() - cached_item['timestamp'] < timedelta(minutes=ttl_minutes):
                # Track cache hit if in Streamlit context
                if self._use_session_state and hasattr(st.session_state, 'performance_monitor'):
                    st.session_state.performance_monitor.track_cache_hit()
                return cached_item['data']
            else:
                # Remove expired item
                del cache_store[key]
        
        # Track cache miss if in Streamlit context
        if self._use_session_state and hasattr(st.session_state, 'performance_monitor'):
            st.session_state.performance_monitor.track_cache_miss()
        
        return None
    
    def set(self, prefix: str, params: Dict[str, Any], data: Any) -> None:
        """Cache data with timestamp"""
        key = self._generate_key(prefix, params)
        
        # Get cache store based on context
        cache_store = st.session_state.cache_store if self._use_session_state else self._cache_store
        
        cache_store[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def clear_expired(self, ttl_minutes: int = 60) -> None:
        """Clear all expired cache entries"""
        try:
            # Get cache store based on context
            cache_store = st.session_state.cache_store if self._use_session_state else self._cache_store
            
            current_time = datetime.now()
            expired_keys = []
            
            for key, item in cache_store.items():
                if current_time - item['timestamp'] > timedelta(minutes=ttl_minutes):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del cache_store[key]
        except (AttributeError, RuntimeError):
            # Silently handle cases where session state is not available
            pass
    
    def clear_all(self) -> None:
        """Clear all cache"""
        try:
            if self._use_session_state:
                st.session_state.cache_store = {}
            else:
                self._cache_store = {}
        except (AttributeError, RuntimeError):
            # Fallback to instance cache
            self._cache_store = {}

# Global cache instance
cache = CacheManager()