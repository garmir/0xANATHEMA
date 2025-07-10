#!/usr/bin/env python3
"""
Performance Monitoring and Caching System for Task Master AI Local Modules
Provides comprehensive performance tracking, caching, and optimization capabilities
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
import hashlib
import pickle
import sqlite3
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "context": self.context
        }

@dataclass
class PerformanceSession:
    """Represents a performance monitoring session"""
    session_id: str
    component: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    metrics: List[PerformanceMetric] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get session duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def add_metric(self, name: str, value: float, unit: str, context: Dict[str, Any] = None):
        """Add a metric to the session"""
        metric = PerformanceMetric(name, value, unit, context=context or {})
        self.metrics.append(metric)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "component": self.component,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "success": self.success,
            "error": self.error
        }

class CacheEntry:
    """Represents a cache entry"""
    
    def __init__(self, key: str, value: Any, ttl: float = 3600, tags: List[str] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.tags = tags or []
        self.access_count = 0
        self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the cached value"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "expires_at": self.created_at + self.ttl,
            "is_expired": self.is_expired
        }

class PerformanceCache:
    """High-performance cache with TTL and tag-based invalidation"""
    
    def __init__(self, 
                 default_ttl: float = 3600,
                 max_size: int = 10000,
                 cleanup_interval: float = 300):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, CacheEntry] = {}
        self._tags_index: Dict[str, set] = defaultdict(set)
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0
        }
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task"""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_expired()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self.stats["cleanups"] += 1
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_entry(self, key: str):
        """Remove a cache entry and update indices"""
        if key in self._cache:
            entry = self._cache[key]
            
            # Remove from tags index
            for tag in entry.tags:
                self._tags_index[tag].discard(key)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]
            
            del self._cache[key]
    
    def _evict_lru(self):
        """Evict least recently used entries to make space"""
        if len(self._cache) < self.max_size:
            return
        
        # Sort by last accessed time
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 10% of entries
        evict_count = max(1, len(entries_by_access) // 10)
        for key, _ in entries_by_access[:evict_count]:
            self._remove_entry(key)
        
        self.stats["evictions"] += evict_count
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired:
                    self.stats["hits"] += 1
                    return entry.access()
                else:
                    self._remove_entry(key)
            
            self.stats["misses"] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: float = None, tags: List[str] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict if necessary
            self._evict_lru()
            
            # Create new entry
            effective_ttl = ttl if ttl is not None else self.default_ttl
            entry = CacheEntry(key, value, effective_ttl, tags)
            
            self._cache[key] = entry
            
            # Update tags index
            for tag in entry.tags:
                self._tags_index[tag].add(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def invalidate_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag"""
        with self._lock:
            if tag not in self._tags_index:
                return 0
            
            keys_to_remove = list(self._tags_index[tag])
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._tags_index.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self.stats["evictions"],
                "cleanups": self.stats["cleanups"],
                "tags_count": len(self._tags_index)
            }

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    Tracks metrics, sessions, and provides analytics
    """
    
    def __init__(self, 
                 data_dir: str = ".taskmaster/local_modules/performance",
                 db_file: str = "performance.db"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / db_file
        
        # Performance tracking
        self.active_sessions: Dict[str, PerformanceSession] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.component_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "last_operation": None
        })
        
        # Initialize database
        self._init_database()
        
        # Monitoring state
        self.monitoring_enabled = True
        self._lock = threading.RLock()
        
        # Background tasks
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize SQLite database for performance data"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_sessions (
                    session_id TEXT PRIMARY KEY,
                    component TEXT,
                    operation TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    success BOOLEAN,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    name TEXT,
                    value REAL,
                    unit TEXT,
                    timestamp REAL,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES performance_sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_component ON performance_sessions(component)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON performance_sessions(start_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(name)
            """)
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def metrics_flush_loop():
            while self.monitoring_enabled:
                try:
                    self._flush_metrics_buffer()
                    time.sleep(30)  # Flush every 30 seconds
                except Exception as e:
                    logger.error(f"Metrics flush error: {e}")
        
        flush_thread = threading.Thread(target=metrics_flush_loop, daemon=True)
        flush_thread.start()
    
    def _flush_metrics_buffer(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        with self._lock:
            metrics_to_flush = list(self.metrics_buffer)
            self.metrics_buffer.clear()
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                for session in metrics_to_flush:
                    # Insert session
                    conn.execute("""
                        INSERT OR REPLACE INTO performance_sessions 
                        (session_id, component, operation, start_time, end_time, duration, success, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session.session_id,
                        session.component,
                        session.operation,
                        session.start_time,
                        session.end_time,
                        session.duration,
                        session.success,
                        session.error
                    ))
                    
                    # Insert metrics
                    for metric in session.metrics:
                        conn.execute("""
                            INSERT INTO performance_metrics 
                            (session_id, name, value, unit, timestamp, context)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            session.session_id,
                            metric.name,
                            metric.value,
                            metric.unit,
                            metric.timestamp,
                            json.dumps(metric.context)
                        ))
        
        except Exception as e:
            logger.error(f"Failed to flush metrics to database: {e}")
    
    @contextmanager
    def session(self, component: str, operation: str):
        """Context manager for performance monitoring sessions"""
        session_id = f"{component}_{operation}_{int(time.time())}_{id(threading.current_thread())}"
        
        session = PerformanceSession(
            session_id=session_id,
            component=component,
            operation=operation,
            start_time=time.time()
        )
        
        with self._lock:
            self.active_sessions[session_id] = session
        
        try:
            yield session
            session.success = True
        except Exception as e:
            session.success = False
            session.error = str(e)
            raise
        finally:
            session.end_time = time.time()
            
            with self._lock:
                # Remove from active sessions
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                
                # Add to buffer for database flush
                self.metrics_buffer.append(session)
                
                # Update component stats
                self._update_component_stats(session)
    
    def _update_component_stats(self, session: PerformanceSession):
        """Update component statistics"""
        stats = self.component_stats[session.component]
        
        stats["total_operations"] += 1
        if session.success:
            stats["successful_operations"] += 1
        else:
            stats["failed_operations"] += 1
        
        duration = session.duration
        stats["total_time"] += duration
        stats["avg_time"] = stats["total_time"] / stats["total_operations"]
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        stats["last_operation"] = session.end_time
    
    def add_metric(self, name: str, value: float, unit: str, 
                  component: str = None, context: Dict[str, Any] = None):
        """Add a standalone metric"""
        # Try to find current session for the component
        current_session = None
        if component:
            with self._lock:
                for session in self.active_sessions.values():
                    if session.component == component:
                        current_session = session
                        break
        
        if current_session:
            current_session.add_metric(name, value, unit, context)
        else:
            # Create a standalone metric session
            session_id = f"metric_{name}_{int(time.time())}"
            session = PerformanceSession(
                session_id=session_id,
                component=component or "standalone",
                operation=f"metric_{name}",
                start_time=time.time(),
                end_time=time.time()
            )
            session.add_metric(name, value, unit, context)
            
            with self._lock:
                self.metrics_buffer.append(session)
    
    def get_component_stats(self, component: str = None) -> Dict[str, Any]:
        """Get statistics for a component or all components"""
        with self._lock:
            if component:
                return dict(self.component_stats.get(component, {}))
            else:
                return {comp: dict(stats) for comp, stats in self.component_stats.items()}
    
    def get_performance_summary(self, 
                               time_window: float = 3600,
                               component: str = None) -> Dict[str, Any]:
        """Get performance summary for a time window"""
        since_time = time.time() - time_window
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Base query
                where_clause = "WHERE start_time >= ?"
                params = [since_time]
                
                if component:
                    where_clause += " AND component = ?"
                    params.append(component)
                
                # Get session stats
                cursor = conn.execute(f"""
                    SELECT 
                        component,
                        COUNT(*) as total_sessions,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_sessions,
                        AVG(duration) as avg_duration,
                        MIN(duration) as min_duration,
                        MAX(duration) as max_duration
                    FROM performance_sessions 
                    {where_clause}
                    GROUP BY component
                """, params)
                
                component_summaries = {}
                for row in cursor:
                    comp_name, total, successful, avg_dur, min_dur, max_dur = row
                    component_summaries[comp_name] = {
                        "total_sessions": total,
                        "successful_sessions": successful,
                        "failed_sessions": total - successful,
                        "success_rate": successful / total if total > 0 else 0,
                        "avg_duration": avg_dur,
                        "min_duration": min_dur,
                        "max_duration": max_dur
                    }
                
                # Get overall stats
                cursor = conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_sessions,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_sessions,
                        AVG(duration) as avg_duration
                    FROM performance_sessions 
                    {where_clause}
                """, params)
                
                total, successful, avg_duration = cursor.fetchone()
                
                return {
                    "time_window_hours": time_window / 3600,
                    "overall": {
                        "total_sessions": total or 0,
                        "successful_sessions": successful or 0,
                        "failed_sessions": (total or 0) - (successful or 0),
                        "success_rate": (successful / total) if total and total > 0 else 0,
                        "avg_duration": avg_duration or 0
                    },
                    "by_component": component_summaries,
                    "active_sessions": len(self.active_sessions)
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def get_metric_trends(self, 
                         metric_name: str,
                         time_window: float = 3600,
                         component: str = None) -> Dict[str, Any]:
        """Get trends for a specific metric"""
        since_time = time.time() - time_window
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                where_clause = "WHERE m.timestamp >= ? AND m.name = ?"
                params = [since_time, metric_name]
                
                if component:
                    where_clause += " AND s.component = ?"
                    params.append(component)
                
                cursor = conn.execute(f"""
                    SELECT m.value, m.timestamp, m.unit, s.component
                    FROM performance_metrics m
                    JOIN performance_sessions s ON m.session_id = s.session_id
                    {where_clause}
                    ORDER BY m.timestamp
                """, params)
                
                values = []
                timestamps = []
                for value, timestamp, unit, comp in cursor:
                    values.append(value)
                    timestamps.append(timestamp)
                
                if not values:
                    return {"error": f"No data found for metric {metric_name}"}
                
                return {
                    "metric_name": metric_name,
                    "time_window_hours": time_window / 3600,
                    "data_points": len(values),
                    "unit": unit if 'unit' in locals() else "unknown",
                    "statistics": {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                    },
                    "trend": {
                        "first_value": values[0],
                        "last_value": values[-1],
                        "change": values[-1] - values[0],
                        "percent_change": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    },
                    "values": values,
                    "timestamps": timestamps
                }
                
        except Exception as e:
            logger.error(f"Failed to get metric trends: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old performance data"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Delete old metrics
                conn.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_time,))
                
                # Delete old sessions
                conn.execute("DELETE FROM performance_sessions WHERE start_time < ?", (cutoff_time,))
                
                # Vacuum database
                conn.execute("VACUUM")
                
            logger.info(f"Cleaned up performance data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def export_performance_data(self, export_path: str, time_window: float = None) -> bool:
        """Export performance data to JSON file"""
        try:
            where_clause = ""
            params = []
            
            if time_window:
                since_time = time.time() - time_window
                where_clause = "WHERE start_time >= ?"
                params = [since_time]
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Export sessions
                cursor = conn.execute(f"""
                    SELECT session_id, component, operation, start_time, end_time, 
                           duration, success, error
                    FROM performance_sessions 
                    {where_clause}
                    ORDER BY start_time
                """, params)
                
                sessions = []
                for row in cursor:
                    session_data = {
                        "session_id": row[0],
                        "component": row[1],
                        "operation": row[2],
                        "start_time": row[3],
                        "end_time": row[4],
                        "duration": row[5],
                        "success": bool(row[6]),
                        "error": row[7]
                    }
                    sessions.append(session_data)
                
                # Export metrics
                cursor = conn.execute(f"""
                    SELECT m.session_id, m.name, m.value, m.unit, m.timestamp, m.context
                    FROM performance_metrics m
                    JOIN performance_sessions s ON m.session_id = s.session_id
                    {where_clause}
                    ORDER BY m.timestamp
                """, params)
                
                metrics = []
                for row in cursor:
                    metric_data = {
                        "session_id": row[0],
                        "name": row[1],
                        "value": row[2],
                        "unit": row[3],
                        "timestamp": row[4],
                        "context": json.loads(row[5]) if row[5] else {}
                    }
                    metrics.append(metric_data)
            
            # Create export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "time_window_hours": time_window / 3600 if time_window else "all",
                "sessions": sessions,
                "metrics": metrics,
                "component_stats": dict(self.component_stats)
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance data exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False

class CachedPerformanceMonitor:
    """
    Combined performance monitor with integrated caching
    Provides both performance tracking and caching capabilities
    """
    
    def __init__(self, 
                 data_dir: str = ".taskmaster/local_modules/performance",
                 cache_ttl: float = 3600,
                 cache_max_size: int = 10000):
        self.performance_monitor = PerformanceMonitor(data_dir)
        self.cache = PerformanceCache(cache_ttl, cache_max_size)
        
    @contextmanager
    def monitored_operation(self, component: str, operation: str):
        """Context manager for monitored operations"""
        with self.performance_monitor.session(component, operation) as session:
            yield session
    
    def cached_call(self, 
                   func: Callable,
                   cache_key: str = None,
                   ttl: float = None,
                   tags: List[str] = None,
                   component: str = None,
                   operation: str = None) -> Any:
        """Execute function with caching and performance monitoring"""
        # Generate cache key if not provided
        if cache_key is None:
            func_name = getattr(func, '__name__', str(func))
            cache_key = f"{func_name}_{hashlib.md5(str(func).encode()).hexdigest()[:8]}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            if component:
                self.performance_monitor.add_metric(
                    "cache_hit", 1, "count", component, {"cache_key": cache_key}
                )
            return cached_result
        
        # Execute function with monitoring
        operation_name = operation or f"cached_call_{func.__name__ if hasattr(func, '__name__') else 'unknown'}"
        component_name = component or "cached_operation"
        
        with self.monitored_operation(component_name, operation_name) as session:
            try:
                result = func()
                
                # Cache the result
                self.cache.set(cache_key, result, ttl, tags)
                
                # Add metrics
                session.add_metric("cache_miss", 1, "count", {"cache_key": cache_key})
                session.add_metric("function_executed", 1, "count", {"function": str(func)})
                
                return result
                
            except Exception as e:
                session.add_metric("function_error", 1, "count", {"error": str(e)})
                raise
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined performance and cache statistics"""
        return {
            "performance": self.performance_monitor.get_component_stats(),
            "cache": self.cache.get_stats(),
            "summary": self.performance_monitor.get_performance_summary()
        }

# Example usage
if __name__ == "__main__":
    def test_performance_monitoring():
        # Initialize combined monitor
        monitor = CachedPerformanceMonitor()
        
        # Test monitored operation
        with monitor.monitored_operation("test_component", "test_operation") as session:
            time.sleep(0.1)  # Simulate work
            session.add_metric("items_processed", 100, "count")
            session.add_metric("processing_rate", 1000, "items/sec")
        
        # Test cached function call
        def expensive_operation():
            time.sleep(0.2)  # Simulate expensive work
            return {"result": "computed_value", "timestamp": time.time()}
        
        # First call (cache miss)
        result1 = monitor.cached_call(
            expensive_operation,
            cache_key="test_operation",
            component="test_cache",
            operation="expensive_computation"
        )
        
        # Second call (cache hit)
        result2 = monitor.cached_call(
            expensive_operation,
            cache_key="test_operation",
            component="test_cache",
            operation="expensive_computation"
        )
        
        print(f"Results equal: {result1 == result2}")
        
        # Get statistics
        stats = monitor.get_combined_stats()
        print(f"Combined stats: {json.dumps(stats, indent=2)}")
        
        # Get performance summary
        summary = monitor.performance_monitor.get_performance_summary()
        print(f"Performance summary: {json.dumps(summary, indent=2)}")
        
        # Export data
        monitor.performance_monitor.export_performance_data("test_performance_export.json")
        print("Performance data exported")
    
    # Run test
    test_performance_monitoring()