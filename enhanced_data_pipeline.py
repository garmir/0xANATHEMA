#!/usr/bin/env python3
"""
Enhanced Data Pipeline and Storage System for Task Master AI

This module provides a scalable, high-performance data pipeline that ingests,
processes, and stores metrics from multiple sources with efficient querying
and real-time streaming capabilities.
"""

import json
import time
import sqlite3
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import gzip
import pickle
from collections import defaultdict, deque
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Individual data point with metadata"""
    timestamp: datetime
    source: str
    metric_type: str
    value: Union[float, int, str, Dict[str, Any]]
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline"""
    # Database settings
    db_path: str = ".taskmaster/analytics/pipeline.db"
    enable_compression: bool = True
    retention_days: int = 30
    
    # Performance settings
    batch_size: int = 100
    flush_interval: float = 5.0  # seconds
    max_queue_size: int = 10000
    worker_threads: int = 4
    
    # Storage settings
    enable_archival: bool = True
    archive_path: str = ".taskmaster/analytics/archive"
    compress_older_than_days: int = 7
    
    # Monitoring settings
    enable_pipeline_metrics: bool = True
    metrics_interval: float = 60.0  # seconds


class MetricsAggregator:
    """Efficient metrics aggregation engine"""
    
    def __init__(self):
        self.aggregations = defaultdict(lambda: defaultdict(list))
        self.time_windows = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '6h': 21600,
            '24h': 86400
        }
        self.lock = threading.Lock()
    
    def add_metric(self, data_point: DataPoint):
        """Add a data point for aggregation"""
        if not isinstance(data_point.value, (int, float)):
            return  # Skip non-numeric values for aggregation
        
        with self.lock:
            timestamp = data_point.timestamp
            metric_key = f"{data_point.source}:{data_point.metric_type}"
            
            for window_name, window_seconds in self.time_windows.items():
                # Calculate window start time
                window_start = timestamp.replace(second=0, microsecond=0)
                window_start = window_start - timedelta(
                    minutes=window_start.minute % (window_seconds // 60),
                    seconds=window_start.second
                )
                
                window_key = f"{metric_key}:{window_name}:{window_start.isoformat()}"
                self.aggregations[window_key]['values'].append(float(data_point.value))
                self.aggregations[window_key]['timestamps'].append(timestamp)
                self.aggregations[window_key]['tags'] = data_point.tags
    
    def get_aggregated_metrics(self, metric_key: str, window: str, 
                              start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get aggregated metrics for a specific time range"""
        results = []
        
        with self.lock:
            for agg_key, agg_data in self.aggregations.items():
                if f"{metric_key}:{window}:" in agg_key:
                    window_time = datetime.fromisoformat(agg_key.split(':')[-1])
                    
                    if start_time <= window_time <= end_time:
                        values = agg_data['values']
                        if values:
                            results.append({
                                'timestamp': window_time,
                                'count': len(values),
                                'min': min(values),
                                'max': max(values),
                                'avg': statistics.mean(values),
                                'sum': sum(values),
                                'median': statistics.median(values),
                                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                                'tags': agg_data['tags']
                            })
        
        return sorted(results, key=lambda x: x['timestamp'])
    
    def cleanup_old_aggregations(self, older_than: datetime):
        """Remove aggregations older than specified time"""
        with self.lock:
            keys_to_remove = []
            
            for agg_key in self.aggregations.keys():
                try:
                    window_time = datetime.fromisoformat(agg_key.split(':')[-1])
                    if window_time < older_than:
                        keys_to_remove.append(agg_key)
                except (ValueError, IndexError):
                    continue
            
            for key in keys_to_remove:
                del self.aggregations[key]
            
            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} old aggregations")


class StreamProcessor:
    """Real-time stream processing engine"""
    
    def __init__(self):
        self.processors: List[Callable[[DataPoint], Optional[DataPoint]]] = []
        self.anomaly_detectors: List[Callable[[DataPoint], bool]] = []
        self.windows = defaultdict(lambda: deque(maxlen=1000))
    
    def add_processor(self, processor: Callable[[DataPoint], Optional[DataPoint]]):
        """Add a stream processor function"""
        self.processors.append(processor)
    
    def add_anomaly_detector(self, detector: Callable[[DataPoint], bool]):
        """Add an anomaly detection function"""
        self.anomaly_detectors.append(detector)
    
    def process_data_point(self, data_point: DataPoint) -> List[DataPoint]:
        """Process a data point through all processors"""
        results = [data_point]
        
        # Apply processors
        for processor in self.processors:
            try:
                processed_results = []
                for dp in results:
                    processed = processor(dp)
                    if processed:
                        processed_results.append(processed)
                results = processed_results
            except Exception as e:
                logger.error(f"Error in stream processor: {e}")
        
        # Check for anomalies
        for data_point in results:
            for detector in self.anomaly_detectors:
                try:
                    if detector(data_point):
                        # Create anomaly data point
                        anomaly_point = DataPoint(
                            timestamp=data_point.timestamp,
                            source="anomaly_detector",
                            metric_type="anomaly",
                            value=1,
                            tags={
                                'original_source': data_point.source,
                                'original_metric': data_point.metric_type,
                                'anomaly_value': str(data_point.value)
                            }
                        )
                        results.append(anomaly_point)
                except Exception as e:
                    logger.error(f"Error in anomaly detector: {e}")
        
        # Update sliding windows for trend analysis
        for data_point in results:
            window_key = f"{data_point.source}:{data_point.metric_type}"
            self.windows[window_key].append((data_point.timestamp, data_point.value))
        
        return results


class EnhancedDatabase:
    """Enhanced database with partitioning and compression"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.db_path = config.db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
        self.write_lock = threading.Lock()
    
    def _init_database(self):
        """Initialize database schema with optimizations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main metrics table with partitioning by date
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value_type TEXT NOT NULL,
                numeric_value REAL,
                text_value TEXT,
                json_value TEXT,
                tags TEXT,
                metadata TEXT,
                date_partition TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create aggregated metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                window_size TEXT NOT NULL,
                count INTEGER,
                min_value REAL,
                max_value REAL,
                avg_value REAL,
                sum_value REAL,
                median_value REAL,
                std_dev REAL,
                tags TEXT,
                date_partition TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
            ON metrics(timestamp, source, metric_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_partition 
            ON metrics(date_partition, source, metric_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregated_partition 
            ON aggregated_metrics(date_partition, source, metric_type, window_size)
        """)
        
        # Create pipeline metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def insert_data_point(self, data_point: DataPoint):
        """Insert a single data point"""
        self.insert_data_points([data_point])
    
    def insert_data_points(self, data_points: List[DataPoint]):
        """Insert multiple data points efficiently"""
        if not data_points:
            return
        
        with self.write_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                records = []
                for dp in data_points:
                    # Determine value type and storage
                    if isinstance(dp.value, (int, float)):
                        value_type = "numeric"
                        numeric_value = float(dp.value)
                        text_value = None
                        json_value = None
                    elif isinstance(dp.value, str):
                        value_type = "text"
                        numeric_value = None
                        text_value = dp.value
                        json_value = None
                    else:
                        value_type = "json"
                        numeric_value = None
                        text_value = None
                        json_value = json.dumps(dp.value)
                    
                    # Create date partition
                    date_partition = dp.timestamp.strftime('%Y-%m-%d')
                    
                    records.append((
                        dp.timestamp.isoformat(),
                        dp.source,
                        dp.metric_type,
                        value_type,
                        numeric_value,
                        text_value,
                        json_value,
                        json.dumps(dp.tags) if dp.tags else None,
                        json.dumps(dp.metadata) if dp.metadata else None,
                        date_partition
                    ))
                
                cursor.executemany("""
                    INSERT INTO metrics 
                    (timestamp, source, metric_type, value_type, numeric_value, 
                     text_value, json_value, tags, metadata, date_partition)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                conn.commit()
                logger.debug(f"Inserted {len(records)} data points")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting data points: {e}")
                raise
            finally:
                conn.close()
    
    def query_metrics(self, source: str = None, metric_type: str = None, 
                     start_time: datetime = None, end_time: datetime = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """Query metrics with flexible filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if source:
                conditions.append("source = ?")
                params.append(source)
            
            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
                SELECT timestamp, source, metric_type, value_type, 
                       numeric_value, text_value, json_value, tags, metadata
                FROM metrics 
                {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            
            params.append(limit)
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                timestamp, source, metric_type, value_type, numeric_value, text_value, json_value, tags, metadata = row
                
                # Reconstruct value based on type
                if value_type == "numeric":
                    value = numeric_value
                elif value_type == "text":
                    value = text_value
                elif value_type == "json":
                    value = json.loads(json_value) if json_value else None
                else:
                    value = None
                
                results.append({
                    'timestamp': datetime.fromisoformat(timestamp),
                    'source': source,
                    'metric_type': metric_type,
                    'value': value,
                    'tags': json.loads(tags) if tags else {},
                    'metadata': json.loads(metadata) if metadata else {}
                })
            
            return results
            
        finally:
            conn.close()
    
    def cleanup_old_data(self, older_than: datetime):
        """Remove data older than specified time"""
        with self.write_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cutoff_date = older_than.isoformat()
                
                # Delete old metrics
                cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_date,))
                deleted_metrics = cursor.rowcount
                
                # Delete old aggregated metrics
                cursor.execute("DELETE FROM aggregated_metrics WHERE timestamp < ?", (cutoff_date,))
                deleted_aggregated = cursor.rowcount
                
                # Delete old pipeline metrics
                cursor.execute("DELETE FROM pipeline_metrics WHERE timestamp < ?", (cutoff_date,))
                deleted_pipeline = cursor.rowcount
                
                conn.commit()
                
                # Vacuum database to reclaim space
                cursor.execute("VACUUM")
                
                logger.info(f"Cleaned up {deleted_metrics} metrics, {deleted_aggregated} aggregated metrics, {deleted_pipeline} pipeline metrics")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error cleaning up old data: {e}")
                raise
            finally:
                conn.close()


class DataPipeline:
    """Enhanced data pipeline with real-time processing"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.database = EnhancedDatabase(self.config)
        self.aggregator = MetricsAggregator()
        self.stream_processor = StreamProcessor()
        
        # Processing queues and workers
        self.input_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.batch_queue = queue.Queue()
        self.workers = []
        self.running = False
        
        # Pipeline metrics
        self.pipeline_metrics = {
            'data_points_processed': 0,
            'data_points_queued': 0,
            'processing_errors': 0,
            'last_flush_time': time.time(),
            'average_processing_time': 0.0
        }
        
        self._setup_default_processors()
    
    def _setup_default_processors(self):
        """Setup default stream processors"""
        
        # Moving average processor
        def moving_average_processor(data_point: DataPoint) -> Optional[DataPoint]:
            if isinstance(data_point.value, (int, float)) and data_point.metric_type.endswith('_raw'):
                # Create moving average version
                window_key = f"{data_point.source}:{data_point.metric_type}"
                window = self.stream_processor.windows.get(window_key, deque())
                
                if len(window) >= 5:  # Need at least 5 points
                    values = [v for _, v in list(window)[-5:]]
                    avg_value = sum(values) / len(values)
                    
                    return DataPoint(
                        timestamp=data_point.timestamp,
                        source=data_point.source,
                        metric_type=data_point.metric_type.replace('_raw', '_avg'),
                        value=avg_value,
                        tags=data_point.tags,
                        metadata={**data_point.metadata, 'window_size': len(values)}
                    )
            return None
        
        # Anomaly detector for CPU spikes
        def cpu_spike_detector(data_point: DataPoint) -> bool:
            if data_point.metric_type == 'cpu_percent' and isinstance(data_point.value, (int, float)):
                return data_point.value > 95  # Detect CPU spikes above 95%
            return False
        
        # Memory pressure detector
        def memory_pressure_detector(data_point: DataPoint) -> bool:
            if data_point.metric_type == 'memory_percent' and isinstance(data_point.value, (int, float)):
                return data_point.value > 90  # Detect memory pressure above 90%
            return False
        
        self.stream_processor.add_processor(moving_average_processor)
        self.stream_processor.add_anomaly_detector(cpu_spike_detector)
        self.stream_processor.add_anomaly_detector(memory_pressure_detector)
    
    def start(self):
        """Start the data pipeline"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting enhanced data pipeline...")
        
        # Start worker threads
        for i in range(self.config.worker_threads):
            worker = threading.Thread(target=self._worker_loop, name=f"Pipeline-Worker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        # Start batch processor
        batch_processor = threading.Thread(target=self._batch_processor_loop, name="Batch-Processor")
        batch_processor.daemon = True
        batch_processor.start()
        
        # Start maintenance tasks
        maintenance_thread = threading.Thread(target=self._maintenance_loop, name="Maintenance")
        maintenance_thread.daemon = True
        maintenance_thread.start()
        
        logger.info(f"Data pipeline started with {self.config.worker_threads} workers")
    
    def stop(self):
        """Stop the data pipeline"""
        if not self.running:
            return
        
        logger.info("Stopping data pipeline...")
        self.running = False
        
        # Flush remaining data
        self._flush_batch_queue()
        
        logger.info("Data pipeline stopped")
    
    def ingest_data_point(self, data_point: DataPoint):
        """Ingest a single data point"""
        try:
            self.input_queue.put(data_point, timeout=1.0)
            self.pipeline_metrics['data_points_queued'] += 1
        except queue.Full:
            logger.warning("Input queue full, dropping data point")
            self.pipeline_metrics['processing_errors'] += 1
    
    def ingest_data_points(self, data_points: List[DataPoint]):
        """Ingest multiple data points"""
        for data_point in data_points:
            self.ingest_data_point(data_point)
    
    def _worker_loop(self):
        """Main worker loop for processing data points"""
        batch = []
        last_flush_time = time.time()
        
        while self.running:
            try:
                # Get data point with timeout
                data_point = self.input_queue.get(timeout=1.0)
                
                # Process through stream processors
                start_time = time.time()
                processed_points = self.stream_processor.process_data_point(data_point)
                processing_time = time.time() - start_time
                
                # Update processing metrics
                self.pipeline_metrics['data_points_processed'] += 1
                self.pipeline_metrics['average_processing_time'] = (
                    self.pipeline_metrics['average_processing_time'] * 0.9 + processing_time * 0.1
                )
                
                # Add to aggregator
                for point in processed_points:
                    self.aggregator.add_metric(point)
                    batch.append(point)
                
                # Flush batch if needed
                current_time = time.time()
                if (len(batch) >= self.config.batch_size or 
                    current_time - last_flush_time >= self.config.flush_interval):
                    
                    if batch:
                        self.batch_queue.put(batch.copy())
                        batch.clear()
                        last_flush_time = current_time
                
            except queue.Empty:
                # Flush any remaining batch on timeout
                if batch:
                    self.batch_queue.put(batch.copy())
                    batch.clear()
                    last_flush_time = time.time()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                self.pipeline_metrics['processing_errors'] += 1
    
    def _batch_processor_loop(self):
        """Process batches of data points for database insertion"""
        while self.running:
            try:
                batch = self.batch_queue.get(timeout=1.0)
                self.database.insert_data_points(batch)
                self.pipeline_metrics['last_flush_time'] = time.time()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                self.pipeline_metrics['processing_errors'] += 1
    
    def _flush_batch_queue(self):
        """Flush all remaining batches"""
        while not self.batch_queue.empty():
            try:
                batch = self.batch_queue.get_nowait()
                self.database.insert_data_points(batch)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error flushing batch: {e}")
    
    def _maintenance_loop(self):
        """Maintenance tasks (cleanup, archival, etc.)"""
        while self.running:
            try:
                # Cleanup old data based on retention policy
                cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
                self.database.cleanup_old_data(cutoff_time)
                self.aggregator.cleanup_old_aggregations(cutoff_time)
                
                # Record pipeline metrics
                if self.config.enable_pipeline_metrics:
                    self._record_pipeline_metrics()
                
                # Sleep until next maintenance cycle
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _record_pipeline_metrics(self):
        """Record pipeline performance metrics"""
        timestamp = datetime.now()
        
        metrics_to_record = [
            DataPoint(timestamp, "pipeline", "data_points_processed", self.pipeline_metrics['data_points_processed']),
            DataPoint(timestamp, "pipeline", "data_points_queued", self.pipeline_metrics['data_points_queued']),
            DataPoint(timestamp, "pipeline", "processing_errors", self.pipeline_metrics['processing_errors']),
            DataPoint(timestamp, "pipeline", "queue_size", self.input_queue.qsize()),
            DataPoint(timestamp, "pipeline", "average_processing_time", self.pipeline_metrics['average_processing_time']),
            DataPoint(timestamp, "pipeline", "time_since_last_flush", time.time() - self.pipeline_metrics['last_flush_time'])
        ]
        
        # Add to batch queue for processing
        self.batch_queue.put(metrics_to_record)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'running': self.running,
            'workers': len(self.workers),
            'queue_size': self.input_queue.qsize(),
            'batch_queue_size': self.batch_queue.qsize(),
            'metrics': self.pipeline_metrics.copy(),
            'config': asdict(self.config)
        }
    
    def query_metrics(self, **kwargs) -> List[Dict[str, Any]]:
        """Query metrics from the database"""
        return self.database.query_metrics(**kwargs)
    
    def get_aggregated_metrics(self, metric_key: str, window: str, 
                              start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get aggregated metrics"""
        return self.aggregator.get_aggregated_metrics(metric_key, window, start_time, end_time)


def create_sample_data_points() -> List[DataPoint]:
    """Create sample data points for testing"""
    timestamp = datetime.now()
    
    return [
        DataPoint(timestamp, "system", "cpu_percent", 45.2, {"host": "localhost"}),
        DataPoint(timestamp, "system", "memory_percent", 67.8, {"host": "localhost"}),
        DataPoint(timestamp, "tasks", "completed_count", 42, {"project": "task-master"}),
        DataPoint(timestamp, "tasks", "execution_time", 125.5, {"task_id": "43"}),
        DataPoint(timestamp, "github", "workflow_duration", 180.2, {"workflow": "ci"}),
        DataPoint(timestamp, "performance", "response_time", 89.3, {"endpoint": "/api/tasks"})
    ]


def main():
    """Main function for testing the enhanced data pipeline"""
    # Create configuration
    config = PipelineConfig(
        batch_size=10,
        flush_interval=2.0,
        worker_threads=2
    )
    
    # Create and start pipeline
    pipeline = DataPipeline(config)
    pipeline.start()
    
    try:
        print("Enhanced Data Pipeline started. Generating sample data...")
        
        # Generate sample data
        for i in range(50):
            sample_points = create_sample_data_points()
            pipeline.ingest_data_points(sample_points)
            time.sleep(0.5)
            
            if i % 10 == 0:
                status = pipeline.get_pipeline_status()
                print(f"Pipeline status: {status['metrics']['data_points_processed']} processed, "
                      f"{status['queue_size']} queued")
        
        # Wait for processing to complete
        time.sleep(5)
        
        # Query some metrics
        print("\nQuerying recent metrics...")
        recent_metrics = pipeline.query_metrics(
            source="system",
            start_time=datetime.now() - timedelta(minutes=5),
            limit=10
        )
        
        for metric in recent_metrics[:5]:
            print(f"  {metric['timestamp']}: {metric['source']}.{metric['metric_type']} = {metric['value']}")
        
        print(f"\nFinal pipeline status:")
        status = pipeline.get_pipeline_status()
        for key, value in status['metrics'].items():
            print(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()