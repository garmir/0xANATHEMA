#!/usr/bin/env python3
"""
Circuit Breaker Patterns for Task-Master System
Implements resilience patterns based on research findings from comprehensive assessment

Based on microservices resilience patterns research, implementing:
1. Circuit Breaker Pattern - Prevents cascading failures
2. Bulkhead Pattern - Isolates resources
3. Timeout Pattern - Prevents hanging requests
4. Retry Pattern with Exponential Backoff
5. Health Check Pattern - Monitors service health
"""

import asyncio
import time
import threading
import logging
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import functools

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failure state, rejecting calls
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opened_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

class TimeoutError(Exception):
    """Raised when operation times out"""
    pass

class CircuitBreaker:
    """
    Circuit Breaker implementation for external API dependencies
    
    Based on Martin Fowler's Circuit Breaker pattern and Netflix Hystrix
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: tuple = (Exception,),
        timeout: Optional[float] = None,
        failure_rate_threshold: float = 0.5
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.timeout = timeout
        self.failure_rate_threshold = failure_rate_threshold
        
        self.stats = CircuitBreakerStats()
        self.lock = threading.RLock()
        
        # For half-open state testing
        self.half_open_max_calls = 3
        self.half_open_calls = 0
        
        logging.info(f"Circuit breaker '{name}' initialized with failure_threshold={failure_threshold}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection"""
        with self.lock:
            # Check circuit state before making call
            if self.stats.current_state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.total_requests += 1
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
            
            elif self.stats.current_state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    self.stats.total_requests += 1
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is HALF_OPEN and max calls exceeded")
        
        # Execute the function call
        start_time = time.time()
        try:
            self.stats.total_requests += 1
            
            # Apply timeout if configured
            if self.timeout:
                result = self._call_with_timeout(func, self.timeout, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            return result
            
        except self.expected_exception as e:
            execution_time = time.time() - start_time
            
            # Check if it was a timeout
            if self.timeout and execution_time >= self.timeout:
                self._record_timeout()
            else:
                self._record_failure(e)
            raise
    
    def _call_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            raise TimeoutError(f"Function call timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def _record_success(self):
        """Record successful call"""
        with self.lock:
            self.stats.successful_requests += 1
            self.stats.last_success_time = datetime.now()
            
            if self.stats.current_state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                # If we've had enough successful calls in half-open, close the circuit
                if self.half_open_calls >= self.half_open_max_calls:
                    self._transition_to_closed()
    
    def _record_failure(self, exception: Exception):
        """Record failed call"""
        with self.lock:
            self.stats.failed_requests += 1
            self.stats.last_failure_time = datetime.now()
            
            logging.warning(f"Circuit breaker '{self.name}' recorded failure: {exception}")
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()
    
    def _record_timeout(self):
        """Record timeout"""
        with self.lock:
            self.stats.failed_requests += 1
            self.stats.timeouts += 1
            self.stats.last_failure_time = datetime.now()
            
            logging.warning(f"Circuit breaker '{self.name}' recorded timeout")
            
            if self._should_open_circuit():
                self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        # Check failure count threshold
        if self.stats.failed_requests >= self.failure_threshold:
            return True
        
        # Check failure rate threshold (if we have enough data)
        if self.stats.total_requests >= 10 and self.stats.failure_rate >= self.failure_rate_threshold:
            return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Determine if we should attempt to reset from OPEN to HALF_OPEN"""
        if not self.stats.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.stats.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        self.stats.current_state = CircuitState.OPEN
        self.stats.circuit_opened_count += 1
        logging.error(f"Circuit breaker '{self.name}' transitioned to OPEN state")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        self.stats.current_state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        logging.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN state")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        self.stats.current_state = CircuitState.CLOSED
        # Reset failure count when circuit closes
        self.stats.failed_requests = 0
        logging.info(f"Circuit breaker '{self.name}' transitioned to CLOSED state")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "name": self.name,
            "state": self.stats.current_state.value,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "timeouts": self.stats.timeouts,
            "failure_rate": self.stats.failure_rate,
            "success_rate": self.stats.success_rate,
            "circuit_opened_count": self.stats.circuit_opened_count,
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self.lock:
            self.stats = CircuitBreakerStats()
            self.half_open_calls = 0
            logging.info(f"Circuit breaker '{self.name}' manually reset")

class BulkheadPattern:
    """
    Bulkhead Pattern implementation for resource isolation
    
    Isolates resources to prevent cascading failures
    """
    
    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_calls = 0
        self.total_calls = 0
        self.rejected_calls = 0
        self.lock = threading.Lock()
    
    async def __aenter__(self):
        """Async context manager entry"""
        with self.lock:
            self.total_calls += 1
            
        try:
            await self.semaphore.acquire()
            with self.lock:
                self.active_calls += 1
            return self
        except Exception:
            with self.lock:
                self.rejected_calls += 1
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        with self.lock:
            self.active_calls -= 1
        self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        with self.lock:
            return {
                "name": self.name,
                "max_concurrent": self.max_concurrent,
                "active_calls": self.active_calls,
                "total_calls": self.total_calls,
                "rejected_calls": self.rejected_calls,
                "utilization": self.active_calls / self.max_concurrent
            }

class RetryPattern:
    """
    Retry Pattern with exponential backoff and jitter
    
    Based on AWS SDK retry logic and Google Cloud best practices
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self.call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Call async function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    # Last attempt, don't delay
                    break
                
                delay = self._calculate_delay(attempt)
                logging.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Call sync function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    break
                
                delay = self._calculate_delay(attempt)
                logging.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            import random
            jitter_factor = 1 + (random.random() - 0.5) * 0.5
            delay *= jitter_factor
        
        return delay

class HealthChecker:
    """
    Health Check Pattern for monitoring service health
    
    Implements health check endpoints and dependency monitoring
    """
    
    def __init__(self, name: str, check_interval: float = 30.0):
        self.name = name
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
        self.check_task = None
    
    def register_check(self, check_name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[check_name] = check_func
        self.health_status[check_name] = {
            "status": "unknown",
            "last_check": None,
            "error": None
        }
        logging.info(f"Registered health check: {check_name}")
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.check_task = asyncio.create_task(self._monitoring_loop())
        logging.info(f"Health monitoring started for {self.name}")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_running = False
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        logging.info(f"Health monitoring stopped for {self.name}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _run_all_checks(self):
        """Run all registered health checks"""
        for check_name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                check_time = time.time() - start_time
                
                self.health_status[check_name] = {
                    "status": "healthy" if result else "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "check_duration": check_time,
                    "error": None
                }
                
            except Exception as e:
                self.health_status[check_name] = {
                    "status": "error",
                    "last_check": datetime.now().isoformat(),
                    "error": str(e)
                }
                logging.error(f"Health check '{check_name}' failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        overall_status = "healthy"
        
        for check_status in self.health_status.values():
            if check_status["status"] in ["unhealthy", "error"]:
                overall_status = "unhealthy"
                break
        
        return {
            "service": self.name,
            "overall_status": overall_status,
            "checks": self.health_status,
            "timestamp": datetime.now().isoformat()
        }

class ResilienceOrchestrator:
    """
    Orchestrates all resilience patterns for the Task-Master system
    
    Combines Circuit Breaker, Bulkhead, Retry, and Health Check patterns
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadPattern] = {}
        self.health_checker = HealthChecker("task-master-system")
        self.retry_policies: Dict[str, RetryPattern] = {}
        
        # Initialize default patterns for Task-Master external dependencies
        self._setup_default_patterns()
    
    def _setup_default_patterns(self):
        """Setup default resilience patterns for Task-Master"""
        
        # Circuit breakers for external APIs
        self.circuit_breakers["perplexity_api"] = CircuitBreaker(
            name="perplexity_api",
            failure_threshold=3,
            recovery_timeout=30,
            timeout=10.0,
            failure_rate_threshold=0.3
        )
        
        self.circuit_breakers["openai_api"] = CircuitBreaker(
            name="openai_api", 
            failure_threshold=5,
            recovery_timeout=60,
            timeout=30.0,
            failure_rate_threshold=0.5
        )
        
        self.circuit_breakers["github_api"] = CircuitBreaker(
            name="github_api",
            failure_threshold=3,
            recovery_timeout=45,
            timeout=15.0,
            failure_rate_threshold=0.4
        )
        
        # Bulkheads for resource isolation
        self.bulkheads["research_operations"] = BulkheadPattern(
            name="research_operations",
            max_concurrent=5
        )
        
        self.bulkheads["task_execution"] = BulkheadPattern(
            name="task_execution", 
            max_concurrent=10
        )
        
        self.bulkheads["file_operations"] = BulkheadPattern(
            name="file_operations",
            max_concurrent=3
        )
        
        # Retry policies
        self.retry_policies["api_calls"] = RetryPattern(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0
        )
        
        self.retry_policies["file_operations"] = RetryPattern(
            max_attempts=2,
            base_delay=0.5,
            max_delay=5.0
        )
        
        # Health checks
        self.health_checker.register_check("disk_space", self._check_disk_space)
        self.health_checker.register_check("memory_usage", self._check_memory_usage)
        self.health_checker.register_check("task_master_files", self._check_task_master_files)
        
        logging.info("Default resilience patterns initialized for Task-Master")
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_bulkhead(self, name: str) -> Optional[BulkheadPattern]:
        """Get bulkhead by name"""
        return self.bulkheads.get(name)
    
    def get_retry_policy(self, name: str) -> Optional[RetryPattern]:
        """Get retry policy by name"""
        return self.retry_policies.get(name)
    
    async def start_health_monitoring(self):
        """Start health monitoring"""
        await self.health_checker.start_monitoring()
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        await self.health_checker.stop_monitoring()
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all resilience patterns"""
        return {
            "circuit_breakers": {name: cb.get_stats() for name, cb in self.circuit_breakers.items()},
            "bulkheads": {name: bh.get_stats() for name, bh in self.bulkheads.items()},
            "health_status": self.health_checker.get_health_status(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        import shutil
        _, _, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        return free_gb > 1  # At least 1GB free
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage"""
        try:
            import subprocess
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
            # Simplified check - in production would parse actual memory usage
            return True  # Assume healthy for now
        except:
            return False
    
    def _check_task_master_files(self) -> bool:
        """Check Task Master essential files exist"""
        import os
        essential_files = [
            ".taskmaster/tasks/tasks.json",
            ".taskmaster/config.json"
        ]
        
        for file_path in essential_files:
            if not os.path.exists(file_path):
                return False
        return True

# Decorators for easy integration
def with_circuit_breaker(breaker_name: str, orchestrator: ResilienceOrchestrator):
    """Decorator to apply circuit breaker to a function"""
    def decorator(func: Callable) -> Callable:
        circuit_breaker = orchestrator.get_circuit_breaker(breaker_name)
        if circuit_breaker:
            return circuit_breaker(func)
        else:
            logging.warning(f"Circuit breaker '{breaker_name}' not found")
            return func
    return decorator

def with_retry(policy_name: str, orchestrator: ResilienceOrchestrator):
    """Decorator to apply retry policy to a function"""
    def decorator(func: Callable) -> Callable:
        retry_policy = orchestrator.get_retry_policy(policy_name)
        if retry_policy:
            return retry_policy(func)
        else:
            logging.warning(f"Retry policy '{policy_name}' not found")
            return func
    return decorator

async def with_bulkhead(bulkhead_name: str, orchestrator: ResilienceOrchestrator, coro):
    """Context manager to apply bulkhead pattern to async operations"""
    bulkhead = orchestrator.get_bulkhead(bulkhead_name)
    if bulkhead:
        async with bulkhead:
            return await coro
    else:
        logging.warning(f"Bulkhead '{bulkhead_name}' not found")
        return await coro

async def main():
    """Demonstration of resilience patterns"""
    print("🛡️ Task-Master Resilience Patterns Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = ResilienceOrchestrator()
    
    # Start health monitoring
    await orchestrator.start_health_monitoring()
    
    # Simulate external API calls with circuit breaker
    @with_circuit_breaker("perplexity_api", orchestrator)
    @with_retry("api_calls", orchestrator)
    def call_perplexity_api():
        """Simulate Perplexity API call"""
        import random
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("API temporarily unavailable")
        return {"response": "Research data retrieved successfully"}
    
    # Test circuit breaker and retry patterns
    print("🔄 Testing Circuit Breaker and Retry Patterns...")
    for i in range(10):
        try:
            result = call_perplexity_api()
            print(f"  ✅ Call {i+1}: {result['response']}")
        except Exception as e:
            print(f"  ❌ Call {i+1}: {e}")
        
        # Small delay between calls
        await asyncio.sleep(0.5)
    
    # Test bulkhead pattern
    print("\n🚧 Testing Bulkhead Pattern...")
    
    async def heavy_operation(operation_id: int):
        """Simulate heavy operation"""
        await asyncio.sleep(1)
        return f"Operation {operation_id} completed"
    
    # Run operations through bulkhead
    tasks = []
    for i in range(8):  # More than bulkhead limit
        task = with_bulkhead("research_operations", orchestrator, heavy_operation(i))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        print(f"  Operation {i}: {result}")
    
    # Show comprehensive status
    print("\n📊 Comprehensive Resilience Status:")
    status = orchestrator.get_comprehensive_status()
    print(json.dumps(status, indent=2, default=str))
    
    # Stop health monitoring
    await orchestrator.stop_health_monitoring()
    
    print("\n✅ Resilience patterns demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())