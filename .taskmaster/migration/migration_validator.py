#!/usr/bin/env python3
"""
Migration Validation Framework for Task Master AI Local LLM Migration
Comprehensive testing and validation of migration phases
"""

import asyncio
import json
import time
import logging
import hashlib
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from api_abstraction import TaskMasterLLMInterface, ModelCapability, TaskComplexity

class MigrationPhase(Enum):
    """Migration phases"""
    FOUNDATION = "foundation"
    RESEARCH_MODULE = "research_module"
    PLANNING_ENGINE = "planning_engine"
    ADVANCED_FEATURES = "advanced_features"
    OPTIMIZATION = "optimization"
    PRODUCTION = "production"

class TestCategory(Enum):
    """Test categories"""
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RELIABILITY = "reliability"
    COMPATIBILITY = "compatibility"

class TestSeverity(Enum):
    """Test failure severity levels"""
    CRITICAL = "critical"      # Blocks migration
    HIGH = "high"             # Significant issue
    MEDIUM = "medium"         # Minor issue
    LOW = "low"              # Cosmetic issue

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    category: TestCategory
    severity: TestSeverity
    success: bool
    score: float  # 0.0 to 1.0
    execution_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class ValidationReport:
    """Complete validation report"""
    phase: MigrationPhase
    timestamp: datetime
    test_results: List[TestResult]
    overall_score: float
    pass_rate: float
    critical_failures: int
    recommendations: List[str]
    ready_for_next_phase: bool
    
    def get_category_score(self, category: TestCategory) -> float:
        """Get score for specific test category"""
        category_results = [r for r in self.test_results if r.category == category]
        if not category_results:
            return 0.0
        return sum(r.score for r in category_results) / len(category_results)
    
    def get_severity_count(self, severity: TestSeverity) -> int:
        """Get count of failures by severity"""
        return sum(1 for r in self.test_results 
                  if not r.success and r.severity == severity)

class QualityMetrics:
    """Quality metrics for response evaluation"""
    
    @staticmethod
    def calculate_relevance_score(query: str, response: str) -> float:
        """Calculate relevance score based on keyword overlap"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= common_words
        response_words -= common_words
        
        if not query_words:
            return 0.0
        
        overlap = query_words.intersection(response_words)
        return len(overlap) / len(query_words)
    
    @staticmethod
    def calculate_completeness_score(response: str, expected_length: int = 100) -> float:
        """Calculate completeness score based on response length and structure"""
        # Length score
        length_score = min(1.0, len(response) / expected_length)
        
        # Structure score (sentences, paragraphs, etc.)
        structure_indicators = [
            '.' in response,  # Has sentences
            '\n' in response or len(response.split()) > 20,  # Has structure
            any(word in response.lower() for word in ['first', 'second', 'then', 'also', 'additionally']),  # Sequential
            len(response.split('.')) > 1  # Multiple sentences
        ]
        structure_score = sum(structure_indicators) / len(structure_indicators)
        
        return (length_score * 0.6 + structure_score * 0.4)
    
    @staticmethod
    def calculate_coherence_score(response: str) -> float:
        """Calculate coherence score based on response consistency"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5  # Default for short responses
        
        # Check for repeated phrases (indicates consistency)
        words = response.lower().split()
        if len(words) < 10:
            return 0.5
        
        # Simple coherence metric based on word repetition and flow
        word_variety = len(set(words)) / len(words)
        repetition_score = 1.0 - word_variety if word_variety < 0.8 else 1.0
        
        # Check for contradictory terms
        contradictions = [
            ('yes', 'no'), ('true', 'false'), ('good', 'bad'),
            ('increase', 'decrease'), ('start', 'stop')
        ]
        
        contradiction_count = 0
        for pos, neg in contradictions:
            if pos in response.lower() and neg in response.lower():
                contradiction_count += 1
        
        contradiction_penalty = contradiction_count * 0.1
        
        return max(0.0, repetition_score - contradiction_penalty)

class MigrationValidator:
    """Comprehensive migration validation framework"""
    
    def __init__(self, config_path: str = ".taskmaster/migration/configs"):
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.quality_metrics = QualityMetrics()
        
        # Test configurations
        self.test_configs = self._load_test_configurations()
        
        # Performance baselines
        self.performance_baselines = {
            'max_response_time': 30.0,  # seconds
            'min_success_rate': 0.90,
            'min_quality_score': 0.70,
            'max_concurrent_requests': 10
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger("migration_validator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(self.config_path / "validation.log")
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_test_configurations(self) -> Dict[str, Any]:
        """Load test configurations"""
        return {
            'functionality_tests': [
                {
                    'name': 'basic_chat',
                    'method': 'research',
                    'args': ['Hello, how are you?', ''],
                    'expected_keywords': ['hello', 'good', 'fine', 'well'],
                    'min_length': 10,
                    'timeout': 15
                },
                {
                    'name': 'simple_math',
                    'method': 'research',
                    'args': ['What is 15 + 27?', 'arithmetic'],
                    'expected_keywords': ['42', 'forty'],
                    'min_length': 5,
                    'timeout': 10
                },
                {
                    'name': 'research_query',
                    'method': 'research',
                    'args': ['Benefits of local LLM deployment', 'AI infrastructure'],
                    'expected_keywords': ['privacy', 'control', 'cost', 'latency', 'security'],
                    'min_length': 100,
                    'timeout': 25
                },
                {
                    'name': 'planning_task',
                    'method': 'plan',
                    'args': ['Implement API rate limiting', 'backend development'],
                    'expected_keywords': ['rate', 'limit', 'request', 'throttle', 'api'],
                    'min_length': 150,
                    'timeout': 30
                },
                {
                    'name': 'code_generation',
                    'method': 'generate_code',
                    'args': ['Create a REST API endpoint for user authentication', 'python'],
                    'expected_keywords': ['def', 'auth', 'user', 'api', 'endpoint'],
                    'min_length': 50,
                    'timeout': 25
                },
                {
                    'name': 'complex_analysis',
                    'method': 'analyze',
                    'args': ['System performance metrics showing 90% CPU, 75% memory, 500ms avg response', 'performance analysis'],
                    'expected_keywords': ['cpu', 'memory', 'performance', 'bottleneck', 'optimization'],
                    'min_length': 100,
                    'timeout': 30
                }
            ],
            'performance_tests': [
                {
                    'name': 'single_request_latency',
                    'concurrent_requests': 1,
                    'max_time': 30.0
                },
                {
                    'name': 'concurrent_requests',
                    'concurrent_requests': 5,
                    'max_time': 45.0
                },
                {
                    'name': 'high_load',
                    'concurrent_requests': 10,
                    'max_time': 60.0
                }
            ],
            'quality_benchmarks': {
                'min_relevance_score': 0.3,
                'min_completeness_score': 0.6,
                'min_coherence_score': 0.7
            }
        }
    
    async def validate_phase(self, phase: MigrationPhase, interface: TaskMasterLLMInterface) -> ValidationReport:
        """Validate specific migration phase"""
        self.logger.info(f"Starting validation for phase: {phase.value}")
        
        test_results = []
        
        # Run functionality tests
        functionality_results = await self._run_functionality_tests(interface)
        test_results.extend(functionality_results)
        
        # Run performance tests
        performance_results = await self._run_performance_tests(interface)
        test_results.extend(performance_results)
        
        # Run quality tests
        quality_results = await self._run_quality_tests(interface)
        test_results.extend(quality_results)
        
        # Run reliability tests
        reliability_results = await self._run_reliability_tests(interface)
        test_results.extend(reliability_results)
        
        # Run compatibility tests
        compatibility_results = await self._run_compatibility_tests(interface)
        test_results.extend(compatibility_results)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(test_results)
        pass_rate = sum(1 for r in test_results if r.success) / len(test_results)
        critical_failures = sum(1 for r in test_results 
                              if not r.success and r.severity == TestSeverity.CRITICAL)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(test_results, phase)
        
        # Determine readiness for next phase
        ready_for_next_phase = self._assess_readiness(test_results, phase)
        
        report = ValidationReport(
            phase=phase,
            timestamp=datetime.now(),
            test_results=test_results,
            overall_score=overall_score,
            pass_rate=pass_rate,
            critical_failures=critical_failures,
            recommendations=recommendations,
            ready_for_next_phase=ready_for_next_phase
        )
        
        # Save report
        await self._save_validation_report(report)
        
        self.logger.info(f"Validation completed for phase: {phase.value}")
        self.logger.info(f"Overall score: {overall_score:.2f}, Pass rate: {pass_rate:.2%}")
        
        return report
    
    async def _run_functionality_tests(self, interface: TaskMasterLLMInterface) -> List[TestResult]:
        """Run functionality tests"""
        self.logger.info("Running functionality tests...")
        results = []
        
        for test_config in self.test_configs['functionality_tests']:
            result = await self._run_single_functionality_test(interface, test_config)
            results.append(result)
        
        return results
    
    async def _run_single_functionality_test(self, interface: TaskMasterLLMInterface, 
                                           config: Dict[str, Any]) -> TestResult:
        """Run single functionality test"""
        test_name = config['name']
        method_name = config['method']
        args = config['args']
        expected_keywords = config['expected_keywords']
        min_length = config['min_length']
        timeout = config['timeout']
        
        try:
            start_time = time.time()
            method = getattr(interface, method_name)
            
            # Run with timeout
            response = await asyncio.wait_for(method(*args), timeout=timeout)
            execution_time = time.time() - start_time
            
            # Evaluate response
            score = self._evaluate_functionality_response(
                response, expected_keywords, min_length
            )
            
            success = score >= 0.5  # 50% threshold for success
            
            return TestResult(
                test_name=test_name,
                category=TestCategory.FUNCTIONALITY,
                severity=TestSeverity.HIGH,
                success=success,
                score=score,
                execution_time=execution_time,
                details={
                    'response_length': len(response),
                    'keywords_found': sum(1 for kw in expected_keywords 
                                        if kw.lower() in response.lower()),
                    'expected_keywords': len(expected_keywords),
                    'response_preview': response[:200] + '...' if len(response) > 200 else response
                }
            )
            
        except asyncio.TimeoutError:
            return TestResult(
                test_name=test_name,
                category=TestCategory.FUNCTIONALITY,
                severity=TestSeverity.CRITICAL,
                success=False,
                score=0.0,
                execution_time=timeout,
                error_message=f"Test timed out after {timeout}s"
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category=TestCategory.FUNCTIONALITY,
                severity=TestSeverity.HIGH,
                success=False,
                score=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _evaluate_functionality_response(self, response: str, expected_keywords: List[str], 
                                       min_length: int) -> float:
        """Evaluate functionality test response"""
        # Length score
        length_score = min(1.0, len(response) / min_length)
        
        # Keyword score
        keywords_found = sum(1 for kw in expected_keywords 
                           if kw.lower() in response.lower())
        keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 1.0
        
        # Quality scores
        relevance_score = self.quality_metrics.calculate_relevance_score(
            ' '.join(expected_keywords), response
        )
        coherence_score = self.quality_metrics.calculate_coherence_score(response)
        
        # Weighted combination
        return (length_score * 0.2 + keyword_score * 0.4 + 
                relevance_score * 0.2 + coherence_score * 0.2)
    
    async def _run_performance_tests(self, interface: TaskMasterLLMInterface) -> List[TestResult]:
        """Run performance tests"""
        self.logger.info("Running performance tests...")
        results = []
        
        for test_config in self.test_configs['performance_tests']:
            result = await self._run_single_performance_test(interface, test_config)
            results.append(result)
        
        return results
    
    async def _run_single_performance_test(self, interface: TaskMasterLLMInterface,
                                         config: Dict[str, Any]) -> TestResult:
        """Run single performance test"""
        test_name = config['name']
        concurrent_requests = config['concurrent_requests']
        max_time = config['max_time']
        
        test_query = "Explain the benefits of asynchronous programming in Python"
        
        try:
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for i in range(concurrent_requests):
                task = interface.research(f"{test_query} (request {i+1})", "performance test")
                tasks.append(task)
            
            # Execute concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            successful_responses = [r for r in responses if isinstance(r, str)]
            failed_responses = [r for r in responses if not isinstance(r, str)]
            
            success_rate = len(successful_responses) / len(responses)
            avg_time_per_request = total_time / concurrent_requests
            
            # Calculate score
            time_score = min(1.0, max_time / total_time) if total_time > 0 else 0.0
            success_score = success_rate
            overall_score = (time_score * 0.6 + success_score * 0.4)
            
            success = (total_time <= max_time and success_rate >= 0.8)
            
            return TestResult(
                test_name=test_name,
                category=TestCategory.PERFORMANCE,
                severity=TestSeverity.MEDIUM,
                success=success,
                score=overall_score,
                execution_time=total_time,
                details={
                    'concurrent_requests': concurrent_requests,
                    'successful_requests': len(successful_responses),
                    'failed_requests': len(failed_responses),
                    'success_rate': success_rate,
                    'avg_time_per_request': avg_time_per_request,
                    'total_time': total_time
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category=TestCategory.PERFORMANCE,
                severity=TestSeverity.HIGH,
                success=False,
                score=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    async def _run_quality_tests(self, interface: TaskMasterLLMInterface) -> List[TestResult]:
        """Run quality tests"""
        self.logger.info("Running quality tests...")
        
        quality_test_cases = [
            {
                'name': 'relevance_test',
                'query': 'Explain machine learning algorithms',
                'context': 'AI education',
                'expected_topics': ['algorithm', 'learning', 'data', 'model', 'training']
            },
            {
                'name': 'completeness_test',
                'query': 'How to deploy a web application?',
                'context': 'DevOps guide',
                'expected_topics': ['server', 'deployment', 'configuration', 'monitoring']
            },
            {
                'name': 'coherence_test',
                'query': 'Benefits and drawbacks of microservices architecture',
                'context': 'software architecture',
                'expected_topics': ['microservices', 'benefits', 'drawbacks', 'architecture']
            }
        ]
        
        results = []
        
        for test_case in quality_test_cases:
            try:
                start_time = time.time()
                response = await interface.research(test_case['query'], test_case['context'])
                execution_time = time.time() - start_time
                
                # Calculate quality metrics
                relevance_score = self.quality_metrics.calculate_relevance_score(
                    test_case['query'], response
                )
                completeness_score = self.quality_metrics.calculate_completeness_score(response)
                coherence_score = self.quality_metrics.calculate_coherence_score(response)
                
                # Overall quality score
                quality_score = (relevance_score + completeness_score + coherence_score) / 3
                
                # Check quality thresholds
                benchmarks = self.test_configs['quality_benchmarks']
                success = (
                    relevance_score >= benchmarks['min_relevance_score'] and
                    completeness_score >= benchmarks['min_completeness_score'] and
                    coherence_score >= benchmarks['min_coherence_score']
                )
                
                results.append(TestResult(
                    test_name=test_case['name'],
                    category=TestCategory.QUALITY,
                    severity=TestSeverity.MEDIUM,
                    success=success,
                    score=quality_score,
                    execution_time=execution_time,
                    details={
                        'relevance_score': relevance_score,
                        'completeness_score': completeness_score,
                        'coherence_score': coherence_score,
                        'response_length': len(response),
                        'query': test_case['query']
                    }
                ))
                
            except Exception as e:
                results.append(TestResult(
                    test_name=test_case['name'],
                    category=TestCategory.QUALITY,
                    severity=TestSeverity.MEDIUM,
                    success=False,
                    score=0.0,
                    execution_time=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    async def _run_reliability_tests(self, interface: TaskMasterLLMInterface) -> List[TestResult]:
        """Run reliability tests"""
        self.logger.info("Running reliability tests...")
        
        # Test repeated requests for consistency
        test_query = "What are the main principles of software engineering?"
        num_iterations = 5
        
        try:
            start_time = time.time()
            responses = []
            
            for i in range(num_iterations):
                response = await interface.research(test_query, f"iteration {i+1}")
                responses.append(response)
            
            execution_time = time.time() - start_time
            
            # Analyze consistency
            consistency_score = self._calculate_consistency_score(responses)
            
            return [TestResult(
                test_name="consistency_test",
                category=TestCategory.RELIABILITY,
                severity=TestSeverity.MEDIUM,
                success=consistency_score >= 0.7,
                score=consistency_score,
                execution_time=execution_time,
                details={
                    'num_iterations': num_iterations,
                    'responses': responses[:2],  # Save first 2 responses
                    'avg_response_length': statistics.mean(len(r) for r in responses)
                }
            )]
            
        except Exception as e:
            return [TestResult(
                test_name="consistency_test",
                category=TestCategory.RELIABILITY,
                severity=TestSeverity.HIGH,
                success=False,
                score=0.0,
                execution_time=0.0,
                error_message=str(e)
            )]
    
    def _calculate_consistency_score(self, responses: List[str]) -> float:
        """Calculate consistency score for multiple responses"""
        if len(responses) < 2:
            return 0.0
        
        # Calculate pairwise similarity
        similarities = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._calculate_text_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple word overlap)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _run_compatibility_tests(self, interface: TaskMasterLLMInterface) -> List[TestResult]:
        """Run compatibility tests"""
        self.logger.info("Running compatibility tests...")
        
        try:
            # Test health status endpoint
            start_time = time.time()
            health_status = await interface.get_health_status()
            execution_time = time.time() - start_time
            
            # Check if health status is valid
            has_providers = len(health_status) > 0
            has_healthy_providers = any(
                status.get('status') == 'healthy' 
                for status in health_status.values()
            )
            
            success = has_providers and has_healthy_providers
            score = 1.0 if success else 0.0
            
            return [TestResult(
                test_name="health_status_test",
                category=TestCategory.COMPATIBILITY,
                severity=TestSeverity.CRITICAL,
                success=success,
                score=score,
                execution_time=execution_time,
                details={
                    'provider_count': len(health_status),
                    'healthy_providers': sum(1 for s in health_status.values() 
                                           if s.get('status') == 'healthy'),
                    'health_status': health_status
                }
            )]
            
        except Exception as e:
            return [TestResult(
                test_name="health_status_test",
                category=TestCategory.COMPATIBILITY,
                severity=TestSeverity.CRITICAL,
                success=False,
                score=0.0,
                execution_time=0.0,
                error_message=str(e)
            )]
    
    def _calculate_overall_score(self, test_results: List[TestResult]) -> float:
        """Calculate overall validation score"""
        if not test_results:
            return 0.0
        
        # Weight by category
        category_weights = {
            TestCategory.FUNCTIONALITY: 0.35,
            TestCategory.PERFORMANCE: 0.25,
            TestCategory.QUALITY: 0.20,
            TestCategory.RELIABILITY: 0.15,
            TestCategory.COMPATIBILITY: 0.05
        }
        
        category_scores = {}
        for category in TestCategory:
            category_results = [r for r in test_results if r.category == category]
            if category_results:
                category_scores[category] = sum(r.score for r in category_results) / len(category_results)
            else:
                category_scores[category] = 0.0
        
        weighted_score = sum(
            category_scores[category] * weight
            for category, weight in category_weights.items()
        )
        
        return weighted_score
    
    def _generate_recommendations(self, test_results: List[TestResult], 
                                 phase: MigrationPhase) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Critical failures
        critical_failures = [r for r in test_results 
                           if not r.success and r.severity == TestSeverity.CRITICAL]
        if critical_failures:
            recommendations.append(
                f"CRITICAL: {len(critical_failures)} critical test(s) failed. "
                "These must be resolved before proceeding."
            )
        
        # Performance issues
        perf_results = [r for r in test_results if r.category == TestCategory.PERFORMANCE]
        avg_perf_score = sum(r.score for r in perf_results) / len(perf_results) if perf_results else 0
        if avg_perf_score < 0.7:
            recommendations.append(
                "Performance optimization needed. Consider model selection tuning "
                "or resource allocation improvements."
            )
        
        # Quality issues
        quality_results = [r for r in test_results if r.category == TestCategory.QUALITY]
        avg_quality_score = sum(r.score for r in quality_results) / len(quality_results) if quality_results else 0
        if avg_quality_score < 0.7:
            recommendations.append(
                "Response quality needs improvement. Consider prompt optimization "
                "or using larger models for complex tasks."
            )
        
        # Reliability issues
        reliability_results = [r for r in test_results if r.category == TestCategory.RELIABILITY]
        if reliability_results and not all(r.success for r in reliability_results):
            recommendations.append(
                "Reliability concerns detected. Review error handling and "
                "implement better fallback mechanisms."
            )
        
        # Phase-specific recommendations
        if phase == MigrationPhase.FOUNDATION:
            recommendations.append("Ensure all providers are properly configured and healthy.")
        elif phase == MigrationPhase.RESEARCH_MODULE:
            recommendations.append("Validate research capabilities with domain-specific queries.")
        elif phase == MigrationPhase.PLANNING_ENGINE:
            recommendations.append("Test planning accuracy with complex, multi-step tasks.")
        
        return recommendations
    
    def _assess_readiness(self, test_results: List[TestResult], phase: MigrationPhase) -> bool:
        """Assess readiness for next migration phase"""
        # No critical failures
        critical_failures = sum(1 for r in test_results 
                              if not r.success and r.severity == TestSeverity.CRITICAL)
        if critical_failures > 0:
            return False
        
        # Minimum pass rate
        pass_rate = sum(1 for r in test_results if r.success) / len(test_results)
        if pass_rate < 0.85:  # 85% pass rate required
            return False
        
        # Category-specific requirements
        category_scores = {}
        for category in TestCategory:
            category_results = [r for r in test_results if r.category == category]
            if category_results:
                category_scores[category] = sum(r.score for r in category_results) / len(category_results)
        
        # Functionality must be strong
        if category_scores.get(TestCategory.FUNCTIONALITY, 0) < 0.8:
            return False
        
        # Performance must be acceptable
        if category_scores.get(TestCategory.PERFORMANCE, 0) < 0.6:
            return False
        
        return True
    
    async def _save_validation_report(self, report: ValidationReport):
        """Save validation report to file"""
        report_file = (self.config_path / 
                      f"validation_report_{report.phase.value}_{int(report.timestamp.timestamp())}.json")
        
        # Convert to serializable format
        report_data = {
            'phase': report.phase.value,
            'timestamp': report.timestamp.isoformat(),
            'overall_score': report.overall_score,
            'pass_rate': report.pass_rate,
            'critical_failures': report.critical_failures,
            'ready_for_next_phase': report.ready_for_next_phase,
            'recommendations': report.recommendations,
            'test_results': [
                {
                    'test_name': r.test_name,
                    'category': r.category.value,
                    'severity': r.severity.value,
                    'success': r.success,
                    'score': r.score,
                    'execution_time': r.execution_time,
                    'error_message': r.error_message,
                    'details': r.details
                }
                for r in report.test_results
            ],
            'category_scores': {
                category.value: report.get_category_score(category)
                for category in TestCategory
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved: {report_file}")
    
    def print_report_summary(self, report: ValidationReport):
        """Print validation report summary"""
        print(f"\n{'='*60}")
        print(f"MIGRATION VALIDATION REPORT - {report.phase.value.upper()}")
        print(f"{'='*60}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Score: {report.overall_score:.2f}/1.0")
        print(f"Pass Rate: {report.pass_rate:.1%}")
        print(f"Critical Failures: {report.critical_failures}")
        print(f"Ready for Next Phase: {'✅ YES' if report.ready_for_next_phase else '❌ NO'}")
        
        print(f"\nCATEGORY SCORES:")
        for category in TestCategory:
            score = report.get_category_score(category)
            print(f"  {category.value.title()}: {score:.2f}")
        
        print(f"\nTEST RESULTS:")
        for result in report.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"  {result.test_name}: {status} (Score: {result.score:.2f}, Time: {result.execution_time:.2f}s)")
        
        if report.recommendations:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"{'='*60}\n")

async def main():
    """Main validation function"""
    print("🧪 Task Master AI - Migration Validation Framework")
    print("=" * 60)
    
    validator = MigrationValidator()
    interface = TaskMasterLLMInterface()
    
    try:
        await interface.initialize()
        
        # Run validation for current phase (example: foundation)
        phase = MigrationPhase.FOUNDATION
        report = await validator.validate_phase(phase, interface)
        
        # Print summary
        validator.print_report_summary(report)
        
        print("📊 Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await interface.cleanup()

if __name__ == "__main__":
    asyncio.run(main())