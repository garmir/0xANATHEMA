#!/usr/bin/env python3
"""
Predictive Analytics Engine
Advanced forecasting system for project timeline, resource needs, and success prediction
"""

import json
import time
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

class ForecastType(Enum):
    """Types of forecasting"""
    TIMELINE = "timeline"
    RESOURCE_USAGE = "resource_usage"
    SUCCESS_PROBABILITY = "success_probability"
    COST_ESTIMATION = "cost_estimation"
    RISK_ASSESSMENT = "risk_assessment"
    TEAM_PERFORMANCE = "team_performance"

class ProjectPhase(Enum):
    """Project phases"""
    PLANNING = "planning"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"

@dataclass
class HistoricalDataPoint:
    """Individual historical data point"""
    timestamp: datetime
    project_id: str
    phase: ProjectPhase
    metrics: Dict[str, float]
    context: Dict[str, Any]

@dataclass
class PredictionModel:
    """Predictive model configuration"""
    model_id: str
    forecast_type: ForecastType
    accuracy_score: float
    training_data_size: int
    model_parameters: Dict[str, Any]
    last_training: datetime
    confidence_threshold: float

@dataclass
class Forecast:
    """Individual forecast result"""
    forecast_id: str
    forecast_type: ForecastType
    target_date: datetime
    predicted_value: Any
    confidence_interval: Tuple[float, float]
    confidence_score: float
    contributing_factors: List[str]
    recommendations: List[str]

@dataclass
class ProjectForecast:
    """Comprehensive project forecast"""
    project_id: str
    forecast_timestamp: datetime
    timeline_forecast: Forecast
    resource_forecast: Forecast
    success_forecast: Forecast
    cost_forecast: Forecast
    risk_forecast: Forecast
    overall_confidence: float
    key_insights: List[str]

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics for project forecasting"""
    
    def __init__(self, analytics_dir: str = '.taskmaster/analytics'):
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.historical_data_file = self.analytics_dir / 'historical_data.json'
        self.models_file = self.analytics_dir / 'prediction_models.json'
        self.forecasts_file = self.analytics_dir / 'forecasts.json'
        
        # Runtime data
        self.historical_data: deque = deque(maxlen=10000)
        self.prediction_models: Dict[ForecastType, PredictionModel] = {}
        self.forecast_cache: Dict[str, ProjectForecast] = {}
        
        # Analytics parameters
        self.min_training_data = 30
        self.model_retrain_interval = timedelta(days=7)
        self.forecast_horizon_days = 90
        
        self.initialize_analytics_engine()
    
    def initialize_analytics_engine(self):
        """Initialize the analytics engine with default models"""
        
        # Load existing data
        self.load_historical_data()
        
        # Initialize prediction models
        self.initialize_prediction_models()
        
        print(f"✅ Initialized predictive analytics with {len(self.historical_data)} historical points")
    
    def record_project_data(self, project_id: str, phase: ProjectPhase, 
                           metrics: Dict[str, float], context: Dict[str, Any] = None) -> str:
        """Record new project data point for learning"""
        
        data_point = HistoricalDataPoint(
            timestamp=datetime.now(),
            project_id=project_id,
            phase=phase,
            metrics=metrics,
            context=context or {}
        )
        
        self.historical_data.append(data_point)
        
        # Trigger model retraining if needed
        self.check_model_retraining()
        
        # Save data periodically
        if len(self.historical_data) % 10 == 0:
            self.save_historical_data()
        
        return f"datapoint_{int(time.time())}"
    
    def generate_project_forecast(self, project_context: Dict[str, Any]) -> ProjectForecast:
        """Generate comprehensive project forecast"""
        
        project_id = project_context.get('project_id', f"project_{int(time.time())}")
        
        print(f"📊 Generating forecast for project: {project_id}")
        
        # Generate individual forecasts
        timeline_forecast = self.forecast_timeline(project_context)
        resource_forecast = self.forecast_resource_usage(project_context)
        success_forecast = self.forecast_success_probability(project_context)
        cost_forecast = self.forecast_cost_estimation(project_context)
        risk_forecast = self.forecast_risk_assessment(project_context)
        
        # Calculate overall confidence
        individual_confidences = [
            timeline_forecast.confidence_score,
            resource_forecast.confidence_score,
            success_forecast.confidence_score,
            cost_forecast.confidence_score,
            risk_forecast.confidence_score
        ]
        overall_confidence = sum(individual_confidences) / len(individual_confidences)
        
        # Generate key insights
        key_insights = self.generate_project_insights(
            timeline_forecast, resource_forecast, success_forecast, 
            cost_forecast, risk_forecast, project_context
        )
        
        # Create comprehensive forecast
        project_forecast = ProjectForecast(
            project_id=project_id,
            forecast_timestamp=datetime.now(),
            timeline_forecast=timeline_forecast,
            resource_forecast=resource_forecast,
            success_forecast=success_forecast,
            cost_forecast=cost_forecast,
            risk_forecast=risk_forecast,
            overall_confidence=overall_confidence,
            key_insights=key_insights
        )
        
        # Cache forecast
        self.forecast_cache[project_id] = project_forecast
        
        # Save forecast
        self.save_forecast(project_forecast)
        
        return project_forecast
    
    def forecast_timeline(self, project_context: Dict[str, Any]) -> Forecast:
        """Forecast project timeline"""
        
        # Extract timeline factors
        team_size = project_context.get('team_size', 3)
        complexity_score = project_context.get('complexity_score', 0.5)
        task_count = project_context.get('task_count', 10)
        has_dependencies = project_context.get('has_dependencies', True)
        
        # Base timeline calculation (simplified model)
        base_days = task_count * 2  # 2 days per task baseline
        
        # Adjustments based on factors
        complexity_multiplier = 1 + (complexity_score * 0.8)  # Up to 80% increase
        team_efficiency = max(0.5, 1.0 - (team_size - 3) * 0.1)  # Efficiency based on team size
        dependency_overhead = 1.2 if has_dependencies else 1.0
        
        predicted_days = base_days * complexity_multiplier * team_efficiency * dependency_overhead
        
        # Add uncertainty
        confidence_score = 0.8 - (complexity_score * 0.3)  # Lower confidence for complex projects
        uncertainty_range = predicted_days * 0.3  # ±30% uncertainty
        
        confidence_interval = (
            predicted_days - uncertainty_range,
            predicted_days + uncertainty_range
        )
        
        # Contributing factors
        contributing_factors = [
            f"Team size: {team_size} members",
            f"Complexity: {complexity_score:.1%}",
            f"Task count: {task_count}",
            "Dependencies present" if has_dependencies else "No dependencies"
        ]
        
        # Recommendations
        recommendations = []
        if predicted_days > 60:
            recommendations.append("Consider breaking project into smaller phases")
        if team_size > 5:
            recommendations.append("Monitor team coordination overhead")
        if complexity_score > 0.7:
            recommendations.append("Allocate extra time for complex requirements")
        
        return Forecast(
            forecast_id=f"timeline_{int(time.time())}",
            forecast_type=ForecastType.TIMELINE,
            target_date=datetime.now() + timedelta(days=predicted_days),
            predicted_value=predicted_days,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
    
    def forecast_resource_usage(self, project_context: Dict[str, Any]) -> Forecast:
        """Forecast resource usage requirements"""
        
        # Extract resource factors
        team_size = project_context.get('team_size', 3)
        project_duration_days = project_context.get('estimated_duration', 30)
        technology_stack = project_context.get('technology_stack', ['python'])
        cloud_usage = project_context.get('cloud_usage', True)
        
        # Resource calculation
        # CPU hours: team_size * hours_per_day * duration
        cpu_hours = team_size * 8 * project_duration_days
        
        # Memory usage (GB-hours): based on technology stack
        memory_multipliers = {
            'python': 1.0,
            'javascript': 0.8,
            'java': 1.5,
            'docker': 1.3,
            'kubernetes': 2.0
        }
        
        memory_multiplier = max([memory_multipliers.get(tech, 1.0) for tech in technology_stack])
        memory_gb_hours = cpu_hours * memory_multiplier * 4  # 4GB per CPU-hour baseline
        
        # Storage usage
        storage_gb = 50 + (team_size * 20) + (project_duration_days * 2)  # Base + team + growth
        
        # Cloud costs (if applicable)
        cloud_cost = 0
        if cloud_usage:
            cloud_cost = (cpu_hours * 0.10) + (memory_gb_hours * 0.02) + (storage_gb * 0.01)
        
        predicted_resources = {
            'cpu_hours': cpu_hours,
            'memory_gb_hours': memory_gb_hours,
            'storage_gb': storage_gb,
            'estimated_cloud_cost': cloud_cost
        }
        
        # Confidence based on data availability
        confidence_score = 0.75
        
        confidence_interval = (
            cloud_cost * 0.8,  # 20% under estimate
            cloud_cost * 1.4   # 40% over estimate
        )
        
        contributing_factors = [
            f"Team size: {team_size} developers",
            f"Duration: {project_duration_days} days",
            f"Tech stack: {', '.join(technology_stack)}",
            "Cloud deployment" if cloud_usage else "On-premise deployment"
        ]
        
        recommendations = []
        if cloud_cost > 1000:
            recommendations.append("Consider cost optimization strategies")
        if memory_gb_hours > 10000:
            recommendations.append("Monitor memory usage and optimize where possible")
        recommendations.append("Set up resource monitoring and alerts")
        
        return Forecast(
            forecast_id=f"resource_{int(time.time())}",
            forecast_type=ForecastType.RESOURCE_USAGE,
            target_date=datetime.now() + timedelta(days=project_duration_days),
            predicted_value=predicted_resources,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
    
    def forecast_success_probability(self, project_context: Dict[str, Any]) -> Forecast:
        """Forecast project success probability"""
        
        # Success factors
        team_experience = project_context.get('team_experience_score', 0.7)  # 0-1
        requirements_clarity = project_context.get('requirements_clarity', 0.8)  # 0-1
        technology_maturity = project_context.get('technology_maturity', 0.9)  # 0-1
        deadline_pressure = project_context.get('deadline_pressure', 0.5)  # 0-1 (higher = more pressure)
        stakeholder_engagement = project_context.get('stakeholder_engagement', 0.7)  # 0-1
        
        # Success probability calculation (weighted factors)
        weights = {
            'team_experience': 0.25,
            'requirements_clarity': 0.20,
            'technology_maturity': 0.15,
            'deadline_pressure': -0.15,  # Negative impact
            'stakeholder_engagement': 0.20,
            'baseline': 0.15
        }
        
        success_probability = (
            weights['team_experience'] * team_experience +
            weights['requirements_clarity'] * requirements_clarity +
            weights['technology_maturity'] * technology_maturity +
            weights['deadline_pressure'] * (1 - deadline_pressure) +  # Invert pressure
            weights['stakeholder_engagement'] * stakeholder_engagement +
            weights['baseline']
        )
        
        success_probability = max(0.1, min(0.95, success_probability))  # Clamp between 10% and 95%
        
        # Confidence in prediction
        factor_variance = [
            abs(team_experience - 0.7),
            abs(requirements_clarity - 0.8),
            abs(technology_maturity - 0.9),
            abs(deadline_pressure - 0.5),
            abs(stakeholder_engagement - 0.7)
        ]
        
        confidence_score = max(0.5, 0.9 - (sum(factor_variance) / len(factor_variance)))
        
        confidence_interval = (
            max(0.1, success_probability - 0.15),
            min(0.95, success_probability + 0.15)
        )
        
        contributing_factors = [
            f"Team experience: {team_experience:.1%}",
            f"Requirements clarity: {requirements_clarity:.1%}",
            f"Technology maturity: {technology_maturity:.1%}",
            f"Deadline pressure: {deadline_pressure:.1%}",
            f"Stakeholder engagement: {stakeholder_engagement:.1%}"
        ]
        
        recommendations = []
        if success_probability < 0.6:
            recommendations.append("High risk project - consider risk mitigation strategies")
        if team_experience < 0.6:
            recommendations.append("Provide additional training or mentoring")
        if requirements_clarity < 0.7:
            recommendations.append("Invest more time in requirements gathering")
        if deadline_pressure > 0.7:
            recommendations.append("Negotiate timeline or reduce scope")
        
        return Forecast(
            forecast_id=f"success_{int(time.time())}",
            forecast_type=ForecastType.SUCCESS_PROBABILITY,
            target_date=datetime.now() + timedelta(days=30),
            predicted_value=success_probability,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
    
    def forecast_cost_estimation(self, project_context: Dict[str, Any]) -> Forecast:
        """Forecast project cost estimation"""
        
        # Cost factors
        team_size = project_context.get('team_size', 3)
        duration_days = project_context.get('estimated_duration', 30)
        average_daily_rate = project_context.get('daily_rate', 800)  # Per developer
        infrastructure_monthly_cost = project_context.get('infrastructure_cost', 500)
        tool_licenses_cost = project_context.get('tools_cost', 200)
        
        # Cost calculation
        labor_cost = team_size * duration_days * average_daily_rate
        infrastructure_cost = (duration_days / 30) * infrastructure_monthly_cost
        tools_cost = (duration_days / 30) * tool_licenses_cost
        
        # Add contingency (15% for unexpected costs)
        contingency = (labor_cost + infrastructure_cost + tools_cost) * 0.15
        
        total_estimated_cost = labor_cost + infrastructure_cost + tools_cost + contingency
        
        # Confidence based on project complexity
        complexity_score = project_context.get('complexity_score', 0.5)
        confidence_score = max(0.6, 0.9 - complexity_score * 0.4)
        
        # Cost range (±20% uncertainty)
        cost_uncertainty = total_estimated_cost * 0.2
        confidence_interval = (
            total_estimated_cost - cost_uncertainty,
            total_estimated_cost + cost_uncertainty
        )
        
        cost_breakdown = {
            'labor_cost': labor_cost,
            'infrastructure_cost': infrastructure_cost,
            'tools_cost': tools_cost,
            'contingency': contingency,
            'total_cost': total_estimated_cost
        }
        
        contributing_factors = [
            f"Team size: {team_size} @ ${average_daily_rate}/day",
            f"Duration: {duration_days} days",
            f"Infrastructure: ${infrastructure_monthly_cost}/month",
            f"Tools: ${tool_licenses_cost}/month",
            "15% contingency included"
        ]
        
        recommendations = []
        if total_estimated_cost > 100000:
            recommendations.append("Consider phased delivery to spread costs")
        if infrastructure_cost > labor_cost * 0.3:
            recommendations.append("Review infrastructure costs for optimization")
        recommendations.append("Monitor actual vs estimated costs throughout project")
        
        return Forecast(
            forecast_id=f"cost_{int(time.time())}",
            forecast_type=ForecastType.COST_ESTIMATION,
            target_date=datetime.now() + timedelta(days=duration_days),
            predicted_value=cost_breakdown,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
    
    def forecast_risk_assessment(self, project_context: Dict[str, Any]) -> Forecast:
        """Forecast risk assessment"""
        
        # Risk factors
        technology_risk = project_context.get('new_technology_usage', 0.3)  # 0-1
        team_turnover_risk = project_context.get('team_stability', 0.8)  # 0-1 (higher = more stable)
        external_dependency_risk = project_context.get('external_dependencies', 0.4)  # 0-1
        scope_change_risk = project_context.get('scope_stability', 0.7)  # 0-1
        deadline_risk = project_context.get('deadline_pressure', 0.5)  # 0-1
        
        # Calculate overall risk score
        risk_weights = {
            'technology': 0.2,
            'team_turnover': 0.25,
            'external_dependencies': 0.2,
            'scope_changes': 0.2,
            'deadline_pressure': 0.15
        }
        
        overall_risk = (
            risk_weights['technology'] * technology_risk +
            risk_weights['team_turnover'] * (1 - team_turnover_risk) +  # Invert stability
            risk_weights['external_dependencies'] * external_dependency_risk +
            risk_weights['scope_changes'] * (1 - scope_change_risk) +  # Invert stability
            risk_weights['deadline_pressure'] * deadline_risk
        )
        
        # Risk categories
        if overall_risk <= 0.3:
            risk_level = "LOW"
        elif overall_risk <= 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Confidence in risk assessment
        confidence_score = 0.8  # Generally high confidence in risk assessment
        
        confidence_interval = (
            max(0, overall_risk - 0.1),
            min(1, overall_risk + 0.1)
        )
        
        risk_breakdown = {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'technology_risk': technology_risk,
            'team_turnover_risk': 1 - team_turnover_risk,
            'external_dependency_risk': external_dependency_risk,
            'scope_change_risk': 1 - scope_change_risk,
            'deadline_risk': deadline_risk
        }
        
        contributing_factors = [
            f"Technology risk: {technology_risk:.1%}",
            f"Team stability: {team_turnover_risk:.1%}",
            f"External dependencies: {external_dependency_risk:.1%}",
            f"Scope stability: {scope_change_risk:.1%}",
            f"Deadline pressure: {deadline_risk:.1%}"
        ]
        
        recommendations = []
        if overall_risk > 0.6:
            recommendations.append("Implement comprehensive risk mitigation plan")
        if technology_risk > 0.5:
            recommendations.append("Conduct proof-of-concept for new technologies")
        if team_turnover_risk < 0.7:
            recommendations.append("Focus on team retention and knowledge sharing")
        if external_dependency_risk > 0.5:
            recommendations.append("Create contingency plans for external dependencies")
        
        return Forecast(
            forecast_id=f"risk_{int(time.time())}",
            forecast_type=ForecastType.RISK_ASSESSMENT,
            target_date=datetime.now() + timedelta(days=7),
            predicted_value=risk_breakdown,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
    
    def generate_project_insights(self, timeline_forecast: Forecast, resource_forecast: Forecast,
                                 success_forecast: Forecast, cost_forecast: Forecast,
                                 risk_forecast: Forecast, project_context: Dict[str, Any]) -> List[str]:
        """Generate key insights from all forecasts"""
        
        insights = []
        
        # Timeline insights
        timeline_days = timeline_forecast.predicted_value
        if timeline_days > 90:
            insights.append(f"Long project duration ({timeline_days:.0f} days) - consider milestone-based delivery")
        elif timeline_days < 14:
            insights.append("Short project timeline - ensure scope is well-defined")
        
        # Success probability insights
        success_prob = success_forecast.predicted_value
        if success_prob < 0.7:
            insights.append(f"Success probability below 70% ({success_prob:.1%}) - high attention required")
        elif success_prob > 0.85:
            insights.append(f"High success probability ({success_prob:.1%}) - well-positioned project")
        
        # Cost insights
        total_cost = cost_forecast.predicted_value['total_cost']
        if total_cost > 50000:
            insights.append(f"High-value project (${total_cost:,.0f}) - implement strict cost controls")
        
        # Risk insights
        risk_level = risk_forecast.predicted_value['risk_level']
        overall_risk = risk_forecast.predicted_value['overall_risk_score']
        insights.append(f"Project risk level: {risk_level} ({overall_risk:.1%})")
        
        # Cross-forecast insights
        if timeline_days > 60 and success_prob < 0.8:
            insights.append("Long timeline + moderate success risk - consider breaking into phases")
        
        if total_cost > 75000 and risk_level == "HIGH":
            insights.append("High cost + high risk - recommend additional oversight and controls")
        
        return insights
    
    def initialize_prediction_models(self):
        """Initialize prediction models"""
        
        # Create basic models for each forecast type
        for forecast_type in ForecastType:
            model = PredictionModel(
                model_id=f"model_{forecast_type.value}",
                forecast_type=forecast_type,
                accuracy_score=0.75,  # Initial estimate
                training_data_size=0,
                model_parameters={'algorithm': 'linear_regression', 'version': '1.0'},
                last_training=datetime.now(),
                confidence_threshold=0.7
            )
            self.prediction_models[forecast_type] = model
    
    def check_model_retraining(self):
        """Check if models need retraining"""
        
        if len(self.historical_data) < self.min_training_data:
            return
        
        # Check if any model needs retraining
        for forecast_type, model in self.prediction_models.items():
            if datetime.now() - model.last_training > self.model_retrain_interval:
                self.retrain_model(forecast_type)
    
    def retrain_model(self, forecast_type: ForecastType):
        """Retrain a specific model"""
        
        print(f"🔄 Retraining {forecast_type.value} model...")
        
        # Get relevant training data
        relevant_data = [
            point for point in self.historical_data
            if forecast_type.value in point.context.get('forecast_types', [forecast_type.value])
        ]
        
        if len(relevant_data) < self.min_training_data:
            print(f"⚠️ Insufficient data for {forecast_type.value} model retraining")
            return
        
        # Update model
        model = self.prediction_models[forecast_type]
        model.training_data_size = len(relevant_data)
        model.last_training = datetime.now()
        
        # Simple accuracy improvement simulation
        data_quality_factor = min(1.0, len(relevant_data) / 100)
        model.accuracy_score = min(0.95, model.accuracy_score + (data_quality_factor * 0.1))
        
        print(f"✅ Retrained {forecast_type.value} model - accuracy: {model.accuracy_score:.1%}")
    
    def load_historical_data(self):
        """Load historical data from disk"""
        try:
            if self.historical_data_file.exists():
                with open(self.historical_data_file, 'r') as f:
                    data = json.load(f)
                
                for point_data in data:
                    point = HistoricalDataPoint(
                        timestamp=datetime.fromisoformat(point_data['timestamp']),
                        project_id=point_data['project_id'],
                        phase=ProjectPhase(point_data['phase']),
                        metrics=point_data['metrics'],
                        context=point_data['context']
                    )
                    self.historical_data.append(point)
                    
        except Exception as e:
            print(f"⚠️ Failed to load historical data: {e}")
    
    def save_historical_data(self):
        """Save historical data to disk"""
        try:
            data = []
            for point in list(self.historical_data)[-1000:]:  # Save last 1000 points
                point_data = asdict(point)
                point_data['timestamp'] = point.timestamp.isoformat()
                point_data['phase'] = point.phase.value
                data.append(point_data)
            
            with open(self.historical_data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save historical data: {e}")
    
    def save_forecast(self, forecast: ProjectForecast):
        """Save forecast to disk"""
        try:
            forecasts = []
            if self.forecasts_file.exists():
                with open(self.forecasts_file, 'r') as f:
                    forecasts = json.load(f)
            
            forecast_data = asdict(forecast)
            
            # Convert datetime objects to strings
            forecast_data['forecast_timestamp'] = forecast.forecast_timestamp.isoformat()
            
            for forecast_field in ['timeline_forecast', 'resource_forecast', 'success_forecast', 
                                 'cost_forecast', 'risk_forecast']:
                if forecast_field in forecast_data:
                    forecast_data[forecast_field]['forecast_type'] = forecast_data[forecast_field]['forecast_type']['value']
                    forecast_data[forecast_field]['target_date'] = forecast_data[forecast_field]['target_date'].isoformat()
            
            forecasts.append(forecast_data)
            
            # Keep only last 100 forecasts
            if len(forecasts) > 100:
                forecasts = forecasts[-100:]
            
            with open(self.forecasts_file, 'w') as f:
                json.dump(forecasts, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save forecast: {e}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics engine summary"""
        return {
            'historical_data_points': len(self.historical_data),
            'prediction_models': len(self.prediction_models),
            'model_accuracy_scores': {
                ft.value: model.accuracy_score 
                for ft, model in self.prediction_models.items()
            },
            'cached_forecasts': len(self.forecast_cache),
            'forecast_types_supported': [ft.value for ft in ForecastType]
        }

def main():
    """Demo of predictive analytics engine"""
    print("Predictive Analytics Engine Demo")
    print("=" * 45)
    
    engine = PredictiveAnalyticsEngine()
    
    # Demo: Record some historical data
    historical_projects = [
        {
            'project_id': 'web_app_1',
            'phase': ProjectPhase.DEVELOPMENT,
            'metrics': {'duration_days': 45, 'team_size': 3, 'success_rate': 0.9, 'cost': 75000},
            'context': {'technology': 'react', 'complexity': 0.6}
        },
        {
            'project_id': 'api_service_1', 
            'phase': ProjectPhase.TESTING,
            'metrics': {'duration_days': 30, 'team_size': 2, 'success_rate': 0.85, 'cost': 45000},
            'context': {'technology': 'python', 'complexity': 0.4}
        }
    ]
    
    for project in historical_projects:
        engine.record_project_data(
            project['project_id'],
            project['phase'],
            project['metrics'],
            project['context']
        )
    
    print(f"📊 Recorded {len(historical_projects)} historical data points")
    
    # Demo: Generate forecast for new project
    new_project_context = {
        'project_id': 'mobile_app_forecast',
        'team_size': 4,
        'complexity_score': 0.7,
        'task_count': 25,
        'estimated_duration': 60,
        'technology_stack': ['react-native', 'python', 'docker'],
        'team_experience_score': 0.8,
        'requirements_clarity': 0.75,
        'technology_maturity': 0.85,
        'deadline_pressure': 0.6,
        'stakeholder_engagement': 0.8,
        'daily_rate': 900,
        'infrastructure_cost': 800,
        'cloud_usage': True
    }
    
    print(f"\n🔮 Generating comprehensive forecast...")
    forecast = engine.generate_project_forecast(new_project_context)
    
    print(f"\n📈 Forecast Results for {forecast.project_id}:")
    print(f"Overall confidence: {forecast.overall_confidence:.1%}")
    
    # Timeline forecast
    timeline_days = forecast.timeline_forecast.predicted_value
    print(f"\n⏱️ Timeline: {timeline_days:.0f} days")
    print(f"   Confidence: {forecast.timeline_forecast.confidence_score:.1%}")
    print(f"   Range: {forecast.timeline_forecast.confidence_interval[0]:.0f}-{forecast.timeline_forecast.confidence_interval[1]:.0f} days")
    
    # Success forecast
    success_prob = forecast.success_forecast.predicted_value
    print(f"\n🎯 Success Probability: {success_prob:.1%}")
    print(f"   Confidence: {forecast.success_forecast.confidence_score:.1%}")
    
    # Cost forecast
    total_cost = forecast.cost_forecast.predicted_value['total_cost']
    print(f"\n💰 Total Cost: ${total_cost:,.0f}")
    print(f"   Confidence: {forecast.cost_forecast.confidence_score:.1%}")
    
    # Risk forecast
    risk_level = forecast.risk_forecast.predicted_value['risk_level']
    risk_score = forecast.risk_forecast.predicted_value['overall_risk_score']
    print(f"\n⚠️ Risk Level: {risk_level} ({risk_score:.1%})")
    
    # Key insights
    print(f"\n💡 Key Insights:")
    for insight in forecast.key_insights:
        print(f"   • {insight}")
    
    # Analytics summary
    summary = engine.get_analytics_summary()
    print(f"\n📊 Analytics Summary:")
    print(f"   Data points: {summary['historical_data_points']}")
    print(f"   Models: {summary['prediction_models']}")
    print(f"   Forecast types: {len(summary['forecast_types_supported'])}")
    
    print(f"\n✅ Predictive analytics demo completed")

if __name__ == "__main__":
    main()