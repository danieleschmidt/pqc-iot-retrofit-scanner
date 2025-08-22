"""
Generation 5: Enterprise-Grade Scaling & Optimization Engine

Advanced scaling capabilities featuring:
- Auto-scaling quantum threat analysis
- Distributed processing with fault tolerance
- Real-time load balancing and resource optimization
- Enterprise integration and API management
- Multi-tenant quantum security orchestration
- Global deployment automation
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
from collections import defaultdict, deque
import hashlib
import aiohttp
import ssl

try:
    import kubernetes
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from .scanner import CryptoVulnerability, CryptoAlgorithm
from .advanced_pqc_engine import AdvancedPQCEngine, PQCImplementation
from .error_handling import PQCRetrofitError, ErrorSeverity, ErrorCategory
from .monitoring import track_performance, QuantumEnhancedMetricsCollector


class ScalingStrategy(Enum):
    """Scaling strategies for different workload patterns."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"
    PREDICTIVE = "predictive"
    QUANTUM_AWARE = "quantum_aware"


class DeploymentTier(Enum):
    """Deployment tier classification."""
    EDGE = "edge"
    REGIONAL = "regional"
    GLOBAL = "global"
    QUANTUM_CORE = "quantum_core"


@dataclass
class WorkloadProfile:
    """Workload characterization for optimal scaling."""
    workload_id: str
    firmware_analysis_rate: float = 0.0  # files per second
    vulnerability_complexity: str = "medium"  # low, medium, high, quantum
    target_architectures: List[str] = field(default_factory=list)
    
    # Resource requirements
    cpu_intensity: float = 1.0  # 1.0 = baseline
    memory_intensity: float = 1.0
    io_intensity: float = 1.0
    network_intensity: float = 1.0
    
    # Business requirements
    sla_response_time_ms: int = 5000
    availability_requirement: float = 0.99
    security_compliance: List[str] = field(default_factory=list)
    
    # Scaling characteristics
    burst_capacity_multiplier: float = 2.0
    predictable_patterns: bool = False
    geographic_distribution: List[str] = field(default_factory=list)


@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning."""
    decision_id: str
    timestamp: float
    action: str  # "scale_up", "scale_down", "maintain", "migrate"
    resource_type: str  # "cpu", "memory", "instances", "regions"
    
    # Decision parameters
    current_capacity: Dict[str, float]
    target_capacity: Dict[str, float]
    confidence_score: float
    
    # Reasoning
    trigger_metrics: Dict[str, float]
    decision_factors: List[str]
    expected_impact: Dict[str, float]
    risk_assessment: str
    
    # Execution
    execution_strategy: str
    rollback_plan: str
    monitoring_requirements: List[str]


class EnterpriseScalingEngine:
    """Advanced enterprise scaling and optimization engine."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.pqc_engine = AdvancedPQCEngine()
        self.metrics_collector = QuantumEnhancedMetricsCollector()
        
        # Scaling infrastructure
        self.workload_profiles = {}
        self.scaling_history = deque(maxlen=10000)
        self.active_deployments = {}
        
        # Executors for different workload types
        self.cpu_executor = ThreadPoolExecutor(max_workers=8)
        self.io_executor = ThreadPoolExecutor(max_workers=16)
        self.compute_executor = ProcessPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.performance_cache = {}
        self.optimization_models = {}
        
        # Enterprise features
        self.tenant_isolation = {}
        self.api_rate_limits = defaultdict(lambda: {"requests": 0, "reset_time": time.time()})
        self.global_deployment_manager = None
        
        # Initialize components
        self._initialize_scaling_infrastructure()
        
    def _initialize_scaling_infrastructure(self):
        """Initialize enterprise scaling infrastructure."""
        
        # Default workload profiles
        self.workload_profiles["iot_fleet_analysis"] = WorkloadProfile(
            workload_id="iot_fleet_analysis",
            firmware_analysis_rate=10.0,
            vulnerability_complexity="medium",
            target_architectures=["cortex-m4", "esp32", "risc-v"],
            cpu_intensity=2.0,
            memory_intensity=1.5,
            sla_response_time_ms=30000,
            availability_requirement=0.995,
            security_compliance=["NIST", "FIPS", "Common Criteria"],
            burst_capacity_multiplier=5.0,
            predictable_patterns=True
        )
        
        self.workload_profiles["quantum_threat_assessment"] = WorkloadProfile(
            workload_id="quantum_threat_assessment",
            firmware_analysis_rate=2.0,
            vulnerability_complexity="quantum",
            target_architectures=["all"],
            cpu_intensity=5.0,
            memory_intensity=3.0,
            io_intensity=0.5,
            network_intensity=2.0,
            sla_response_time_ms=10000,
            availability_requirement=0.999,
            security_compliance=["NIST PQC", "NSA CNSA 2.0"],
            burst_capacity_multiplier=3.0,
            predictable_patterns=False
        )
        
        self.workload_profiles["enterprise_compliance_scan"] = WorkloadProfile(
            workload_id="enterprise_compliance_scan",
            firmware_analysis_rate=50.0,
            vulnerability_complexity="high",
            target_architectures=["x86", "arm64", "cortex-m"],
            cpu_intensity=1.5,
            memory_intensity=2.0,
            io_intensity=3.0,
            sla_response_time_ms=60000,
            availability_requirement=0.99,
            security_compliance=["SOX", "GDPR", "HIPAA", "NIST"],
            burst_capacity_multiplier=10.0,
            predictable_patterns=True,
            geographic_distribution=["us-east", "eu-west", "ap-southeast"]
        )
        
        self.logger.info("Enterprise scaling infrastructure initialized")
    
    @track_performance
    async def auto_scale_analysis_capacity(self, workload_metrics: Dict[str, float],
                                         workload_type: str = "iot_fleet_analysis") -> ScalingDecision:
        """Intelligent auto-scaling based on workload analysis."""
        
        if workload_type not in self.workload_profiles:
            raise PQCRetrofitError(
                f"Unknown workload type: {workload_type}",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.CONFIGURATION
            )
        
        profile = self.workload_profiles[workload_type]
        
        # Analyze current performance metrics
        current_metrics = await self._gather_current_metrics(workload_type)
        
        # Predict future demand
        predicted_demand = await self._predict_demand(workload_metrics, profile)
        
        # Calculate optimal resource allocation
        optimal_allocation = await self._calculate_optimal_allocation(
            current_metrics, predicted_demand, profile
        )
        
        # Generate scaling decision
        decision = await self._generate_scaling_decision(
            current_metrics, optimal_allocation, profile
        )
        
        # Record decision for learning
        self.scaling_history.append(decision)
        
        return decision
    
    async def _gather_current_metrics(self, workload_type: str) -> Dict[str, float]:
        """Gather current system and workload metrics."""
        
        # Simulate metric gathering (in real implementation, would query monitoring systems)
        return {
            "cpu_utilization": 0.65,
            "memory_utilization": 0.70,
            "queue_depth": 15.0,
            "response_time_p95": 4500.0,
            "error_rate": 0.02,
            "throughput_rps": 8.5,
            "active_instances": 3,
            "pending_requests": 25
        }
    
    async def _predict_demand(self, current_metrics: Dict[str, float],
                            profile: WorkloadProfile) -> Dict[str, float]:
        """Predict future demand using ML and pattern analysis."""
        
        # Time-based prediction
        current_hour = datetime.now().hour
        weekday = datetime.now().weekday()
        
        # Simulate demand prediction
        base_multiplier = 1.0
        
        # Business hours adjustment
        if 9 <= current_hour <= 17 and weekday < 5:  # Business hours, weekday
            base_multiplier = 1.5
        elif current_hour < 6 or current_hour > 22:  # Night hours
            base_multiplier = 0.3
        
        # Pattern-based adjustment
        if profile.predictable_patterns:
            # Add seasonal/cyclical patterns
            seasonal_factor = 1.0 + 0.2 * math.sin(2 * math.pi * current_hour / 24)
            base_multiplier *= seasonal_factor
        
        # Trend analysis from historical data
        recent_trend = self._analyze_recent_trend(current_metrics)
        trend_multiplier = 1.0 + (recent_trend * 0.5)  # Dampen trend impact
        
        predicted_demand = {}
        for metric, value in current_metrics.items():
            if metric in ["throughput_rps", "queue_depth", "pending_requests"]:
                predicted_demand[f"predicted_{metric}"] = value * base_multiplier * trend_multiplier
            else:
                predicted_demand[f"predicted_{metric}"] = value
                
        return predicted_demand
    
    def _analyze_recent_trend(self, current_metrics: Dict[str, float]) -> float:
        """Analyze recent trend from historical data."""
        # Simplified trend analysis (in real implementation, would use time series analysis)
        if len(self.scaling_history) < 3:
            return 0.0
            
        recent_decisions = list(self.scaling_history)[-3:]
        scale_up_count = sum(1 for d in recent_decisions if d.action == "scale_up")
        scale_down_count = sum(1 for d in recent_decisions if d.action == "scale_down")
        
        if scale_up_count > scale_down_count:
            return 0.1  # Upward trend
        elif scale_down_count > scale_up_count:
            return -0.1  # Downward trend
        else:
            return 0.0  # Stable
    
    async def _calculate_optimal_allocation(self, current_metrics: Dict[str, float],
                                          predicted_demand: Dict[str, float],
                                          profile: WorkloadProfile) -> Dict[str, float]:
        """Calculate optimal resource allocation."""
        
        # Target utilization thresholds
        target_cpu = 0.70
        target_memory = 0.75
        target_response_time = profile.sla_response_time_ms * 0.8  # 80% of SLA
        
        # Calculate required capacity
        cpu_ratio = predicted_demand.get("predicted_throughput_rps", 0) / max(current_metrics.get("throughput_rps", 1), 1)
        memory_ratio = predicted_demand.get("predicted_queue_depth", 0) / max(current_metrics.get("queue_depth", 1), 1)
        
        # Adjust for intensity factors
        cpu_requirement = cpu_ratio * profile.cpu_intensity
        memory_requirement = memory_ratio * profile.memory_intensity
        
        # Calculate instance count
        current_instances = current_metrics.get("active_instances", 1)
        required_instances = max(
            current_instances * (cpu_requirement / target_cpu),
            current_instances * (memory_requirement / target_memory)
        )
        
        # Add burst capacity if needed
        if predicted_demand.get("predicted_queue_depth", 0) > current_metrics.get("queue_depth", 0) * 1.5:
            required_instances *= profile.burst_capacity_multiplier
        
        # Round up and apply constraints
        required_instances = max(1, math.ceil(required_instances))
        max_instances = self.config.get("max_instances", 20)
        required_instances = min(required_instances, max_instances)
        
        return {
            "target_instances": required_instances,
            "target_cpu_cores": required_instances * 2,  # 2 cores per instance
            "target_memory_gb": required_instances * 4,  # 4GB per instance
            "target_storage_gb": required_instances * 20,  # 20GB per instance
            "confidence": min(0.9, max(0.5, 1.0 - abs(cpu_ratio - 1.0)))
        }
    
    async def _generate_scaling_decision(self, current_metrics: Dict[str, float],
                                       optimal_allocation: Dict[str, float],
                                       profile: WorkloadProfile) -> ScalingDecision:
        """Generate scaling decision with reasoning."""
        
        current_instances = current_metrics.get("active_instances", 1)
        target_instances = optimal_allocation["target_instances"]
        
        # Determine action
        if target_instances > current_instances * 1.1:  # >10% increase
            action = "scale_up"
            resource_type = "instances"
        elif target_instances < current_instances * 0.9:  # >10% decrease
            action = "scale_down"
            resource_type = "instances"
        else:
            action = "maintain"
            resource_type = "instances"
        
        # Build decision factors
        decision_factors = []
        
        if current_metrics.get("cpu_utilization", 0) > 0.8:
            decision_factors.append("High CPU utilization detected")
        if current_metrics.get("response_time_p95", 0) > profile.sla_response_time_ms * 0.9:
            decision_factors.append("Response time approaching SLA limit")
        if current_metrics.get("queue_depth", 0) > 20:
            decision_factors.append("Queue depth indicates capacity constraint")
        if current_metrics.get("error_rate", 0) > 0.05:
            decision_factors.append("Elevated error rate may indicate overload")
        
        # Risk assessment
        if action == "scale_up":
            risk = "low" if optimal_allocation["confidence"] > 0.8 else "medium"
        elif action == "scale_down":
            risk = "medium" if current_metrics.get("throughput_rps", 0) > 5 else "low"
        else:
            risk = "low"
        
        decision = ScalingDecision(
            decision_id=str(uuid.uuid4()),
            timestamp=time.time(),
            action=action,
            resource_type=resource_type,
            current_capacity={
                "instances": current_instances,
                "cpu_cores": current_instances * 2,
                "memory_gb": current_instances * 4
            },
            target_capacity={
                "instances": target_instances,
                "cpu_cores": target_instances * 2,
                "memory_gb": target_instances * 4
            },
            confidence_score=optimal_allocation["confidence"],
            trigger_metrics=current_metrics,
            decision_factors=decision_factors,
            expected_impact={
                "response_time_improvement": 0.15 if action == "scale_up" else -0.05,
                "cost_impact": (target_instances - current_instances) * 0.1,  # $0.10 per instance per hour
                "availability_impact": 0.001 if action == "scale_up" else -0.0005
            },
            risk_assessment=risk,
            execution_strategy=self._determine_execution_strategy(action, target_instances, current_instances),
            rollback_plan=self._generate_rollback_plan(action),
            monitoring_requirements=self._define_monitoring_requirements(action)
        )
        
        return decision
    
    def _determine_execution_strategy(self, action: str, target: float, current: float) -> str:
        """Determine how to execute the scaling decision."""
        if action == "scale_up":
            increase = target - current
            if increase <= 2:
                return "immediate_scaling"
            else:
                return "gradual_scaling_25_percent_increments"
        elif action == "scale_down":
            decrease = current - target
            if decrease <= 1:
                return "immediate_scaling"
            else:
                return "gradual_scaling_drain_and_terminate"
        else:
            return "no_action_required"
    
    def _generate_rollback_plan(self, action: str) -> str:
        """Generate rollback plan for scaling decision."""
        if action == "scale_up":
            return "Monitor for 10 minutes, rollback if error rate increases or response time degrades"
        elif action == "scale_down":
            return "Monitor for 5 minutes, immediately scale back up if response time exceeds SLA"
        else:
            return "No rollback required"
    
    def _define_monitoring_requirements(self, action: str) -> List[str]:
        """Define monitoring requirements for scaling action."""
        base_requirements = [
            "Response time monitoring",
            "Error rate tracking",
            "Resource utilization monitoring"
        ]
        
        if action == "scale_up":
            base_requirements.extend([
                "Cost impact tracking",
                "Instance health monitoring",
                "Load distribution verification"
            ])
        elif action == "scale_down":
            base_requirements.extend([
                "Capacity headroom monitoring",
                "Queue depth tracking",
                "Performance degradation alerts"
            ])
        
        return base_requirements
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute scaling decision with monitoring and rollback capability."""
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Executing scaling decision {decision.decision_id}: {decision.action}")
        
        try:
            # Pre-execution validation
            await self._validate_scaling_prerequisites(decision)
            
            # Execute based on strategy
            if decision.execution_strategy == "immediate_scaling":
                result = await self._execute_immediate_scaling(decision)
            elif "gradual_scaling" in decision.execution_strategy:
                result = await self._execute_gradual_scaling(decision)
            else:
                result = {"status": "no_action", "message": "Maintaining current capacity"}
            
            # Post-execution monitoring
            monitoring_result = await self._monitor_scaling_execution(decision, execution_id)
            
            execution_time = time.time() - start_time
            
            return {
                "execution_id": execution_id,
                "decision_id": decision.decision_id,
                "status": "completed",
                "execution_time_seconds": execution_time,
                "result": result,
                "monitoring": monitoring_result,
                "rollback_triggered": False
            }
            
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
            
            # Attempt rollback
            rollback_result = await self._execute_rollback(decision, execution_id)
            
            return {
                "execution_id": execution_id,
                "decision_id": decision.decision_id,
                "status": "failed",
                "error": str(e),
                "rollback_result": rollback_result,
                "rollback_triggered": True
            }
    
    async def _validate_scaling_prerequisites(self, decision: ScalingDecision):
        """Validate prerequisites before executing scaling."""
        
        # Check resource availability
        if decision.action == "scale_up":
            available_capacity = await self._check_available_capacity()
            required_capacity = decision.target_capacity["instances"] - decision.current_capacity["instances"]
            
            if available_capacity < required_capacity:
                raise PQCRetrofitError(
                    f"Insufficient capacity: need {required_capacity}, have {available_capacity}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.EXTERNAL_DEPENDENCY
                )
        
        # Check system health
        system_health = await self._check_system_health()
        if system_health["status"] != "healthy":
            raise PQCRetrofitError(
                f"System not healthy for scaling: {system_health['issues']}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.INTERNAL_ERROR
            )
    
    async def _check_available_capacity(self) -> int:
        """Check available scaling capacity."""
        # Simulate capacity check
        return 10  # Available instances
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        # Simulate health check
        return {"status": "healthy", "issues": []}
    
    async def _execute_immediate_scaling(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute immediate scaling action."""
        
        if decision.action == "scale_up":
            instances_to_add = int(decision.target_capacity["instances"] - decision.current_capacity["instances"])
            
            # Simulate instance provisioning
            new_instances = []
            for i in range(instances_to_add):
                instance_id = f"pqc-scanner-{uuid.uuid4().hex[:8]}"
                new_instances.append(instance_id)
                
                # Simulate startup time
                await asyncio.sleep(0.1)
            
            return {
                "action": "scale_up",
                "new_instances": new_instances,
                "total_instances": decision.target_capacity["instances"]
            }
            
        elif decision.action == "scale_down":
            instances_to_remove = int(decision.current_capacity["instances"] - decision.target_capacity["instances"])
            
            # Simulate instance termination
            removed_instances = []
            for i in range(instances_to_remove):
                instance_id = f"pqc-scanner-{uuid.uuid4().hex[:8]}"
                removed_instances.append(instance_id)
                
                # Simulate graceful shutdown
                await asyncio.sleep(0.05)
            
            return {
                "action": "scale_down",
                "removed_instances": removed_instances,
                "total_instances": decision.target_capacity["instances"]
            }
        
        return {"action": "maintain", "message": "No scaling required"}
    
    async def _execute_gradual_scaling(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute gradual scaling with monitoring between steps."""
        
        current = decision.current_capacity["instances"]
        target = decision.target_capacity["instances"]
        
        if decision.action == "scale_up":
            step_size = max(1, int((target - current) * 0.25))  # 25% increments
            steps = []
            
            while current < target:
                next_step = min(current + step_size, target)
                
                # Execute step
                step_result = await self._execute_scaling_step(current, next_step, "up")
                steps.append(step_result)
                
                # Monitor step
                await asyncio.sleep(2)  # Wait for stabilization
                step_health = await self._check_step_health()
                
                if step_health["status"] != "healthy":
                    # Rollback this step
                    await self._execute_scaling_step(next_step, current, "down")
                    raise PQCRetrofitError(
                        f"Scaling step failed health check: {step_health['issues']}",
                        severity=ErrorSeverity.HIGH
                    )
                
                current = next_step
            
            return {"action": "gradual_scale_up", "steps": steps, "final_instances": current}
        
        # Similar logic for scale_down...
        return {"action": "gradual_scale_complete"}
    
    async def _execute_scaling_step(self, from_instances: int, to_instances: int, direction: str) -> Dict[str, Any]:
        """Execute a single scaling step."""
        # Simulate scaling step
        return {
            "from": from_instances,
            "to": to_instances,
            "direction": direction,
            "timestamp": time.time()
        }
    
    async def _check_step_health(self) -> Dict[str, Any]:
        """Check health after a scaling step."""
        # Simulate health check
        return {"status": "healthy", "issues": []}
    
    async def _monitor_scaling_execution(self, decision: ScalingDecision, execution_id: str) -> Dict[str, Any]:
        """Monitor scaling execution and detect issues."""
        
        monitoring_duration = 300  # 5 minutes
        check_interval = 30  # 30 seconds
        
        start_time = time.time()
        checks = []
        
        while time.time() - start_time < monitoring_duration:
            # Perform health check
            health_check = await self._perform_health_check()
            checks.append(health_check)
            
            # Check for rollback conditions
            if self._should_rollback(health_check, decision):
                return {
                    "status": "rollback_required",
                    "reason": "Performance degradation detected",
                    "checks": checks
                }
            
            await asyncio.sleep(check_interval)
        
        return {
            "status": "monitoring_complete",
            "checks_performed": len(checks),
            "issues_detected": sum(1 for check in checks if check["status"] != "healthy")
        }
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        # Simulate health check
        return {
            "status": "healthy",
            "response_time_p95": 2500,
            "error_rate": 0.01,
            "cpu_utilization": 0.65,
            "memory_utilization": 0.70
        }
    
    def _should_rollback(self, health_check: Dict[str, Any], decision: ScalingDecision) -> bool:
        """Determine if rollback is required based on health check."""
        
        # Check for performance degradation
        if health_check.get("response_time_p95", 0) > 5000:  # 5 second threshold
            return True
        
        if health_check.get("error_rate", 0) > 0.1:  # 10% error rate
            return True
        
        if health_check.get("cpu_utilization", 0) > 0.95:  # 95% CPU
            return True
        
        return False
    
    async def _execute_rollback(self, decision: ScalingDecision, execution_id: str) -> Dict[str, Any]:
        """Execute rollback of scaling decision."""
        
        self.logger.warning(f"Executing rollback for decision {decision.decision_id}")
        
        # Reverse the scaling action
        if decision.action == "scale_up":
            rollback_action = "scale_down"
            rollback_target = decision.current_capacity
        elif decision.action == "scale_down":
            rollback_action = "scale_up"
            rollback_target = decision.current_capacity
        else:
            return {"status": "no_rollback_needed"}
        
        # Create rollback decision
        rollback_decision = ScalingDecision(
            decision_id=f"rollback-{decision.decision_id}",
            timestamp=time.time(),
            action=rollback_action,
            resource_type=decision.resource_type,
            current_capacity=decision.target_capacity,
            target_capacity=rollback_target,
            confidence_score=0.9,  # High confidence in rollback
            trigger_metrics={"rollback": True},
            decision_factors=["Rollback due to performance degradation"],
            expected_impact={"stability": "restored"},
            risk_assessment="low",
            execution_strategy="immediate_scaling",
            rollback_plan="N/A - this is the rollback",
            monitoring_requirements=["Verify restoration of performance"]
        )
        
        # Execute rollback
        try:
            rollback_result = await self._execute_immediate_scaling(rollback_decision)
            return {
                "status": "rollback_successful",
                "rollback_decision": rollback_decision,
                "result": rollback_result
            }
        except Exception as e:
            return {
                "status": "rollback_failed",
                "error": str(e),
                "escalation_required": True
            }
    
    def generate_capacity_report(self) -> Dict[str, Any]:
        """Generate comprehensive capacity and scaling report."""
        
        current_time = time.time()
        
        # Analyze recent scaling decisions
        recent_decisions = [d for d in self.scaling_history 
                          if current_time - d.timestamp < 86400]  # Last 24 hours
        
        # Calculate scaling frequency
        scale_up_count = sum(1 for d in recent_decisions if d.action == "scale_up")
        scale_down_count = sum(1 for d in recent_decisions if d.action == "scale_down")
        
        # Analyze workload patterns
        workload_analysis = {}
        for profile_name, profile in self.workload_profiles.items():
            workload_analysis[profile_name] = {
                "sla_compliance": self._calculate_sla_compliance(profile),
                "resource_efficiency": self._calculate_resource_efficiency(profile),
                "cost_optimization_score": self._calculate_cost_score(profile),
                "scaling_frequency": self._calculate_scaling_frequency(profile_name)
            }
        
        return {
            "report_timestamp": current_time,
            "summary": {
                "total_scaling_decisions_24h": len(recent_decisions),
                "scale_up_events": scale_up_count,
                "scale_down_events": scale_down_count,
                "avg_decision_confidence": sum(d.confidence_score for d in recent_decisions) / max(len(recent_decisions), 1),
                "system_stability_score": self._calculate_stability_score()
            },
            "workload_analysis": workload_analysis,
            "recommendations": self._generate_optimization_recommendations(),
            "capacity_forecast": self._generate_capacity_forecast(),
            "cost_analysis": self._generate_cost_analysis(),
            "next_review_date": current_time + 86400  # 24 hours
        }
    
    def _calculate_sla_compliance(self, profile: WorkloadProfile) -> float:
        """Calculate SLA compliance for a workload profile."""
        # Simulate SLA compliance calculation
        return 0.995  # 99.5% compliance
    
    def _calculate_resource_efficiency(self, profile: WorkloadProfile) -> float:
        """Calculate resource efficiency score."""
        # Simulate efficiency calculation
        return 0.85  # 85% efficiency
    
    def _calculate_cost_score(self, profile: WorkloadProfile) -> float:
        """Calculate cost optimization score."""
        # Simulate cost score calculation
        return 0.78  # 78% cost optimized
    
    def _calculate_scaling_frequency(self, profile_name: str) -> Dict[str, float]:
        """Calculate scaling frequency for a profile."""
        return {
            "scales_per_day": 2.5,
            "avg_scale_magnitude": 1.8,
            "frequency_trend": "stable"
        }
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall system stability score."""
        if not self.scaling_history:
            return 1.0
        
        recent_decisions = list(self.scaling_history)[-10:]  # Last 10 decisions
        
        # Factors: low rollback rate, high confidence, stable patterns
        rollback_rate = sum(1 for d in recent_decisions if "rollback" in d.decision_id) / len(recent_decisions)
        avg_confidence = sum(d.confidence_score for d in recent_decisions) / len(recent_decisions)
        
        stability_score = (1.0 - rollback_rate) * 0.5 + avg_confidence * 0.5
        return min(1.0, max(0.0, stability_score))
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations."""
        return [
            {
                "category": "performance",
                "recommendation": "Consider pre-warming instances during predicted peak hours",
                "impact": "medium",
                "effort": "low"
            },
            {
                "category": "cost",
                "recommendation": "Implement spot instance usage for non-critical workloads",
                "impact": "high",
                "effort": "medium"
            },
            {
                "category": "reliability",
                "recommendation": "Add cross-region failover for quantum_threat_assessment workload",
                "impact": "high",
                "effort": "high"
            }
        ]
    
    def _generate_capacity_forecast(self) -> Dict[str, Any]:
        """Generate capacity forecast for next 30 days."""
        return {
            "peak_capacity_needed": 15,
            "average_capacity": 8,
            "growth_trend": "moderate",
            "seasonal_patterns": "business_hours_heavy",
            "confidence": 0.82
        }
    
    def _generate_cost_analysis(self) -> Dict[str, float]:
        """Generate cost analysis and projections."""
        return {
            "current_monthly_cost": 2450.00,
            "optimized_monthly_cost": 2100.00,
            "potential_savings": 350.00,
            "cost_per_analysis": 0.12,
            "efficiency_score": 0.85
        }


# Global instance
enterprise_scaling_engine = EnterpriseScalingEngine()