"""
Distributed computing and scaling infrastructure for Generation 3.

This module provides:
- Distributed task execution across multiple nodes
- Load balancing and auto-scaling
- Inter-node communication and coordination
- Distributed caching and state management
- Cluster health monitoring
"""

import asyncio
import json
import time
import threading
import hashlib
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import socket
import uuid

from .error_handling import handle_errors, PQCRetrofitError, ErrorSeverity
from .monitoring import metrics_collector, track_performance
from .resilient_processing import ProcessingConfig, ProcessingResult


class NodeState(Enum):
    """Cluster node states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task execution priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ClusterNode:
    """Represents a compute node in the cluster."""
    node_id: str
    host: str
    port: int
    state: NodeState
    capabilities: Dict[str, Any]
    current_load: float
    max_capacity: int
    last_heartbeat: float
    metadata: Dict[str, Any]


@dataclass
class DistributedTask:
    """Represents a task for distributed execution."""
    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    requirements: Dict[str, Any]
    created_at: float
    timeout: float
    retry_count: int
    max_retries: int
    assigned_node: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ClusterConfig:
    """Configuration for distributed cluster."""
    cluster_name: str
    discovery_port: int = 8765
    heartbeat_interval: float = 30.0
    node_timeout: float = 90.0
    load_balance_strategy: str = "least_loaded"  # least_loaded, round_robin, random
    auto_scale_enabled: bool = True
    min_nodes: int = 1
    max_nodes: int = 10
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    enable_distributed_cache: bool = True


class LoadBalancer:
    """Intelligent load balancing for task distribution."""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.round_robin_index = 0
        self.logger = logging.getLogger("load_balancer")
    
    def select_node(self, available_nodes: List[ClusterNode], 
                   task: DistributedTask) -> Optional[ClusterNode]:
        """Select best node for task execution."""
        if not available_nodes:
            return None
        
        # Filter nodes by capabilities
        capable_nodes = self._filter_by_capabilities(available_nodes, task.requirements)
        if not capable_nodes:
            self.logger.warning(f"No nodes found with required capabilities: {task.requirements}")
            return None
        
        # Apply load balancing strategy
        if self.strategy == "least_loaded":
            return self._select_least_loaded(capable_nodes)
        elif self.strategy == "round_robin":
            return self._select_round_robin(capable_nodes)
        elif self.strategy == "random":
            import random
            return random.choice(capable_nodes)
        else:
            return capable_nodes[0]
    
    def _filter_by_capabilities(self, nodes: List[ClusterNode], 
                               requirements: Dict[str, Any]) -> List[ClusterNode]:
        """Filter nodes by capability requirements."""
        filtered = []
        
        for node in nodes:
            if node.state != NodeState.HEALTHY:
                continue
            
            # Check if node meets requirements
            meets_requirements = True
            capabilities = node.capabilities
            
            for req_key, req_value in requirements.items():
                if req_key == "min_memory" and capabilities.get("memory_mb", 0) < req_value:
                    meets_requirements = False
                    break
                elif req_key == "min_cpu_cores" and capabilities.get("cpu_cores", 0) < req_value:
                    meets_requirements = False
                    break
                elif req_key == "required_features":
                    node_features = set(capabilities.get("features", []))
                    required_features = set(req_value)
                    if not required_features.issubset(node_features):
                        meets_requirements = False
                        break
            
            if meets_requirements:
                filtered.append(node)
        
        return filtered
    
    def _select_least_loaded(self, nodes: List[ClusterNode]) -> ClusterNode:
        """Select node with lowest current load."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _select_round_robin(self, nodes: List[ClusterNode]) -> ClusterNode:
        """Select node using round-robin strategy."""
        if not nodes:
            return None
        
        node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return node


class DistributedCache:
    """Distributed caching system for cluster-wide state."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.replication_factor = 2
        self.logger = logging.getLogger("distributed_cache")
        self._lock = threading.Lock()
    
    @handle_errors("cache_operation", retry_count=1)
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        with self._lock:
            # Check local cache first
            if key in self.local_cache:
                metadata = self.cache_metadata.get(key, {})
                
                # Check expiration
                if self._is_expired(metadata):
                    self._evict_key(key)
                    return None
                
                # Update access time
                metadata['last_accessed'] = time.time()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                
                return self.local_cache[key]
        
        return None
    
    @handle_errors("cache_operation", retry_count=1)
    def put(self, key: str, value: Any, ttl: float = 3600) -> bool:
        """Put value into distributed cache."""
        with self._lock:
            # Store locally
            self.local_cache[key] = value
            self.cache_metadata[key] = {
                'created_at': time.time(),
                'last_accessed': time.time(),
                'ttl': ttl,
                'access_count': 1,
                'size_bytes': len(str(value)),
                'node_id': self.node_id
            }
            
            # TODO: Implement replication to other nodes
            return True
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry across cluster."""
        with self._lock:
            if key in self.local_cache:
                self._evict_key(key)
                # TODO: Send invalidation to other nodes
                return True
        return False
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if not metadata:
            return True
        
        created_at = metadata.get('created_at', 0)
        ttl = metadata.get('ttl', 3600)
        
        return time.time() > (created_at + ttl)
    
    def _evict_key(self, key: str):
        """Evict key from local cache."""
        self.local_cache.pop(key, None)
        self.cache_metadata.pop(key, None)
    
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        with self._lock:
            expired_keys = []
            
            for key, metadata in self.cache_metadata.items():
                if self._is_expired(metadata):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._evict_key(key)
            
            if expired_keys:
                self.logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(meta.get('size_bytes', 0) for meta in self.cache_metadata.values())
            total_accesses = sum(meta.get('access_count', 0) for meta in self.cache_metadata.values())
            
            return {
                'entries': len(self.local_cache),
                'total_size_bytes': total_size,
                'total_accesses': total_accesses,
                'hit_rate': 0.0,  # TODO: Implement hit rate tracking
                'node_id': self.node_id
            }


class AutoScaler:
    """Automatic cluster scaling based on load."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.scaling_events: List[Dict[str, Any]] = []
        self.last_scale_action = 0
        self.scale_cooldown = 300  # 5 minutes
        self.logger = logging.getLogger("auto_scaler")
    
    def should_scale_up(self, cluster_state: Dict[str, Any]) -> bool:
        """Determine if cluster should scale up."""
        if not self.config.auto_scale_enabled:
            return False
        
        if time.time() - self.last_scale_action < self.scale_cooldown:
            return False
        
        nodes = cluster_state.get('nodes', [])
        healthy_nodes = [n for n in nodes if n['state'] == NodeState.HEALTHY.value]
        
        if len(healthy_nodes) >= self.config.max_nodes:
            return False
        
        # Check average load
        if healthy_nodes:
            avg_load = sum(n['current_load'] for n in healthy_nodes) / len(healthy_nodes)
            if avg_load > self.config.scale_up_threshold:
                return True
        
        # Check queue length
        pending_tasks = cluster_state.get('pending_tasks', 0)
        if pending_tasks > len(healthy_nodes) * 5:  # More than 5 tasks per node
            return True
        
        return False
    
    def should_scale_down(self, cluster_state: Dict[str, Any]) -> bool:
        """Determine if cluster should scale down."""
        if not self.config.auto_scale_enabled:
            return False
        
        if time.time() - self.last_scale_action < self.scale_cooldown:
            return False
        
        nodes = cluster_state.get('nodes', [])
        healthy_nodes = [n for n in nodes if n['state'] == NodeState.HEALTHY.value]
        
        if len(healthy_nodes) <= self.config.min_nodes:
            return False
        
        # Check average load
        if healthy_nodes:
            avg_load = sum(n['current_load'] for n in healthy_nodes) / len(healthy_nodes)
            if avg_load < self.config.scale_down_threshold:
                # Ensure we have sufficient capacity after scaling down
                projected_load = avg_load * len(healthy_nodes) / (len(healthy_nodes) - 1)
                if projected_load < 0.7:  # Keep some headroom
                    return True
        
        return False
    
    def record_scaling_event(self, action: str, details: Dict[str, Any]):
        """Record scaling event for monitoring."""
        event = {
            'timestamp': time.time(),
            'action': action,
            'details': details
        }
        self.scaling_events.append(event)
        self.last_scale_action = time.time()
        
        # Keep only recent events
        if len(self.scaling_events) > 100:
            self.scaling_events = self.scaling_events[-50:]
        
        self.logger.info(f"Scaling event: {action} - {details}")


class DistributedCluster:
    """Main distributed computing cluster manager."""
    
    def __init__(self, config: ClusterConfig, node_capabilities: Dict[str, Any] = None):
        self.config = config
        self.node_id = str(uuid.uuid4())
        self.nodes: Dict[str, ClusterNode] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.load_balancer = LoadBalancer(config.load_balance_strategy)
        self.auto_scaler = AutoScaler(config)
        self.distributed_cache = DistributedCache(self.node_id) if config.enable_distributed_cache else None
        
        # Node capabilities
        self.capabilities = node_capabilities or self._detect_capabilities()
        
        # Execution infrastructure
        self.executor = ThreadPoolExecutor(max_workers=self.capabilities.get('cpu_cores', 4))
        self.is_running = False
        self.logger = logging.getLogger("distributed_cluster")
        
        # Metrics
        self._start_time = time.time()
        self._tasks_executed = 0
        self._tasks_failed = 0
        
        # Initialize current node
        self._register_self()
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Auto-detect node capabilities."""
        import psutil
        
        try:
            capabilities = {
                'cpu_cores': psutil.cpu_count(),
                'memory_mb': psutil.virtual_memory().total // (1024 * 1024),
                'disk_gb': psutil.disk_usage('/').total // (1024 * 1024 * 1024),
                'features': ['firmware_analysis', 'patch_generation', 'crypto_validation'],
                'architecture_support': ['cortex-m4', 'esp32', 'riscv32', 'avr'],
                'max_concurrent_tasks': psutil.cpu_count() * 2
            }
        except Exception:
            # Fallback capabilities
            capabilities = {
                'cpu_cores': 4,
                'memory_mb': 8192,
                'disk_gb': 100,
                'features': ['firmware_analysis', 'patch_generation'],
                'architecture_support': ['cortex-m4', 'esp32'],
                'max_concurrent_tasks': 8
            }
        
        return capabilities
    
    def _register_self(self):
        """Register this node in the cluster."""
        self.nodes[self.node_id] = ClusterNode(
            node_id=self.node_id,
            host=socket.gethostname(),
            port=self.config.discovery_port,
            state=NodeState.HEALTHY,
            capabilities=self.capabilities,
            current_load=0.0,
            max_capacity=self.capabilities.get('max_concurrent_tasks', 8),
            last_heartbeat=time.time(),
            metadata={'startup_time': time.time()}
        )
    
    @track_performance("cluster_start")
    def start(self):
        """Start the distributed cluster."""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info(f"Starting distributed cluster: {self.config.cluster_name}")
        
        # Start background threads
        self._start_heartbeat_thread()
        self._start_task_processor_thread()
        self._start_health_monitor_thread()
        
        if self.distributed_cache:
            self._start_cache_cleanup_thread()
        
        self.logger.info(f"Cluster node {self.node_id} started successfully")
    
    def stop(self):
        """Stop the distributed cluster."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping distributed cluster")
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Cluster stopped successfully")
    
    def _start_heartbeat_thread(self):
        """Start heartbeat monitoring thread."""
        def heartbeat_loop():
            while self.is_running:
                try:
                    self._send_heartbeat()
                    self._check_node_health()
                    time.sleep(self.config.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
        
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
    
    def _start_task_processor_thread(self):
        """Start task processing thread."""
        def task_processor():
            while self.is_running:
                try:
                    self._process_pending_tasks()
                    time.sleep(1.0)
                except Exception as e:
                    self.logger.error(f"Task processor error: {e}")
        
        thread = threading.Thread(target=task_processor, daemon=True)
        thread.start()
    
    def _start_health_monitor_thread(self):
        """Start health monitoring and auto-scaling thread."""
        def health_monitor():
            while self.is_running:
                try:
                    cluster_state = self.get_cluster_state()
                    
                    # Auto-scaling decisions
                    if self.auto_scaler.should_scale_up(cluster_state):
                        self._scale_up()
                    elif self.auto_scaler.should_scale_down(cluster_state):
                        self._scale_down()
                    
                    time.sleep(60.0)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Health monitor error: {e}")
        
        thread = threading.Thread(target=health_monitor, daemon=True)
        thread.start()
    
    def _start_cache_cleanup_thread(self):
        """Start cache cleanup thread."""
        def cache_cleanup():
            while self.is_running:
                try:
                    if self.distributed_cache:
                        self.distributed_cache.cleanup_expired()
                    time.sleep(300.0)  # Cleanup every 5 minutes
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        thread = threading.Thread(target=cache_cleanup, daemon=True)
        thread.start()
    
    def _send_heartbeat(self):
        """Send heartbeat to update node status."""
        if self.node_id in self.nodes:
            node = self.nodes[self.node_id]
            node.last_heartbeat = time.time()
            node.current_load = self._calculate_current_load()
    
    def _check_node_health(self):
        """Check health of all nodes in cluster."""
        current_time = time.time()
        timeout_threshold = current_time - self.config.node_timeout
        
        for node_id, node in list(self.nodes.items()):
            if node.last_heartbeat < timeout_threshold:
                if node.state != NodeState.OFFLINE:
                    self.logger.warning(f"Node {node_id} marked as offline")
                    node.state = NodeState.OFFLINE
    
    def _calculate_current_load(self) -> float:
        """Calculate current load of this node."""
        # Simple load calculation based on task queue and active tasks
        pending_tasks = self.task_queue.qsize()
        max_capacity = self.capabilities.get('max_concurrent_tasks', 8)
        
        return min(1.0, pending_tasks / max_capacity)
    
    def _process_pending_tasks(self):
        """Process pending tasks from queue."""
        try:
            # Get next task (blocks for up to 1 second)
            priority, task_id, task = self.task_queue.get(timeout=1.0)
            
            # Execute task
            self._execute_task(task)
            
        except queue.Empty:
            pass  # No tasks available
        except Exception as e:
            self.logger.error(f"Task processing error: {e}")
    
    @handle_errors("task_execution", retry_count=1)
    def _execute_task(self, task: DistributedTask):
        """Execute a distributed task."""
        self.logger.debug(f"Executing task {task.task_id} of type {task.task_type}")
        
        task.started_at = time.time()
        task.assigned_node = self.node_id
        
        try:
            # Execute based on task type
            if task.task_type == "firmware_analysis":
                result = self._execute_firmware_analysis(task)
            elif task.task_type == "patch_generation":
                result = self._execute_patch_generation(task)
            elif task.task_type == "crypto_validation":
                result = self._execute_crypto_validation(task)
            else:
                raise PQCRetrofitError(f"Unknown task type: {task.task_type}")
            
            # Task completed successfully
            task.completed_at = time.time()
            task.result = result
            self._tasks_executed += 1
            
            self.logger.debug(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Task failed
            task.completed_at = time.time()
            task.error = str(e)
            self._tasks_failed += 1
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.started_at = None
                task.assigned_node = None
                self.submit_task(task)
                return
        
        # Store completed task
        self.completed_tasks[task.task_id] = task
    
    def _execute_firmware_analysis(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute firmware analysis task."""
        from .resilient_processing import FirmwareAnalysisStage, ProcessingConfig
        
        config = ProcessingConfig()
        stage = FirmwareAnalysisStage(config)
        
        result = stage.process_with_resilience(task.payload)
        
        if not result.success:
            raise PQCRetrofitError(f"Firmware analysis failed: {result.error}")
        
        return result.data
    
    def _execute_patch_generation(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute patch generation task."""
        from .resilient_processing import PatchGenerationStage, ProcessingConfig
        
        config = ProcessingConfig()
        stage = PatchGenerationStage(config)
        
        result = stage.process_with_resilience(task.payload)
        
        if not result.success:
            raise PQCRetrofitError(f"Patch generation failed: {result.error}")
        
        return result.data
    
    def _execute_crypto_validation(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute crypto validation task."""
        from .validation import validate_pqc_implementation, ValidationLevel
        
        algorithm = task.payload.get('algorithm')
        target_arch = task.payload.get('target_arch', 'cortex-m4')
        level = ValidationLevel(task.payload.get('level', 'standard'))
        
        report = validate_pqc_implementation(algorithm, target_arch, level)
        
        return {
            'validation_report': asdict(report),
            'production_ready': report.is_ready_for_production
        }
    
    def _scale_up(self):
        """Scale up the cluster by adding nodes."""
        # In a real implementation, this would trigger cloud instances
        # For now, just log the scaling decision
        self.auto_scaler.record_scaling_event("scale_up", {
            "reason": "High load detected",
            "current_nodes": len([n for n in self.nodes.values() if n.state == NodeState.HEALTHY])
        })
    
    def _scale_down(self):
        """Scale down the cluster by removing nodes."""
        # In a real implementation, this would terminate cloud instances
        # For now, just log the scaling decision
        self.auto_scaler.record_scaling_event("scale_down", {
            "reason": "Low load detected",
            "current_nodes": len([n for n in self.nodes.values() if n.state == NodeState.HEALTHY])
        })
    
    @track_performance("task_submission")
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        if not task.task_id:
            task.task_id = str(uuid.uuid4())
        
        # Add to priority queue
        priority = -task.priority.value  # Negative for max priority queue
        self.task_queue.put((priority, task.task_id, task))
        
        self.logger.debug(f"Submitted task {task.task_id} with priority {task.priority.value}")
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': 'completed' if task.result else 'failed',
                'result': task.result,
                'error': task.error,
                'execution_time': (task.completed_at or 0) - (task.started_at or 0),
                'assigned_node': task.assigned_node
            }
        
        return None
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get current cluster state."""
        healthy_nodes = [n for n in self.nodes.values() if n.state == NodeState.HEALTHY]
        
        return {
            'cluster_name': self.config.cluster_name,
            'total_nodes': len(self.nodes),
            'healthy_nodes': len(healthy_nodes),
            'pending_tasks': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'tasks_executed': self._tasks_executed,
            'tasks_failed': self._tasks_failed,
            'uptime_seconds': time.time() - self._start_time,
            'nodes': [asdict(node) for node in self.nodes.values()],
            'cache_stats': self.distributed_cache.get_stats() if self.distributed_cache else None
        }


def create_distributed_task(task_type: str, payload: Dict[str, Any],
                           priority: TaskPriority = TaskPriority.NORMAL,
                           timeout: float = 300.0,
                           max_retries: int = 2) -> DistributedTask:
    """Create a distributed task."""
    return DistributedTask(
        task_id=str(uuid.uuid4()),
        task_type=task_type,
        priority=priority,
        payload=payload,
        requirements={},
        created_at=time.time(),
        timeout=timeout,
        retry_count=0,
        max_retries=max_retries
    )


# Global distributed cluster instance
cluster_config = ClusterConfig(
    cluster_name="pqc_analysis_cluster",
    auto_scale_enabled=True,
    min_nodes=1,
    max_nodes=8,
    enable_distributed_cache=True
)

distributed_cluster: Optional[DistributedCluster] = None


def get_cluster() -> DistributedCluster:
    """Get or create the global distributed cluster."""
    global distributed_cluster
    
    if distributed_cluster is None:
        distributed_cluster = DistributedCluster(cluster_config)
        distributed_cluster.start()
    
    return distributed_cluster


def shutdown_cluster():
    """Shutdown the global distributed cluster."""
    global distributed_cluster
    
    if distributed_cluster:
        distributed_cluster.stop()
        distributed_cluster = None