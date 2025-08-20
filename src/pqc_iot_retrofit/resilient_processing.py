"""
Resilient processing engine for Generation 2 robustness.

This module provides:
- Fault-tolerant processing pipelines
- Graceful degradation mechanisms
- Recovery and retry strategies
- Circuit breaker patterns
- Health monitoring integration
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import queue

from .error_handling import (
    PQCRetrofitError, ValidationError, ErrorSeverity, ErrorCategory,
    CircuitBreaker, handle_errors, global_error_handler
)
from .monitoring import metrics_collector, track_performance


T = TypeVar('T')
R = TypeVar('R')


class ProcessingState(Enum):
    """Processing pipeline states."""
    IDLE = "idle"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class FailureMode(Enum):
    """Failure handling modes."""
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    BEST_EFFORT = "best_effort"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


@dataclass
class ProcessingConfig:
    """Configuration for resilient processing."""
    max_workers: int = 4
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    backoff_multiplier: float = 2.0
    failure_mode: FailureMode = FailureMode.GRACEFUL_DEGRADATION
    health_check_interval: float = 60.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_metrics: bool = True


@dataclass
class ProcessingResult(Generic[T]):
    """Result of resilient processing operation."""
    success: bool
    data: Optional[T] = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    retry_count: int = 0
    circuit_breaker_triggered: bool = False
    degraded_mode: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingStage(ABC, Generic[T, R]):
    """Abstract base class for processing stages."""
    
    def __init__(self, name: str, config: ProcessingConfig):
        self.name = name
        self.config = config
        self.state = ProcessingState.IDLE
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        self.logger = logging.getLogger(f"processing.{name}")
        self._metrics_enabled = config.enable_metrics
    
    @abstractmethod
    def process(self, input_data: T) -> R:
        """Process input data and return result."""
        pass
    
    def process_with_resilience(self, input_data: T) -> ProcessingResult[R]:
        """Process with full resilience features."""
        start_time = time.time()
        retry_count = 0
        
        if self._metrics_enabled:
            operation_id = metrics_collector.record_operation_start(f"stage.{self.name}")
        
        try:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                self.logger.warning(f"Circuit breaker open for stage {self.name}")
                return ProcessingResult(
                    success=False,
                    error=PQCRetrofitError(f"Circuit breaker open for {self.name}"),
                    processing_time=time.time() - start_time,
                    circuit_breaker_triggered=True
                )
            
            # Attempt processing with retries
            while retry_count <= self.config.retry_attempts:
                try:
                    self.state = ProcessingState.RUNNING
                    
                    # Execute processing
                    result = self.process(input_data)
                    
                    # Success path
                    self.circuit_breaker.record_success()
                    self.state = ProcessingState.IDLE
                    
                    processing_time = time.time() - start_time
                    
                    if self._metrics_enabled:
                        metrics_collector.record_operation_end(f"stage.{self.name}", operation_id, True)
                    
                    return ProcessingResult(
                        success=True,
                        data=result,
                        processing_time=processing_time,
                        retry_count=retry_count
                    )
                
                except Exception as e:
                    retry_count += 1
                    self.circuit_breaker.record_failure()
                    
                    if retry_count <= self.config.retry_attempts:
                        # Apply backoff
                        backoff_time = self.config.backoff_multiplier ** (retry_count - 1)
                        self.logger.warning(
                            f"Stage {self.name} failed (attempt {retry_count}), "
                            f"retrying in {backoff_time:.1f}s: {e}"
                        )
                        time.sleep(backoff_time)
                        continue
                    else:
                        # Final failure
                        self.state = ProcessingState.FAILED
                        processing_time = time.time() - start_time
                        
                        if self._metrics_enabled:
                            metrics_collector.record_operation_end(f"stage.{self.name}", operation_id, False)
                        
                        return ProcessingResult(
                            success=False,
                            error=e,
                            processing_time=processing_time,
                            retry_count=retry_count
                        )
        
        except Exception as e:
            # Unexpected error
            self.state = ProcessingState.FAILED
            processing_time = time.time() - start_time
            
            if self._metrics_enabled:
                metrics_collector.record_operation_end(f"stage.{self.name}", operation_id, False)
            
            return ProcessingResult(
                success=False,
                error=e,
                processing_time=processing_time,
                retry_count=retry_count
            )
    
    def can_degrade_gracefully(self) -> bool:
        """Check if this stage supports graceful degradation."""
        return False
    
    def process_degraded(self, input_data: T) -> Optional[R]:
        """Process in degraded mode (fallback implementation)."""
        return None


class FirmwareAnalysisStage(ProcessingStage[Dict[str, Any], Dict[str, Any]]):
    """Resilient firmware analysis stage."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__("firmware_analysis", config)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process firmware analysis."""
        firmware_path = input_data.get('firmware_path')
        architecture = input_data.get('architecture')
        base_address = input_data.get('base_address', 0)
        
        if not firmware_path or not architecture:
            raise ValidationError("Missing required input data for firmware analysis")
        
        try:
            # Import scanner here to avoid circular imports
            from .scanner import FirmwareScanner
            
            # Create scanner
            scanner = FirmwareScanner(architecture, input_data.get('memory_constraints'))
            
            # Perform scan
            vulnerabilities = scanner.scan_firmware(firmware_path, base_address)
            report = scanner.generate_report()
            
            return {
                'vulnerabilities': vulnerabilities,
                'report': report,
                'scan_metadata': {
                    'firmware_path': firmware_path,
                    'architecture': architecture,
                    'base_address': base_address,
                    'vulnerabilities_count': len(vulnerabilities)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Firmware analysis failed: {e}")
            raise
    
    def can_degrade_gracefully(self) -> bool:
        """Firmware analysis supports basic pattern matching fallback."""
        return True
    
    def process_degraded(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Degraded mode: basic pattern matching only."""
        try:
            firmware_path = input_data.get('firmware_path')
            if not firmware_path:
                return None
            
            # Simple pattern-based analysis
            from pathlib import Path
            firmware_data = Path(firmware_path).read_bytes()
            
            # Basic vulnerability indicators
            vulnerabilities = []
            
            # Simple pattern matching for common crypto strings
            crypto_patterns = [b'RSA', b'ECDSA', b'ECDH', b'DH']
            for pattern in crypto_patterns:
                if pattern in firmware_data:
                    vulnerabilities.append({
                        'algorithm': pattern.decode(),
                        'address': firmware_data.find(pattern),
                        'confidence': 'low',
                        'source': 'degraded_analysis'
                    })
            
            return {
                'vulnerabilities': vulnerabilities,
                'report': {
                    'scan_summary': {
                        'total_vulnerabilities': len(vulnerabilities),
                        'analysis_mode': 'degraded'
                    }
                },
                'degraded_mode': True
            }
            
        except Exception as e:
            self.logger.error(f"Degraded firmware analysis failed: {e}")
            return None


class PatchGenerationStage(ProcessingStage[Dict[str, Any], Dict[str, Any]]):
    """Resilient patch generation stage."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__("patch_generation", config)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process patch generation."""
        vulnerabilities = input_data.get('vulnerabilities', [])
        device = input_data.get('device')
        optimization = input_data.get('optimization', 'balanced')
        
        if not vulnerabilities or not device:
            raise ValidationError("Missing required input data for patch generation")
        
        try:
            # Import patcher here to avoid circular imports
            from .patcher import PQCPatcher
            
            # Create patcher
            patcher = PQCPatcher(device, optimization)
            
            # Generate patches
            patches = []
            failed_patches = []
            
            for vuln in vulnerabilities:
                try:
                    # Determine appropriate patch type
                    if 'RSA' in str(vuln.get('algorithm', '')):
                        patch = patcher.create_dilithium_patch(vuln, security_level=2)
                    else:
                        patch = patcher.create_kyber_patch(vuln, security_level=1)
                    
                    patches.append(patch)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate patch for vulnerability: {e}")
                    failed_patches.append({
                        'vulnerability': vuln,
                        'error': str(e)
                    })
            
            return {
                'patches': patches,
                'failed_patches': failed_patches,
                'patch_metadata': {
                    'device': device,
                    'optimization': optimization,
                    'success_count': len(patches),
                    'failure_count': len(failed_patches)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Patch generation failed: {e}")
            raise
    
    def can_degrade_gracefully(self) -> bool:
        """Patch generation supports simplified patching."""
        return True
    
    def process_degraded(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Degraded mode: generate simplified recommendations only."""
        try:
            vulnerabilities = input_data.get('vulnerabilities', [])
            
            recommendations = []
            for vuln in vulnerabilities:
                algorithm = vuln.get('algorithm', 'unknown')
                if 'RSA' in algorithm:
                    recommendations.append(f"Replace {algorithm} with Dilithium2")
                elif 'ECC' in algorithm or 'ECDSA' in algorithm:
                    recommendations.append(f"Replace {algorithm} with Dilithium2")
                elif 'ECDH' in algorithm:
                    recommendations.append(f"Replace {algorithm} with Kyber512")
                else:
                    recommendations.append(f"Assess {algorithm} for quantum resistance")
            
            return {
                'patches': [],
                'recommendations': recommendations,
                'patch_metadata': {
                    'mode': 'degraded',
                    'recommendation_count': len(recommendations)
                },
                'degraded_mode': True
            }
            
        except Exception as e:
            self.logger.error(f"Degraded patch generation failed: {e}")
            return None


class ResilientPipeline:
    """Resilient processing pipeline with fault tolerance."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.stages: List[ProcessingStage] = []
        self.state = ProcessingState.IDLE
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.logger = logging.getLogger("resilient_pipeline")
        self._shutdown = False
        
        # Health monitoring
        self._start_health_monitor()
    
    def add_stage(self, stage: ProcessingStage) -> 'ResilientPipeline':
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)
        return self
    
    def _start_health_monitor(self):
        """Start health monitoring thread."""
        def health_monitor():
            while not self._shutdown:
                try:
                    self._check_pipeline_health()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitor error: {e}")
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
    
    def _check_pipeline_health(self):
        """Check pipeline health and update state."""
        failed_stages = [stage for stage in self.stages if stage.state == ProcessingState.FAILED]
        degraded_stages = [stage for stage in self.stages if stage.state == ProcessingState.DEGRADED]
        
        if failed_stages:
            if len(failed_stages) >= len(self.stages) // 2:
                self.state = ProcessingState.FAILED
            else:
                self.state = ProcessingState.DEGRADED
        elif degraded_stages:
            self.state = ProcessingState.DEGRADED
        else:
            self.state = ProcessingState.IDLE
        
        # Log health status
        if self.state != ProcessingState.IDLE:
            self.logger.warning(f"Pipeline health: {self.state.value}")
    
    @track_performance("resilient_pipeline_execution")
    @handle_errors("pipeline_execution", retry_count=1)
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline with resilience features."""
        start_time = time.time()
        
        if not self.stages:
            raise ValidationError("No stages configured in pipeline")
        
        self.logger.info(f"Starting pipeline execution with {len(self.stages)} stages")
        
        try:
            # Execute stages sequentially with resilience
            current_data = input_data
            results = []
            
            for i, stage in enumerate(self.stages):
                self.logger.debug(f"Executing stage {i+1}/{len(self.stages)}: {stage.name}")
                
                # Execute stage with resilience
                stage_result = stage.process_with_resilience(current_data)
                results.append(stage_result)
                
                if stage_result.success:
                    # Continue with successful result
                    current_data.update(stage_result.data or {})
                    
                elif stage.can_degrade_gracefully() and self.config.failure_mode == FailureMode.GRACEFUL_DEGRADATION:
                    # Try degraded processing
                    self.logger.warning(f"Stage {stage.name} failed, attempting degraded processing")
                    
                    degraded_result = stage.process_degraded(current_data)
                    if degraded_result:
                        current_data.update(degraded_result)
                        stage_result.degraded_mode = True
                        self.logger.info(f"Stage {stage.name} completed in degraded mode")
                    else:
                        # Degraded processing also failed
                        if self.config.failure_mode == FailureMode.FAIL_FAST:
                            raise stage_result.error or PQCRetrofitError(f"Stage {stage.name} failed")
                        else:
                            self.logger.warning(f"Stage {stage.name} failed completely, continuing with best effort")
                
                elif self.config.failure_mode == FailureMode.FAIL_FAST:
                    # Fail fast mode
                    raise stage_result.error or PQCRetrofitError(f"Stage {stage.name} failed")
                
                else:
                    # Best effort mode - continue despite failure
                    self.logger.warning(f"Stage {stage.name} failed, continuing with best effort")
            
            # Compile final results
            execution_time = time.time() - start_time
            
            # Check if any critical stages failed
            critical_failures = [r for r in results if not r.success and not r.degraded_mode]
            success = len(critical_failures) == 0
            
            final_result = {
                'success': success,
                'execution_time': execution_time,
                'stages_executed': len(results),
                'stages_succeeded': sum(1 for r in results if r.success),
                'stages_degraded': sum(1 for r in results if r.degraded_mode),
                'stages_failed': sum(1 for r in results if not r.success and not r.degraded_mode),
                'stage_results': results,
                'pipeline_state': self.state.value,
                'data': current_data
            }
            
            self.logger.info(f"Pipeline execution completed: success={success}, time={execution_time:.2f}s")
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Pipeline execution failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'pipeline_state': ProcessingState.FAILED.value,
                'data': input_data
            }
    
    def execute_async(self, input_data: Dict[str, Any]) -> Future[Dict[str, Any]]:
        """Execute pipeline asynchronously."""
        return self.executor.submit(self.execute, input_data)
    
    def execute_batch(self, input_batch: List[Dict[str, Any]], 
                     max_concurrent: int = None) -> List[Dict[str, Any]]:
        """Execute pipeline on batch of inputs with concurrency control."""
        max_concurrent = max_concurrent or self.config.max_workers
        
        self.logger.info(f"Starting batch execution: {len(input_batch)} items, {max_concurrent} concurrent")
        
        results = []
        
        # Submit all tasks
        futures = []
        for i, input_data in enumerate(input_batch):
            future = self.executor.submit(self.execute, input_data)
            futures.append((i, future))
        
        # Collect results as they complete
        for i, future in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                results.append((i, result))
            except Exception as e:
                self.logger.error(f"Batch item {i} failed: {e}")
                results.append((i, {'success': False, 'error': str(e)}))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def shutdown(self):
        """Shutdown the pipeline and cleanup resources."""
        self.logger.info("Shutting down resilient pipeline")
        self._shutdown = True
        self.executor.shutdown(wait=True)


def create_firmware_analysis_pipeline(config: ProcessingConfig = None) -> ResilientPipeline:
    """Create a resilient firmware analysis pipeline."""
    config = config or ProcessingConfig()
    
    pipeline = ResilientPipeline(config)
    
    # Add standard stages
    pipeline.add_stage(FirmwareAnalysisStage(config))
    pipeline.add_stage(PatchGenerationStage(config))
    
    return pipeline


def create_batch_processor(pipeline: ResilientPipeline, 
                          batch_size: int = 10,
                          max_concurrent: int = 4) -> Callable:
    """Create a batch processor function."""
    
    def process_batch(input_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process items in batches."""
        results = []
        
        # Process in chunks
        for i in range(0, len(input_items), batch_size):
            batch = input_items[i:i + batch_size]
            batch_results = pipeline.execute_batch(batch, max_concurrent)
            results.extend(batch_results)
        
        return results
    
    return process_batch


# Global resilient pipeline instance
default_pipeline_config = ProcessingConfig(
    max_workers=4,
    timeout_seconds=60.0,
    retry_attempts=2,
    failure_mode=FailureMode.GRACEFUL_DEGRADATION,
    health_check_interval=30.0
)

default_firmware_pipeline = create_firmware_analysis_pipeline(default_pipeline_config)