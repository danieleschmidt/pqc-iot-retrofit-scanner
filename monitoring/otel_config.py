"""
OpenTelemetry configuration for PQC IoT Retrofit Scanner.

Provides distributed tracing, metrics, and logs integration with OTLP exporters
for comprehensive observability in cloud-native environments.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.sdk.trace import TracerProvider, sampling
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.psutil import PsutilInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.propagators.jaeger import JaegerPropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.semconv.trace import SpanAttributes
    from opentelemetry.util.http import get_excluded_urls
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    
    # Mock classes for when OpenTelemetry is not available
    class MockTracer:
        def start_as_current_span(self, name, **kwargs):
            return MockSpan()
        
        def start_span(self, name, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def set_attribute(self, key, value):
            pass
        
        def set_status(self, status):
            pass
        
        def record_exception(self, exception):
            pass
        
        def end(self):
            pass
    
    class MockMeter:
        def create_counter(self, *args, **kwargs):
            return MockInstrument()
        
        def create_histogram(self, *args, **kwargs):
            return MockInstrument()
        
        def create_gauge(self, *args, **kwargs):
            return MockInstrument()
    
    class MockInstrument:
        def add(self, *args, **kwargs):
            pass
        
        def record(self, *args, **kwargs):
            pass
        
        def set(self, *args, **kwargs):
            pass
    
    trace = type('trace', (), {'get_tracer': lambda name: MockTracer()})()
    metrics = type('metrics', (), {'get_meter': lambda name: MockMeter()})()


class OTelConfig:
    """OpenTelemetry configuration manager."""
    
    def __init__(self):
        """Initialize OpenTelemetry configuration."""
        self.enabled = OTEL_AVAILABLE and os.getenv("PQC_OTEL_ENABLED", "false").lower() == "true"
        self.service_name = os.getenv("PQC_SERVICE_NAME", "pqc-iot-retrofit-scanner")
        self.service_version = os.getenv("VERSION", "0.1.0")
        self.environment = os.getenv("PQC_ENVIRONMENT", "development")
        
        # OTLP exporter configuration
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.otlp_headers = self._parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
        
        # Jaeger configuration (alternative)
        self.jaeger_endpoint = os.getenv("JAEGER_AGENT_HOST")
        self.jaeger_port = int(os.getenv("JAEGER_AGENT_PORT", "14268"))
        
        # Sampling configuration
        self.trace_sample_rate = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "0.1"))  # 10% sampling
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    def _parse_headers(self, headers_str: str) -> Dict[str, str]:
        """Parse OTLP headers from environment variable."""
        headers = {}
        if headers_str:
            for header in headers_str.split(","):
                key, value = header.split("=", 1)
                headers[key.strip()] = value.strip()
        return headers
    
    def initialize(self):
        """Initialize OpenTelemetry with configured providers."""
        if not self.enabled:
            self.logger.info("OpenTelemetry disabled")
            return
        
        if self.initialized:
            return
        
        try:
            # Set up resource information
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
                DEPLOYMENT_ENVIRONMENT: self.environment,
                "service.instance.id": os.getenv("HOSTNAME", "unknown"),
                "pqc.architecture": os.getenv("PQC_ARCHITECTURE", "unknown"),
            })
            
            # Configure tracing
            self._setup_tracing(resource)
            
            # Configure metrics
            self._setup_metrics(resource)
            
            # Configure propagators
            self._setup_propagators()
            
            # Auto-instrument libraries
            self._setup_auto_instrumentation()
            
            self.initialized = True
            self.logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")
    
    def _setup_tracing(self, resource: Resource):
        """Set up tracing with appropriate exporters."""
        # Create tracer provider with sampling
        sampler = sampling.TraceIdRatioBased(self.trace_sample_rate)
        tracer_provider = TracerProvider(resource=resource, sampler=sampler)
        
        # Configure span exporters
        span_processors = []
        
        # OTLP exporter (primary)
        if self.otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint,
                    headers=self.otlp_headers,
                    timeout=30
                )
                span_processors.append(BatchSpanProcessor(otlp_exporter))
                self.logger.info(f"OTLP trace exporter configured: {self.otlp_endpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to configure OTLP trace exporter: {e}")
        
        # Console exporter (development)
        if self.environment == "development":
            console_exporter = ConsoleSpanExporter()
            span_processors.append(BatchSpanProcessor(console_exporter))
        
        # Add processors to tracer provider
        for processor in span_processors:
            tracer_provider.add_span_processor(processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
    
    def _setup_metrics(self, resource: Resource):
        """Set up metrics with appropriate exporters."""
        # Configure metric exporters
        readers = []
        
        # OTLP metric exporter (primary)
        if self.otlp_endpoint:
            try:
                otlp_metric_exporter = OTLPMetricExporter(
                    endpoint=self.otlp_endpoint,
                    headers=self.otlp_headers,
                    timeout=30
                )
                readers.append(PeriodicExportingMetricReader(
                    exporter=otlp_metric_exporter,
                    export_interval_millis=60000  # Export every 60 seconds
                ))
                self.logger.info(f"OTLP metric exporter configured: {self.otlp_endpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to configure OTLP metric exporter: {e}")
        
        # Console exporter (development)
        if self.environment == "development":
            console_metric_exporter = ConsoleMetricExporter()
            readers.append(PeriodicExportingMetricReader(
                exporter=console_metric_exporter,
                export_interval_millis=30000  # Export every 30 seconds
            ))
        
        # Create meter provider
        if readers:
            meter_provider = MeterProvider(resource=resource, metric_readers=readers)
            metrics.set_meter_provider(meter_provider)
    
    def _setup_propagators(self):
        """Set up trace context propagators."""
        # Configure multiple propagators for compatibility
        propagators = [
            B3MultiFormat(),
            JaegerPropagator(),
        ]
        
        # Set composite propagator
        set_global_textmap(CompositePropagator(propagators))
    
    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        try:
            # Instrument HTTP requests
            RequestsInstrumentor().instrument()
            
            # Instrument logging
            LoggingInstrumentor().instrument(set_logging_format=True)
            
            # Instrument system metrics
            PsutilInstrumentor().instrument()
            
            self.logger.info("Auto-instrumentation configured")
        except Exception as e:
            self.logger.warning(f"Failed to configure auto-instrumentation: {e}")


class PQCTracer:
    """Custom tracer for PQC IoT Retrofit Scanner operations."""
    
    def __init__(self):
        """Initialize PQC tracer."""
        self.enabled = OTEL_AVAILABLE and os.getenv("PQC_OTEL_ENABLED", "false").lower() == "true"
        if self.enabled:
            self.tracer = trace.get_tracer("pqc_iot_retrofit")
        else:
            self.tracer = MockTracer()
    
    @contextmanager
    def trace_firmware_analysis(self, firmware_path: str, architecture: str):
        """Trace firmware analysis operation."""
        with self.tracer.start_as_current_span("firmware_analysis") as span:
            if self.enabled:
                span.set_attribute("pqc.firmware.path", firmware_path)
                span.set_attribute("pqc.firmware.architecture", architecture)
                span.set_attribute("operation.type", "firmware_analysis")
            
            try:
                yield span
            except Exception as e:
                if self.enabled:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                raise
    
    @contextmanager
    def trace_vulnerability_detection(self, algorithm: str, confidence: float):
        """Trace vulnerability detection operation."""
        with self.tracer.start_as_current_span("vulnerability_detection") as span:
            if self.enabled:
                span.set_attribute("pqc.vulnerability.algorithm", algorithm)
                span.set_attribute("pqc.vulnerability.confidence", confidence)
                span.set_attribute("operation.type", "vulnerability_detection")
            
            try:
                yield span
            except Exception as e:
                if self.enabled:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                raise
    
    @contextmanager
    def trace_patch_generation(self, algorithm: str, target_device: str):
        """Trace patch generation operation."""
        with self.tracer.start_as_current_span("patch_generation") as span:
            if self.enabled:
                span.set_attribute("pqc.patch.algorithm", algorithm)
                span.set_attribute("pqc.patch.target_device", target_device)
                span.set_attribute("operation.type", "patch_generation")
            
            try:
                yield span
            except Exception as e:
                if self.enabled:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                raise
    
    @contextmanager
    def trace_crypto_operation(self, operation: str, algorithm: str):
        """Trace cryptographic operation."""
        with self.tracer.start_as_current_span("crypto_operation") as span:
            if self.enabled:
                span.set_attribute("pqc.crypto.operation", operation)
                span.set_attribute("pqc.crypto.algorithm", algorithm)
                span.set_attribute("operation.type", "crypto_operation")
            
            try:
                yield span
            except Exception as e:
                if self.enabled:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                raise
    
    def add_baggage(self, key: str, value: str):
        """Add baggage to current trace context."""
        if self.enabled and OTEL_AVAILABLE:
            baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage from current trace context."""
        if self.enabled and OTEL_AVAILABLE:
            return baggage.get_baggage(key)
        return None


class PQCMeter:
    """Custom meter for PQC IoT Retrofit Scanner metrics."""
    
    def __init__(self):
        """Initialize PQC meter."""
        self.enabled = OTEL_AVAILABLE and os.getenv("PQC_OTEL_ENABLED", "false").lower() == "true"
        if self.enabled:
            self.meter = metrics.get_meter("pqc_iot_retrofit")
        else:
            self.meter = MockMeter()
        
        # Create instruments
        self.firmware_analyses_counter = self.meter.create_counter(
            "pqc_firmware_analyses",
            description="Number of firmware analyses performed",
            unit="1"
        )
        
        self.firmware_analysis_duration = self.meter.create_histogram(
            "pqc_firmware_analysis_duration",
            description="Duration of firmware analysis operations",
            unit="ms"
        )
        
        self.vulnerabilities_detected_counter = self.meter.create_counter(
            "pqc_vulnerabilities_detected",
            description="Number of vulnerabilities detected",
            unit="1"
        )
        
        self.patches_generated_counter = self.meter.create_counter(
            "pqc_patches_generated",
            description="Number of patches generated",
            unit="1"
        )
        
        self.crypto_operations_counter = self.meter.create_counter(
            "pqc_crypto_operations",
            description="Number of cryptographic operations",
            unit="1"
        )
        
        self.active_analyses_gauge = self.meter.create_gauge(
            "pqc_active_analyses",
            description="Number of currently active analyses",
            unit="1"
        )
    
    def record_firmware_analysis(self, architecture: str, duration_ms: float, status: str):
        """Record firmware analysis metrics."""
        attributes = {
            "architecture": architecture,
            "status": status
        }
        
        self.firmware_analyses_counter.add(1, attributes)
        self.firmware_analysis_duration.record(duration_ms, attributes)
    
    def record_vulnerability(self, algorithm: str, severity: str, architecture: str):
        """Record vulnerability detection metrics."""
        attributes = {
            "algorithm": algorithm,
            "severity": severity,
            "architecture": architecture
        }
        
        self.vulnerabilities_detected_counter.add(1, attributes)
    
    def record_patch_generation(self, algorithm: str, target_device: str, status: str):
        """Record patch generation metrics."""
        attributes = {
            "algorithm": algorithm,
            "target_device": target_device,
            "status": status
        }
        
        self.patches_generated_counter.add(1, attributes)
    
    def record_crypto_operation(self, operation: str, algorithm: str, status: str):
        """Record cryptographic operation metrics."""
        attributes = {
            "operation": operation,
            "algorithm": algorithm,
            "status": status
        }
        
        self.crypto_operations_counter.add(1, attributes)
    
    def set_active_analyses(self, count: int):
        """Update active analyses gauge."""
        self.active_analyses_gauge.set(count)


# Global instances
otel_config = OTelConfig()
pqc_tracer = PQCTracer()
pqc_meter = PQCMeter()


def initialize_otel():
    """Initialize OpenTelemetry configuration."""
    otel_config.initialize()


def trace_operation(operation_type: str):
    """Decorator for tracing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with pqc_tracer.tracer.start_as_current_span(f"{operation_type}_{func.__name__}") as span:
                if pqc_tracer.enabled:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("operation.type", operation_type)
                
                try:
                    result = func(*args, **kwargs)
                    if pqc_tracer.enabled:
                        span.set_attribute("success", True)
                    return result
                except Exception as e:
                    if pqc_tracer.enabled:
                        span.record_exception(e)
                        span.set_attribute("error", True)
                        span.set_attribute("error.type", type(e).__name__)
                    raise
        
        return wrapper
    return decorator


# Convenience functions
def trace_firmware_analysis(firmware_path: str, architecture: str):
    """Context manager for tracing firmware analysis."""
    return pqc_tracer.trace_firmware_analysis(firmware_path, architecture)


def trace_vulnerability_detection(algorithm: str, confidence: float):
    """Context manager for tracing vulnerability detection."""
    return pqc_tracer.trace_vulnerability_detection(algorithm, confidence)


def trace_patch_generation(algorithm: str, target_device: str):
    """Context manager for tracing patch generation."""
    return pqc_tracer.trace_patch_generation(algorithm, target_device)


def trace_crypto_operation(operation: str, algorithm: str):
    """Context manager for tracing crypto operations."""
    return pqc_tracer.trace_crypto_operation(operation, algorithm)


def record_firmware_analysis_metrics(architecture: str, duration_ms: float, status: str = "success"):
    """Record firmware analysis metrics."""
    pqc_meter.record_firmware_analysis(architecture, duration_ms, status)


def record_vulnerability_metrics(algorithm: str, severity: str, architecture: str):
    """Record vulnerability detection metrics."""
    pqc_meter.record_vulnerability(algorithm, severity, architecture)


def record_patch_generation_metrics(algorithm: str, target_device: str, status: str = "success"):
    """Record patch generation metrics."""
    pqc_meter.record_patch_generation(algorithm, target_device, status)


def record_crypto_operation_metrics(operation: str, algorithm: str, status: str = "success"):
    """Record cryptographic operation metrics."""
    pqc_meter.record_crypto_operation(operation, algorithm, status)


# Initialize on import if enabled
if os.getenv("PQC_OTEL_AUTO_INIT", "true").lower() == "true":
    initialize_otel()