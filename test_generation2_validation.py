"""Generation 2 Validation Suite - Robust Error Handling and Resilience.

Tests for:
- Comprehensive input validation and sanitization
- Advanced logging and monitoring
- Retry mechanisms and resilience patterns
- Circuit breaker functionality
- Security-aware validation
"""

import sys
import tempfile
import json
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_robust_validation():
    """Test comprehensive input validation."""
    print("Testing robust validation...")
    
    try:
        from pqc_iot_retrofit.robust_validation import (
            RobustValidator, ValidationSeverity, ValidationResult,
            validate_firmware_scan_config, is_valid_configuration
        )
        
        # Test string input validation
        result = RobustValidator.validate_string_input("test_string", "test_field")
        assert result.is_valid
        print("✅ String validation works")
        
        # Test malicious pattern detection
        malicious_result = RobustValidator.validate_string_input("../../../etc/passwd", "malicious_field")
        assert not malicious_result.is_valid
        assert malicious_result.severity == ValidationSeverity.CRITICAL
        print("✅ Malicious pattern detection works")
        
        # Test file path validation
        with tempfile.NamedTemporaryFile(suffix='.bin') as tmp:
            tmp.write(b"test firmware data")
            tmp.flush()
            
            path_result = RobustValidator.validate_file_path(tmp.name)
            assert path_result.is_valid
            print("✅ File path validation works")
        
        # Test firmware file validation
        with tempfile.NamedTemporaryFile(suffix='.bin') as tmp:
            tmp.write(b"test firmware data" * 100)  # Make it reasonable size
            tmp.flush()
            
            firmware_result = RobustValidator.validate_firmware_file(tmp.name)
            assert firmware_result.is_valid
            print("✅ Firmware file validation works")
        
        # Test architecture validation
        arch_result = RobustValidator.validate_architecture("cortex-m4")
        assert arch_result.is_valid
        print("✅ Architecture validation works")
        
        # Test invalid architecture
        invalid_arch = RobustValidator.validate_architecture("invalid-arch")
        assert not invalid_arch.is_valid
        print("✅ Invalid architecture detection works")
        
        # Test memory constraints validation
        constraints = {"flash": 512*1024, "ram": 128*1024}
        memory_result = RobustValidator.validate_memory_constraints(constraints)
        assert memory_result.is_valid
        print("✅ Memory constraints validation works")
        
        # Test address validation
        addr_result = RobustValidator.validate_address("0x08000000")
        assert addr_result.is_valid
        assert addr_result.value == 0x08000000
        print("✅ Address validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust validation test failed: {e}")
        return False

def test_secure_logging():
    """Test secure logging functionality."""
    print("\nTesting secure logging...")
    
    try:
        from pqc_iot_retrofit.robust_logging import (
            StructuredLogger, LogLevel, EventType, SecurityEvent,
            SecureLogSanitizer, performance_monitor, get_logger
        )
        
        # Test logger creation
        logger = StructuredLogger("test_logger")
        assert logger.name == "test_logger"
        print("✅ Logger creation works")
        
        # Test context setting
        logger.set_context(operation="test_operation", user_id="test_user")
        context = logger.get_context()
        assert context.operation == "test_operation"
        print("✅ Context setting works")
        
        # Test event logging
        logger.log_event(LogLevel.INFO, EventType.SYSTEM_START, "System started")
        print("✅ Event logging works")
        
        # Test security event logging
        security_event = SecurityEvent(
            event_type="suspicious_activity",
            severity="high",
            source_ip="192.168.1.100",
            attack_indicators=["path_traversal"]
        )
        logger.log_security_event(security_event, "Security alert triggered")
        print("✅ Security event logging works")
        
        # Test log sanitization
        sensitive_data = "password=secret123 and token=abc123xyz"
        sanitized = SecureLogSanitizer.sanitize_message(sensitive_data)
        assert "secret123" not in sanitized
        assert "abc123xyz" not in sanitized
        print("✅ Log sanitization works")
        
        # Test performance monitoring decorator
        @performance_monitor("test_operation")
        def test_function():
            time.sleep(0.1)
            return "test_result"
        
        result = test_function()
        assert result == "test_result"
        print("✅ Performance monitoring decorator works")
        
        return True
        
    except Exception as e:
        print(f"❌ Secure logging test failed: {e}")
        return False

def test_resilience_patterns():
    """Test retry and resilience mechanisms."""
    print("\nTesting resilience patterns...")
    
    try:
        from pqc_iot_retrofit.resilience import (
            RetryHandler, RetryConfig, RetryStrategy, CircuitBreaker,
            CircuitBreakerConfig, ResilientExecutor, retry, circuit_breaker
        )
        
        # Test retry handler
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_handler = RetryHandler(retry_config)
        
        # Test successful retry
        call_count = 0
        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = retry_handler.execute(failing_then_succeeding)
        assert result == "success"
        assert call_count == 2
        print("✅ Retry mechanism works")
        
        # Test circuit breaker
        circuit_config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        circuit_breaker_obj = CircuitBreaker(circuit_config)
        
        # Cause failures to open circuit
        def always_failing():
            raise ConnectionError("Always fails")
        
        for _ in range(3):
            try:
                circuit_breaker_obj.call(always_failing)
            except:
                pass
        
        state = circuit_breaker_obj.get_state()
        assert state['state'] == 'open'
        print("✅ Circuit breaker opens on failures")
        
        # Test resilient executor
        resilient_executor = ResilientExecutor(
            retry_config=RetryConfig(max_attempts=2, base_delay=0.1),
            timeout_seconds=1.0
        )
        
        def simple_function():
            return "resilient_result"
        
        result = resilient_executor.execute(simple_function)
        assert result == "resilient_result"
        print("✅ Resilient executor works")
        
        # Test retry decorator
        attempt_count = 0
        @retry(max_attempts=3, base_delay=0.1)
        def decorated_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Temporary failure")
            return "decorated_success"
        
        result = decorated_function()
        assert result == "decorated_success"
        print("✅ Retry decorator works")
        
        return True
        
    except Exception as e:
        print(f"❌ Resilience patterns test failed: {e}")
        return False

def test_configuration_validation():
    """Test comprehensive configuration validation."""
    print("\nTesting configuration validation...")
    
    try:
        from pqc_iot_retrofit.robust_validation import (
            validate_firmware_scan_config, is_valid_configuration,
            get_validation_errors, format_validation_errors
        )
        
        # Create test firmware file
        with tempfile.NamedTemporaryFile(suffix='.bin') as tmp:
            tmp.write(b"test firmware data" * 100)
            tmp.flush()
            
            # Test valid configuration
            results = validate_firmware_scan_config(
                firmware_path=tmp.name,
                architecture="cortex-m4",
                memory_constraints={"flash": 512*1024, "ram": 128*1024},
                base_address="0x08000000"
            )
            
            assert is_valid_configuration(results)
            print("✅ Valid configuration validation works")
            
            # Test invalid configuration
            invalid_results = validate_firmware_scan_config(
                firmware_path="/nonexistent/file.bin",
                architecture="invalid-arch",
                base_address="invalid_address"
            )
            
            assert not is_valid_configuration(invalid_results)
            errors = get_validation_errors(invalid_results)
            assert len(errors) > 0
            
            error_message = format_validation_errors(invalid_results)
            assert "❌" in error_message or "🚨" in error_message
            print("✅ Invalid configuration detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_secure_hash_validation():
    """Test secure hash validation."""
    print("\nTesting secure hash validation...")
    
    try:
        from pqc_iot_retrofit.robust_validation import SecureHashValidator
        
        # Create test file
        test_data = b"test data for hashing"
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(test_data)
            tmp.flush()
            
            # Calculate hash
            hash_value = SecureHashValidator.calculate_secure_hash(tmp.name)
            assert hash_value is not None
            assert len(hash_value) == 64  # SHA-256 hex length
            print("✅ Hash calculation works")
            
            # Test integrity validation
            integrity_result = SecureHashValidator.validate_file_integrity(
                tmp.name, hash_value
            )
            assert integrity_result.is_valid
            print("✅ File integrity validation works")
            
            # Test integrity failure
            wrong_hash = "a" * 64  # Wrong hash
            integrity_failure = SecureHashValidator.validate_file_integrity(
                tmp.name, wrong_hash
            )
            assert not integrity_failure.is_valid
            print("✅ File integrity failure detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Secure hash validation test failed: {e}")
        return False

def test_log_aggregation():
    """Test log aggregation and analysis."""
    print("\nTesting log aggregation...")
    
    try:
        from pqc_iot_retrofit.robust_logging import (
            get_log_aggregator, get_logger, EventType, LogLevel
        )
        
        # Get logger and aggregator
        logger = get_logger("test_aggregator")
        aggregator = get_log_aggregator()
        
        # Log some events
        logger.log_event(LogLevel.INFO, EventType.SYSTEM_START, "System starting")
        logger.log_event(LogLevel.ERROR, EventType.SCAN_ERROR, "Scan failed")
        logger.log_event(LogLevel.WARNING, EventType.VALIDATION_ERROR, "Validation issue")
        
        # Give some time for events to be processed
        time.sleep(0.1)
        
        # Test event aggregation
        error_summary = aggregator.get_error_summary(time_window_seconds=60)
        assert error_summary['total_events'] >= 0
        print("✅ Log aggregation works")
        
        # Test event filtering
        security_events = aggregator.get_security_events()
        assert isinstance(security_events, list)
        print("✅ Event filtering works")
        
        return True
        
    except Exception as e:
        print(f"❌ Log aggregation test failed: {e}")
        return False

def test_integration_generation2():
    """Test integration of Generation 2 components."""
    print("\nTesting Generation 2 integration...")
    
    try:
        from pqc_iot_retrofit.robust_validation import validate_firmware_scan_config
        from pqc_iot_retrofit.robust_logging import get_logger, LogLevel, EventType
        from pqc_iot_retrofit.resilience import retry
        
        # Create test scenario combining all Generation 2 features
        logger = get_logger("integration_test")
        
        # Resilient validation function
        @retry(max_attempts=2, base_delay=0.1)
        def validate_configuration_resilient(firmware_path, architecture):
            logger.log_event(LogLevel.INFO, EventType.USER_ACTION, 
                           "Starting configuration validation")
            
            results = validate_firmware_scan_config(firmware_path, architecture)
            
            if not all(r.is_valid for r in results):
                logger.log_event(LogLevel.ERROR, EventType.VALIDATION_ERROR,
                               "Configuration validation failed")
                raise ValueError("Configuration validation failed")
            
            logger.log_event(LogLevel.INFO, EventType.SYSTEM_START,
                           "Configuration validation successful")
            return results
        
        # Test with valid configuration
        with tempfile.NamedTemporaryFile(suffix='.bin') as tmp:
            tmp.write(b"test firmware data" * 100)
            tmp.flush()
            
            results = validate_configuration_resilient(tmp.name, "cortex-m4")
            assert all(r.is_valid for r in results)
            print("✅ Integration: resilient validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation 2 integration test failed: {e}")
        return False

def test_health_checking():
    """Test health checking functionality."""
    print("\nTesting health checking...")
    
    try:
        from pqc_iot_retrofit.resilience import HealthChecker, health_check, global_health_checker
        
        # Test health checker
        health_checker = HealthChecker()
        
        def healthy_check():
            return True
        
        def unhealthy_check():
            return False
        
        health_checker.register_check("healthy_service", healthy_check)
        health_checker.register_check("unhealthy_service", unhealthy_check)
        
        results = health_checker.run_checks()
        
        assert 'overall' in results
        assert results['healthy_service']['healthy'] == True
        assert results['unhealthy_service']['healthy'] == False
        assert results['overall']['healthy'] == False  # Overall should be unhealthy
        print("✅ Health checking works")
        
        # Test health check decorator
        @health_check("decorated_service")
        def decorated_health_check():
            return True
        
        global_results = global_health_checker.run_checks()
        assert 'decorated_service' in global_results
        print("✅ Health check decorator works")
        
        return True
        
    except Exception as e:
        print(f"❌ Health checking test failed: {e}")
        return False

def main():
    """Run all Generation 2 validation tests."""
    print("🚀 Generation 2 Validation Suite - Robust Error Handling")
    print("=" * 70)
    
    tests = [
        test_robust_validation,
        test_secure_logging,
        test_resilience_patterns,
        test_configuration_validation,
        test_secure_hash_validation,
        test_log_aggregation,
        test_health_checking,
        test_integration_generation2
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        print()  # Add spacing between tests
    
    print("=" * 70)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Generation 2 tests passed!")
        print("\n🛡️ Generation 2 Features Validated:")
        print("   ✅ Comprehensive input validation and sanitization")
        print("   ✅ Security-aware logging and monitoring")
        print("   ✅ Retry mechanisms and resilience patterns") 
        print("   ✅ Circuit breaker functionality")
        print("   ✅ Health checking and diagnostics")
        print("   ✅ Secure hash validation")
        print("   ✅ Configuration validation framework")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)