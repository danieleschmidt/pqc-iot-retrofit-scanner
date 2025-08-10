"""Generation 4 Integration Tests.

Comprehensive testing of AI-powered features, quantum resilience analysis,
and autonomous research capabilities.
"""

import pytest
import numpy as np
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import Generation 4 components
from pqc_iot_retrofit.adaptive_ai import (
    AdaptiveAI, EnsembleDetector, AnomalyDetector, AdaptiveOptimizer,
    AIModelType, LearningStrategy, FirmwareFingerprint
)
from pqc_iot_retrofit.quantum_resilience import (
    QuantumResilienceAnalyzer, QuantumThreatModel, PQCAlgorithmDatabase,
    QuantumThreatLevel, MigrationStrategy
)
from pqc_iot_retrofit.autonomous_research import (
    AutonomousResearcher, ExperimentalFramework, BenchmarkingSuite,
    ResearchObjective, ExperimentStatus
)

# Import base components
from pqc_iot_retrofit.scanner import CryptoAlgorithm, CryptoVulnerability, RiskLevel


class TestAdaptiveAI:
    """Test suite for Adaptive AI system."""
    
    def test_adaptive_ai_initialization(self):
        """Test AI system initialization."""
        ai = AdaptiveAI()
        
        assert ai.ensemble_detector is not None
        assert ai.anomaly_detector is not None
        assert ai.adaptive_optimizer is not None
        assert ai.learning_strategy == LearningStrategy.HYBRID
        assert len(ai.ensemble_detector.models) >= 2  # Default detectors loaded
    
    def test_ensemble_detector(self):
        """Test ensemble vulnerability detection."""
        detector = EnsembleDetector()
        
        # Add test detector
        def test_detector(firmware_data, context):
            return [(self._create_test_vulnerability(), 0.8)]
        
        detector.add_detector("test", test_detector, 1.0)
        
        # Test detection
        firmware_data = b"RSA" * 100  # Simple test data
        detections = detector.detect(firmware_data, {})
        
        assert len(detections) > 0
        assert all(len(detection) == 2 for detection in detections)  # (vuln, confidence)
        assert all(0 <= confidence <= 1 for _, confidence in detections)
    
    def test_anomaly_detector_training(self):
        """Test anomaly detector baseline training."""
        detector = AnomalyDetector()
        
        # Generate synthetic firmware samples
        samples = [
            np.random.bytes(1000) for _ in range(10)
        ]
        
        detector.fit_baseline(samples)
        
        assert detector.baseline_features is not None
        assert detector.anomaly_threshold is not None
        assert detector.anomaly_threshold > 0
    
    def test_anomaly_detection(self):
        """Test anomaly detection on trained baseline."""
        detector = AnomalyDetector()
        
        # Train with normal samples
        normal_samples = [
            np.random.normal(128, 50, 1000).astype(np.uint8).tobytes() 
            for _ in range(5)
        ]
        detector.fit_baseline(normal_samples)
        
        # Test with normal sample
        test_normal = np.random.normal(128, 50, 1000).astype(np.uint8).tobytes()
        is_anomalous, score, analysis = detector.detect_anomalies(test_normal)
        
        assert isinstance(is_anomalous, bool)
        assert isinstance(score, float)
        assert score >= 0
        assert 'anomaly_score' in analysis
        assert 'threshold' in analysis
    
    def test_adaptive_optimizer(self):
        """Test adaptive patch optimization."""
        optimizer = AdaptiveOptimizer()
        
        test_vuln = self._create_test_vulnerability()
        constraints = {
            'max_memory': 64000,
            'min_performance': 0.7
        }
        
        patch = optimizer.optimize_patch(test_vuln, constraints)
        
        assert patch.patch_id is not None
        assert patch.target_vulnerability == test_vuln
        assert patch.algorithm_replacement in ['Dilithium2', 'Dilithium3', 'Kyber512', 'Kyber768']
        assert 0 <= patch.memory_efficiency <= 1
        assert 0 <= patch.performance_gain <= 1
        assert 0 <= patch.success_probability <= 1
    
    def test_firmware_analysis_integration(self):
        """Test complete AI-powered firmware analysis."""
        ai = AdaptiveAI()
        
        # Generate test firmware data
        firmware_data = self._generate_test_firmware()
        context = {'architecture': 'cortex-m4', 'file_size': len(firmware_data)}
        
        analysis = ai.analyze_firmware(firmware_data, context)
        
        # Verify analysis structure
        assert 'firmware_fingerprint' in analysis
        assert 'ensemble_detection' in analysis
        assert 'anomaly_analysis' in analysis
        assert 'optimized_patches' in analysis
        assert 'ai_metadata' in analysis
        
        # Verify fingerprint
        fingerprint = analysis['firmware_fingerprint']
        assert fingerprint['size_bytes'] == len(firmware_data)
        assert 0 <= fingerprint['entropy'] <= 8
        assert fingerprint['architecture'] == 'cortex-m4'
        
        # Verify AI metadata
        metadata = analysis['ai_metadata']
        assert 'analysis_time_seconds' in metadata
        assert metadata['analysis_time_seconds'] > 0
    
    def _create_test_vulnerability(self):
        """Create a test vulnerability object."""
        return CryptoVulnerability(
            algorithm=CryptoAlgorithm.RSA_2048,
            address=0x08001000,
            function_name="test_rsa_function",
            risk_level=RiskLevel.HIGH,
            key_size=2048,
            description="Test RSA vulnerability",
            mitigation="Replace with Dilithium",
            stack_usage=256,
            available_stack=32768
        )
    
    def _generate_test_firmware(self, size=10000):
        """Generate synthetic test firmware data."""
        # Create firmware-like data with mixed content
        firmware = bytearray()
        
        # Add some structured data (simulating code sections)
        for _ in range(size // 4):
            firmware.extend(np.random.randint(0x4000, 0x8000, dtype=np.uint16).tobytes())
        
        # Add some string-like data
        for _ in range(size // 8):
            if np.random.random() < 0.01:  # 1% chance of crypto string
                firmware.extend(b'RSA')
            else:
                firmware.extend(np.random.randint(32, 127, dtype=np.uint8).tobytes())
        
        # Fill remaining with random data
        remaining = size - len(firmware)
        firmware.extend(np.random.bytes(remaining))
        
        return bytes(firmware)


class TestQuantumResilience:
    """Test suite for Quantum Resilience Analysis."""
    
    def test_quantum_threat_model(self):
        """Test quantum threat modeling."""
        threat_model = QuantumThreatModel()
        
        # Test threat assessment
        vulnerability = threat_model.assess_quantum_vulnerability(CryptoAlgorithm.RSA_2048, 2048)
        
        assert vulnerability.algorithm == CryptoAlgorithm.RSA_2048
        assert vulnerability.threat_level in list(QuantumThreatLevel)
        assert vulnerability.shor_applicable is True
        assert vulnerability.estimated_qubits_required > 0
        assert vulnerability.cryptanalytic_margin > 0
    
    def test_threat_timeline_projection(self):
        """Test quantum threat timeline projection."""
        threat_model = QuantumThreatModel()
        
        timeline = threat_model.project_threat_timeline(CryptoAlgorithm.RSA_2048, 2024)
        
        assert isinstance(timeline, dict)
        assert len(timeline) > 0
        
        # Check timeline structure
        for year, data in timeline.items():
            assert isinstance(year, int)
            assert year >= 2024
            assert 'projected_qubits' in data
            assert 'threat_active' in data
            assert 'risk_level' in data
            assert 'recommended_action' in data
    
    def test_pqc_algorithm_database(self):
        """Test PQC algorithm database."""
        db = PQCAlgorithmDatabase()
        
        # Test algorithm retrieval
        dilithium2 = db.get_algorithm('Dilithium2')
        assert dilithium2 is not None
        assert dilithium2.name == 'Dilithium2'
        assert dilithium2.nist_security_level == 2
        assert dilithium2.algorithm_type == 'signature'
        
        # Test algorithm search
        suitable = db.find_suitable_algorithms({
            'algorithm_type': 'signature',
            'min_security_level': 2,
            'max_memory_kb': 32
        })
        
        assert len(suitable) > 0
        assert all(alg.algorithm_type == 'signature' for alg in suitable)
        assert all(alg.nist_security_level >= 2 for alg in suitable)
    
    def test_resilience_assessment(self):
        """Test complete resilience assessment."""
        analyzer = QuantumResilienceAnalyzer()
        
        # Create test vulnerabilities
        vulnerabilities = [
            self._create_test_vulnerability(CryptoAlgorithm.RSA_2048),
            self._create_test_vulnerability(CryptoAlgorithm.ECDSA_P256)
        ]
        
        assessment = analyzer.assess_system_resilience(
            vulnerabilities,
            {'architecture': 'cortex-m4', 'max_memory_kb': 64}
        )
        
        assert 0 <= assessment.overall_resilience_score <= 1
        assert len(assessment.quantum_vulnerabilities) > 0
        assert assessment.crypto_agility_level is not None
        assert 0 <= assessment.migration_readiness <= 1
        assert assessment.recommended_strategy in list(MigrationStrategy)
    
    def test_migration_plan_generation(self):
        """Test migration plan generation."""
        analyzer = QuantumResilienceAnalyzer()
        
        # Create mock assessment
        vulnerabilities = [self._create_test_vulnerability(CryptoAlgorithm.RSA_2048)]
        assessment = analyzer.assess_system_resilience(vulnerabilities)
        
        migration_plan = analyzer.generate_migration_plan(
            assessment,
            {'system_name': 'Test System', 'max_memory_kb': 64}
        )
        
        assert migration_plan.plan_id is not None
        assert migration_plan.target_system == 'Test System'
        assert len(migration_plan.current_algorithms) > 0
        assert len(migration_plan.recommended_algorithms) > 0
        assert len(migration_plan.migration_phases) > 0
        assert migration_plan.estimated_timeline.total_seconds() > 0
    
    def _create_test_vulnerability(self, algorithm):
        """Create test vulnerability for quantum analysis."""
        return CryptoVulnerability(
            algorithm=algorithm,
            address=0x08001000,
            function_name=f"test_{algorithm.value.lower()}_function",
            risk_level=RiskLevel.HIGH,
            key_size=2048 if 'RSA' in algorithm.value else 256,
            description=f"Test {algorithm.value} vulnerability",
            mitigation="Replace with PQC algorithm",
            stack_usage=256,
            available_stack=32768
        )


class TestAutonomousResearch:
    """Test suite for Autonomous Research system."""
    
    def test_experimental_framework(self):
        """Test experimental framework initialization."""
        framework = ExperimentalFramework()
        
        assert framework.experiment_database is not None
        assert framework.statistical_analyzer is not None
        assert framework.benchmarking_suite is not None
        assert framework.executor is not None
    
    def test_benchmarking_suite(self):
        """Test algorithm benchmarking."""
        suite = BenchmarkingSuite()
        
        parameters = {
            'algorithm': 'Dilithium2',
            'platform': 'cortex-m4',
            'optimization': 'balanced'
        }
        
        metrics = suite.benchmark_algorithm(parameters)
        
        # Verify required metrics exist
        required_metrics = [
            'keygen_cycles', 'keygen_time_us',
            'sign_cycles', 'sign_time_us', 
            'verify_cycles', 'verify_time_us',
            'public_key_bytes', 'private_key_bytes'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert metrics[metric] >= 0
    
    def test_statistical_analysis(self):
        """Test statistical analysis of experimental results."""
        from pqc_iot_retrofit.autonomous_research import (
            StatisticalAnalyzer, ExperimentResult, ResearchHypothesis
        )
        
        analyzer = StatisticalAnalyzer()
        
        # Create mock experimental results
        results = []
        for i in range(20):
            result = ExperimentResult(
                experiment_id="test_exp",
                run_id=f"run_{i}",
                parameters={'algorithm': 'Dilithium2', 'platform': 'cortex-m4'},
                measurements={
                    'execution_cycles': np.random.normal(100000, 10000),
                    'memory_usage': np.random.normal(15000, 1500)
                },
                execution_time=np.random.uniform(0.1, 1.0),
                success=True,
                error_message=None,
                raw_data={},
                timestamp=time.time()
            )
            results.append(result)
        
        # Create mock hypothesis
        hypothesis = ResearchHypothesis(
            hypothesis_id="test_hypothesis",
            objective=ResearchObjective.PERFORMANCE_BENCHMARKING,
            null_hypothesis="No performance difference",
            alternative_hypothesis="Performance difference exists",
            independent_variables=["algorithm"],
            dependent_variables=["execution_cycles", "memory_usage"],
            expected_outcome="Better performance",
            confidence_level=0.95,
            sample_size_required=20,
            experiment_design={},
            success_criteria=[],
            created_timestamp=time.time()
        )
        
        analyses = analyzer.analyze_experiment_results(results, hypothesis)
        
        assert len(analyses) > 0
        for analysis in analyses:
            assert analysis.analysis_id is not None
            assert analysis.hypothesis_tested is not None
            assert isinstance(analysis.p_value, float)
            assert 0 <= analysis.p_value <= 1
    
    def test_autonomous_researcher_integration(self):
        """Test autonomous research system integration."""
        researcher = AutonomousResearcher()
        
        # Test with limited scope for faster testing
        focus_areas = [ResearchObjective.PERFORMANCE_BENCHMARKING]
        
        # Mock the time-intensive research process
        with patch.object(researcher.experimental_framework, 'conduct_experiment') as mock_experiment:
            # Mock experimental results
            mock_results = [
                self._create_mock_experiment_result(i) for i in range(5)
            ]
            mock_experiment.return_value = mock_results
            
            research_results = researcher.conduct_autonomous_research(focus_areas)
            
            assert 'research_results' in research_results
            assert 'summary' in research_results
            assert 'metadata' in research_results
            
            # Verify metadata
            metadata = research_results['metadata']
            assert metadata['total_experiments'] >= 0
            assert metadata['successful_experiments'] >= 0
            assert len(metadata['focus_areas']) == len(focus_areas)
    
    def _create_mock_experiment_result(self, run_id):
        """Create mock experimental result."""
        from pqc_iot_retrofit.autonomous_research import ExperimentResult
        
        return ExperimentResult(
            experiment_id="mock_exp",
            run_id=f"mock_run_{run_id}",
            parameters={'algorithm': 'Dilithium2', 'platform': 'cortex-m4'},
            measurements={
                'execution_cycles': np.random.normal(100000, 10000),
                'memory_usage': np.random.normal(15000, 1500)
            },
            execution_time=0.5,
            success=True,
            error_message=None,
            raw_data={},
            timestamp=time.time()
        )


class TestGeneration4Integration:
    """Integration tests for Generation 4 features."""
    
    def test_end_to_end_ai_analysis(self):
        """Test complete end-to-end AI analysis pipeline."""
        ai = AdaptiveAI()
        
        # Generate test firmware
        firmware_data = self._generate_complex_firmware()
        context = {
            'architecture': 'cortex-m4',
            'file_size': len(firmware_data),
            'expected_algorithms': ['RSA', 'AES']
        }
        
        # Run complete analysis
        analysis = ai.analyze_firmware(firmware_data, context)
        
        # Verify comprehensive analysis
        assert 'firmware_fingerprint' in analysis
        assert 'ensemble_detection' in analysis
        assert 'anomaly_analysis' in analysis
        assert 'optimized_patches' in analysis
        
        # Check for realistic fingerprint values
        fingerprint = analysis['firmware_fingerprint']
        assert 0 < fingerprint['entropy'] < 8
        assert 0 < fingerprint['instruction_density'] < 1
        assert 0 < fingerprint['string_density'] < 1
    
    def test_quantum_ai_integration(self):
        """Test integration between AI and quantum resilience analysis."""
        ai = AdaptiveAI()
        quantum_analyzer = QuantumResilienceAnalyzer()
        
        # AI detects vulnerabilities
        firmware_data = self._generate_complex_firmware()
        ai_analysis = ai.analyze_firmware(firmware_data)
        
        # Extract vulnerabilities for quantum analysis
        vulnerabilities = []
        for detection in ai_analysis['ensemble_detection']['detections']:
            # Convert AI detection to vulnerability format (simplified)
            vuln_data = detection['vulnerability']
            vuln = CryptoVulnerability(
                algorithm=CryptoAlgorithm(vuln_data['algorithm']),
                address=int(vuln_data['address'], 16),
                function_name=vuln_data['function_name'],
                risk_level=RiskLevel(vuln_data['risk_level']),
                key_size=vuln_data.get('key_size'),
                description=vuln_data['description'],
                mitigation=vuln_data['mitigation'],
                stack_usage=vuln_data['memory_impact']['stack_usage'],
                available_stack=vuln_data['memory_impact']['available_stack']
            )
            vulnerabilities.append(vuln)
        
        # Quantum resilience analysis
        if vulnerabilities:
            assessment = quantum_analyzer.assess_system_resilience(vulnerabilities)
            
            assert assessment is not None
            assert 0 <= assessment.overall_resilience_score <= 1
    
    def test_research_ai_integration(self):
        """Test integration between research and AI systems."""
        researcher = AutonomousResearcher()
        ai = AdaptiveAI()
        
        # Mock research focusing on AI optimization
        with patch.object(researcher, '_generate_research_hypotheses') as mock_hypotheses:
            from pqc_iot_retrofit.autonomous_research import ResearchHypothesis
            
            mock_hypothesis = ResearchHypothesis(
                hypothesis_id="ai_integration_test",
                objective=ResearchObjective.ALGORITHM_OPTIMIZATION,
                null_hypothesis="AI optimization has no effect",
                alternative_hypothesis="AI optimization improves performance",
                independent_variables=["ai_enabled", "optimization_level"],
                dependent_variables=["performance_gain", "accuracy"],
                expected_outcome="Improved performance with AI",
                confidence_level=0.95,
                sample_size_required=10,
                experiment_design={'type': 'factorial'},
                success_criteria=["p < 0.05"],
                created_timestamp=time.time()
            )
            
            mock_hypotheses.return_value = [mock_hypothesis]
            
            # Run research
            results = researcher.conduct_autonomous_research([ResearchObjective.ALGORITHM_OPTIMIZATION])
            
            assert 'research_results' in results
            assert ResearchObjective.ALGORITHM_OPTIMIZATION.value in results['research_results']
    
    def test_performance_under_load(self):
        """Test system performance under heavy load."""
        ai = AdaptiveAI()
        
        # Generate multiple large firmware samples
        firmware_samples = [
            self._generate_complex_firmware(50000) for _ in range(5)
        ]
        
        start_time = time.time()
        
        analyses = []
        for firmware in firmware_samples:
            analysis = ai.analyze_firmware(firmware)
            analyses.append(analysis)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert len(analyses) == len(firmware_samples)
        assert total_time < 60  # Should complete within 1 minute
        assert all('ai_metadata' in analysis for analysis in analyses)
        
        # Check that analysis times are reasonable
        for analysis in analyses:
            analysis_time = analysis['ai_metadata']['analysis_time_seconds']
            assert analysis_time < 15  # Each analysis should take < 15 seconds
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency of Generation 4 features."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Initialize all Generation 4 components
        ai = AdaptiveAI()
        quantum_analyzer = QuantumResilienceAnalyzer()
        researcher = AutonomousResearcher()
        
        # Perform analyses
        firmware_data = self._generate_complex_firmware()
        ai.analyze_firmware(firmware_data)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable (< 500MB increase)
        assert memory_increase < 500 * 1024 * 1024  # 500MB
    
    def test_error_handling_robustness(self):
        """Test error handling across Generation 4 components."""
        ai = AdaptiveAI()
        
        # Test with malformed firmware data
        malformed_data = b"invalid firmware data"
        
        try:
            analysis = ai.analyze_firmware(malformed_data)
            # Should not crash, may return minimal analysis
            assert 'firmware_fingerprint' in analysis
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            assert isinstance(e, (ValueError, TypeError))
        
        # Test with empty data
        empty_analysis = ai.analyze_firmware(b"")
        assert 'firmware_fingerprint' in empty_analysis
        assert empty_analysis['firmware_fingerprint']['size_bytes'] == 0
    
    def _generate_complex_firmware(self, size=10000):
        """Generate complex synthetic firmware for testing."""
        firmware = bytearray()
        
        # Add realistic firmware sections
        
        # 1. Vector table (ARM Cortex-M)
        for _ in range(64):  # 64 interrupt vectors
            firmware.extend(np.random.randint(0x08000000, 0x08010000, dtype=np.uint32).tobytes())
        
        # 2. Code section with ARM Thumb instructions
        code_size = size // 3
        for _ in range(0, code_size, 2):
            # Generate realistic ARM Thumb instruction patterns
            if np.random.random() < 0.1:  # Branch instructions
                instr = np.random.randint(0xD000, 0xD100, dtype=np.uint16)
            elif np.random.random() < 0.2:  # Load/Store
                instr = np.random.randint(0x6000, 0x6FFF, dtype=np.uint16)
            else:  # Data processing
                instr = np.random.randint(0x4000, 0x4FFF, dtype=np.uint16)
            
            firmware.extend(instr.tobytes())
        
        # 3. Data section with strings and constants
        strings = [b'RSA_SIGN', b'ECDSA_VERIFY', b'AES_ENCRYPT', b'SHA256_HASH']
        for _ in range(size // 20):
            if np.random.random() < 0.05:  # 5% crypto strings
                firmware.extend(np.random.choice(strings))
            else:
                # Random printable characters
                firmware.extend(np.random.randint(32, 126, size=np.random.randint(4, 16), dtype=np.uint8).tobytes())
            firmware.extend(b'\x00')  # Null terminator
        
        # 4. BSS section (zeros)
        bss_size = size // 10
        firmware.extend(b'\x00' * bss_size)
        
        # 5. Random data to fill remaining space
        remaining = max(0, size - len(firmware))
        firmware.extend(np.random.bytes(remaining))
        
        return bytes(firmware[:size])


# Performance benchmarking tests
class TestGeneration4Performance:
    """Performance tests for Generation 4 features."""
    
    def test_ai_analysis_performance(self):
        """Benchmark AI analysis performance."""
        ai = AdaptiveAI()
        
        # Test different firmware sizes
        sizes = [1000, 10000, 100000]  # 1KB, 10KB, 100KB
        performance_data = []
        
        for size in sizes:
            firmware_data = np.random.bytes(size)
            
            start_time = time.time()
            analysis = ai.analyze_firmware(firmware_data)
            analysis_time = time.time() - start_time
            
            performance_data.append({
                'size': size,
                'time': analysis_time,
                'throughput': size / analysis_time  # bytes per second
            })
            
            # Performance assertions
            assert analysis_time < 5.0  # Should complete in < 5 seconds
            assert analysis['ai_metadata']['analysis_time_seconds'] > 0
        
        # Throughput should be reasonable
        avg_throughput = np.mean([d['throughput'] for d in performance_data])
        assert avg_throughput > 1000  # At least 1KB/second
    
    def test_quantum_analysis_performance(self):
        """Benchmark quantum analysis performance."""
        quantum_analyzer = QuantumResilienceAnalyzer()
        
        # Create varying numbers of vulnerabilities
        vuln_counts = [1, 5, 10, 20]
        
        for count in vuln_counts:
            vulnerabilities = [
                self._create_test_vulnerability(CryptoAlgorithm.RSA_2048)
                for _ in range(count)
            ]
            
            start_time = time.time()
            assessment = quantum_analyzer.assess_system_resilience(vulnerabilities)
            analysis_time = time.time() - start_time
            
            # Performance assertions
            assert analysis_time < 2.0  # Should complete quickly
            assert assessment is not None
    
    def test_research_framework_performance(self):
        """Benchmark research framework performance."""
        from pqc_iot_retrofit.autonomous_research import BenchmarkingSuite
        
        suite = BenchmarkingSuite()
        
        # Benchmark different algorithms
        algorithms = ['Dilithium2', 'Kyber512', 'Falcon-512']
        
        for algorithm in algorithms:
            parameters = {
                'algorithm': algorithm,
                'platform': 'cortex-m4',
                'optimization': 'balanced'
            }
            
            start_time = time.time()
            metrics = suite.benchmark_algorithm(parameters)
            benchmark_time = time.time() - start_time
            
            # Performance assertions
            assert benchmark_time < 1.0  # Should complete quickly
            assert len(metrics) > 5  # Should return multiple metrics
    
    def _create_test_vulnerability(self, algorithm):
        """Create test vulnerability for performance testing."""
        return CryptoVulnerability(
            algorithm=algorithm,
            address=0x08001000,
            function_name=f"test_{algorithm.value.lower()}",
            risk_level=RiskLevel.HIGH,
            key_size=2048 if 'RSA' in algorithm.value else 256,
            description=f"Test {algorithm.value}",
            mitigation="Replace with PQC",
            stack_usage=256,
            available_stack=32768
        )


# Fixture for test data
@pytest.fixture
def sample_firmware():
    """Provide sample firmware data for testing."""
    return np.random.bytes(10000)


@pytest.fixture
def test_vulnerabilities():
    """Provide test vulnerability data."""
    return [
        CryptoVulnerability(
            algorithm=CryptoAlgorithm.RSA_2048,
            address=0x08001000,
            function_name="rsa_sign",
            risk_level=RiskLevel.CRITICAL,
            key_size=2048,
            description="RSA signature function",
            mitigation="Replace with Dilithium2",
            stack_usage=512,
            available_stack=32768
        ),
        CryptoVulnerability(
            algorithm=CryptoAlgorithm.ECDSA_P256,
            address=0x08002000,
            function_name="ecdsa_verify",
            risk_level=RiskLevel.HIGH,
            key_size=256,
            description="ECDSA verification function", 
            mitigation="Replace with Dilithium2",
            stack_usage=256,
            available_stack=32768
        )
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])