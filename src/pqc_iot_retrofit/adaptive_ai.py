"""Generation 4: Adaptive AI Module for PQC IoT Retrofit Scanner.

Advanced machine learning-based vulnerability detection and patch optimization
using ensemble methods, anomaly detection, and continuous learning capabilities.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle
import hashlib
import time
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from .scanner import CryptoVulnerability, RiskLevel, CryptoAlgorithm
from .error_handling import handle_errors, ValidationError
from .monitoring import track_performance, metrics_collector


class AIModelType(Enum):
    """Types of AI models available."""
    VULNERABILITY_CLASSIFIER = "vulnerability_classifier"
    RISK_PREDICTOR = "risk_predictor"
    PATCH_OPTIMIZER = "patch_optimizer"
    ANOMALY_DETECTOR = "anomaly_detector"
    PERFORMANCE_PREDICTOR = "performance_predictor"


class LearningStrategy(Enum):
    """Learning strategy for model updates."""
    ONLINE = "online"          # Real-time learning
    BATCH = "batch"            # Periodic batch learning
    FEDERATED = "federated"    # Distributed learning
    HYBRID = "hybrid"          # Combination approach


@dataclass
class FirmwareFingerprint:
    """Unique fingerprint for firmware analysis."""
    size_bytes: int
    entropy: float
    architecture: str
    instruction_density: float
    string_density: float
    crypto_density: float
    function_count: int
    section_count: int
    compiler_signature: Optional[str]
    build_timestamp: Optional[int]
    checksum: str


@dataclass
class VulnerabilityPattern:
    """Pattern learned from vulnerability detection."""
    pattern_id: str
    algorithm_type: CryptoAlgorithm
    instruction_sequence: List[str]
    memory_layout: Dict[str, int]
    confidence: float
    false_positive_rate: float
    detection_count: int
    last_seen: float
    context_features: Dict[str, Any]


@dataclass
class AdaptivePatch:
    """AI-optimized patch with adaptive parameters."""
    patch_id: str
    target_vulnerability: CryptoVulnerability
    algorithm_replacement: str
    optimization_level: str
    memory_efficiency: float
    performance_gain: float
    security_level: int
    success_probability: float
    resource_requirements: Dict[str, int]
    deployment_metadata: Dict[str, Any]


class EnsembleDetector:
    """Ensemble-based vulnerability detection system."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_history = defaultdict(list)
        self.confidence_threshold = 0.7
        self.logger = logging.getLogger(__name__)
    
    def add_detector(self, name: str, detector_func, weight: float = 1.0):
        """Add a detection model to the ensemble."""
        self.models[name] = detector_func
        self.weights[name] = weight
        self.performance_history[name] = deque(maxlen=1000)
    
    def detect(self, firmware_data: bytes, context: Dict[str, Any]) -> List[Tuple[CryptoVulnerability, float]]:
        """Run ensemble detection on firmware."""
        all_detections = []
        model_predictions = {}
        
        # Run all models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(model, firmware_data, context): name 
                for name, model in self.models.items()
            }
            
            for future in futures:
                name = futures[future]
                try:
                    detections = future.result(timeout=30)
                    model_predictions[name] = detections
                except Exception as e:
                    self.logger.warning(f"Model {name} failed: {e}")
                    model_predictions[name] = []
        
        # Aggregate predictions using weighted voting
        vulnerability_scores = defaultdict(lambda: defaultdict(float))
        
        for model_name, detections in model_predictions.items():
            weight = self.weights[model_name]
            for vuln, confidence in detections:
                vuln_key = (vuln.algorithm, vuln.address, vuln.function_name)
                vulnerability_scores[vuln_key]['score'] += confidence * weight
                vulnerability_scores[vuln_key]['count'] += 1
                if 'vuln' not in vulnerability_scores[vuln_key]:
                    vulnerability_scores[vuln_key]['vuln'] = vuln
        
        # Convert to final detections
        for vuln_data in vulnerability_scores.values():
            if vuln_data['score'] >= self.confidence_threshold:
                normalized_confidence = min(1.0, vuln_data['score'] / sum(self.weights.values()))
                all_detections.append((vuln_data['vuln'], normalized_confidence))
        
        return sorted(all_detections, key=lambda x: x[1], reverse=True)
    
    def update_performance(self, model_name: str, accuracy: float):
        """Update model performance metrics."""
        self.performance_history[model_name].append(accuracy)
        
        # Adaptive weight adjustment
        if len(self.performance_history[model_name]) >= 10:
            recent_performance = np.mean(list(self.performance_history[model_name])[-10:])
            self.weights[model_name] = max(0.1, min(2.0, recent_performance))


class AnomalyDetector:
    """Unsupervised anomaly detection for unknown crypto patterns."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.baseline_features = None
        self.anomaly_threshold = None
        self.feature_history = deque(maxlen=10000)
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, firmware_data: bytes) -> np.ndarray:
        """Extract statistical features from firmware."""
        # Byte frequency analysis
        byte_freqs = np.bincount(firmware_data, minlength=256) / len(firmware_data)
        
        # Entropy calculation
        entropy = -np.sum(byte_freqs[byte_freqs > 0] * np.log2(byte_freqs[byte_freqs > 0]))
        
        # N-gram analysis (2-gram and 3-gram)
        bigram_counts = defaultdict(int)
        trigram_counts = defaultdict(int)
        
        for i in range(len(firmware_data) - 2):
            bigram = tuple(firmware_data[i:i+2])
            trigram = tuple(firmware_data[i:i+3])
            bigram_counts[bigram] += 1
            trigram_counts[trigram] += 1
        
        bigram_entropy = -sum((count/len(bigram_counts)) * np.log2(count/len(bigram_counts)) 
                             for count in bigram_counts.values())
        trigram_entropy = -sum((count/len(trigram_counts)) * np.log2(count/len(trigram_counts)) 
                              for count in trigram_counts.values())
        
        # Statistical moments
        data_float = firmware_data.astype(np.float64)
        mean_val = np.mean(data_float)
        std_val = np.std(data_float)
        skew_val = np.mean(((data_float - mean_val) / std_val) ** 3) if std_val > 0 else 0
        kurtosis_val = np.mean(((data_float - mean_val) / std_val) ** 4) if std_val > 0 else 0
        
        # Combine features
        features = np.array([
            entropy, bigram_entropy, trigram_entropy,
            mean_val, std_val, skew_val, kurtosis_val,
            *byte_freqs[:32]  # Top 32 byte frequencies
        ])
        
        return features
    
    def fit_baseline(self, firmware_samples: List[bytes]):
        """Establish baseline from known-good firmware samples."""
        feature_matrix = np.array([self.extract_features(fw) for fw in firmware_samples])
        
        # Calculate baseline statistics
        self.baseline_features = {
            'mean': np.mean(feature_matrix, axis=0),
            'std': np.std(feature_matrix, axis=0),
            'min': np.min(feature_matrix, axis=0),
            'max': np.max(feature_matrix, axis=0)
        }
        
        # Set anomaly threshold using Mahalanobis distance
        covariance = np.cov(feature_matrix.T)
        try:
            inv_cov = np.linalg.inv(covariance + np.eye(covariance.shape[0]) * 1e-6)
            
            # Calculate distances for baseline samples
            distances = []
            for features in feature_matrix:
                diff = features - self.baseline_features['mean']
                distance = np.sqrt(diff.T @ inv_cov @ diff)
                distances.append(distance)
            
            self.anomaly_threshold = np.percentile(distances, (1 - self.contamination) * 100)
            
        except np.linalg.LinAlgError:
            # Fallback to simpler threshold
            self.anomaly_threshold = 3.0  # 3-sigma rule
        
        self.logger.info(f"Established anomaly baseline with threshold {self.anomaly_threshold:.3f}")
    
    def detect_anomalies(self, firmware_data: bytes) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect if firmware contains anomalous patterns."""
        if self.baseline_features is None:
            raise ValueError("Must call fit_baseline() first")
        
        features = self.extract_features(firmware_data)
        
        # Normalized deviation from baseline
        normalized_diff = (features - self.baseline_features['mean']) / (self.baseline_features['std'] + 1e-8)
        anomaly_score = np.linalg.norm(normalized_diff)
        
        is_anomalous = anomaly_score > self.anomaly_threshold
        
        # Detailed analysis
        analysis = {
            'anomaly_score': float(anomaly_score),
            'threshold': float(self.anomaly_threshold),
            'is_anomalous': is_anomalous,
            'most_anomalous_features': self._get_top_anomalous_features(normalized_diff),
            'confidence': min(1.0, anomaly_score / (self.anomaly_threshold * 2))
        }
        
        return is_anomalous, anomaly_score, analysis
    
    def _get_top_anomalous_features(self, normalized_diff: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Identify the most anomalous feature dimensions."""
        feature_names = [
            'entropy', 'bigram_entropy', 'trigram_entropy', 
            'mean', 'std', 'skew', 'kurtosis'
        ] + [f'byte_freq_{i}' for i in range(32)]
        
        abs_diff = np.abs(normalized_diff)
        top_indices = np.argsort(abs_diff)[-top_k:][::-1]
        
        return [
            {
                'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                'deviation': float(normalized_diff[i]),
                'abs_deviation': float(abs_diff[i])
            }
            for i in top_indices
        ]


class AdaptiveOptimizer:
    """AI-driven patch optimization system."""
    
    def __init__(self):
        self.optimization_history = defaultdict(list)
        self.performance_models = {}
        self.constraint_solver = ConstraintSolver()
        self.logger = logging.getLogger(__name__)
    
    @track_performance("adaptive_patch_optimization")
    def optimize_patch(self, vulnerability: CryptoVulnerability, 
                      target_constraints: Dict[str, Any]) -> AdaptivePatch:
        """Generate optimized patch using AI-driven analysis."""
        
        # Analyze historical performance
        similar_cases = self._find_similar_cases(vulnerability)
        
        # Multi-objective optimization
        optimization_objectives = {
            'memory_efficiency': 0.3,
            'performance_gain': 0.4,
            'security_level': 0.2,
            'deployment_ease': 0.1
        }
        
        # Generate candidate patches
        candidates = self._generate_patch_candidates(vulnerability, target_constraints)
        
        # Score candidates
        best_patch = None
        best_score = -1
        
        for candidate in candidates:
            score = self._score_patch(candidate, optimization_objectives, similar_cases)
            if score > best_score:
                best_score = score
                best_patch = candidate
        
        # Record optimization result
        self.optimization_history[vulnerability.algorithm].append({
            'patch': best_patch,
            'score': best_score,
            'timestamp': time.time()
        })
        
        return best_patch
    
    def _find_similar_cases(self, vulnerability: CryptoVulnerability) -> List[Dict[str, Any]]:
        """Find historically similar vulnerabilities for learning."""
        algorithm_history = self.optimization_history.get(vulnerability.algorithm, [])
        
        similar_cases = []
        for case in algorithm_history[-50:]:  # Last 50 cases
            similarity = self._calculate_similarity(vulnerability, case['patch'].target_vulnerability)
            if similarity > 0.7:  # 70% similarity threshold
                similar_cases.append({
                    'case': case,
                    'similarity': similarity
                })
        
        return sorted(similar_cases, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_similarity(self, vuln1: CryptoVulnerability, vuln2: CryptoVulnerability) -> float:
        """Calculate similarity between vulnerabilities."""
        score = 0.0
        
        # Algorithm match
        if vuln1.algorithm == vuln2.algorithm:
            score += 0.4
        
        # Key size similarity
        if vuln1.key_size and vuln2.key_size:
            size_ratio = min(vuln1.key_size, vuln2.key_size) / max(vuln1.key_size, vuln2.key_size)
            score += 0.2 * size_ratio
        
        # Risk level match
        if vuln1.risk_level == vuln2.risk_level:
            score += 0.2
        
        # Memory constraints similarity
        if vuln1.stack_usage and vuln2.stack_usage:
            memory_ratio = min(vuln1.stack_usage, vuln2.stack_usage) / max(vuln1.stack_usage, vuln2.stack_usage)
            score += 0.2 * memory_ratio
        
        return score
    
    def _generate_patch_candidates(self, vulnerability: CryptoVulnerability, 
                                 constraints: Dict[str, Any]) -> List[AdaptivePatch]:
        """Generate multiple patch candidates for optimization."""
        candidates = []
        
        # Determine replacement algorithms
        replacements = self._get_pqc_replacements(vulnerability.algorithm)
        
        for replacement in replacements:
            for optimization_level in ['size', 'speed', 'balanced', 'memory']:
                for security_level in [1, 2, 3]:
                    if self._meets_constraints(replacement, optimization_level, security_level, constraints):
                        patch = AdaptivePatch(
                            patch_id=f"{replacement}_{optimization_level}_L{security_level}_{int(time.time())}",
                            target_vulnerability=vulnerability,
                            algorithm_replacement=replacement,
                            optimization_level=optimization_level,
                            memory_efficiency=self._estimate_memory_efficiency(replacement, optimization_level),
                            performance_gain=self._estimate_performance_gain(replacement, optimization_level),
                            security_level=security_level,
                            success_probability=self._estimate_success_probability(replacement, vulnerability),
                            resource_requirements=self._estimate_resources(replacement, optimization_level, security_level),
                            deployment_metadata=self._generate_deployment_metadata(replacement, constraints)
                        )
                        candidates.append(patch)
        
        return candidates
    
    def _get_pqc_replacements(self, algorithm: CryptoAlgorithm) -> List[str]:
        """Get appropriate PQC replacements for classical algorithm."""
        replacements = {
            CryptoAlgorithm.RSA_1024: ['Dilithium2', 'Falcon-512'],
            CryptoAlgorithm.RSA_2048: ['Dilithium3', 'Falcon-1024'],
            CryptoAlgorithm.RSA_4096: ['Dilithium5'],
            CryptoAlgorithm.ECDSA_P256: ['Dilithium2', 'Falcon-512'],
            CryptoAlgorithm.ECDSA_P384: ['Dilithium3', 'Falcon-1024'],
            CryptoAlgorithm.ECDH_P256: ['Kyber512', 'NTRU'],
            CryptoAlgorithm.ECDH_P384: ['Kyber768', 'NTRU'],
            CryptoAlgorithm.DH_1024: ['Kyber512'],
            CryptoAlgorithm.DH_2048: ['Kyber768', 'Kyber1024'],
        }
        return replacements.get(algorithm, ['Dilithium2', 'Kyber512'])
    
    def _score_patch(self, patch: AdaptivePatch, objectives: Dict[str, float], 
                     similar_cases: List[Dict[str, Any]]) -> float:
        """Score patch candidate using multi-objective optimization."""
        score = 0.0
        
        # Objective-based scoring
        score += objectives['memory_efficiency'] * patch.memory_efficiency
        score += objectives['performance_gain'] * patch.performance_gain
        score += objectives['security_level'] * (patch.security_level / 5.0)
        score += objectives['deployment_ease'] * patch.success_probability
        
        # Historical performance bonus
        if similar_cases:
            avg_historical_score = np.mean([case['case']['score'] for case in similar_cases])
            historical_bonus = 0.1 * avg_historical_score
            score += historical_bonus
        
        # Constraint penalty
        penalty = self._calculate_constraint_penalty(patch)
        score -= penalty
        
        return max(0.0, min(1.0, score))
    
    def _estimate_memory_efficiency(self, algorithm: str, optimization: str) -> float:
        """Estimate memory efficiency for algorithm/optimization combination."""
        base_efficiency = {
            'Dilithium2': 0.85, 'Dilithium3': 0.75, 'Dilithium5': 0.65,
            'Falcon-512': 0.90, 'Falcon-1024': 0.80,
            'Kyber512': 0.95, 'Kyber768': 0.85, 'Kyber1024': 0.75,
            'NTRU': 0.80
        }
        
        optimization_multiplier = {
            'memory': 1.2, 'size': 1.1, 'balanced': 1.0, 'speed': 0.9
        }
        
        return min(1.0, base_efficiency.get(algorithm, 0.7) * optimization_multiplier.get(optimization, 1.0))
    
    def _estimate_performance_gain(self, algorithm: str, optimization: str) -> float:
        """Estimate performance improvement."""
        base_performance = {
            'Dilithium2': 0.80, 'Dilithium3': 0.75, 'Dilithium5': 0.70,
            'Falcon-512': 0.85, 'Falcon-1024': 0.80,
            'Kyber512': 0.90, 'Kyber768': 0.85, 'Kyber1024': 0.80,
            'NTRU': 0.75
        }
        
        optimization_multiplier = {
            'speed': 1.3, 'balanced': 1.1, 'size': 1.0, 'memory': 0.9
        }
        
        return min(1.0, base_performance.get(algorithm, 0.7) * optimization_multiplier.get(optimization, 1.0))
    
    def _estimate_success_probability(self, algorithm: str, vulnerability: CryptoVulnerability) -> float:
        """Estimate patch deployment success probability."""
        base_probability = 0.85
        
        # Adjust based on algorithm maturity
        maturity_bonus = {
            'Dilithium2': 0.1, 'Dilithium3': 0.1, 'Dilithium5': 0.05,
            'Kyber512': 0.1, 'Kyber768': 0.08, 'Kyber1024': 0.05
        }
        
        # Risk level adjustment
        risk_penalty = {
            RiskLevel.CRITICAL: 0.0,
            RiskLevel.HIGH: 0.05,
            RiskLevel.MEDIUM: 0.1,
            RiskLevel.LOW: 0.15
        }
        
        probability = base_probability + maturity_bonus.get(algorithm, 0.0) - risk_penalty.get(vulnerability.risk_level, 0.0)
        return max(0.1, min(1.0, probability))
    
    def _meets_constraints(self, algorithm: str, optimization: str, security_level: int, 
                          constraints: Dict[str, Any]) -> bool:
        """Check if patch configuration meets constraints."""
        if not constraints:
            return True
        
        # Memory constraints
        if 'max_memory' in constraints:
            estimated_memory = self._estimate_memory_usage(algorithm, optimization, security_level)
            if estimated_memory > constraints['max_memory']:
                return False
        
        # Performance constraints
        if 'min_performance' in constraints:
            estimated_performance = self._estimate_performance_gain(algorithm, optimization)
            if estimated_performance < constraints['min_performance']:
                return False
        
        return True
    
    def _estimate_memory_usage(self, algorithm: str, optimization: str, security_level: int) -> int:
        """Estimate memory usage in bytes."""
        base_memory = {
            'Dilithium2': 12000, 'Dilithium3': 18000, 'Dilithium5': 32000,
            'Falcon-512': 8000, 'Falcon-1024': 14000,
            'Kyber512': 6000, 'Kyber768': 9000, 'Kyber1024': 12000,
            'NTRU': 10000
        }
        
        optimization_multiplier = {
            'memory': 0.8, 'size': 0.9, 'balanced': 1.0, 'speed': 1.2
        }
        
        security_multiplier = 1.0 + (security_level - 1) * 0.1
        
        return int(base_memory.get(algorithm, 15000) * 
                  optimization_multiplier.get(optimization, 1.0) * 
                  security_multiplier)
    
    def _estimate_resources(self, algorithm: str, optimization: str, security_level: int) -> Dict[str, int]:
        """Estimate resource requirements."""
        return {
            'flash_bytes': self._estimate_memory_usage(algorithm, optimization, security_level),
            'ram_bytes': self._estimate_memory_usage(algorithm, optimization, security_level) // 4,
            'cpu_cycles': int(50000 * (1.0 + security_level * 0.2)),
            'stack_bytes': int(2048 * (1.0 + security_level * 0.15))
        }
    
    def _generate_deployment_metadata(self, algorithm: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment metadata."""
        return {
            'algorithm_family': 'lattice' if 'Dilithium' in algorithm or 'Kyber' in algorithm else 'other',
            'implementation_complexity': 'medium',
            'hardware_requirements': ['fpu'] if 'Falcon' in algorithm else [],
            'estimated_integration_time': '2-4 weeks',
            'testing_requirements': ['functional', 'performance', 'security'],
            'rollback_plan': 'hybrid_mode_30_days'
        }
    
    def _calculate_constraint_penalty(self, patch: AdaptivePatch) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0
        
        # Resource constraint penalties
        if patch.resource_requirements['ram_bytes'] > 64000:  # 64KB limit
            penalty += 0.2
        
        if patch.resource_requirements['flash_bytes'] > 256000:  # 256KB limit
            penalty += 0.2
        
        if patch.success_probability < 0.7:  # Low success probability
            penalty += 0.3
        
        return penalty


class ConstraintSolver:
    """Constraint satisfaction solver for patch optimization."""
    
    def __init__(self):
        self.constraints = []
        self.variables = {}
        self.logger = logging.getLogger(__name__)
    
    def add_constraint(self, constraint_func, description: str):
        """Add a constraint function."""
        self.constraints.append({
            'func': constraint_func,
            'description': description
        })
    
    def solve(self, variables: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Solve constraint satisfaction problem."""
        self.variables = variables.copy()
        
        violations = []
        for constraint in self.constraints:
            try:
                if not constraint['func'](self.variables):
                    violations.append(constraint['description'])
            except Exception as e:
                violations.append(f"Constraint evaluation error: {e}")
        
        is_satisfied = len(violations) == 0
        
        result = {
            'satisfied': is_satisfied,
            'violations': violations,
            'variables': self.variables
        }
        
        return is_satisfied, result


class AdaptiveAI:
    """Main adaptive AI system coordinating all components."""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("./models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.ensemble_detector = EnsembleDetector()
        self.anomaly_detector = AnomalyDetector()
        self.adaptive_optimizer = AdaptiveOptimizer()
        
        self.learning_strategy = LearningStrategy.HYBRID
        self.model_versions = {}
        self.performance_metrics = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default detectors
        self._initialize_default_detectors()
        
        # Load saved models
        self._load_models()
    
    def _initialize_default_detectors(self):
        """Initialize default detection models."""
        
        def pattern_detector(firmware_data: bytes, context: Dict[str, Any]) -> List[Tuple[CryptoVulnerability, float]]:
            """Pattern-based detection model."""
            # Simplified pattern detection
            detections = []
            
            # Look for RSA patterns
            if b'RSA' in firmware_data:
                vuln = CryptoVulnerability(
                    algorithm=CryptoAlgorithm.RSA_2048,
                    address=firmware_data.find(b'RSA'),
                    function_name="rsa_pattern_detected",
                    risk_level=RiskLevel.CRITICAL,
                    key_size=2048,
                    description="RSA pattern detected",
                    mitigation="Replace with Dilithium",
                    stack_usage=256,
                    available_stack=32768
                )
                detections.append((vuln, 0.8))
            
            return detections
        
        def entropy_detector(firmware_data: bytes, context: Dict[str, Any]) -> List[Tuple[CryptoVulnerability, float]]:
            """Entropy-based detection model."""
            # High entropy regions might indicate crypto
            chunk_size = 1024
            high_entropy_threshold = 7.5
            detections = []
            
            for i in range(0, len(firmware_data), chunk_size):
                chunk = firmware_data[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    continue
                
                # Calculate entropy
                byte_counts = np.bincount(chunk)
                probabilities = byte_counts[byte_counts > 0] / len(chunk)
                entropy = -np.sum(probabilities * np.log2(probabilities))
                
                if entropy > high_entropy_threshold:
                    vuln = CryptoVulnerability(
                        algorithm=CryptoAlgorithm.ECDSA_P256,
                        address=i,
                        function_name=f"high_entropy_region_0x{i:08x}",
                        risk_level=RiskLevel.HIGH,
                        key_size=256,
                        description=f"High entropy region (entropy={entropy:.2f})",
                        mitigation="Investigate potential crypto implementation",
                        stack_usage=128,
                        available_stack=32768
                    )
                    detections.append((vuln, min(1.0, (entropy - high_entropy_threshold) / 2.0)))
            
            return detections
        
        self.ensemble_detector.add_detector("pattern", pattern_detector, 1.0)
        self.ensemble_detector.add_detector("entropy", entropy_detector, 0.7)
    
    @track_performance("adaptive_ai_analysis")
    def analyze_firmware(self, firmware_data: bytes, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive AI-powered firmware analysis."""
        context = context or {}
        start_time = time.time()
        
        self.logger.info(f"Starting adaptive AI analysis of {len(firmware_data)} bytes")
        
        # Generate firmware fingerprint
        fingerprint = self._generate_fingerprint(firmware_data, context)
        
        # Ensemble detection
        ensemble_results = self.ensemble_detector.detect(firmware_data, context)
        
        # Anomaly detection
        try:
            is_anomalous, anomaly_score, anomaly_analysis = self.anomaly_detector.detect_anomalies(firmware_data)
        except ValueError:
            # No baseline established yet
            is_anomalous, anomaly_score, anomaly_analysis = False, 0.0, {"message": "No baseline established"}
        
        # Generate optimized patches for detected vulnerabilities
        optimized_patches = []
        for vuln, confidence in ensemble_results:
            try:
                patch = self.adaptive_optimizer.optimize_patch(vuln, context)
                optimized_patches.append({
                    'vulnerability': asdict(vuln),
                    'patch': asdict(patch),
                    'detection_confidence': confidence
                })
            except Exception as e:
                self.logger.warning(f"Patch optimization failed for {vuln.function_name}: {e}")
        
        analysis_time = time.time() - start_time
        
        results = {
            'firmware_fingerprint': asdict(fingerprint),
            'ensemble_detection': {
                'vulnerabilities_found': len(ensemble_results),
                'detections': [
                    {
                        'vulnerability': asdict(vuln),
                        'confidence': confidence
                    }
                    for vuln, confidence in ensemble_results
                ]
            },
            'anomaly_analysis': anomaly_analysis,
            'optimized_patches': optimized_patches,
            'ai_metadata': {
                'analysis_time_seconds': analysis_time,
                'model_versions': self.model_versions,
                'learning_strategy': self.learning_strategy.value,
                'confidence_threshold': self.ensemble_detector.confidence_threshold
            }
        }
        
        # Record metrics
        metrics_collector.record_metric("ai_analysis.vulnerabilities_detected", len(ensemble_results), "count")
        metrics_collector.record_metric("ai_analysis.anomalies_detected", int(is_anomalous), "count")
        metrics_collector.record_metric("ai_analysis.patches_generated", len(optimized_patches), "count")
        metrics_collector.record_metric("ai_analysis.processing_time", analysis_time, "seconds")
        
        self.logger.info(f"AI analysis completed in {analysis_time:.2f}s: {len(ensemble_results)} vulnerabilities, {len(optimized_patches)} patches")
        
        return results
    
    def _generate_fingerprint(self, firmware_data: bytes, context: Dict[str, Any]) -> FirmwareFingerprint:
        """Generate unique firmware fingerprint for analysis."""
        
        # Calculate entropy
        byte_counts = np.bincount(firmware_data)
        probabilities = byte_counts[byte_counts > 0] / len(firmware_data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Estimate instruction density (rough heuristic)
        instruction_bytes = 0
        for i in range(0, len(firmware_data) - 1, 2):
            word = struct.unpack('<H', firmware_data[i:i+2])[0]
            if 0x4000 <= word <= 0xFFFF:  # ARM Thumb instruction range
                instruction_bytes += 2
        instruction_density = instruction_bytes / len(firmware_data)
        
        # String density analysis
        printable_bytes = sum(1 for b in firmware_data if 32 <= b <= 126)
        string_density = printable_bytes / len(firmware_data)
        
        # Crypto pattern density (rough estimate)
        crypto_patterns = [b'RSA', b'AES', b'SHA', b'ECDSA', b'DH']
        crypto_matches = sum(firmware_data.count(pattern) for pattern in crypto_patterns)
        crypto_density = crypto_matches / (len(firmware_data) / 1024)  # per KB
        
        # Function count estimation (very rough)
        function_count = firmware_data.count(b'\xF0\x00')  # Common ARM function prologue
        
        # Section count (rough estimate based on alignment patterns)
        section_count = max(1, firmware_data.count(b'\x00' * 16) // 4)
        
        # Checksum
        checksum = hashlib.sha256(firmware_data).hexdigest()[:16]
        
        return FirmwareFingerprint(
            size_bytes=len(firmware_data),
            entropy=entropy,
            architecture=context.get('architecture', 'unknown'),
            instruction_density=instruction_density,
            string_density=string_density,
            crypto_density=crypto_density,
            function_count=function_count,
            section_count=section_count,
            compiler_signature=None,  # Would need more sophisticated analysis
            build_timestamp=None,     # Would need binary format parsing
            checksum=checksum
        )
    
    def train_anomaly_baseline(self, firmware_samples: List[bytes]):
        """Train anomaly detection baseline on known-good firmware."""
        self.logger.info(f"Training anomaly detection baseline on {len(firmware_samples)} samples")
        self.anomaly_detector.fit_baseline(firmware_samples)
        self._save_models()
    
    def update_model_performance(self, model_name: str, accuracy: float):
        """Update model performance metrics."""
        self.ensemble_detector.update_performance(model_name, accuracy)
        self.performance_metrics[model_name].append({
            'accuracy': accuracy,
            'timestamp': time.time()
        })
    
    def _save_models(self):
        """Save trained models to disk."""
        model_state = {
            'anomaly_detector': {
                'baseline_features': self.anomaly_detector.baseline_features,
                'anomaly_threshold': self.anomaly_detector.anomaly_threshold,
                'contamination': self.anomaly_detector.contamination
            },
            'ensemble_weights': self.ensemble_detector.weights,
            'performance_history': dict(self.ensemble_detector.performance_history),
            'model_versions': self.model_versions,
            'timestamp': time.time()
        }
        
        model_file = self.model_dir / "adaptive_ai_models.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_state, f)
        
        self.logger.info(f"Models saved to {model_file}")
    
    def _load_models(self):
        """Load saved models from disk."""
        model_file = self.model_dir / "adaptive_ai_models.pkl"
        
        if not model_file.exists():
            self.logger.info("No saved models found, starting fresh")
            return
        
        try:
            with open(model_file, 'rb') as f:
                model_state = pickle.load(f)
            
            # Restore anomaly detector
            if 'anomaly_detector' in model_state:
                ad_state = model_state['anomaly_detector']
                self.anomaly_detector.baseline_features = ad_state.get('baseline_features')
                self.anomaly_detector.anomaly_threshold = ad_state.get('anomaly_threshold')
                self.anomaly_detector.contamination = ad_state.get('contamination', 0.1)
            
            # Restore ensemble weights
            if 'ensemble_weights' in model_state:
                self.ensemble_detector.weights = model_state['ensemble_weights']
            
            # Restore performance history
            if 'performance_history' in model_state:
                for model_name, history in model_state['performance_history'].items():
                    self.ensemble_detector.performance_history[model_name] = deque(history, maxlen=1000)
            
            # Restore model versions
            self.model_versions = model_state.get('model_versions', {})
            
            self.logger.info(f"Models loaded from {model_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'models_loaded': len(self.ensemble_detector.models),
            'anomaly_baseline_trained': self.anomaly_detector.baseline_features is not None,
            'learning_strategy': self.learning_strategy.value,
            'model_versions': self.model_versions,
            'performance_metrics': {
                model: {
                    'recent_accuracy': np.mean(list(self.ensemble_detector.performance_history[model])[-10:]) if self.ensemble_detector.performance_history[model] else 0.0,
                    'weight': self.ensemble_detector.weights.get(model, 0.0),
                    'sample_count': len(self.ensemble_detector.performance_history[model])
                }
                for model in self.ensemble_detector.models.keys()
            }
        }


# Global AI instance for application use
adaptive_ai = AdaptiveAI()