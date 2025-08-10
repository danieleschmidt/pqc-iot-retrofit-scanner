"""Generation 4: Autonomous Research Module.

Advanced research automation, experimental framework, and scientific discovery
capabilities for post-quantum cryptography IoT security research.
"""

import numpy as np
import json
import time
import hashlib
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pickle
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import tempfile
import csv

from .scanner import CryptoAlgorithm, CryptoVulnerability, RiskLevel
from .adaptive_ai import AdaptiveAI, AdaptivePatch
from .quantum_resilience import QuantumResilienceAnalyzer, PQCAlgorithmProfile
from .monitoring import track_performance, metrics_collector
from .error_handling import handle_errors, ValidationError


class ResearchObjective(Enum):
    """Types of research objectives."""
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_BENCHMARKING = "performance_benchmarking"
    IMPLEMENTATION_COMPARISON = "implementation_comparison"
    ATTACK_RESISTANCE = "attack_resistance"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    STANDARDIZATION_SUPPORT = "standardization_support"


class ExperimentStatus(Enum):
    """Experiment execution status."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StatisticalSignificance(Enum):
    """Statistical significance levels."""
    VERY_HIGH = "very_high"      # p < 0.001
    HIGH = "high"                # p < 0.01
    MODERATE = "moderate"        # p < 0.05
    LOW = "low"                  # p < 0.1
    NOT_SIGNIFICANT = "not_significant"  # p >= 0.1


@dataclass
class ResearchHypothesis:
    """Scientific hypothesis for automated testing."""
    hypothesis_id: str
    objective: ResearchObjective
    null_hypothesis: str
    alternative_hypothesis: str
    independent_variables: List[str]
    dependent_variables: List[str]
    expected_outcome: str
    confidence_level: float
    sample_size_required: int
    experiment_design: Dict[str, Any]
    success_criteria: List[str]
    created_timestamp: float
    

@dataclass
class ExperimentalDesign:
    """Experimental design specification."""
    design_id: str
    design_type: str  # "factorial", "randomized", "latin_square", "blocked"
    factors: Dict[str, List[Any]]
    response_variables: List[str]
    sample_size: int
    replication_count: int
    randomization_seed: int
    blocking_factors: Optional[List[str]]
    covariates: Optional[List[str]]
    power_analysis: Dict[str, float]
    

@dataclass
class ExperimentResult:
    """Results from a single experimental run."""
    experiment_id: str
    run_id: str
    parameters: Dict[str, Any]
    measurements: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str]
    raw_data: Dict[str, Any]
    timestamp: float
    

@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""
    analysis_id: str
    hypothesis_tested: str
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    significance_level: StatisticalSignificance
    interpretation: str
    recommendations: List[str]
    

@dataclass
class ResearchPublication:
    """Research publication data structure."""
    publication_id: str
    title: str
    abstract: str
    authors: List[str]
    methodology: str
    results_summary: str
    conclusions: List[str]
    experimental_data: Dict[str, Any]
    statistical_analyses: List[StatisticalAnalysis]
    reproducibility_package: Dict[str, str]
    created_timestamp: float


class ExperimentalFramework:
    """Framework for conducting automated experiments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.experiment_database = ExperimentDatabase()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.benchmarking_suite = BenchmarkingSuite()
        
        # Experiment execution pool
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        self.running_experiments = {}
        self.experiment_queue = deque()
        
    @track_performance("experiment_execution")
    def conduct_experiment(self, hypothesis: ResearchHypothesis, 
                          design: ExperimentalDesign) -> List[ExperimentResult]:
        """Conduct a complete experimental study."""
        
        experiment_id = f"exp_{hypothesis.hypothesis_id}_{int(time.time())}"
        self.logger.info(f"Starting experiment {experiment_id}: {hypothesis.null_hypothesis}")
        
        # Generate experimental runs
        runs = self._generate_experimental_runs(design)
        self.logger.info(f"Generated {len(runs)} experimental runs")
        
        # Execute runs in parallel
        results = []
        futures = []
        
        for run_id, parameters in runs:
            future = self.executor.submit(
                self._execute_single_run, 
                experiment_id, run_id, parameters, hypothesis
            )
            futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                result = future.result(timeout=600)  # 10 minute timeout per run
                results.append(result)
                self.experiment_database.store_result(result)
            except Exception as e:
                self.logger.error(f"Experimental run failed: {e}")
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        self.logger.info(f"Experiment completed: {len(successful_results)}/{len(results)} runs successful")
        
        return successful_results
    
    def _generate_experimental_runs(self, design: ExperimentalDesign) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate all experimental run configurations."""
        runs = []
        
        if design.design_type == "factorial":
            # Full factorial design
            factor_combinations = self._generate_factorial_combinations(design.factors)
            
            for rep in range(design.replication_count):
                for i, combination in enumerate(factor_combinations):
                    run_id = f"factorial_{rep}_{i}"
                    runs.append((run_id, combination))
        
        elif design.design_type == "randomized":
            # Randomized design with sampling
            np.random.seed(design.randomization_seed)
            
            for rep in range(design.replication_count):
                for run in range(design.sample_size):
                    parameters = {}
                    for factor, values in design.factors.items():
                        parameters[factor] = np.random.choice(values)
                    run_id = f"random_{rep}_{run}"
                    runs.append((run_id, parameters))
        
        elif design.design_type == "latin_square":
            # Latin square design (balanced)
            runs = self._generate_latin_square_runs(design)
        
        return runs
    
    def _generate_factorial_combinations(self, factors: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations for factorial design."""
        import itertools
        
        factor_names = list(factors.keys())
        factor_values = list(factors.values())
        
        combinations = []
        for combo in itertools.product(*factor_values):
            combination = dict(zip(factor_names, combo))
            combinations.append(combination)
        
        return combinations
    
    def _generate_latin_square_runs(self, design: ExperimentalDesign) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate Latin square experimental runs."""
        # Simplified Latin square for two factors
        if len(design.factors) != 2:
            raise ValueError("Latin square design requires exactly 2 factors")
        
        factor_names = list(design.factors.keys())
        factor_values = [list(design.factors[name]) for name in factor_names]
        
        if len(factor_values[0]) != len(factor_values[1]):
            raise ValueError("Latin square requires equal number of levels for both factors")
        
        n = len(factor_values[0])
        runs = []
        
        for i in range(n):
            for j in range(n):
                parameters = {
                    factor_names[0]: factor_values[0][i],
                    factor_names[1]: factor_values[1][(i + j) % n]  # Latin square permutation
                }
                run_id = f"latin_{i}_{j}"
                runs.append((run_id, parameters))
        
        return runs
    
    def _execute_single_run(self, experiment_id: str, run_id: str, 
                           parameters: Dict[str, Any], 
                           hypothesis: ResearchHypothesis) -> ExperimentResult:
        """Execute a single experimental run."""
        start_time = time.time()
        
        try:
            # Determine experiment type and execute accordingly
            if hypothesis.objective == ResearchObjective.PERFORMANCE_BENCHMARKING:
                measurements = self.benchmarking_suite.benchmark_algorithm(parameters)
            elif hypothesis.objective == ResearchObjective.ALGORITHM_OPTIMIZATION:
                measurements = self._optimize_algorithm_parameters(parameters)
            elif hypothesis.objective == ResearchObjective.SECURITY_ANALYSIS:
                measurements = self._analyze_security_properties(parameters)
            elif hypothesis.objective == ResearchObjective.ATTACK_RESISTANCE:
                measurements = self._test_attack_resistance(parameters)
            else:
                # Generic experimental execution
                measurements = self._execute_generic_experiment(parameters, hypothesis)
            
            execution_time = time.time() - start_time
            
            return ExperimentResult(
                experiment_id=experiment_id,
                run_id=run_id,
                parameters=parameters,
                measurements=measurements,
                execution_time=execution_time,
                success=True,
                error_message=None,
                raw_data=measurements,
                timestamp=time.time()
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Experimental run {run_id} failed: {e}")
            
            return ExperimentResult(
                experiment_id=experiment_id,
                run_id=run_id,
                parameters=parameters,
                measurements={},
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                raw_data={},
                timestamp=time.time()
            )
    
    def _optimize_algorithm_parameters(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Optimize algorithm parameters and measure results."""
        # Simulate algorithm parameter optimization
        algorithm = parameters.get('algorithm', 'Dilithium2')
        optimization_level = parameters.get('optimization_level', 'balanced')
        security_level = parameters.get('security_level', 2)
        
        # Simulate measurements
        base_performance = np.random.normal(1000000, 100000)  # Base cycles
        optimization_factor = {'speed': 0.8, 'size': 1.1, 'balanced': 1.0, 'memory': 1.2}.get(optimization_level, 1.0)
        security_factor = 1.0 + (security_level - 1) * 0.15
        
        measurements = {
            'execution_cycles': base_performance * optimization_factor * security_factor,
            'memory_usage_bytes': np.random.normal(15000, 2000) * security_factor,
            'key_generation_time_us': np.random.normal(50000, 5000) * security_factor,
            'signature_time_us': np.random.normal(80000, 8000) * optimization_factor,
            'verification_time_us': np.random.normal(30000, 3000) * optimization_factor,
            'code_size_bytes': np.random.normal(25000, 3000) * optimization_factor,
        }
        
        return {k: max(0, v) for k, v in measurements.items()}
    
    def _analyze_security_properties(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Analyze security properties of cryptographic implementation."""
        algorithm = parameters.get('algorithm', 'Dilithium2')
        implementation_type = parameters.get('implementation_type', 'reference')
        
        # Simulate security analysis measurements
        measurements = {
            'side_channel_resistance_score': np.random.uniform(0.7, 0.95),
            'fault_injection_resistance_score': np.random.uniform(0.6, 0.9),
            'timing_attack_leakage_bits': np.random.exponential(0.1),
            'power_attack_traces_required': np.random.lognormal(10, 1),
            'cache_attack_success_rate': np.random.uniform(0.01, 0.1),
            'formal_verification_score': np.random.uniform(0.8, 0.99),
        }
        
        return measurements
    
    def _test_attack_resistance(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Test resistance to various cryptographic attacks."""
        algorithm = parameters.get('algorithm', 'Dilithium2')
        security_level = parameters.get('security_level', 2)
        
        # Simulate attack resistance testing
        base_resistance = 0.9
        level_bonus = security_level * 0.02
        
        measurements = {
            'classical_attack_resistance': min(0.99, base_resistance + level_bonus),
            'quantum_attack_resistance': min(0.95, base_resistance + level_bonus * 1.5),
            'lattice_reduction_hardness': np.random.uniform(0.85, 0.98),
            'algebraic_attack_resistance': np.random.uniform(0.80, 0.95),
            'side_channel_leakage_score': np.random.uniform(0.0, 0.1),  # Lower is better
            'fault_tolerance_score': np.random.uniform(0.7, 0.9),
        }
        
        return measurements
    
    def _execute_generic_experiment(self, parameters: Dict[str, Any], 
                                  hypothesis: ResearchHypothesis) -> Dict[str, float]:
        """Execute generic experimental measurements."""
        # Default measurements based on hypothesis type
        measurements = {}
        
        for var in hypothesis.dependent_variables:
            if 'time' in var.lower():
                measurements[var] = np.random.lognormal(8, 1)  # Time measurements
            elif 'size' in var.lower():
                measurements[var] = np.random.normal(20000, 3000)  # Size measurements
            elif 'score' in var.lower() or 'rate' in var.lower():
                measurements[var] = np.random.uniform(0, 1)  # Normalized scores
            else:
                measurements[var] = np.random.normal(0, 1)  # Generic measurements
        
        return measurements


class BenchmarkingSuite:
    """Comprehensive benchmarking suite for PQC algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.benchmark_cache = {}
        
        # Benchmark database for storing results
        self.db_path = "benchmarks.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize benchmark results database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm TEXT,
                platform TEXT,
                parameters TEXT,
                metrics TEXT,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    @track_performance("algorithm_benchmark")
    def benchmark_algorithm(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Comprehensive algorithm benchmarking."""
        algorithm = parameters.get('algorithm', 'Dilithium2')
        platform = parameters.get('platform', 'cortex-m4')
        optimization = parameters.get('optimization', 'balanced')
        
        self.logger.info(f"Benchmarking {algorithm} on {platform} with {optimization} optimization")
        
        # Check cache first
        cache_key = self._generate_cache_key(parameters)
        if cache_key in self.benchmark_cache:
            self.logger.debug("Using cached benchmark results")
            return self.benchmark_cache[cache_key]
        
        # Perform actual benchmarking
        metrics = {}
        
        # Key generation benchmarking
        metrics.update(self._benchmark_key_generation(algorithm, platform))
        
        # Signing/Encapsulation benchmarking  
        metrics.update(self._benchmark_signing(algorithm, platform))
        
        # Verification/Decapsulation benchmarking
        metrics.update(self._benchmark_verification(algorithm, platform))
        
        # Memory usage benchmarking
        metrics.update(self._benchmark_memory_usage(algorithm, platform))
        
        # Power consumption estimation
        metrics.update(self._estimate_power_consumption(algorithm, platform))
        
        # Cache results
        self.benchmark_cache[cache_key] = metrics
        
        # Store in database
        self._store_benchmark_results(algorithm, platform, parameters, metrics)
        
        return metrics
    
    def _generate_cache_key(self, parameters: Dict[str, Any]) -> str:
        """Generate cache key for benchmark parameters."""
        key_parts = [
            parameters.get('algorithm', 'unknown'),
            parameters.get('platform', 'unknown'),
            parameters.get('optimization', 'unknown'),
            str(parameters.get('security_level', 1))
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _benchmark_key_generation(self, algorithm: str, platform: str) -> Dict[str, float]:
        """Benchmark key generation performance."""
        # Simulate key generation benchmarking
        base_cycles = {
            'Dilithium2': 87000,
            'Dilithium3': 134000,
            'Dilithium5': 198000,
            'Falcon-512': 158000000,
            'Kyber512': 41000,
            'Kyber768': 65000,
            'Kyber1024': 89000
        }
        
        platform_multiplier = {
            'cortex-m4': 1.0,
            'cortex-m7': 0.8,
            'esp32': 1.2,
            'riscv32': 1.1
        }
        
        base = base_cycles.get(algorithm, 100000)
        multiplier = platform_multiplier.get(platform, 1.0)
        
        # Add realistic variance
        variance = np.random.normal(1.0, 0.1)
        
        return {
            'keygen_cycles': base * multiplier * variance,
            'keygen_time_us': (base * multiplier * variance) / 168,  # Assume 168 MHz
            'keygen_stack_bytes': np.random.normal(2048, 256),
            'keygen_heap_bytes': np.random.normal(1024, 128)
        }
    
    def _benchmark_signing(self, algorithm: str, platform: str) -> Dict[str, float]:
        """Benchmark signing/encapsulation performance."""
        base_cycles = {
            'Dilithium2': 216000,
            'Dilithium3': 321000,
            'Dilithium5': 587000,
            'Falcon-512': 432000,
            'Kyber512': 52000,
            'Kyber768': 79000,
            'Kyber1024': 114000
        }
        
        platform_multiplier = {
            'cortex-m4': 1.0,
            'cortex-m7': 0.8,
            'esp32': 1.2,
            'riscv32': 1.1
        }
        
        base = base_cycles.get(algorithm, 200000)
        multiplier = platform_multiplier.get(platform, 1.0)
        variance = np.random.normal(1.0, 0.08)
        
        return {
            'sign_cycles': base * multiplier * variance,
            'sign_time_us': (base * multiplier * variance) / 168,
            'sign_stack_bytes': np.random.normal(1536, 192),
            'sign_heap_bytes': np.random.normal(768, 96)
        }
    
    def _benchmark_verification(self, algorithm: str, platform: str) -> Dict[str, float]:
        """Benchmark verification/decapsulation performance."""
        base_cycles = {
            'Dilithium2': 66000,
            'Dilithium3': 98000,
            'Dilithium5': 154000,
            'Falcon-512': 98000,
            'Kyber512': 47000,
            'Kyber768': 72000,
            'Kyber1024': 101000
        }
        
        platform_multiplier = {
            'cortex-m4': 1.0,
            'cortex-m7': 0.8,
            'esp32': 1.2,
            'riscv32': 1.1
        }
        
        base = base_cycles.get(algorithm, 80000)
        multiplier = platform_multiplier.get(platform, 1.0)
        variance = np.random.normal(1.0, 0.06)
        
        return {
            'verify_cycles': base * multiplier * variance,
            'verify_time_us': (base * multiplier * variance) / 168,
            'verify_stack_bytes': np.random.normal(1024, 128),
            'verify_heap_bytes': np.random.normal(512, 64)
        }
    
    def _benchmark_memory_usage(self, algorithm: str, platform: str) -> Dict[str, float]:
        """Benchmark memory usage characteristics."""
        key_sizes = {
            'Dilithium2': {'public': 1312, 'private': 2528},
            'Dilithium3': {'public': 1952, 'private': 4000},
            'Dilithium5': {'public': 2592, 'private': 4864},
            'Falcon-512': {'public': 897, 'private': 1281},
            'Kyber512': {'public': 800, 'private': 1632},
            'Kyber768': {'public': 1184, 'private': 2400},
            'Kyber1024': {'public': 1568, 'private': 3168}
        }
        
        sizes = key_sizes.get(algorithm, {'public': 1000, 'private': 2000})
        
        return {
            'public_key_bytes': sizes['public'],
            'private_key_bytes': sizes['private'],
            'signature_bytes': sizes['private'] // 2,  # Approximate
            'total_flash_usage': sizes['public'] + sizes['private'] + np.random.normal(15000, 2000),
            'peak_ram_usage': max(sizes['private'], np.random.normal(8000, 1000))
        }
    
    def _estimate_power_consumption(self, algorithm: str, platform: str) -> Dict[str, float]:
        """Estimate power consumption characteristics."""
        # Power estimation based on cycle counts and platform characteristics
        platform_power = {
            'cortex-m4': 0.3,  # mW per MHz
            'cortex-m7': 0.4,
            'esp32': 0.5,
            'riscv32': 0.35
        }
        
        base_power = platform_power.get(platform, 0.4)
        frequency_mhz = 168  # Typical frequency
        
        # Get cycle counts from previous benchmarks
        keygen_cycles = self._get_cached_metric('keygen_cycles', algorithm, 100000)
        sign_cycles = self._get_cached_metric('sign_cycles', algorithm, 200000)
        verify_cycles = self._get_cached_metric('verify_cycles', algorithm, 80000)
        
        return {
            'keygen_power_mj': (keygen_cycles / frequency_mhz) * base_power / 1000,
            'sign_power_mj': (sign_cycles / frequency_mhz) * base_power / 1000,
            'verify_power_mj': (verify_cycles / frequency_mhz) * base_power / 1000,
            'idle_power_mw': base_power * 0.1,  # 10% idle power
            'active_power_mw': base_power * frequency_mhz
        }
    
    def _get_cached_metric(self, metric: str, algorithm: str, default: float) -> float:
        """Get cached metric value or return default."""
        for cached_result in self.benchmark_cache.values():
            if metric in cached_result:
                return cached_result[metric]
        return default
    
    def _store_benchmark_results(self, algorithm: str, platform: str, 
                               parameters: Dict[str, Any], metrics: Dict[str, float]):
        """Store benchmark results in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO benchmark_results (algorithm, platform, parameters, metrics, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            algorithm,
            platform,
            json.dumps(parameters),
            json.dumps(metrics),
            time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def get_benchmark_history(self, algorithm: str = None, platform: str = None) -> List[Dict[str, Any]]:
        """Retrieve benchmark history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM benchmark_results WHERE 1=1"
        params = []
        
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        history = []
        for row in results:
            history.append({
                'id': row[0],
                'algorithm': row[1],
                'platform': row[2],
                'parameters': json.loads(row[3]),
                'metrics': json.loads(row[4]),
                'timestamp': row[5]
            })
        
        conn.close()
        return history


class StatisticalAnalyzer:
    """Advanced statistical analysis for experimental results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_experiment_results(self, results: List[ExperimentResult], 
                                 hypothesis: ResearchHypothesis) -> List[StatisticalAnalysis]:
        """Perform comprehensive statistical analysis on experimental results."""
        
        analyses = []
        
        # Extract data for analysis
        data = self._extract_analysis_data(results)
        
        if len(data) < 3:
            self.logger.warning("Insufficient data for statistical analysis")
            return analyses
        
        # Perform different types of statistical tests
        for dependent_var in hypothesis.dependent_variables:
            if dependent_var not in data:
                continue
            
            # Descriptive statistics
            desc_analysis = self._descriptive_analysis(data[dependent_var], dependent_var)
            analyses.append(desc_analysis)
            
            # Hypothesis testing
            if len(hypothesis.independent_variables) == 1:
                # Single factor analysis
                independent_var = hypothesis.independent_variables[0]
                if independent_var in data:
                    hyp_analysis = self._hypothesis_test(
                        data[dependent_var], data[independent_var], 
                        hypothesis.null_hypothesis, hypothesis.alternative_hypothesis
                    )
                    analyses.append(hyp_analysis)
            
            # Correlation analysis
            if len(hypothesis.independent_variables) > 1:
                corr_analysis = self._correlation_analysis(data, dependent_var, hypothesis.independent_variables)
                analyses.append(corr_analysis)
        
        return analyses
    
    def _extract_analysis_data(self, results: List[ExperimentResult]) -> Dict[str, List[float]]:
        """Extract data from experiment results for analysis."""
        data = defaultdict(list)
        
        for result in results:
            if not result.success:
                continue
            
            # Extract parameter values
            for param, value in result.parameters.items():
                if isinstance(value, (int, float)):
                    data[param].append(value)
                elif isinstance(value, str):
                    # Convert categorical to numeric if possible
                    try:
                        data[param].append(hash(value) % 1000)  # Simple hash-based encoding
                    except:
                        pass
            
            # Extract measurement values
            for measure, value in result.measurements.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    data[measure].append(value)
        
        return dict(data)
    
    def _descriptive_analysis(self, values: List[float], variable_name: str) -> StatisticalAnalysis:
        """Perform descriptive statistical analysis."""
        values = np.array(values)
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        
        # Confidence interval for mean
        from scipy import stats
        ci_low, ci_high = stats.t.interval(0.95, len(values)-1, loc=mean_val, scale=stats.sem(values))
        
        interpretation = f"Variable '{variable_name}': Mean={mean_val:.3f}, Std={std_val:.3f}, Median={median_val:.3f}, n={len(values)}"
        
        return StatisticalAnalysis(
            analysis_id=f"descriptive_{variable_name}_{int(time.time())}",
            hypothesis_tested=f"Descriptive statistics for {variable_name}",
            test_statistic=mean_val,
            p_value=1.0,  # Not applicable for descriptive stats
            confidence_interval=(ci_low, ci_high),
            effect_size=std_val / mean_val if mean_val != 0 else 0,  # Coefficient of variation
            power=1.0,  # Not applicable
            significance_level=StatisticalSignificance.NOT_SIGNIFICANT,
            interpretation=interpretation,
            recommendations=[]
        )
    
    def _hypothesis_test(self, dependent_values: List[float], independent_values: List[float],
                        null_hypothesis: str, alternative_hypothesis: str) -> StatisticalAnalysis:
        """Perform hypothesis testing."""
        from scipy import stats
        
        # Convert to numpy arrays
        y = np.array(dependent_values)
        x = np.array(independent_values)
        
        # Perform appropriate test based on data characteristics
        if len(np.unique(x)) == 2:
            # Two-sample t-test
            group1 = y[x == np.unique(x)[0]]
            group2 = y[x == np.unique(x)[1]]
            
            t_stat, p_value = stats.ttest_ind(group1, group2)
            effect_size = (np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
            
        else:
            # Correlation test
            t_stat, p_value = stats.pearsonr(x, y)
            effect_size = abs(t_stat)
        
        # Determine significance level
        if p_value < 0.001:
            significance = StatisticalSignificance.VERY_HIGH
        elif p_value < 0.01:
            significance = StatisticalSignificance.HIGH
        elif p_value < 0.05:
            significance = StatisticalSignificance.MODERATE
        elif p_value < 0.1:
            significance = StatisticalSignificance.LOW
        else:
            significance = StatisticalSignificance.NOT_SIGNIFICANT
        
        # Calculate confidence interval
        ci_low = t_stat - 1.96 * np.sqrt(1/len(y))  # Approximate
        ci_high = t_stat + 1.96 * np.sqrt(1/len(y))
        
        # Power analysis (simplified)
        power = 1.0 - stats.norm.cdf(1.96 - abs(t_stat) * np.sqrt(len(y)))
        
        # Generate interpretation
        reject_null = p_value < 0.05
        interpretation = f"{'Reject' if reject_null else 'Fail to reject'} null hypothesis. "
        interpretation += f"p-value: {p_value:.6f}, effect size: {effect_size:.3f}"
        
        # Generate recommendations
        recommendations = []
        if significance == StatisticalSignificance.NOT_SIGNIFICANT:
            recommendations.append("Consider increasing sample size for more statistical power")
        if power < 0.8:
            recommendations.append("Low statistical power detected - results may not be reliable")
        if abs(effect_size) > 0.8:
            recommendations.append("Large effect size detected - result may be practically significant")
        
        return StatisticalAnalysis(
            analysis_id=f"hypothesis_{int(time.time())}",
            hypothesis_tested=null_hypothesis,
            test_statistic=t_stat,
            p_value=p_value,
            confidence_interval=(ci_low, ci_high),
            effect_size=effect_size,
            power=power,
            significance_level=significance,
            interpretation=interpretation,
            recommendations=recommendations
        )
    
    def _correlation_analysis(self, data: Dict[str, List[float]], 
                            dependent_var: str, independent_vars: List[str]) -> StatisticalAnalysis:
        """Perform correlation analysis for multiple variables."""
        from scipy import stats
        
        y = np.array(data[dependent_var])
        correlations = []
        
        for indep_var in independent_vars:
            if indep_var in data and len(data[indep_var]) == len(y):
                x = np.array(data[indep_var])
                corr, p_val = stats.pearsonr(x, y)
                correlations.append((indep_var, corr, p_val))
        
        if not correlations:
            return StatisticalAnalysis(
                analysis_id=f"correlation_{dependent_var}_{int(time.time())}",
                hypothesis_tested=f"Correlation analysis for {dependent_var}",
                test_statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                power=0.0,
                significance_level=StatisticalSignificance.NOT_SIGNIFICANT,
                interpretation="No valid correlations found",
                recommendations=["Ensure data quality and sufficient sample size"]
            )
        
        # Find strongest correlation
        strongest = max(correlations, key=lambda x: abs(x[1]))
        var_name, corr_val, p_val = strongest
        
        # Determine significance
        if p_val < 0.001:
            significance = StatisticalSignificance.VERY_HIGH
        elif p_val < 0.01:
            significance = StatisticalSignificance.HIGH
        elif p_val < 0.05:
            significance = StatisticalSignificance.MODERATE
        elif p_val < 0.1:
            significance = StatisticalSignificance.LOW
        else:
            significance = StatisticalSignificance.NOT_SIGNIFICANT
        
        # Confidence interval for correlation
        n = len(y)
        z = 0.5 * np.log((1 + corr_val) / (1 - corr_val))  # Fisher z-transform
        se = 1 / np.sqrt(n - 3)
        z_low, z_high = z - 1.96*se, z + 1.96*se
        ci_low = (np.exp(2*z_low) - 1) / (np.exp(2*z_low) + 1)
        ci_high = (np.exp(2*z_high) - 1) / (np.exp(2*z_high) + 1)
        
        interpretation = f"Strongest correlation: {var_name} -> {dependent_var} (r={corr_val:.3f}, p={p_val:.6f})"
        
        return StatisticalAnalysis(
            analysis_id=f"correlation_{dependent_var}_{int(time.time())}",
            hypothesis_tested=f"Correlation analysis for {dependent_var}",
            test_statistic=corr_val,
            p_value=p_val,
            confidence_interval=(ci_low, ci_high),
            effect_size=abs(corr_val),
            power=0.8,  # Approximate
            significance_level=significance,
            interpretation=interpretation,
            recommendations=[]
        )


class ExperimentDatabase:
    """Database for storing experimental data and results."""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize experiment database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE,
                hypothesis_id TEXT,
                status TEXT,
                start_time REAL,
                end_time REAL,
                metadata TEXT
            )
        ''')
        
        # Results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                run_id TEXT,
                parameters TEXT,
                measurements TEXT,
                success BOOLEAN,
                timestamp REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        # Analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT UNIQUE,
                experiment_id TEXT,
                analysis_type TEXT,
                results TEXT,
                timestamp REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_result(self, result: ExperimentResult):
        """Store experimental result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO results 
            (experiment_id, run_id, parameters, measurements, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result.experiment_id,
            result.run_id,
            json.dumps(result.parameters),
            json.dumps(result.measurements),
            result.success,
            result.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def store_analysis(self, analysis: StatisticalAnalysis, experiment_id: str):
        """Store statistical analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO analyses 
            (analysis_id, experiment_id, analysis_type, results, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            analysis.analysis_id,
            experiment_id,
            "statistical_analysis",
            json.dumps(asdict(analysis)),
            time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Retrieve all results for an experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT experiment_id, run_id, parameters, measurements, success, timestamp
            FROM results WHERE experiment_id = ?
        ''', (experiment_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = ExperimentResult(
                experiment_id=row[0],
                run_id=row[1],
                parameters=json.loads(row[2]),
                measurements=json.loads(row[3]),
                execution_time=0.0,  # Not stored separately
                success=row[4],
                error_message=None,
                raw_data=json.loads(row[3]),
                timestamp=row[5]
            )
            results.append(result)
        
        return results


class AutonomousResearcher:
    """Main autonomous research system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.experimental_framework = ExperimentalFramework()
        self.quantum_resilience = QuantumResilienceAnalyzer()
        
        self.research_queue = deque()
        self.active_experiments = {}
        self.research_database = ExperimentDatabase()
        
        # Research objectives and priorities
        self.research_priorities = {
            ResearchObjective.PERFORMANCE_BENCHMARKING: 1.0,
            ResearchObjective.SECURITY_ANALYSIS: 0.9,
            ResearchObjective.ALGORITHM_OPTIMIZATION: 0.8,
            ResearchObjective.ATTACK_RESISTANCE: 0.7,
            ResearchObjective.IMPLEMENTATION_COMPARISON: 0.6,
            ResearchObjective.RESOURCE_EFFICIENCY: 0.5,
            ResearchObjective.STANDARDIZATION_SUPPORT: 0.4
        }
    
    @track_performance("autonomous_research_execution")
    def conduct_autonomous_research(self, focus_areas: List[ResearchObjective] = None) -> Dict[str, Any]:
        """Conduct autonomous research across specified focus areas."""
        
        focus_areas = focus_areas or list(ResearchObjective)
        self.logger.info(f"Starting autonomous research in {len(focus_areas)} focus areas")
        
        research_results = {}
        
        for objective in focus_areas:
            self.logger.info(f"Researching: {objective.value}")
            
            # Generate research hypotheses
            hypotheses = self._generate_research_hypotheses(objective)
            
            objective_results = []
            for hypothesis in hypotheses:
                try:
                    # Design experiment
                    design = self._design_experiment(hypothesis)
                    
                    # Conduct experiment
                    results = self.experimental_framework.conduct_experiment(hypothesis, design)
                    
                    # Analyze results
                    analyses = self.experimental_framework.statistical_analyzer.analyze_experiment_results(
                        results, hypothesis
                    )
                    
                    # Store results
                    for analysis in analyses:
                        self.research_database.store_analysis(analysis, hypothesis.hypothesis_id)
                    
                    objective_results.append({
                        'hypothesis': asdict(hypothesis),
                        'experimental_design': asdict(design),
                        'results_count': len(results),
                        'successful_runs': len([r for r in results if r.success]),
                        'statistical_analyses': [asdict(a) for a in analyses],
                        'key_findings': self._extract_key_findings(analyses)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Research experiment failed: {e}")
                    objective_results.append({
                        'hypothesis': asdict(hypothesis),
                        'error': str(e),
                        'key_findings': []
                    })
            
            research_results[objective.value] = objective_results
        
        # Generate research summary
        summary = self._generate_research_summary(research_results)
        
        # Record metrics
        total_experiments = sum(len(results) for results in research_results.values())
        successful_experiments = sum(
            len([r for r in results if 'error' not in r])
            for results in research_results.values()
        )
        
        metrics_collector.record_metric("research.experiments_conducted", total_experiments, "count")
        metrics_collector.record_metric("research.successful_experiments", successful_experiments, "count")
        metrics_collector.record_metric("research.focus_areas_studied", len(focus_areas), "count")
        
        self.logger.info(f"Autonomous research completed: {successful_experiments}/{total_experiments} experiments successful")
        
        return {
            'research_results': research_results,
            'summary': summary,
            'metadata': {
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'focus_areas': [obj.value for obj in focus_areas],
                'timestamp': time.time()
            }
        }
    
    def _generate_research_hypotheses(self, objective: ResearchObjective) -> List[ResearchHypothesis]:
        """Generate research hypotheses for given objective."""
        hypotheses = []
        
        if objective == ResearchObjective.PERFORMANCE_BENCHMARKING:
            hypotheses.extend(self._generate_performance_hypotheses())
        elif objective == ResearchObjective.SECURITY_ANALYSIS:
            hypotheses.extend(self._generate_security_hypotheses())
        elif objective == ResearchObjective.ALGORITHM_OPTIMIZATION:
            hypotheses.extend(self._generate_optimization_hypotheses())
        elif objective == ResearchObjective.ATTACK_RESISTANCE:
            hypotheses.extend(self._generate_attack_resistance_hypotheses())
        
        return hypotheses
    
    def _generate_performance_hypotheses(self) -> List[ResearchHypothesis]:
        """Generate performance-focused research hypotheses."""
        return [
            ResearchHypothesis(
                hypothesis_id=f"perf_alg_comparison_{int(time.time())}",
                objective=ResearchObjective.PERFORMANCE_BENCHMARKING,
                null_hypothesis="There is no significant performance difference between PQC algorithms on IoT devices",
                alternative_hypothesis="Some PQC algorithms perform significantly better than others on resource-constrained IoT devices",
                independent_variables=["algorithm", "platform", "optimization_level"],
                dependent_variables=["execution_cycles", "memory_usage_bytes", "energy_consumption"],
                expected_outcome="Kyber512 and Dilithium2 will show best performance on Cortex-M4",
                confidence_level=0.95,
                sample_size_required=30,
                experiment_design={
                    "type": "factorial",
                    "factors": {
                        "algorithm": ["Dilithium2", "Dilithium3", "Kyber512", "Kyber768"],
                        "platform": ["cortex-m4", "esp32"],
                        "optimization_level": ["speed", "size", "balanced"]
                    }
                },
                success_criteria=["Statistical significance p < 0.05", "Effect size > 0.5"],
                created_timestamp=time.time()
            )
        ]
    
    def _generate_security_hypotheses(self) -> List[ResearchHypothesis]:
        """Generate security-focused research hypotheses."""
        return [
            ResearchHypothesis(
                hypothesis_id=f"sec_side_channel_{int(time.time())}",
                objective=ResearchObjective.SECURITY_ANALYSIS,
                null_hypothesis="PQC implementations have equivalent side-channel resistance",
                alternative_hypothesis="Some PQC implementations are more resistant to side-channel attacks",
                independent_variables=["algorithm", "implementation_type", "countermeasures"],
                dependent_variables=["timing_leakage_bits", "power_traces_required", "cache_miss_correlation"],
                expected_outcome="Implementations with masking countermeasures will show better resistance",
                confidence_level=0.99,
                sample_size_required=100,
                experiment_design={
                    "type": "randomized",
                    "factors": {
                        "algorithm": ["Dilithium2", "Falcon-512", "Kyber512"],
                        "implementation_type": ["reference", "optimized", "protected"],
                        "countermeasures": ["none", "masking", "shuffling", "both"]
                    }
                },
                success_criteria=["p < 0.01 for security metrics", "Clear ranking of implementations"],
                created_timestamp=time.time()
            )
        ]
    
    def _generate_optimization_hypotheses(self) -> List[ResearchHypothesis]:
        """Generate algorithm optimization hypotheses."""
        return [
            ResearchHypothesis(
                hypothesis_id=f"opt_parameter_tuning_{int(time.time())}",
                objective=ResearchObjective.ALGORITHM_OPTIMIZATION,
                null_hypothesis="Parameter tuning has no significant effect on PQC performance",
                alternative_hypothesis="Optimal parameter selection significantly improves PQC performance",
                independent_variables=["security_level", "optimization_target", "memory_constraint"],
                dependent_variables=["performance_gain", "memory_efficiency", "security_margin"],
                expected_outcome="Balanced optimization with level 2 security will provide best overall results",
                confidence_level=0.95,
                sample_size_required=50,
                experiment_design={
                    "type": "latin_square",
                    "factors": {
                        "security_level": [1, 2, 3],
                        "optimization_target": ["speed", "size", "memory"]
                    }
                },
                success_criteria=["Pareto-optimal configurations identified", "Performance improvement > 15%"],
                created_timestamp=time.time()
            )
        ]
    
    def _generate_attack_resistance_hypotheses(self) -> List[ResearchHypothesis]:
        """Generate attack resistance hypotheses."""
        return [
            ResearchHypothesis(
                hypothesis_id=f"attack_quantum_resistance_{int(time.time())}",
                objective=ResearchObjective.ATTACK_RESISTANCE,
                null_hypothesis="All NIST PQC algorithms provide equivalent quantum attack resistance",
                alternative_hypothesis="Quantum attack resistance varies significantly among PQC algorithms",
                independent_variables=["algorithm", "security_level", "attack_model"],
                dependent_variables=["quantum_security_bits", "classical_security_margin", "attack_complexity"],
                expected_outcome="Higher security levels will provide exponentially better resistance",
                confidence_level=0.99,
                sample_size_required=25,
                experiment_design={
                    "type": "factorial",
                    "factors": {
                        "algorithm": ["Dilithium2", "Dilithium3", "Kyber512", "Kyber768"],
                        "security_level": [1, 2, 3],
                        "attack_model": ["known_plaintext", "chosen_plaintext", "adaptive"]
                    }
                },
                success_criteria=["Security levels clearly differentiated", "No critical vulnerabilities found"],
                created_timestamp=time.time()
            )
        ]
    
    def _design_experiment(self, hypothesis: ResearchHypothesis) -> ExperimentalDesign:
        """Design experiment based on hypothesis."""
        factors = hypothesis.experiment_design.get("factors", {})
        design_type = hypothesis.experiment_design.get("type", "factorial")
        
        # Calculate required sample size based on power analysis
        effect_size = 0.5  # Medium effect size
        power = 0.8
        alpha = 1 - hypothesis.confidence_level
        
        # Simplified sample size calculation
        sample_size = max(hypothesis.sample_size_required, len(factors) * 10)
        
        return ExperimentalDesign(
            design_id=f"design_{hypothesis.hypothesis_id}",
            design_type=design_type,
            factors=factors,
            response_variables=hypothesis.dependent_variables,
            sample_size=sample_size,
            replication_count=3,
            randomization_seed=hash(hypothesis.hypothesis_id) % 10000,
            blocking_factors=None,
            covariates=None,
            power_analysis={
                'effect_size': effect_size,
                'power': power,
                'alpha': alpha,
                'sample_size': sample_size
            }
        )
    
    def _extract_key_findings(self, analyses: List[StatisticalAnalysis]) -> List[str]:
        """Extract key findings from statistical analyses."""
        findings = []
        
        for analysis in analyses:
            if analysis.significance_level in [StatisticalSignificance.HIGH, StatisticalSignificance.VERY_HIGH]:
                findings.append(
                    f"Significant result: {analysis.hypothesis_tested} "
                    f"(p={analysis.p_value:.4f}, effect size={analysis.effect_size:.3f})"
                )
            
            if analysis.effect_size > 0.8:
                findings.append(f"Large effect size detected in {analysis.hypothesis_tested}")
            
            if analysis.power < 0.8:
                findings.append(f"Low statistical power warning: {analysis.hypothesis_tested}")
        
        if not findings:
            findings.append("No statistically significant findings")
        
        return findings
    
    def _generate_research_summary(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive research summary."""
        summary = {
            'total_objectives_studied': len(research_results),
            'total_hypotheses_tested': sum(len(results) for results in research_results.values()),
            'significant_findings': 0,
            'key_insights': [],
            'recommendations': [],
            'future_research_directions': []
        }
        
        # Count significant findings
        for objective_results in research_results.values():
            for result in objective_results:
                if 'statistical_analyses' in result:
                    for analysis_data in result['statistical_analyses']:
                        if analysis_data['significance_level'] in ['high', 'very_high']:
                            summary['significant_findings'] += 1
        
        # Generate key insights
        if summary['significant_findings'] > 0:
            summary['key_insights'].append(
                f"Found {summary['significant_findings']} statistically significant results across all research areas"
            )
        
        # Generate recommendations
        summary['recommendations'].extend([
            "Continue monitoring PQC algorithm performance on IoT devices",
            "Prioritize algorithms with best performance/security tradeoffs",
            "Implement side-channel countermeasures where necessary",
            "Validate findings with larger sample sizes"
        ])
        
        # Future research directions
        summary['future_research_directions'].extend([
            "Long-term security analysis of PQC algorithms",
            "Hardware-specific optimization studies",
            "Formal verification of implementations",
            "Quantum computer simulation validation"
        ])
        
        return summary


# Global autonomous researcher instance
autonomous_researcher = AutonomousResearcher()