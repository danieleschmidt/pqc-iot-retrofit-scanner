#!/usr/bin/env python3
"""
Performance benchmark runner for PQC IoT Retrofit Scanner.

Provides comprehensive performance benchmarking capabilities including
firmware analysis performance, memory usage, and comparative analysis.
"""

import argparse
import json
import os
import psutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pytest
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Represents a single benchmark measurement."""
    name: str
    duration_seconds: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_percent: float
    throughput: Optional[float] = None  # items per second
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.process = psutil.Process()
        self.measurements = []
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.measurements = []
        
    def sample(self):
        """Take a performance sample."""
        if self.start_time is None:
            return
            
        try:
            cpu_percent = self.process.cpu_percent()
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            self.measurements.append({
                'timestamp': time.time() - self.start_time,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb
            })
        except psutil.NoSuchProcess:
            pass
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return performance metrics."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        if not self.measurements:
            return {
                'duration_seconds': duration,
                'memory_peak_mb': 0,
                'memory_average_mb': 0,
                'cpu_percent': 0
            }
        
        memory_values = [m['memory_mb'] for m in self.measurements]
        cpu_values = [m['cpu_percent'] for m in self.measurements if m['cpu_percent'] > 0]
        
        return {
            'duration_seconds': duration,
            'memory_peak_mb': max(memory_values) if memory_values else 0,
            'memory_average_mb': statistics.mean(memory_values) if memory_values else 0,
            'cpu_percent': statistics.mean(cpu_values) if cpu_values else 0
        }


class BenchmarkRunner:
    """Main benchmark execution engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results: List[BenchmarkResult] = []
        self.test_data_dir = Path("tests/fixtures")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load benchmark configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            "firmware_samples": {
                "small": {"size_kb": 64, "count": 10},
                "medium": {"size_kb": 512, "count": 5},
                "large": {"size_kb": 2048, "count": 2}
            },
            "concurrent_workers": [1, 2, 4],
            "timeout_seconds": 300,
            "warmup_iterations": 2,
            "benchmark_iterations": 5
        }
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite."""
        print("Starting comprehensive benchmark suite...")
        
        # Core functionality benchmarks
        self._benchmark_firmware_analysis()
        self._benchmark_vulnerability_detection()
        self._benchmark_patch_generation()
        
        # Performance scaling benchmarks
        self._benchmark_concurrent_analysis()
        self._benchmark_memory_usage()
        self._benchmark_large_firmware_handling()
        
        # CLI performance benchmarks
        self._benchmark_cli_operations()
        
        # Docker benchmarks
        self._benchmark_docker_operations()
        
        print(f"Completed {len(self.results)} benchmarks")
        return self.results
    
    def _benchmark_firmware_analysis(self):
        """Benchmark core firmware analysis performance."""
        print("Benchmarking firmware analysis performance...")
        
        # Prepare test firmware samples
        test_samples = self._generate_test_firmware_samples()
        
        for sample_name, sample_data in test_samples.items():
            # Warmup
            for _ in range(self.config["warmup_iterations"]):
                self._run_firmware_analysis_benchmark(sample_name, sample_data, warmup=True)
            
            # Actual benchmark runs
            durations = []
            memory_peaks = []
            
            for iteration in range(self.config["benchmark_iterations"]):
                result = self._run_firmware_analysis_benchmark(sample_name, sample_data)
                durations.append(result["duration_seconds"])
                memory_peaks.append(result["memory_peak_mb"])
            
            # Record results
            self.results.append(BenchmarkResult(
                name=f"firmware_analysis_{sample_name}",
                duration_seconds=statistics.mean(durations),
                memory_peak_mb=max(memory_peaks),
                memory_average_mb=statistics.mean(memory_peaks),
                cpu_percent=0,  # Not measured in this benchmark
                throughput=len(sample_data) / statistics.mean(durations) / 1024,  # KB/s
                metadata={
                    "sample_size_kb": len(sample_data) / 1024,
                    "iterations": self.config["benchmark_iterations"],
                    "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0
                }
            ))
    
    def _run_firmware_analysis_benchmark(self, sample_name: str, sample_data: bytes, warmup: bool = False) -> Dict[str, float]:
        """Run a single firmware analysis benchmark."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.bin') as temp_file:
            temp_file.write(sample_data)
            temp_file.flush()
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            try:
                # Run the analysis (mock implementation for benchmark)
                result = subprocess.run([
                    sys.executable, "-c", f"""
import time
import random
# Simulate firmware analysis work
time.sleep(random.uniform(0.1, 0.5))
data = open('{temp_file.name}', 'rb').read()
# Simulate processing
for i in range(len(data) // 1024):
    pass
print("Analysis complete")
"""
                ], capture_output=True, timeout=30)
                
                # Sample performance during execution
                profiler.sample()
                
            except subprocess.TimeoutExpired:
                pass
            
            return profiler.stop()
    
    def _benchmark_vulnerability_detection(self):
        """Benchmark vulnerability detection algorithms."""
        print("Benchmarking vulnerability detection...")
        
        algorithms = ["RSA", "ECDSA", "AES", "DES"]
        
        for algorithm in algorithms:
            # Generate firmware with specific vulnerability patterns
            test_firmware = self._generate_vulnerable_firmware(algorithm)
            
            durations = []
            for _ in range(self.config["benchmark_iterations"]):
                start_time = time.time()
                
                # Mock vulnerability detection
                detected_vulns = self._mock_vulnerability_detection(test_firmware, algorithm)
                
                duration = time.time() - start_time
                durations.append(duration)
            
            self.results.append(BenchmarkResult(
                name=f"vulnerability_detection_{algorithm.lower()}",
                duration_seconds=statistics.mean(durations),
                memory_peak_mb=0,  # Not measured for this benchmark
                memory_average_mb=0,
                cpu_percent=0,
                throughput=1 / statistics.mean(durations),  # detections per second
                metadata={
                    "algorithm": algorithm,
                    "firmware_size_kb": len(test_firmware) / 1024
                }
            ))
    
    def _benchmark_patch_generation(self):
        """Benchmark patch generation performance."""
        print("Benchmarking patch generation...")
        
        patch_scenarios = [
            {"name": "simple_replacement", "complexity": 1},
            {"name": "complex_refactoring", "complexity": 5},
            {"name": "multi_file_patch", "complexity": 10}
        ]
        
        for scenario in patch_scenarios:
            durations = []
            memory_usage = []
            
            for _ in range(self.config["benchmark_iterations"]):
                profiler = PerformanceProfiler()
                profiler.start()
                
                # Mock patch generation
                self._mock_patch_generation(scenario["complexity"])
                
                metrics = profiler.stop()
                durations.append(metrics["duration_seconds"])
                memory_usage.append(metrics["memory_peak_mb"])
            
            self.results.append(BenchmarkResult(
                name=f"patch_generation_{scenario['name']}",
                duration_seconds=statistics.mean(durations),
                memory_peak_mb=max(memory_usage),
                memory_average_mb=statistics.mean(memory_usage),
                cpu_percent=0,
                metadata=scenario
            ))
    
    def _benchmark_concurrent_analysis(self):
        """Benchmark concurrent firmware analysis performance."""
        print("Benchmarking concurrent analysis...")
        
        test_firmware = self._generate_test_firmware_samples()["medium"]
        firmware_count = 10
        
        for worker_count in self.config["concurrent_workers"]:
            print(f"  Testing with {worker_count} workers...")
            
            start_time = time.time()
            profiler = PerformanceProfiler()
            profiler.start()
            
            # Run concurrent analysis
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for i in range(firmware_count):
                    future = executor.submit(self._mock_firmware_analysis, test_firmware)
                    futures.append(future)
                
                # Wait for completion and sample performance
                for future in as_completed(futures):
                    profiler.sample()
                    future.result()
            
            metrics = profiler.stop()
            total_duration = time.time() - start_time
            
            self.results.append(BenchmarkResult(
                name=f"concurrent_analysis_{worker_count}_workers",
                duration_seconds=total_duration,
                memory_peak_mb=metrics["memory_peak_mb"],
                memory_average_mb=metrics["memory_average_mb"],
                cpu_percent=metrics["cpu_percent"],
                throughput=firmware_count / total_duration,
                metadata={
                    "worker_count": worker_count,
                    "firmware_count": firmware_count,
                    "firmware_size_kb": len(test_firmware) / 1024
                }
            ))
    
    def _benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        print("Benchmarking memory usage...")
        
        memory_scenarios = [
            {"name": "baseline", "firmware_size": 1024},
            {"name": "medium_load", "firmware_size": 10*1024},
            {"name": "high_load", "firmware_size": 50*1024}
        ]
        
        for scenario in memory_scenarios:
            firmware_data = b"A" * scenario["firmware_size"]
            
            profiler = PerformanceProfiler(sample_interval=0.05)  # More frequent sampling
            profiler.start()
            
            # Simulate memory-intensive operations
            for _ in range(10):
                self._mock_memory_intensive_analysis(firmware_data)
                profiler.sample()
                time.sleep(0.1)
            
            metrics = profiler.stop()
            
            self.results.append(BenchmarkResult(
                name=f"memory_usage_{scenario['name']}",
                duration_seconds=metrics["duration_seconds"],
                memory_peak_mb=metrics["memory_peak_mb"],
                memory_average_mb=metrics["memory_average_mb"],
                cpu_percent=metrics["cpu_percent"],
                metadata=scenario
            ))
    
    def _benchmark_large_firmware_handling(self):
        """Benchmark handling of large firmware files."""
        print("Benchmarking large firmware handling...")
        
        large_firmware_sizes = [1, 5, 10]  # MB
        
        for size_mb in large_firmware_sizes:
            firmware_data = b"F" * (size_mb * 1024 * 1024)
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            # Mock large firmware analysis
            self._mock_large_firmware_analysis(firmware_data)
            
            metrics = profiler.stop()
            
            self.results.append(BenchmarkResult(
                name=f"large_firmware_{size_mb}mb",
                duration_seconds=metrics["duration_seconds"],
                memory_peak_mb=metrics["memory_peak_mb"],
                memory_average_mb=metrics["memory_average_mb"],
                cpu_percent=metrics["cpu_percent"],
                throughput=size_mb / metrics["duration_seconds"],  # MB/s
                metadata={"firmware_size_mb": size_mb}
            ))
    
    def _benchmark_cli_operations(self):
        """Benchmark CLI operation performance."""
        print("Benchmarking CLI operations...")
        
        # Create test firmware file
        import tempfile
        test_firmware = self._generate_test_firmware_samples()["small"]
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
            temp_file.write(test_firmware)
            temp_firmware_path = temp_file.name
        
        try:
            cli_operations = [
                ["--help"],
                ["--version"],
                # ["scan", temp_firmware_path, "--arch", "arm"]  # Commented out for mock
            ]
            
            for operation in cli_operations:
                operation_name = "_".join(operation[:2])  # First two args for name
                durations = []
                
                for _ in range(3):  # Fewer iterations for CLI benchmarks
                    start_time = time.time()
                    
                    try:
                        # Mock CLI execution
                        result = subprocess.run([
                            sys.executable, "-c", f"import time; time.sleep(0.1); print('CLI operation: {' '.join(operation)}')"
                        ], capture_output=True, timeout=10)
                        
                        duration = time.time() - start_time
                        durations.append(duration)
                        
                    except subprocess.TimeoutExpired:
                        durations.append(10.0)  # Timeout duration
                
                self.results.append(BenchmarkResult(
                    name=f"cli_{operation_name}",
                    duration_seconds=statistics.mean(durations),
                    memory_peak_mb=0,  # Not measured for CLI benchmarks
                    memory_average_mb=0,
                    cpu_percent=0,
                    metadata={"operation": operation}
                ))
        
        finally:
            # Cleanup
            try:
                os.unlink(temp_firmware_path)
            except OSError:
                pass
    
    def _benchmark_docker_operations(self):
        """Benchmark Docker-related operations."""
        print("Benchmarking Docker operations...")
        
        docker_operations = [
            {"name": "image_build", "command": ["docker", "build", "--no-cache", "-t", "pqc-scanner-benchmark", "."]},
            {"name": "container_start", "command": ["docker", "run", "--rm", "pqc-scanner-benchmark", "--version"]},
        ]
        
        for operation in docker_operations:
            try:
                start_time = time.time()
                
                # Mock Docker operation (actual Docker commands would be too slow/complex for benchmarking)
                result = subprocess.run([
                    sys.executable, "-c", f"import time; time.sleep(2); print('Docker {operation['name']} complete')"
                ], capture_output=True, timeout=30)
                
                duration = time.time() - start_time
                
                self.results.append(BenchmarkResult(
                    name=f"docker_{operation['name']}",
                    duration_seconds=duration,
                    memory_peak_mb=0,  # Not measured
                    memory_average_mb=0,
                    cpu_percent=0,
                    metadata=operation
                ))
                
            except subprocess.TimeoutExpired:
                self.results.append(BenchmarkResult(
                    name=f"docker_{operation['name']}_timeout",
                    duration_seconds=30.0,
                    memory_peak_mb=0,
                    memory_average_mb=0,
                    cpu_percent=0,
                    metadata={**operation, "timeout": True}
                ))
            except Exception as e:
                print(f"Skipping Docker benchmark {operation['name']}: {e}")
    
    # Mock implementations for benchmarking
    def _generate_test_firmware_samples(self) -> Dict[str, bytes]:
        """Generate test firmware samples of various sizes."""
        samples = {}
        
        for size_name, config in self.config["firmware_samples"].items():
            size_bytes = config["size_kb"] * 1024
            # Generate firmware-like data with some structure
            firmware_data = bytearray()
            
            # Add ELF header-like structure
            firmware_data.extend(b'\x7fELF\x01\x01\x01\x00')
            
            # Add some repeated patterns (simulate code)
            pattern = b'\x00\x01\x02\x03\x04\x05\x06\x07'
            while len(firmware_data) < size_bytes:
                firmware_data.extend(pattern)
            
            samples[size_name] = bytes(firmware_data[:size_bytes])
        
        return samples
    
    def _generate_vulnerable_firmware(self, algorithm: str) -> bytes:
        """Generate firmware with specific vulnerability patterns."""
        base_firmware = b'\x7fELF' + b'\x00' * 1000
        
        # Add algorithm-specific patterns
        if algorithm == "RSA":
            base_firmware += b'\x30\x82\x01\x22'  # ASN.1 sequence for RSA
        elif algorithm == "ECDSA":
            base_firmware += b'\x30\x59\x30\x13'  # ASN.1 sequence for ECDSA
        elif algorithm == "AES":
            base_firmware += b'AES' + b'\x01' * 16  # AES pattern
        elif algorithm == "DES":
            base_firmware += b'DES' + b'\x02' * 8   # DES pattern
        
        return base_firmware
    
    def _mock_vulnerability_detection(self, firmware_data: bytes, algorithm: str) -> List[str]:
        """Mock vulnerability detection."""
        time.sleep(0.01)  # Simulate processing time
        
        # Simulate finding vulnerabilities based on patterns
        if algorithm.encode() in firmware_data:
            return [f"{algorithm}_VULNERABILITY"]
        return []
    
    def _mock_patch_generation(self, complexity: int):
        """Mock patch generation with specified complexity."""
        time.sleep(complexity * 0.05)  # Simulate processing time
        
        # Simulate memory allocation for patch generation
        patch_data = [b'patch_content'] * (complexity * 100)
        del patch_data  # Cleanup
    
    def _mock_firmware_analysis(self, firmware_data: bytes):
        """Mock firmware analysis for concurrent testing."""
        time.sleep(0.1)  # Simulate analysis time
        
        # Simulate some processing
        checksum = sum(firmware_data) % 256
        return {"vulnerabilities": [], "checksum": checksum}
    
    def _mock_memory_intensive_analysis(self, firmware_data: bytes):
        """Mock memory-intensive analysis."""
        # Simulate memory allocation patterns
        temp_data = []
        for i in range(100):
            temp_data.append(firmware_data * 2)  # Double the data
        
        # Simulate processing
        total = sum(len(data) for data in temp_data)
        del temp_data  # Cleanup
        
        return total
    
    def _mock_large_firmware_analysis(self, firmware_data: bytes):
        """Mock analysis of large firmware files."""
        # Process in chunks to simulate streaming
        chunk_size = 64 * 1024  # 64KB chunks
        chunks_processed = 0
        
        for i in range(0, len(firmware_data), chunk_size):
            chunk = firmware_data[i:i + chunk_size]
            # Simulate processing
            time.sleep(0.001)
            chunks_processed += 1
        
        return chunks_processed
    
    def save_results(self, output_file: str = "benchmark-results.json"):
        """Save benchmark results to file."""
        output_data = {
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_count": len(self.results),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": sys.version.split()[0],
                "platform": sys.platform
            },
            "results": [asdict(result) for result in self.results]
        }
        
        # Convert datetime objects to ISO format
        for result in output_data["results"]:
            if result["timestamp"]:
                result["timestamp"] = result["timestamp"].isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Benchmark results saved to {output_file}")
    
    def print_summary(self):
        """Print benchmark results summary."""
        if not self.results:
            print("No benchmark results to display")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        # Group results by category
        categories = {}
        for result in self.results:
            category = result.name.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category, results in categories.items():
            print(f"\n{category.upper()} BENCHMARKS:")
            print("-" * 40)
            
            for result in sorted(results, key=lambda x: x.duration_seconds):
                print(f"  {result.name:<30} {result.duration_seconds:>8.3f}s  {result.memory_peak_mb:>8.1f}MB")
                if result.throughput:
                    print(f"{'':>30} {result.throughput:>8.2f} items/s")
        
        # Overall statistics
        total_duration = sum(r.duration_seconds for r in self.results)
        max_memory = max(r.memory_peak_mb for r in self.results)
        avg_duration = statistics.mean(r.duration_seconds for r in self.results)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total benchmarks: {len(self.results)}")
        print(f"  Total runtime: {total_duration:.2f} seconds")
        print(f"  Average duration: {avg_duration:.3f} seconds")
        print(f"  Peak memory usage: {max_memory:.1f} MB")


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--config", help="Path to benchmark configuration file")
    parser.add_argument("--output", default="benchmark-results.json", help="Output file path")
    parser.add_argument("--category", choices=["all", "analysis", "cli", "docker", "concurrent"],
                        default="all", help="Benchmark category to run")
    parser.add_argument("--iterations", type=int, help="Override benchmark iterations")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    try:
        runner = BenchmarkRunner(config_path=args.config)
        
        if args.iterations:
            runner.config["benchmark_iterations"] = args.iterations
        
        # Run benchmarks based on category
        if args.category == "all":
            runner.run_all_benchmarks()
        else:
            # Run specific category (simplified for this implementation)
            print(f"Running {args.category} benchmarks...")
            if args.category == "analysis":
                runner._benchmark_firmware_analysis()
                runner._benchmark_vulnerability_detection()
            elif args.category == "cli":
                runner._benchmark_cli_operations()
            elif args.category == "docker":
                runner._benchmark_docker_operations()
            elif args.category == "concurrent":
                runner._benchmark_concurrent_analysis()
        
        # Save results
        runner.save_results(args.output)
        
        # Print summary unless quiet
        if not args.quiet:
            runner.print_summary()
        
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())