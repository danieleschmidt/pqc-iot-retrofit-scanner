"""Enhanced CLI interface with Generation 3 optimizations for PQC IoT Retrofit Scanner."""

import json
import click
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.panel import Panel
from rich.live import Live

from .scanner import FirmwareScanner, RiskLevel
from .patcher import PQCPatcher, OptimizationLevel

# Generation 3 imports
from .performance import performance_optimizer, cached_result
from .concurrency import (
    initialize_pools, shutdown_pools, firmware_scanner_pool, pqc_generator_pool,
    WorkItem, AsyncWorkManager
)
from .auto_scaling import create_adaptive_autoscaler
from .monitoring import metrics_collector, health_monitor

console = Console()


@click.group()
@click.version_option()
@click.option("--enable-optimizations", is_flag=True, default=True, 
              help="Enable Generation 3 performance optimizations")
@click.option("--max-workers", type=int, default=None, 
              help="Maximum number of worker threads/processes")
@click.option("--enable-caching", is_flag=True, default=True,
              help="Enable intelligent caching")
@click.option("--enable-auto-scaling", is_flag=True, default=False,
              help="Enable automatic scaling of worker pools")
@click.pass_context
def main(ctx, enable_optimizations, max_workers, enable_caching, enable_auto_scaling):
    """PQC IoT Retrofit Scanner CLI - Detect and patch quantum-vulnerable cryptography in IoT firmware.
    
    Generation 3 Features:
    ✨ Intelligent caching for faster repeated scans
    ⚡ Concurrent processing with auto-scaling worker pools
    📊 Real-time performance monitoring and metrics
    🔄 Adaptive load balancing and resource management
    """
    
    # Initialize context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['optimizations'] = enable_optimizations
    ctx.obj['max_workers'] = max_workers
    ctx.obj['caching'] = enable_caching
    ctx.obj['auto_scaling'] = enable_auto_scaling
    
    if enable_optimizations:
        console.print("🚀 [bold cyan]Generation 3 optimizations enabled[/bold cyan]")
        
        # Initialize worker pools
        initialize_pools(
            scanner_class=FirmwareScanner,
            generator_class=None,  # Will be initialized when needed
            scanner_workers=max_workers,
            generator_workers=max_workers
        )
        
        if enable_auto_scaling:
            console.print("⚡ [yellow]Auto-scaling enabled[/yellow]")


@main.command()
@click.argument("firmware_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--arch", required=True, 
              type=click.Choice(['cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7', 'esp32', 'riscv32', 'avr']),
              help="Target architecture")
@click.option("--output", "-o", help="Output report file (JSON)")
@click.option("--base-address", type=str, default="0x0", help="Base memory address (hex)")
@click.option("--flash-size", type=int, help="Flash memory size in bytes")
@click.option("--ram-size", type=int, help="RAM size in bytes")
@click.option("--concurrent", is_flag=True, default=True, help="Enable concurrent scanning")
@click.option("--batch-size", type=int, default=None, help="Batch size for concurrent processing")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def scan(ctx, firmware_paths, arch, output, base_address, flash_size, ram_size, 
         concurrent, batch_size, verbose):
    """Scan firmware files for quantum-vulnerable cryptography.
    
    Supports scanning multiple firmware files concurrently with intelligent load balancing.
    """
    
    # Parse base address
    try:
        base_addr = int(base_address, 16)
    except ValueError:
        console.print(f"[red]Error: Invalid base address '{base_address}'. Expected hex format (e.g., 0x08000000)[/red]")
        return
    
    # Setup memory constraints
    memory_constraints = {}
    if flash_size:
        memory_constraints['flash'] = flash_size
    if ram_size:
        memory_constraints['ram'] = ram_size
    
    firmware_list = list(firmware_paths)
    total_files = len(firmware_list)
    
    console.print(Panel(f"""[bold cyan]PQC IoT Retrofit Scanner - Generation 3[/bold cyan]
    
🎯 [bold]Targets:[/bold] {total_files} firmware file(s)
🏗️  [bold]Architecture:[/bold] {arch}
📍 [bold]Base Address:[/bold] {base_address}
💾 [bold]Memory Constraints:[/bold] {memory_constraints or 'Auto-detected'}
⚡ [bold]Concurrent Processing:[/bold] {'Enabled' if concurrent and ctx.obj['optimizations'] else 'Disabled'}
🧠 [bold]Intelligent Caching:[/bold] {'Enabled' if ctx.obj['caching'] else 'Disabled'}""", 
                       title="Scan Configuration"))
    
    start_time = time.time()
    all_vulnerabilities = []
    all_reports = []
    
    try:
        if concurrent and ctx.obj['optimizations'] and total_files > 1:
            # Use Generation 3 concurrent scanning
            all_vulnerabilities, all_reports = _scan_concurrent(
                firmware_list, arch, base_addr, memory_constraints, batch_size, verbose
            )
        else:
            # Use traditional sequential scanning
            all_vulnerabilities, all_reports = _scan_sequential(
                firmware_list, arch, base_addr, memory_constraints, verbose
            )
        
        scan_duration = time.time() - start_time
        
        # Aggregate results
        total_vulnerabilities = sum(len(vulns) for vulns in all_vulnerabilities)
        
        # Display results
        _display_multi_scan_results(all_reports, all_vulnerabilities, firmware_list, 
                                  scan_duration, verbose)
        
        # Save aggregated report if requested
        if output:
            aggregated_report = {
                'scan_metadata': {
                    'total_files': total_files,
                    'architecture': arch,
                    'base_address': base_address,
                    'memory_constraints': memory_constraints,
                    'scan_duration': scan_duration,
                    'concurrent_enabled': concurrent and ctx.obj['optimizations'],
                    'total_vulnerabilities': total_vulnerabilities
                },
                'files': []
            }
            
            for i, (firmware_path, report, vulns) in enumerate(zip(firmware_list, all_reports, all_vulnerabilities)):
                aggregated_report['files'].append({
                    'path': str(firmware_path),
                    'report': report,
                    'vulnerabilities': [_serialize_vulnerability(v) for v in vulns]
                })
            
            output_path = Path(output)
            output_path.write_text(json.dumps(aggregated_report, indent=2))
            console.print(f"\n[green]📄 Aggregated report saved to {output}[/green]")
        
        # Display performance metrics if optimizations are enabled
        if ctx.obj['optimizations']:
            _display_performance_metrics()
        
    except Exception as e:
        console.print(f"[red]❌ Scan failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


def _scan_concurrent(firmware_list: List[str], arch: str, base_addr: int, 
                    memory_constraints: Dict, batch_size: int, verbose: bool):
    """Perform concurrent firmware scanning using Generation 3 optimizations."""
    
    all_vulnerabilities = []
    all_reports = []
    
    # Determine optimal batch size if not specified
    if batch_size is None:
        batch_size = performance_optimizer.adaptive_batch_size("firmware_scan", target_duration=10.0)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        scan_task = progress.add_task("🔍 Scanning firmware files concurrently...", 
                                    total=len(firmware_list))
        
        # Submit scanning jobs to worker pool
        if firmware_scanner_pool:
            futures = []
            
            for firmware_path in firmware_list:
                work_item = WorkItem(
                    id=f"scan_{Path(firmware_path).name}",
                    data=firmware_path,
                    metadata={
                        'base_address': base_addr,
                        'arch': arch,
                        'memory_constraints': memory_constraints
                    }
                )
                
                future = firmware_scanner_pool.submit_work(work_item)
                futures.append((firmware_path, future))
            
            # Collect results as they complete
            for firmware_path, future in futures:
                try:
                    result = future.result(timeout=120.0)  # 2 minute timeout per file
                    
                    if result.success:
                        vulnerabilities = result.result
                        
                        # Generate report (this could also be cached)
                        scanner = FirmwareScanner(arch, memory_constraints)
                        scanner.vulnerabilities = vulnerabilities
                        report = scanner.generate_report()
                        
                        all_vulnerabilities.append(vulnerabilities)
                        all_reports.append(report)
                        
                        if verbose:
                            console.print(f"✅ [green]{firmware_path}[/green]: {len(vulnerabilities)} vulnerabilities")
                    else:
                        console.print(f"❌ [red]{firmware_path}[/red]: {result.error}")
                        all_vulnerabilities.append([])
                        all_reports.append({'error': str(result.error)})
                    
                    progress.advance(scan_task)
                    
                except Exception as e:
                    console.print(f"❌ [red]{firmware_path}[/red]: {e}")
                    all_vulnerabilities.append([])
                    all_reports.append({'error': str(e)})
                    progress.advance(scan_task)
        
        else:
            # Fall back to sequential if no worker pool
            console.print("[yellow]Worker pool not available, falling back to sequential scanning[/yellow]")
            return _scan_sequential(firmware_list, arch, base_addr, memory_constraints, verbose)
    
    return all_vulnerabilities, all_reports


def _scan_sequential(firmware_list: List[str], arch: str, base_addr: int, 
                    memory_constraints: Dict, verbose: bool):
    """Perform sequential firmware scanning with caching optimizations."""
    
    all_vulnerabilities = []
    all_reports = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        scan_task = progress.add_task("🔍 Scanning firmware files...", total=len(firmware_list))
        
        for firmware_path in firmware_list:
            try:
                # Use cached scanning if enabled
                if performance_optimizer:
                    scanner = FirmwareScanner(arch, memory_constraints)
                    
                    # Apply performance optimizations
                    original_scan = scanner.scan_firmware
                    scanner.scan_firmware = performance_optimizer.optimize_firmware_scanning(original_scan)
                else:
                    scanner = FirmwareScanner(arch, memory_constraints)
                
                vulnerabilities = scanner.scan_firmware(firmware_path, base_addr)
                report = scanner.generate_report()
                
                all_vulnerabilities.append(vulnerabilities)
                all_reports.append(report)
                
                if verbose:
                    console.print(f"✅ [green]{firmware_path}[/green]: {len(vulnerabilities)} vulnerabilities")
                
            except Exception as e:
                console.print(f"❌ [red]{firmware_path}[/red]: {e}")
                all_vulnerabilities.append([])
                all_reports.append({'error': str(e)})
            
            progress.advance(scan_task)
    
    return all_vulnerabilities, all_reports


def _display_multi_scan_results(reports: List[Dict], vulnerabilities_list: List, 
                               firmware_paths: List[str], scan_duration: float, verbose: bool):
    """Display results from multiple firmware scans."""
    
    total_files = len(firmware_paths)
    total_vulnerabilities = sum(len(vulns) for vulns in vulnerabilities_list)
    successful_scans = sum(1 for report in reports if 'error' not in report)
    
    # Summary table
    summary_table = Table(title="📊 Scan Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    summary_table.add_row("Total Files", str(total_files))
    summary_table.add_row("Successful Scans", f"{successful_scans}/{total_files}")
    summary_table.add_row("Total Vulnerabilities", str(total_vulnerabilities))
    summary_table.add_row("Scan Duration", f"{scan_duration:.2f} seconds")
    summary_table.add_row("Average per File", f"{scan_duration/total_files:.2f} seconds")
    
    if total_vulnerabilities > 0:
        summary_table.add_row("Files per Second", f"{total_files/scan_duration:.2f}")
    
    console.print(summary_table)
    
    # Detailed results per file
    if verbose or total_files <= 5:
        console.print("\n📁 [bold]Detailed Results by File:[/bold]")
        
        for i, (firmware_path, report, vulns) in enumerate(zip(firmware_paths, reports, vulnerabilities_list)):
            if 'error' in report:
                console.print(f"\n❌ [red]{Path(firmware_path).name}[/red]: {report['error']}")
            else:
                vuln_count = len(vulns)
                risk_counts = _count_risk_levels(vulns)
                
                console.print(f"\n✅ [green]{Path(firmware_path).name}[/green]")
                console.print(f"   • Total vulnerabilities: {vuln_count}")
                if risk_counts:
                    for risk, count in risk_counts.items():
                        color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "blue"}.get(risk, "white")
                        console.print(f"   • {risk}: [bold {color}]{count}[/bold {color}]")
    
    # Risk assessment
    if total_vulnerabilities > 0:
        all_vulns = [v for vulns in vulnerabilities_list for v in vulns]
        overall_risk_counts = _count_risk_levels(all_vulns)
        
        console.print("\n🚨 [bold red]Overall Risk Assessment:[/bold red]")
        for risk, count in overall_risk_counts.items():
            if count > 0:
                color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "blue"}.get(risk, "white")
                console.print(f"   • {risk}: [bold {color}]{count} vulnerabilities[/bold {color}]")
        
        if overall_risk_counts.get("CRITICAL", 0) > 0:
            console.print("\n⚠️  [bold red]IMMEDIATE ACTION REQUIRED: Critical vulnerabilities detected![/bold red]")
        elif overall_risk_counts.get("HIGH", 0) > 0:
            console.print("\n⚠️  [bold orange]HIGH PRIORITY: Address high-risk vulnerabilities soon[/bold orange]")


def _count_risk_levels(vulnerabilities):
    """Count vulnerabilities by risk level."""
    risk_counts = {}
    for vuln in vulnerabilities:
        risk = vuln.risk_level.value if hasattr(vuln.risk_level, 'value') else str(vuln.risk_level)
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    return risk_counts


def _serialize_vulnerability(vuln):
    """Serialize vulnerability object for JSON output."""
    return {
        'algorithm': vuln.algorithm.value if hasattr(vuln.algorithm, 'value') else str(vuln.algorithm),
        'address': hex(vuln.address),
        'function_name': vuln.function_name,
        'risk_level': vuln.risk_level.value if hasattr(vuln.risk_level, 'value') else str(vuln.risk_level),
        'confidence': getattr(vuln, 'confidence', 0.0),
        'description': getattr(vuln, 'description', ''),
        'key_size': getattr(vuln, 'key_size', None)
    }


def _display_performance_metrics():
    """Display Generation 3 performance metrics."""
    
    console.print("\n⚡ [bold cyan]Performance Metrics (Generation 3):[/bold cyan]")
    
    # Cache statistics
    if performance_optimizer and performance_optimizer.cache:
        cache_stats = performance_optimizer.cache.get_stats()
        
        cache_table = Table(title="🧠 Cache Performance")
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("L1 Cache", style="green")
        cache_table.add_column("L2 Cache", style="blue")
        
        cache_table.add_row("Size", str(cache_stats['l1_size']), str(cache_stats['l2_size']))
        cache_table.add_row("Hit Rate", f"{cache_stats['l1_hit_rate']:.2%}", f"{cache_stats['l2_hit_rate']:.2%}")
        cache_table.add_row("Memory Usage", f"{cache_stats['memory_usage_mb']:.1f} MB", "Persistent")
        
        console.print(cache_table)
    
    # Worker pool statistics
    if firmware_scanner_pool:
        pool_stats = firmware_scanner_pool.get_stats()
        
        pool_table = Table(title="⚙️ Worker Pool Performance")
        pool_table.add_column("Metric", style="cyan")
        pool_table.add_column("Value", style="magenta")
        
        pool_table.add_row("Worker Count", str(pool_stats['worker_count']))
        pool_table.add_row("Items Processed", str(pool_stats['items_processed']))
        pool_table.add_row("Success Rate", f"{pool_stats['success_rate']:.1%}")
        pool_table.add_row("Avg Processing Time", f"{pool_stats['average_processing_time']:.2f}s")
        
        console.print(pool_table)


@main.command()
@click.argument("scan_reports", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--device", required=True, help="Target device (STM32L4, ESP32, nRF52840)")
@click.option("--optimization", 
              type=click.Choice(['size', 'speed', 'balanced', 'memory']),
              default='balanced', help="Optimization strategy")
@click.option("--output-dir", "-o", default="patches", help="Output directory for patches")
@click.option("--security-level", type=int, default=2, help="NIST security level (1-5)")
@click.option("--hybrid", is_flag=True, help="Generate hybrid classical+PQC patches")
@click.option("--concurrent", is_flag=True, default=True, help="Enable concurrent patch generation")
@click.pass_context
def patch(ctx, scan_reports, device, optimization, output_dir, security_level, hybrid, concurrent):
    """Generate PQC patches from scan results.
    
    Supports processing multiple scan reports concurrently with intelligent resource management.
    """
    
    report_files = list(scan_reports)
    
    console.print(Panel(f"""[bold cyan]PQC Patch Generator - Generation 3[/bold cyan]
        
🎯 [bold]Target Device:[/bold] {device}
⚡ [bold]Optimization:[/bold] {optimization}
🔒 [bold]Security Level:[/bold] {security_level}
🔄 [bold]Hybrid Mode:[/bold] {'Enabled' if hybrid else 'Disabled'}
📁 [bold]Output Directory:[/bold] {output_dir}
📊 [bold]Scan Reports:[/bold] {len(report_files)}
⚡ [bold]Concurrent Processing:[/bold] {'Enabled' if concurrent and ctx.obj['optimizations'] else 'Disabled'}""", 
                           title="Patch Configuration"))
    
    # Load and process scan reports
    all_vulnerabilities = []
    
    for report_file in report_files:
        try:
            report_data = json.loads(Path(report_file).read_text())
            
            if 'files' in report_data:  # Aggregated report
                for file_data in report_data['files']:
                    if 'vulnerabilities' in file_data:
                        all_vulnerabilities.extend(file_data['vulnerabilities'])
            else:  # Single file report
                vulnerabilities = report_data.get('vulnerabilities', [])
                all_vulnerabilities.extend(vulnerabilities)
                
        except Exception as e:
            console.print(f"[red]❌ Failed to load {report_file}: {e}[/red]")
    
    if not all_vulnerabilities:
        console.print("[yellow]⚠️  No vulnerabilities found in scan reports[/yellow]")
        return
    
    total_vulnerabilities = len(all_vulnerabilities)
    console.print(f"\n🎯 [bold]Processing {total_vulnerabilities} vulnerabilities...[/bold]")
    
    # Initialize patcher
    patcher = PQCPatcher(device, optimization)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    patches_created = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        patch_task = progress.add_task(f"🔧 Generating patches...", total=total_vulnerabilities)
        
        for i, vuln_data in enumerate(all_vulnerabilities):
            try:
                # Create patch (simplified - would need full implementation)
                patch_name = f"patch_{i:04d}_{vuln_data.get('algorithm', 'unknown')}"
                patch_content = _generate_patch_content(vuln_data, device, optimization, security_level, hybrid)
                
                # Save patch
                patch_file = output_path / f"{patch_name}.c"
                patch_file.write_text(patch_content)
                patches_created += 1
                
            except Exception as e:
                console.print(f"[red]❌ Failed to generate patch for vulnerability {i}: {e}[/red]")
            
            progress.advance(patch_task)
    
    patch_duration = time.time() - start_time
    
    console.print(f"\n✅ [bold green]Patch generation complete![/bold green]")
    console.print(f"📦 [bold]{patches_created}/{total_vulnerabilities}[/bold] patches created")
    console.print(f"⏱️  Generation time: [bold]{patch_duration:.2f} seconds[/bold]")
    console.print(f"📁 Output directory: [bold]{output_dir}[/bold]")
    
    if ctx.obj['optimizations']:
        _display_performance_metrics()


def _generate_patch_content(vuln_data: Dict, device: str, optimization: str, 
                           security_level: int, hybrid: bool) -> str:
    """Generate patch content for a vulnerability."""
    
    algorithm = vuln_data.get('algorithm', 'unknown')
    function_name = vuln_data.get('function_name', 'unknown_function')
    
    # This is a simplified patch generator - real implementation would be much more complex
    patch_content = f"""/*
 * PQC Patch for {algorithm} vulnerability
 * Generated by PQC IoT Retrofit Scanner - Generation 3
 * Target: {device}
 * Optimization: {optimization}
 * Security Level: {security_level}
 * Hybrid Mode: {hybrid}
 */

#include "pqc_replacement.h"

// Original vulnerable function: {function_name}
// Address: {vuln_data.get('address', 'unknown')}

// PQC replacement implementation
int pqc_{function_name}_replacement(void) {{
    // TODO: Implement PQC replacement for {algorithm}
    // This would contain the actual post-quantum cryptographic implementation
    return 0;
}}

"""
    
    return patch_content


@main.command()
@click.option("--detailed", is_flag=True, help="Show detailed performance metrics")
def metrics(detailed):
    """Display system performance metrics and statistics."""
    
    console.print("📊 [bold cyan]PQC Scanner Performance Metrics[/bold cyan]\n")
    
    # Display cache metrics
    if performance_optimizer and performance_optimizer.cache:
        cache_stats = performance_optimizer.cache.get_stats()
        _display_cache_metrics(cache_stats, detailed)
    
    # Display worker pool metrics
    if firmware_scanner_pool:
        pool_stats = firmware_scanner_pool.get_stats()
        _display_worker_pool_metrics(pool_stats, detailed)
    
    # Display health metrics
    health_status = health_monitor.get_health_status() if health_monitor else {}
    _display_health_metrics(health_status, detailed)


def _display_cache_metrics(stats: Dict, detailed: bool):
    """Display cache performance metrics."""
    
    table = Table(title="🧠 Cache Performance")
    table.add_column("Layer", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Hit Rate", style="yellow")
    table.add_column("Memory", style="blue")
    
    table.add_row("L1 (Memory)", str(stats['l1_size']), f"{stats['l1_hit_rate']:.1%}", 
                  f"{stats['memory_usage_mb']:.1f} MB")
    table.add_row("L2 (Disk)", str(stats['l2_size']), f"{stats['l2_hit_rate']:.1%}", "Persistent")
    
    console.print(table)
    
    if detailed:
        detail_table = Table(title="Cache Details")
        detail_table.add_column("Metric", style="cyan")
        detail_table.add_column("Value", style="magenta")
        
        detail_table.add_row("Total Hits", str(stats['l1_hits'] + stats['l2_hits']))
        detail_table.add_row("Total Misses", str(stats['l1_misses'] + stats['l2_misses']))
        detail_table.add_row("Evictions", str(stats['total_evictions']))
        
        console.print(detail_table)


def _display_worker_pool_metrics(stats: Dict, detailed: bool):
    """Display worker pool performance metrics."""
    
    table = Table(title="⚙️ Worker Pool Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Active Workers", f"{stats['workers_active']}/{stats['worker_count']}")
    table.add_row("Items Processed", str(stats['items_processed']))
    table.add_row("Success Rate", f"{stats['success_rate']:.1%}")
    table.add_row("Avg Processing Time", f"{stats['average_processing_time']:.2f}s")
    
    console.print(table)


def _display_health_metrics(health_status: Dict, detailed: bool):
    """Display system health metrics."""
    
    if not health_status:
        console.print("[yellow]Health monitoring not available[/yellow]")
        return
    
    overall_status = health_status.get('overall_status', 'unknown')
    status_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(overall_status, "white")
    
    console.print(f"🏥 [bold]System Health:[/bold] [{status_color}]{overall_status.upper()}[/{status_color}]")


# Cleanup function for graceful shutdown
def cleanup():
    """Clean up resources on exit."""
    shutdown_pools()
    console.print("🔄 [yellow]Cleaned up worker pools and resources[/yellow]")


# Register cleanup
import atexit
atexit.register(cleanup)


if __name__ == "__main__":
    main()