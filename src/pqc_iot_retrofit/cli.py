"""CLI interface for PQC IoT Retrofit Scanner with Generation 3 optimizations."""

import json
import click
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.panel import Panel

from .scanner import FirmwareScanner, RiskLevel
from .patcher import PQCPatcher, OptimizationLevel

# Generation 3 optimizations
from .performance import performance_optimizer
from .concurrency import initialize_pools, shutdown_pools, firmware_scanner_pool
from .monitoring import metrics_collector

console = Console()

# Global flag for optimizations
_optimizations_enabled = True


@click.group()
@click.version_option()
@click.option("--enable-gen3", is_flag=True, default=True, help="Enable Generation 3 optimizations")
@click.option("--max-workers", type=int, default=None, help="Maximum worker threads")
def main(enable_gen3, max_workers):
    """PQC IoT Retrofit Scanner CLI - Detect and patch quantum-vulnerable cryptography in IoT firmware.
    
    🚀 Generation 3 Features:
    ✨ Intelligent caching for faster repeated scans
    ⚡ Concurrent processing with auto-scaling
    📊 Real-time performance monitoring
    """
    
    global _optimizations_enabled
    _optimizations_enabled = enable_gen3
    
    if enable_gen3:
        console.print("🚀 [bold cyan]Generation 3 optimizations enabled[/bold cyan]")
        
        # Initialize worker pools
        try:
            initialize_pools(
                scanner_class=FirmwareScanner,
                scanner_workers=max_workers
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize worker pools: {e}[/yellow]")
            _optimizations_enabled = False


@main.command()
@click.argument("firmware_path", type=click.Path(exists=True))
@click.option("--arch", required=True, 
              type=click.Choice(['cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7', 'esp32', 'riscv32', 'avr']),
              help="Target architecture")
@click.option("--output", "-o", help="Output report file (JSON)")
@click.option("--base-address", type=str, default="0x0", help="Base memory address (hex)")
@click.option("--flash-size", type=int, help="Flash memory size in bytes")
@click.option("--ram-size", type=int, help="RAM size in bytes")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def scan(firmware_path, arch, output, base_address, flash_size, ram_size, verbose):
    """Scan firmware for quantum-vulnerable cryptography."""
    
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
    
    gen3_status = "Enabled" if _optimizations_enabled else "Disabled"
    console.print(Panel(f"""[bold cyan]PQC IoT Retrofit Scanner - Generation 3[/bold cyan]
    
🎯 [bold]Target:[/bold] {firmware_path}
🏗️  [bold]Architecture:[/bold] {arch}
📍 [bold]Base Address:[/bold] {base_address}
💾 [bold]Memory Constraints:[/bold] {memory_constraints or 'Auto-detected'}
⚡ [bold]Gen3 Optimizations:[/bold] {gen3_status}""", 
                       title="Scan Configuration"))
    
    start_time = time.time()
    
    try:
        # Initialize scanner with potential optimizations
        scanner = FirmwareScanner(arch, memory_constraints)
        
        # Apply Generation 3 optimizations if enabled
        if _optimizations_enabled and performance_optimizer:
            original_scan = scanner.scan_firmware
            scanner.scan_firmware = performance_optimizer.optimize_firmware_scanning(original_scan)
            
            if verbose:
                console.print("🧠 [blue]Applied intelligent caching optimizations[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            # Scanning phases
            scan_task = progress.add_task("🔍 Analyzing firmware binary...", total=100)
            
            # Simulate progressive scanning
            progress.update(scan_task, advance=20, description="📖 Loading firmware...")
            vulnerabilities = scanner.scan_firmware(str(firmware_path), base_addr)
            progress.update(scan_task, advance=60, description="🔍 Pattern matching...")
            
            # Generate detailed report
            report = scanner.generate_report()
            progress.update(scan_task, advance=20, description="✅ Analysis complete")
        
        scan_duration = time.time() - start_time
        
        # Display results with performance info
        _display_scan_results(report, vulnerabilities, verbose, scan_duration)
        
        # Display Generation 3 metrics if enabled
        if _optimizations_enabled and verbose:
            _display_gen3_metrics()
        
        # Save report if requested
        if output:
            # Enhance report with Generation 3 metadata
            enhanced_report = report.copy()
            enhanced_report['scan_metadata'] = {
                'generation_3_enabled': _optimizations_enabled,
                'scan_duration': scan_duration,
                'timestamp': time.time()
            }
            
            output_path = Path(output)
            output_path.write_text(json.dumps(enhanced_report, indent=2))
            console.print(f"\n[green]📄 Report saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Scan failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


@main.command()
@click.argument("scan_report", type=click.Path(exists=True))
@click.option("--device", required=True, help="Target device (STM32L4, ESP32, nRF52840)")
@click.option("--optimization", 
              type=click.Choice(['size', 'speed', 'balanced', 'memory']),
              default='balanced', help="Optimization strategy")
@click.option("--output-dir", "-o", default="patches", help="Output directory for patches")
@click.option("--security-level", type=int, default=2, help="NIST security level (1-5)")
@click.option("--hybrid", is_flag=True, help="Generate hybrid classical+PQC patches")
def patch(scan_report, device, optimization, output_dir, security_level, hybrid):
    """Generate PQC patches from scan results."""
    
    try:
        # Load scan report
        report_data = json.loads(Path(scan_report).read_text())
        vulnerabilities = report_data.get('vulnerabilities', [])
        
        if not vulnerabilities:
            console.print("[yellow]⚠️  No vulnerabilities found in scan report[/yellow]")
            return
        
        console.print(Panel(f"""[bold cyan]PQC Patch Generator[/bold cyan]
        
🎯 [bold]Target Device:[/bold] {device}
⚡ [bold]Optimization:[/bold] {optimization}
🔒 [bold]Security Level:[/bold] {security_level}
🔄 [bold]Hybrid Mode:[/bold] {'Enabled' if hybrid else 'Disabled'}
📁 [bold]Output Directory:[/bold] {output_dir}""", 
                           title="Patch Configuration"))
        
        # Initialize patcher
        patcher = PQCPatcher(device, optimization)
        patches_created = 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            patch_task = progress.add_task(f"🔧 Generating patches for {len(vulnerabilities)} vulnerabilities...", 
                                         total=len(vulnerabilities))
            
            for i, vuln_data in enumerate(vulnerabilities):
                try:
                    # Reconstruct vulnerability object (simplified)
                    from .scanner import CryptoVulnerability, CryptoAlgorithm, RiskLevel
                    
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
                    
                    # Generate appropriate patch
                    if hybrid:
                        patch = patcher.generate_hybrid_patch(vuln, transition_period=30)
                    elif vuln.algorithm.value.startswith('RSA') or vuln.algorithm.value.startswith('ECDSA'):
                        patch = patcher.create_dilithium_patch(vuln, security_level)
                    else:
                        patch = patcher.create_kyber_patch(vuln, security_level)
                    
                    # Optimize patch for target device
                    patch = patcher.optimize_for_device(patch)
                    
                    # Validate constraints
                    violations = patcher.validate_patch_constraints(patch)
                    if violations:
                        console.print(f"[yellow]⚠️  Patch validation warnings for {vuln.function_name}:[/yellow]")
                        for violation in violations:
                            console.print(f"   • {violation}")
                    
                    # Save patch
                    patch_filename = f"{vuln.function_name}_{patch.patch_metadata['algorithm']}.patch"
                    patch.save(output_path / patch_filename)
                    patches_created += 1
                    
                    progress.update(patch_task, advance=1, 
                                  description=f"✅ Generated patch: {patch_filename}")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to generate patch for {vuln_data['function_name']}: {e}[/red]")
                    progress.update(patch_task, advance=1)
        
        console.print(f"\n[green]🎉 Successfully generated {patches_created} patches in {output_dir}[/green]")
        
        # Display patch summary
        _display_patch_summary(output_path)
        
    except Exception as e:
        console.print(f"[red]❌ Patch generation failed: {e}[/red]")


@main.command()
@click.argument("firmware_path", type=click.Path(exists=True))
@click.option("--arch", required=True, help="Target architecture")
@click.option("--device", required=True, help="Target device")
@click.option("--output-dir", "-o", default="analysis", help="Output directory")
@click.option("--generate-patches", is_flag=True, help="Generate patches automatically")
@click.option("--security-level", type=int, default=2, help="Security level for patches")
def analyze(firmware_path, arch, device, output_dir, generate_patches, security_level):
    """Complete firmware analysis and optional patch generation."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel(f"""[bold cyan]Complete Firmware Analysis[/bold cyan]
    
🎯 [bold]Firmware:[/bold] {firmware_path}
🏗️  [bold]Architecture:[/bold] {arch}
📱 [bold]Device:[/bold] {device}
🔧 [bold]Generate Patches:[/bold] {'Yes' if generate_patches else 'No'}""", 
                       title="Analysis Configuration"))
    
    try:
        # Step 1: Scan firmware
        scanner = FirmwareScanner(arch)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            scan_task = progress.add_task("🔍 Scanning firmware...", total=None)
            vulnerabilities = scanner.scan_firmware(str(firmware_path))
            report = scanner.generate_report()
            progress.update(scan_task, description="✅ Scan complete")
            
            # Save scan report
            scan_report_path = output_path / "scan_report.json"
            scan_report_path.write_text(json.dumps(report, indent=2))
            
            if generate_patches and vulnerabilities:
                # Step 2: Generate patches
                patcher = PQCPatcher(device)
                patch_task = progress.add_task(f"🔧 Generating patches...", total=len(vulnerabilities))
                
                patches_dir = output_path / "patches"
                patches_dir.mkdir(exist_ok=True)
                
                for vuln in vulnerabilities:
                    try:
                        if vuln.algorithm.value.startswith('RSA') or vuln.algorithm.value.startswith('ECDSA'):
                            patch = patcher.create_dilithium_patch(vuln, security_level)
                        else:
                            patch = patcher.create_kyber_patch(vuln, security_level)
                        
                        patch_filename = f"{vuln.function_name}_{patch.patch_metadata['algorithm']}.patch"
                        patch.save(patches_dir / patch_filename)
                        progress.update(patch_task, advance=1)
                        
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Skipped patch for {vuln.function_name}: {e}[/yellow]")
                        progress.update(patch_task, advance=1)
        
        # Display comprehensive results
        _display_scan_results(report, vulnerabilities, verbose=True)
        
        if generate_patches:
            _display_patch_summary(patches_dir)
        
        console.print(f"\n[green]📁 Analysis complete! Results saved to {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")


def _display_scan_results(report, vulnerabilities, verbose=False, scan_duration=None):
    """Display scan results in a formatted table."""
    
    summary = report['scan_summary']
    
    # Summary panel
    risk_counts = summary['risk_distribution']
    total_vulns = summary['total_vulnerabilities']
    
    if total_vulns == 0:
        console.print("\n[green]🎉 No quantum vulnerabilities detected![/green]")
        return
    
    # Risk distribution
    risk_table = Table(title="Risk Distribution")
    risk_table.add_column("Risk Level", style="bold")
    risk_table.add_column("Count", justify="right")
    risk_table.add_column("Percentage", justify="right")
    
    for level, count in risk_counts.items():
        if count > 0:
            percentage = (count / total_vulns) * 100
            style = "red" if level == "critical" else "yellow" if level == "high" else "blue"
            risk_table.add_row(
                f"[{style}]{level.upper()}[/{style}]",
                str(count),
                f"{percentage:.1f}%"
            )
    
    console.print("\n")
    console.print(risk_table)
    
    # Detailed vulnerabilities
    if verbose and vulnerabilities:
        vuln_table = Table(title=f"Detected Vulnerabilities ({total_vulns})")
        vuln_table.add_column("Function", style="cyan")
        vuln_table.add_column("Algorithm", style="yellow")
        vuln_table.add_column("Address", style="magenta")
        vuln_table.add_column("Risk", style="red")
        vuln_table.add_column("Mitigation", style="green")
        
        for vuln in vulnerabilities[:10]:  # Show first 10
            risk_style = "red" if vuln.risk_level == RiskLevel.CRITICAL else "yellow"
            vuln_table.add_row(
                vuln.function_name,
                vuln.algorithm.value,
                f"0x{vuln.address:08x}",
                f"[{risk_style}]{vuln.risk_level.value}[/{risk_style}]",
                vuln.mitigation[:50] + "..." if len(vuln.mitigation) > 50 else vuln.mitigation
            )
        
        if total_vulns > 10:
            vuln_table.add_row("...", "...", "...", "...", f"({total_vulns - 10} more)")
        
        console.print("\n")
        console.print(vuln_table)
    
    # Performance info if available
    if scan_duration is not None:
        console.print(f"\n⏱️  [bold]Scan Duration:[/bold] {scan_duration:.2f} seconds")
    
    # Recommendations
    if report.get('recommendations'):
        console.print("\n[bold yellow]🔍 Recommendations:[/bold yellow]")
        for i, rec in enumerate(report['recommendations'], 1):
            console.print(f"  {i}. {rec}")


def _display_patch_summary(patches_dir):
    """Display summary of generated patches."""
    
    patch_files = list(Path(patches_dir).glob("*.patch"))
    
    if not patch_files:
        console.print("[yellow]⚠️  No patches generated[/yellow]")
        return
    
    console.print(f"\n[bold green]📦 Generated Patches ({len(patch_files)}):[/bold green]")
    
    patch_table = Table()
    patch_table.add_column("Patch File", style="cyan")  
    patch_table.add_column("Algorithm", style="yellow")
    patch_table.add_column("Size", justify="right")
    
    for patch_file in sorted(patch_files):
        # Load metadata
        metadata_file = patch_file.with_suffix('.json')
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                algorithm = metadata.get('algorithm', 'unknown')
                size = f"{patch_file.stat().st_size} bytes"
                patch_table.add_row(patch_file.name, algorithm, size)
            except:
                patch_table.add_row(patch_file.name, "unknown", f"{patch_file.stat().st_size} bytes")
        else:
            patch_table.add_row(patch_file.name, "unknown", f"{patch_file.stat().st_size} bytes")
    
    console.print(patch_table)
    console.print(f"\n[dim]💡 Use the installation scripts (.sh files) to apply patches[/dim]")


def _display_gen3_metrics():
    """Display Generation 3 performance metrics."""
    
    console.print("\n⚡ [bold cyan]Generation 3 Performance Metrics:[/bold cyan]")
    
    # Cache performance
    if performance_optimizer and performance_optimizer.cache:
        cache_stats = performance_optimizer.cache.get_stats()
        
        cache_table = Table(title="🧠 Cache Performance")
        cache_table.add_column("Layer", style="cyan")
        cache_table.add_column("Hit Rate", style="green")
        cache_table.add_column("Size", style="yellow")
        
        cache_table.add_row("L1 (Memory)", f"{cache_stats['l1_hit_rate']:.1%}", str(cache_stats['l1_size']))
        cache_table.add_row("L2 (Disk)", f"{cache_stats['l2_hit_rate']:.1%}", str(cache_stats['l2_size']))
        
        console.print(cache_table)
    
    # Worker pool performance
    if firmware_scanner_pool:
        pool_stats = firmware_scanner_pool.get_stats()
        
        pool_table = Table(title="⚙️ Worker Pool Performance")
        pool_table.add_column("Metric", style="cyan")
        pool_table.add_column("Value", style="magenta")
        
        pool_table.add_row("Workers", f"{pool_stats['workers_active']}/{pool_stats['worker_count']}")
        pool_table.add_row("Items Processed", str(pool_stats['items_processed']))
        pool_table.add_row("Success Rate", f"{pool_stats['success_rate']:.1%}")
        
        console.print(pool_table)


@main.command()
def metrics():
    """Display Generation 3 performance metrics."""
    _display_gen3_metrics()


# Cleanup function for graceful shutdown
def cleanup():
    """Clean up Generation 3 resources on exit."""
    if _optimizations_enabled:
        try:
            shutdown_pools()
            console.print("🔄 [dim]Cleaned up Generation 3 resources[/dim]")
        except Exception:
            pass  # Ignore cleanup errors


# Register cleanup handler
import atexit
atexit.register(cleanup)


if __name__ == "__main__":
    main()