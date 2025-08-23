"""Basic CLI interface for PQC IoT Retrofit Scanner.

Simple, reliable CLI interface with essential functionality:
- Firmware scanning with automatic architecture detection
- Basic report generation 
- Error handling and validation
- Progress indication
"""

import json
import sys
import click
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import core components
from .scanner import FirmwareScanner, RiskLevel
from .utils import (
    ArchitectureDetector, create_firmware_info, validate_firmware_path,
    format_size, format_address, FirmwareInfo
)

# Setup console and logging
console = Console()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0", package_name="pqc-iot-retrofit-scanner")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(verbose):
    """PQC IoT Retrofit Scanner - Basic CLI
    
    Scan IoT firmware for quantum-vulnerable cryptography and get 
    recommendations for post-quantum cryptography migration.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")


@main.command()
@click.argument("firmware_path", type=click.Path(exists=True))
@click.option("--arch", 
              type=click.Choice(['cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7', 
                               'esp32', 'riscv32', 'avr', 'auto']),
              default='auto',
              help="Target architecture (auto-detect if not specified)")
@click.option("--output", "-o", type=click.Path(), 
              help="Output report file (JSON format)")
@click.option("--base-address", type=str, default="0x0",
              help="Base memory address in hex (default: 0x0)")
@click.option("--format", "output_format", 
              type=click.Choice(['json', 'table', 'summary']),
              default='table',
              help="Output format")
def scan(firmware_path: str, arch: str, output: Optional[str], 
         base_address: str, output_format: str):
    """Scan firmware binary for quantum-vulnerable cryptography.
    
    Examples:
      pqc-iot scan firmware.bin --arch cortex-m4
      pqc-iot scan firmware.elf --arch auto --output report.json
    """
    
    # Validate inputs
    if not validate_firmware_path(firmware_path):
        console.print("[red]❌ Invalid firmware file[/red]")
        sys.exit(1)
    
    # Parse base address
    try:
        base_addr = int(base_address, 16)
    except ValueError:
        console.print(f"[red]❌ Invalid base address '{base_address}'. Use hex format (e.g., 0x08000000)[/red]")
        sys.exit(1)
    
    # Create firmware info
    firmware_info = create_firmware_info(firmware_path)
    
    # Auto-detect architecture if requested
    if arch == 'auto':
        detected_arch = firmware_info.architecture
        if detected_arch:
            arch = detected_arch
            console.print(f"[blue]🔍 Auto-detected architecture: {arch}[/blue]")
        else:
            console.print("[yellow]⚠️  Could not auto-detect architecture. Please specify with --arch[/yellow]")
            console.print("Supported architectures: cortex-m0, cortex-m3, cortex-m4, cortex-m7, esp32, riscv32, avr")
            sys.exit(1)
    
    # Display scan configuration
    _display_scan_config(firmware_info, arch, base_addr)
    
    # Perform scan
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            scan_task = progress.add_task("🔍 Scanning firmware...", total=None)
            
            # Initialize scanner
            scanner = FirmwareScanner(arch)
            
            # Scan firmware
            progress.update(scan_task, description="📖 Loading and analyzing firmware...")
            vulnerabilities = scanner.scan_firmware(firmware_path, base_addr)
            
            progress.update(scan_task, description="📊 Generating report...")
            report = scanner.generate_report()
            
            progress.update(scan_task, description="✅ Scan complete")
        
        # Display results
        _display_results(report, vulnerabilities, output_format)
        
        # Save output if requested
        if output:
            _save_report(report, output)
            console.print(f"\n[green]📄 Report saved to {output}[/green]")
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        console.print(f"[red]❌ Scan failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("firmware_path", type=click.Path(exists=True))
def info(firmware_path: str):
    """Display detailed information about firmware file.
    
    Shows file format, architecture, size, and other metadata.
    """
    
    if not validate_firmware_path(firmware_path):
        console.print("[red]❌ Invalid firmware file[/red]")
        sys.exit(1)
    
    firmware_info = create_firmware_info(firmware_path)
    _display_firmware_info(firmware_info)


@main.command()
def architectures():
    """List supported target architectures."""
    
    arch_table = Table(title="Supported Architectures")
    arch_table.add_column("Architecture", style="cyan")
    arch_table.add_column("Description", style="white")
    arch_table.add_column("Common Devices", style="yellow")
    
    architectures = [
        ("cortex-m0", "ARM Cortex-M0", "Simple microcontrollers"),
        ("cortex-m3", "ARM Cortex-M3", "STM32F1, LPC1700 series"),
        ("cortex-m4", "ARM Cortex-M4", "STM32F4, nRF52, ATSAM"),
        ("cortex-m7", "ARM Cortex-M7", "STM32F7/H7 series"),
        ("esp32", "Espressif ESP32", "ESP32, ESP32-S2, ESP32-S3"),
        ("riscv32", "RISC-V 32-bit", "SiFive, GigaDevice GD32V"),
        ("avr", "Atmel AVR", "Arduino, ATmega series"),
    ]
    
    for arch, desc, devices in architectures:
        arch_table.add_row(arch, desc, devices)
    
    console.print(arch_table)
    console.print("\n[dim]💡 Use 'auto' to enable automatic architecture detection[/dim]")


def _display_scan_config(firmware_info: FirmwareInfo, arch: str, base_addr: int):
    """Display scan configuration panel."""
    
    config_text = f"""[bold cyan]Firmware Analysis Configuration[/bold cyan]

📁 [bold]File:[/bold] {firmware_info.path}
📊 [bold]Size:[/bold] {format_size(firmware_info.size)}
🔧 [bold]Format:[/bold] {firmware_info.format.value.upper()}
🏗️  [bold]Architecture:[/bold] {arch}
📍 [bold]Base Address:[/bold] {format_address(base_addr)}
🔍 [bold]Checksum:[/bold] {firmware_info.checksum[:16]}..."""
    
    console.print(Panel(config_text, title="Scan Configuration"))


def _display_firmware_info(firmware_info: FirmwareInfo):
    """Display detailed firmware information."""
    
    info_table = Table(title=f"Firmware Information: {Path(firmware_info.path).name}")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("File Path", firmware_info.path)
    info_table.add_row("File Size", format_size(firmware_info.size))
    info_table.add_row("Format", firmware_info.format.value.upper())
    info_table.add_row("Architecture", firmware_info.architecture or "Unknown")
    info_table.add_row("Checksum (SHA256)", firmware_info.checksum)
    
    if firmware_info.entry_point:
        info_table.add_row("Entry Point", format_address(firmware_info.entry_point))
    
    console.print(info_table)


def _display_results(report: Dict[str, Any], vulnerabilities: list, 
                    output_format: str):
    """Display scan results in specified format."""
    
    summary = report['scan_summary']
    total_vulns = summary['total_vulnerabilities']
    
    if total_vulns == 0:
        console.print("\n[green]🎉 No quantum vulnerabilities detected![/green]")
        console.print("[dim]Your firmware appears to be using quantum-safe cryptography.[/dim]")
        return
    
    if output_format == 'json':
        console.print(json.dumps(report, indent=2))
    elif output_format == 'summary':
        _display_summary(summary)
    else:  # table format (default)
        _display_table_results(report, vulnerabilities)


def _display_summary(summary: Dict[str, Any]):
    """Display summary results."""
    
    total_vulns = summary['total_vulnerabilities']
    risk_dist = summary['risk_distribution']
    
    console.print(f"\n[bold]📊 Scan Results Summary[/bold]")
    console.print(f"Total vulnerabilities found: [red]{total_vulns}[/red]")
    
    if risk_dist.get('critical', 0) > 0:
        console.print(f"[red]🚨 Critical risks: {risk_dist['critical']}[/red]")
    if risk_dist.get('high', 0) > 0:
        console.print(f"[yellow]⚠️  High risks: {risk_dist['high']}[/yellow]")
    if risk_dist.get('medium', 0) > 0:
        console.print(f"[blue]ℹ️  Medium risks: {risk_dist['medium']}[/blue]")


def _display_table_results(report: Dict[str, Any], vulnerabilities: list):
    """Display detailed table results."""
    
    summary = report['scan_summary']
    total_vulns = summary['total_vulnerabilities']
    
    # Risk distribution table
    risk_table = Table(title="Vulnerability Risk Distribution")
    risk_table.add_column("Risk Level", style="bold")
    risk_table.add_column("Count", justify="right")
    risk_table.add_column("Percentage", justify="right")
    
    risk_dist = summary['risk_distribution']
    for level, count in risk_dist.items():
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
    
    # Detailed vulnerabilities (show first 5)
    if vulnerabilities:
        vuln_table = Table(title=f"Detected Vulnerabilities (showing first 5 of {total_vulns})")
        vuln_table.add_column("Algorithm", style="yellow")
        vuln_table.add_column("Address", style="magenta")
        vuln_table.add_column("Risk", style="red")
        vuln_table.add_column("Recommended Fix", style="green")
        
        for vuln in vulnerabilities[:5]:
            risk_style = "red" if vuln.risk_level == RiskLevel.CRITICAL else "yellow"
            vuln_table.add_row(
                vuln.algorithm.value,
                format_address(vuln.address),
                f"[{risk_style}]{vuln.risk_level.value.upper()}[/{risk_style}]",
                vuln.mitigation[:60] + "..." if len(vuln.mitigation) > 60 else vuln.mitigation
            )
        
        console.print("\n")
        console.print(vuln_table)
        
        if total_vulns > 5:
            console.print(f"[dim]... and {total_vulns - 5} more vulnerabilities[/dim]")
    
    # Recommendations
    if report.get('recommendations'):
        console.print("\n[bold yellow]🔍 Recommendations:[/bold yellow]")
        for i, rec in enumerate(report['recommendations'], 1):
            console.print(f"  {i}. {rec}")


def _save_report(report: Dict[str, Any], output_path: str):
    """Save report to file."""
    
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        report['metadata'] = {
            'version': '1.0.0',
            'tool': 'pqc-iot-retrofit-scanner-basic',
            'format_version': '1.0'
        }
        
        output_file.write_text(json.dumps(report, indent=2))
        logger.info(f"Report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        raise


if __name__ == "__main__":
    main()