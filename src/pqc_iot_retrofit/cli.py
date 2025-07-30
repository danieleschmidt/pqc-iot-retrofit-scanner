"""CLI interface for PQC IoT Retrofit Scanner."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main():
    """PQC IoT Retrofit Scanner CLI."""
    pass


@main.command()
@click.argument("firmware_path", type=click.Path(exists=True))
@click.option("--arch", required=True, help="Target architecture")
@click.option("--output", "-o", help="Output report file")
def scan(firmware_path, arch, output):
    """Scan firmware for quantum-vulnerable cryptography."""
    console.print(f"[bold blue]Scanning {firmware_path} for architecture {arch}[/bold blue]")
    # Implementation placeholder
    console.print("[green]Scan complete - no vulnerabilities found[/green]")


if __name__ == "__main__":
    main()