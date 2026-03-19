"""
pqc-scan CLI — Post-Quantum Cryptography IoT Firmware Scanner

Usage:
    pqc-scan /path/to/firmware/
    pqc-scan /path/to/firmware/ --format json
    pqc-scan /path/to/firmware/ --format text --min-severity HIGH
    pqc-scan /path/to/firmware/ --output report.json
"""

import sys
import json
from pathlib import Path

import click

from .scanner import PQCScanner, Severity


SEVERITY_CHOICES = [s.value for s in Severity]


@click.command(name="pqc-scan")
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output", "-o", "output_file",
    type=click.Path(),
    default=None,
    help="Write results to this file (default: stdout).",
)
@click.option(
    "--min-severity",
    type=click.Choice(SEVERITY_CHOICES, case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Minimum severity to include in output.",
)
@click.option(
    "--exclude", multiple=True,
    default=[".git", "__pycache__", "node_modules", ".venv", "venv"],
    show_default=True,
    help="Directory names to skip (repeatable).",
)
@click.option(
    "--fail-on",
    type=click.Choice(SEVERITY_CHOICES, case_sensitive=False),
    default=None,
    help="Exit with code 1 if findings at or above this severity exist (CI integration).",
)
def main(target, output_format, output_file, min_severity, exclude, fail_on):
    """Scan TARGET (file or directory) for quantum-vulnerable cryptographic usage
    and recommend post-quantum replacements."""

    sev = Severity(min_severity.upper())
    scanner = PQCScanner(
        path=Path(target),
        exclude_dirs=list(exclude),
        min_severity=sev,
    )

    click.echo(f"Scanning {target} …", err=True)
    report = scanner.scan()
    click.echo(f"Done. {report.files_scanned} files, {report.total} findings.", err=True)

    if output_format == "json":
        result_text = report.to_json()
    else:
        result_text = report.to_text()

    if output_file:
        Path(output_file).write_text(result_text)
        click.echo(f"Results written to {output_file}", err=True)
    else:
        click.echo(result_text)

    # CI exit-code support
    if fail_on:
        fail_sev = Severity(fail_on.upper())
        sev_order = [Severity.INFO, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        threshold_idx = sev_order.index(fail_sev)
        for f in report.findings:
            if sev_order.index(f.severity) >= threshold_idx:
                sys.exit(1)


if __name__ == "__main__":
    main()
