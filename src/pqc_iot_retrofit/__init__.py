"""PQC IoT Retrofit Scanner.

A CLI and library for auditing embedded firmware and suggesting 
post-quantum cryptography drop-ins.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from .scanner import FirmwareScanner
from .patcher import PQCPatcher

__all__ = ["FirmwareScanner", "PQCPatcher"]