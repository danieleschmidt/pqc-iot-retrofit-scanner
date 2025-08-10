"""PQC IoT Retrofit Scanner - Generation 4.

Advanced AI-powered CLI and library for auditing embedded firmware and 
generating post-quantum cryptography solutions with adaptive intelligence,
quantum resilience analysis, and autonomous research capabilities.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from .scanner import FirmwareScanner
from .patcher import PQCPatcher

# Generation 4: Advanced AI and Research Capabilities
from .adaptive_ai import adaptive_ai, AdaptiveAI, EnsembleDetector, AnomalyDetector
from .quantum_resilience import quantum_resilience, QuantumResilienceAnalyzer
from .autonomous_research import autonomous_researcher, AutonomousResearcher

__all__ = [
    # Core Components
    "FirmwareScanner", 
    "PQCPatcher",
    
    # Generation 4: AI & Research
    "adaptive_ai",
    "AdaptiveAI", 
    "EnsembleDetector",
    "AnomalyDetector",
    "quantum_resilience",
    "QuantumResilienceAnalyzer",
    "autonomous_researcher", 
    "AutonomousResearcher"
]