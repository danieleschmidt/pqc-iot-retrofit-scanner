"""PQC IoT Retrofit Scanner - Generation 5.

Revolutionary AI-powered CLI and library for auditing embedded firmware and 
generating post-quantum cryptography solutions with quantum-enhanced ML analysis,
autonomous research breakthroughs, and real-time security orchestration.
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from .scanner import FirmwareScanner
from .patcher import PQCPatcher

# Generation 4: Advanced AI and Research Capabilities
from .adaptive_ai import adaptive_ai, AdaptiveAI, EnsembleDetector, AnomalyDetector
from .quantum_resilience import quantum_resilience, QuantumResilienceAnalyzer
from .autonomous_research import autonomous_researcher, AutonomousResearcher

# Generation 5: Breakthrough Capabilities
from .quantum_ml_analysis import (
    quantum_enhanced_analysis, QuantumCryptographicAnalyzer, 
    QuantumNeuralNetwork, adaptive_quantum_analysis
)
from .research_breakthrough import (
    autonomous_research_breakthrough, AdvancedCryptographicResearcher,
    NovelAlgorithm, ResearchBreakthrough
)
from .realtime_security_orchestrator import (
    create_security_orchestrator, RealTimeSecurityOrchestrator,
    IoTDevice, SecurityThreat
)

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
    "AutonomousResearcher",
    
    # Generation 5: Breakthrough Technologies
    "quantum_enhanced_analysis",
    "QuantumCryptographicAnalyzer",
    "QuantumNeuralNetwork", 
    "adaptive_quantum_analysis",
    "autonomous_research_breakthrough",
    "AdvancedCryptographicResearcher",
    "NovelAlgorithm",
    "ResearchBreakthrough",
    "create_security_orchestrator",
    "RealTimeSecurityOrchestrator",
    "IoTDevice",
    "SecurityThreat"
]