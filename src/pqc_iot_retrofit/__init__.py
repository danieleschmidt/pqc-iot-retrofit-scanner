"""PQC IoT Retrofit Scanner - Generation 5.

Revolutionary AI-powered CLI and library for auditing embedded firmware and 
generating post-quantum cryptography solutions with quantum-enhanced ML analysis,
autonomous research breakthroughs, and real-time security orchestration.
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

# Core components (always available)
from .scanner import FirmwareScanner
from .patcher import PQCPatcher

# Advanced features (import with graceful fallback)
_advanced_features_available = True

try:
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
except ImportError as e:
    # Graceful fallback for missing dependencies
    _advanced_features_available = False
    
    # Create placeholder functions/classes
    def _missing_dependency_error(feature_name):
        def wrapper(*args, **kwargs):
            raise ImportError(f"Advanced feature '{feature_name}' requires additional dependencies. "
                            f"Install with: pip install pqc-iot-retrofit-scanner[analysis]")
        return wrapper
    
    # Placeholder for advanced features
    adaptive_ai = _missing_dependency_error("adaptive_ai")
    AdaptiveAI = _missing_dependency_error("AdaptiveAI")
    EnsembleDetector = _missing_dependency_error("EnsembleDetector")
    AnomalyDetector = _missing_dependency_error("AnomalyDetector")
    quantum_resilience = _missing_dependency_error("quantum_resilience")
    QuantumResilienceAnalyzer = _missing_dependency_error("QuantumResilienceAnalyzer")
    autonomous_researcher = _missing_dependency_error("autonomous_researcher")
    AutonomousResearcher = _missing_dependency_error("AutonomousResearcher")
    quantum_enhanced_analysis = _missing_dependency_error("quantum_enhanced_analysis")
    QuantumCryptographicAnalyzer = _missing_dependency_error("QuantumCryptographicAnalyzer")
    QuantumNeuralNetwork = _missing_dependency_error("QuantumNeuralNetwork")
    adaptive_quantum_analysis = _missing_dependency_error("adaptive_quantum_analysis")
    autonomous_research_breakthrough = _missing_dependency_error("autonomous_research_breakthrough")
    AdvancedCryptographicResearcher = _missing_dependency_error("AdvancedCryptographicResearcher")
    NovelAlgorithm = _missing_dependency_error("NovelAlgorithm")
    ResearchBreakthrough = _missing_dependency_error("ResearchBreakthrough")
    create_security_orchestrator = _missing_dependency_error("create_security_orchestrator")
    RealTimeSecurityOrchestrator = _missing_dependency_error("RealTimeSecurityOrchestrator")
    IoTDevice = _missing_dependency_error("IoTDevice")
    SecurityThreat = _missing_dependency_error("SecurityThreat")

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