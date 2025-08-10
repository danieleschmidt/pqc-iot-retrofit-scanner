"""Generation 4: Quantum Resilience Module.

Advanced quantum threat modeling, cryptographic agility framework,
and future-proof security assessment capabilities.
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from .scanner import CryptoAlgorithm, RiskLevel
from .adaptive_ai import AdaptiveAI, AdaptivePatch
from .monitoring import track_performance, metrics_collector
from .error_handling import handle_errors, ValidationError


class QuantumThreatLevel(Enum):
    """Quantum threat assessment levels based on NIST timeline."""
    IMMEDIATE = "immediate"      # Quantum computer exists now
    NEAR_TERM = "near_term"      # 5-10 years
    MID_TERM = "mid_term"        # 10-20 years  
    LONG_TERM = "long_term"      # 20+ years
    NEGLIGIBLE = "negligible"    # Quantum-safe


class CryptoAgility(Enum):
    """Cryptographic agility levels."""
    RIGID = "rigid"              # Hard-coded algorithms
    CONFIGURABLE = "configurable" # Algorithm selection at build time
    DYNAMIC = "dynamic"          # Runtime algorithm switching
    ADAPTIVE = "adaptive"        # AI-driven algorithm selection


class MigrationStrategy(Enum):
    """PQC migration strategy types."""
    IMMEDIATE_REPLACEMENT = "immediate_replacement"
    GRADUAL_TRANSITION = "gradual_transition"
    HYBRID_COEXISTENCE = "hybrid_coexistence"
    STAGED_ROLLOUT = "staged_rollout"


@dataclass
class QuantumVulnerability:
    """Quantum-specific vulnerability assessment."""
    algorithm: CryptoAlgorithm
    threat_level: QuantumThreatLevel
    cryptanalytic_margin: float  # Years until quantum break
    shor_applicable: bool
    grover_applicable: bool
    estimated_qubits_required: int
    estimated_gate_count: int
    current_quantum_attacks: List[str]
    projected_quantum_attacks: List[str]
    quantum_advantage_threshold: float
    

@dataclass
class ResilienceAssessment:
    """Comprehensive quantum resilience assessment."""
    overall_resilience_score: float
    quantum_vulnerabilities: List[QuantumVulnerability]
    crypto_agility_level: CryptoAgility
    migration_readiness: float
    estimated_migration_cost: float
    recommended_strategy: MigrationStrategy
    timeline_recommendations: Dict[str, str]
    compliance_status: Dict[str, bool]
    

@dataclass
class PQCAlgorithmProfile:
    """Post-quantum algorithm security profile."""
    name: str
    nist_security_level: int
    algorithm_type: str  # signature, kem, hash
    key_sizes: Dict[str, int]
    signature_sizes: Dict[str, int]
    performance_characteristics: Dict[str, float]
    security_assumptions: List[str]
    known_attacks: List[str]
    standardization_status: str
    implementation_maturity: str
    hardware_requirements: Dict[str, Any]
    patent_status: str
    

@dataclass 
class MigrationPlan:
    """Detailed PQC migration plan."""
    plan_id: str
    target_system: str
    current_algorithms: List[CryptoAlgorithm]
    recommended_algorithms: List[str]
    migration_phases: List[Dict[str, Any]]
    estimated_timeline: timedelta
    resource_requirements: Dict[str, Any]
    risk_assessment: Dict[str, float]
    testing_requirements: List[str]
    rollback_procedures: List[str]
    success_criteria: List[str]


class QuantumThreatModel:
    """Quantum computing threat modeling and assessment."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quantum algorithm impact database
        self.quantum_algorithms = {
            'shor': {
                'breaks': ['RSA', 'ECDSA', 'ECDH', 'DH'],
                'complexity': 'polynomial',
                'qubits_required': self._calculate_shor_qubits,
                'gate_count': self._calculate_shor_gates
            },
            'grover': {
                'breaks': ['AES', 'SHA', 'symmetric'],
                'complexity': 'square_root',
                'security_reduction': 0.5  # Halves effective key length
            },
            'simon': {
                'breaks': ['block_ciphers'],
                'complexity': 'polynomial',
                'conditions': 'specific_constructions'
            }
        }
        
        # Current quantum computing capabilities (updated regularly)
        self.quantum_capabilities = {
            'current_logical_qubits': 100,  # Conservative estimate
            'current_coherence_time': 0.1,  # milliseconds
            'current_gate_fidelity': 0.999,
            'projection_2025': {'qubits': 1000, 'coherence': 1.0, 'fidelity': 0.9999},
            'projection_2030': {'qubits': 10000, 'coherence': 10.0, 'fidelity': 0.99999},
            'projection_2035': {'qubits': 100000, 'coherence': 100.0, 'fidelity': 0.999999}
        }
        
        # Algorithm cryptanalytic margins (years until quantum break)
        self.cryptanalytic_margins = {
            CryptoAlgorithm.RSA_1024: 5,   # Already broken classically
            CryptoAlgorithm.RSA_2048: 10,
            CryptoAlgorithm.RSA_4096: 15,
            CryptoAlgorithm.ECDSA_P256: 12,
            CryptoAlgorithm.ECDSA_P384: 18,
            CryptoAlgorithm.ECDH_P256: 12,
            CryptoAlgorithm.ECDH_P384: 18,
            CryptoAlgorithm.DH_1024: 8,
            CryptoAlgorithm.DH_2048: 12,
        }
    
    def _calculate_shor_qubits(self, key_size_bits: int) -> int:
        """Calculate qubits required for Shor's algorithm."""
        # Conservative estimate: 2n + O(log n) qubits for n-bit integer
        return 2 * key_size_bits + int(np.log2(key_size_bits)) * 10
    
    def _calculate_shor_gates(self, key_size_bits: int) -> int:
        """Calculate gate count for Shor's algorithm."""
        # Conservative estimate: O(n³) gates for n-bit integer
        return key_size_bits ** 3
    
    @track_performance("quantum_threat_assessment")
    def assess_quantum_vulnerability(self, algorithm: CryptoAlgorithm, 
                                   key_size: Optional[int] = None) -> QuantumVulnerability:
        """Assess quantum vulnerability of specific algorithm."""
        
        # Determine applicable quantum attacks
        shor_applicable = algorithm.value in ['RSA-1024', 'RSA-2048', 'RSA-4096', 'ECDSA-P256', 'ECDSA-P384', 'ECDH-P256', 'ECDH-P384', 'DH-1024', 'DH-2048']
        grover_applicable = False  # These are public key algorithms
        
        # Calculate resource requirements
        if shor_applicable and key_size:
            qubits_required = self._calculate_shor_qubits(key_size)
            gate_count = self._calculate_shor_gates(key_size)
        else:
            qubits_required = 0
            gate_count = 0
        
        # Determine threat level
        margin = self.cryptanalytic_margins.get(algorithm, 20)
        if margin <= 5:
            threat_level = QuantumThreatLevel.IMMEDIATE
        elif margin <= 10:
            threat_level = QuantumThreatLevel.NEAR_TERM
        elif margin <= 20:
            threat_level = QuantumThreatLevel.MID_TERM
        else:
            threat_level = QuantumThreatLevel.LONG_TERM
        
        # Current and projected attacks
        current_attacks = []
        projected_attacks = []
        
        if shor_applicable:
            if qubits_required <= self.quantum_capabilities['current_logical_qubits'] * 10:
                current_attacks.append("Shor's algorithm (theoretical)")
            projected_attacks.append("Shor's algorithm (practical)")
        
        # Quantum advantage threshold
        advantage_threshold = 0.1 if threat_level == QuantumThreatLevel.IMMEDIATE else min(1.0, margin / 20)
        
        return QuantumVulnerability(
            algorithm=algorithm,
            threat_level=threat_level,
            cryptanalytic_margin=margin,
            shor_applicable=shor_applicable,
            grover_applicable=grover_applicable,
            estimated_qubits_required=qubits_required,
            estimated_gate_count=gate_count,
            current_quantum_attacks=current_attacks,
            projected_quantum_attacks=projected_attacks,
            quantum_advantage_threshold=advantage_threshold
        )
    
    def project_threat_timeline(self, algorithm: CryptoAlgorithm, 
                              current_year: int = None) -> Dict[int, Dict[str, Any]]:
        """Project quantum threat evolution over time."""
        current_year = current_year or datetime.now().year
        timeline = {}
        
        vulnerability = self.assess_quantum_vulnerability(algorithm)
        
        for year in range(current_year, current_year + 30, 5):
            # Project quantum capabilities for this year
            years_from_now = year - current_year
            
            # Exponential growth model for quantum capabilities
            projected_qubits = self.quantum_capabilities['current_logical_qubits'] * (2 ** (years_from_now / 3))
            projected_coherence = self.quantum_capabilities['current_coherence_time'] * (2 ** (years_from_now / 5))
            projected_fidelity = min(0.999999, self.quantum_capabilities['current_gate_fidelity'] * (1 + years_from_now * 0.001))
            
            # Determine threat status for this year
            threat_active = projected_qubits >= vulnerability.estimated_qubits_required
            
            timeline[year] = {
                'projected_qubits': int(projected_qubits),
                'projected_coherence_ms': projected_coherence,
                'projected_fidelity': projected_fidelity,
                'threat_active': threat_active,
                'risk_level': 'CRITICAL' if threat_active else 'HIGH' if years_from_now <= 10 else 'MEDIUM',
                'recommended_action': self._get_timeline_recommendation(vulnerability, years_from_now, threat_active)
            }
        
        return timeline


    def _get_timeline_recommendation(self, vulnerability: QuantumVulnerability, 
                                   years_from_now: int, threat_active: bool) -> str:
        """Get recommendation for specific timeline point."""
        if threat_active:
            return "URGENT: Quantum threat active - complete PQC migration required"
        elif years_from_now <= 5:
            return "Begin immediate PQC migration planning and pilot deployments"
        elif years_from_now <= 10:
            return "Initiate PQC algorithm selection and testing procedures"
        else:
            return "Monitor quantum developments and prepare crypto-agility measures"


class PQCAlgorithmDatabase:
    """Database of post-quantum cryptographic algorithms."""
    
    def __init__(self):
        self.algorithms = self._initialize_algorithm_database()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_algorithm_database(self) -> Dict[str, PQCAlgorithmProfile]:
        """Initialize comprehensive PQC algorithm database."""
        return {
            'Dilithium2': PQCAlgorithmProfile(
                name='Dilithium2',
                nist_security_level=2,
                algorithm_type='signature',
                key_sizes={'public': 1312, 'private': 2528},
                signature_sizes={'signature': 2420},
                performance_characteristics={
                    'keygen_cycles': 87000,
                    'sign_cycles': 216000,
                    'verify_cycles': 66000,
                    'memory_kb': 12
                },
                security_assumptions=['Module-LWE', 'SIS'],
                known_attacks=['Lattice reduction', 'Algebraic attacks'],
                standardization_status='NIST Round 3 Winner',
                implementation_maturity='Production Ready',
                hardware_requirements={'ram_kb': 16, 'flash_kb': 32},
                patent_status='Royalty-free'
            ),
            
            'Dilithium3': PQCAlgorithmProfile(
                name='Dilithium3',
                nist_security_level=3,
                algorithm_type='signature',
                key_sizes={'public': 1952, 'private': 4000},
                signature_sizes={'signature': 3293},
                performance_characteristics={
                    'keygen_cycles': 134000,
                    'sign_cycles': 321000,
                    'verify_cycles': 98000,
                    'memory_kb': 18
                },
                security_assumptions=['Module-LWE', 'SIS'],
                known_attacks=['Lattice reduction', 'Algebraic attacks'],
                standardization_status='NIST Round 3 Winner',
                implementation_maturity='Production Ready',
                hardware_requirements={'ram_kb': 24, 'flash_kb': 48},
                patent_status='Royalty-free'
            ),
            
            'Kyber512': PQCAlgorithmProfile(
                name='Kyber512',
                nist_security_level=1,
                algorithm_type='kem',
                key_sizes={'public': 800, 'private': 1632},
                signature_sizes={'ciphertext': 768, 'shared_secret': 32},
                performance_characteristics={
                    'keygen_cycles': 41000,
                    'encaps_cycles': 52000,
                    'decaps_cycles': 47000,
                    'memory_kb': 6
                },
                security_assumptions=['Module-LWE'],
                known_attacks=['Lattice reduction', 'Primal attacks', 'Dual attacks'],
                standardization_status='NIST Round 3 Winner',
                implementation_maturity='Production Ready',
                hardware_requirements={'ram_kb': 8, 'flash_kb': 16},
                patent_status='Royalty-free'
            ),
            
            'Kyber768': PQCAlgorithmProfile(
                name='Kyber768',
                nist_security_level=3,
                algorithm_type='kem',
                key_sizes={'public': 1184, 'private': 2400},
                signature_sizes={'ciphertext': 1088, 'shared_secret': 32},
                performance_characteristics={
                    'keygen_cycles': 65000,
                    'encaps_cycles': 79000,
                    'decaps_cycles': 72000,
                    'memory_kb': 9
                },
                security_assumptions=['Module-LWE'],
                known_attacks=['Lattice reduction', 'Primal attacks', 'Dual attacks'],
                standardization_status='NIST Round 3 Winner',
                implementation_maturity='Production Ready',
                hardware_requirements={'ram_kb': 12, 'flash_kb': 24},
                patent_status='Royalty-free'
            ),
            
            'Falcon-512': PQCAlgorithmProfile(
                name='Falcon-512',
                nist_security_level=1,
                algorithm_type='signature',
                key_sizes={'public': 897, 'private': 1281},
                signature_sizes={'signature': 690},
                performance_characteristics={
                    'keygen_cycles': 158000000,  # Very slow key generation
                    'sign_cycles': 432000,
                    'verify_cycles': 98000,
                    'memory_kb': 8
                },
                security_assumptions=['SIS over NTRU lattices'],
                known_attacks=['Lattice reduction'],
                standardization_status='NIST Round 3 Winner',
                implementation_maturity='Limited Production',
                hardware_requirements={'ram_kb': 12, 'flash_kb': 20, 'fpu': True},
                patent_status='Royalty-free'
            ),
            
            'SPHINCS+': PQCAlgorithmProfile(
                name='SPHINCS+',
                nist_security_level=1,
                algorithm_type='signature',
                key_sizes={'public': 32, 'private': 64},
                signature_sizes={'signature': 17088},  # Large signatures
                performance_characteristics={
                    'keygen_cycles': 15000,
                    'sign_cycles': 54000000,  # Very slow signing
                    'verify_cycles': 1400000,
                    'memory_kb': 2
                },
                security_assumptions=['Hash function security'],
                known_attacks=['Hash collision attacks'],
                standardization_status='NIST Round 3 Winner',
                implementation_maturity='Research',
                hardware_requirements={'ram_kb': 4, 'flash_kb': 8},
                patent_status='Royalty-free'
            )
        }
    
    def get_algorithm(self, name: str) -> Optional[PQCAlgorithmProfile]:
        """Get algorithm profile by name."""
        return self.algorithms.get(name)
    
    def find_suitable_algorithms(self, requirements: Dict[str, Any]) -> List[PQCAlgorithmProfile]:
        """Find algorithms meeting specific requirements."""
        suitable = []
        
        for algo in self.algorithms.values():
            # Check security level requirement
            if requirements.get('min_security_level', 1) > algo.nist_security_level:
                continue
            
            # Check memory constraints
            if requirements.get('max_memory_kb', float('inf')) < algo.performance_characteristics.get('memory_kb', 0):
                continue
            
            # Check algorithm type
            if requirements.get('algorithm_type') and requirements['algorithm_type'] != algo.algorithm_type:
                continue
            
            # Check performance requirements
            if requirements.get('max_sign_cycles') and algo.performance_characteristics.get('sign_cycles', 0) > requirements['max_sign_cycles']:
                continue
            
            # Check hardware requirements
            hw_reqs = algo.hardware_requirements
            if requirements.get('no_fpu', False) and hw_reqs.get('fpu', False):
                continue
            
            suitable.append(algo)
        
        # Sort by suitability score
        return sorted(suitable, key=lambda x: self._calculate_suitability_score(x, requirements), reverse=True)
    
    def _calculate_suitability_score(self, algorithm: PQCAlgorithmProfile, requirements: Dict[str, Any]) -> float:
        """Calculate suitability score for algorithm given requirements."""
        score = 0.0
        
        # Security level bonus
        score += algorithm.nist_security_level * 0.2
        
        # Memory efficiency bonus
        memory_usage = algorithm.performance_characteristics.get('memory_kb', 10)
        max_memory = requirements.get('max_memory_kb', 64)
        if memory_usage <= max_memory:
            score += (1 - memory_usage / max_memory) * 0.3
        
        # Performance bonus
        sign_cycles = algorithm.performance_characteristics.get('sign_cycles', 1000000)
        if sign_cycles < 500000:
            score += 0.3
        elif sign_cycles < 1000000:
            score += 0.15
        
        # Maturity bonus
        if algorithm.implementation_maturity == 'Production Ready':
            score += 0.2
        elif algorithm.implementation_maturity == 'Limited Production':
            score += 0.1
        
        return score


class QuantumResilienceAnalyzer:
    """Main quantum resilience analysis system."""
    
    def __init__(self):
        self.threat_model = QuantumThreatModel()
        self.algorithm_db = PQCAlgorithmDatabase()
        self.adaptive_ai = AdaptiveAI()
        self.logger = logging.getLogger(__name__)
    
    @handle_errors(operation_name="quantum_resilience_assessment")
    @track_performance("quantum_resilience_assessment")
    def assess_system_resilience(self, vulnerabilities: List, 
                               system_constraints: Dict[str, Any] = None) -> ResilienceAssessment:
        """Comprehensive quantum resilience assessment."""
        system_constraints = system_constraints or {}
        
        self.logger.info(f"Assessing quantum resilience for {len(vulnerabilities)} vulnerabilities")
        
        # Analyze quantum vulnerabilities
        quantum_vulns = []
        total_threat_score = 0.0
        
        for vuln in vulnerabilities:
            if hasattr(vuln, 'algorithm') and hasattr(vuln, 'key_size'):
                q_vuln = self.threat_model.assess_quantum_vulnerability(vuln.algorithm, vuln.key_size)
                quantum_vulns.append(q_vuln)
                
                # Weight threat score by vulnerability criticality
                threat_weight = {
                    QuantumThreatLevel.IMMEDIATE: 1.0,
                    QuantumThreatLevel.NEAR_TERM: 0.8,
                    QuantumThreatLevel.MID_TERM: 0.5,
                    QuantumThreatLevel.LONG_TERM: 0.2,
                    QuantumThreatLevel.NEGLIGIBLE: 0.0
                }
                total_threat_score += threat_weight.get(q_vuln.threat_level, 0.5)
        
        # Calculate overall resilience score
        if len(quantum_vulns) == 0:
            overall_resilience = 1.0
        else:
            # Inverse relationship with threat score
            overall_resilience = max(0.0, 1.0 - (total_threat_score / len(quantum_vulns)))
        
        # Assess crypto agility
        agility_level = self._assess_crypto_agility(system_constraints)
        
        # Calculate migration readiness
        migration_readiness = self._calculate_migration_readiness(system_constraints, agility_level)
        
        # Estimate migration cost
        migration_cost = self._estimate_migration_cost(quantum_vulns, system_constraints)
        
        # Recommend migration strategy
        strategy = self._recommend_migration_strategy(quantum_vulns, agility_level, system_constraints)
        
        # Generate timeline recommendations
        timeline_recs = self._generate_timeline_recommendations(quantum_vulns)
        
        # Check compliance status
        compliance = self._check_compliance_status(quantum_vulns, system_constraints)
        
        assessment = ResilienceAssessment(
            overall_resilience_score=overall_resilience,
            quantum_vulnerabilities=quantum_vulns,
            crypto_agility_level=agility_level,
            migration_readiness=migration_readiness,
            estimated_migration_cost=migration_cost,
            recommended_strategy=strategy,
            timeline_recommendations=timeline_recs,
            compliance_status=compliance
        )
        
        # Record metrics
        metrics_collector.record_metric("quantum.resilience_score", overall_resilience, "score")
        metrics_collector.record_metric("quantum.vulnerabilities_assessed", len(quantum_vulns), "count")
        metrics_collector.record_metric("quantum.migration_readiness", migration_readiness, "score")
        
        self.logger.info(f"Quantum resilience assessment complete: {overall_resilience:.2f} resilience score")
        
        return assessment
    
    def _assess_crypto_agility(self, system_constraints: Dict[str, Any]) -> CryptoAgility:
        """Assess current cryptographic agility level."""
        # Analyze system characteristics to determine agility
        
        # Check for configuration options
        has_crypto_config = system_constraints.get('configurable_crypto', False)
        has_runtime_selection = system_constraints.get('runtime_crypto_selection', False)
        has_ai_optimization = system_constraints.get('ai_crypto_optimization', False)
        
        if has_ai_optimization:
            return CryptoAgility.ADAPTIVE
        elif has_runtime_selection:
            return CryptoAgility.DYNAMIC
        elif has_crypto_config:
            return CryptoAgility.CONFIGURABLE
        else:
            return CryptoAgility.RIGID
    
    def _calculate_migration_readiness(self, system_constraints: Dict[str, Any], 
                                     agility: CryptoAgility) -> float:
        """Calculate migration readiness score (0-1)."""
        readiness = 0.0
        
        # Agility bonus
        agility_scores = {
            CryptoAgility.ADAPTIVE: 1.0,
            CryptoAgility.DYNAMIC: 0.8,
            CryptoAgility.CONFIGURABLE: 0.5,
            CryptoAgility.RIGID: 0.2
        }
        readiness += agility_scores[agility] * 0.4
        
        # Development team readiness
        team_readiness = system_constraints.get('team_pqc_experience', 0.3)  # 0-1 scale
        readiness += team_readiness * 0.2
        
        # Testing infrastructure
        has_testing = system_constraints.get('crypto_testing_framework', False)
        readiness += (0.15 if has_testing else 0.0)
        
        # Budget allocation
        budget_allocated = system_constraints.get('pqc_budget_allocated', 0.5)  # 0-1 scale
        readiness += budget_allocated * 0.15
        
        # Regulatory compliance requirements
        compliance_pressure = system_constraints.get('compliance_pressure', 0.5)  # 0-1 scale
        readiness += compliance_pressure * 0.1
        
        return min(1.0, readiness)
    
    def _estimate_migration_cost(self, quantum_vulns: List[QuantumVulnerability], 
                               constraints: Dict[str, Any]) -> float:
        """Estimate migration cost in normalized units (0-1)."""
        base_cost = 0.3  # Baseline migration cost
        
        # Cost increases with number of vulnerabilities
        vuln_cost = min(0.4, len(quantum_vulns) * 0.05)
        
        # Cost varies by threat urgency
        urgency_multiplier = 1.0
        urgent_threats = sum(1 for v in quantum_vulns if v.threat_level in [QuantumThreatLevel.IMMEDIATE, QuantumThreatLevel.NEAR_TERM])
        if urgent_threats > 0:
            urgency_multiplier = 1.5
        
        # System complexity factor
        complexity = constraints.get('system_complexity', 0.5)  # 0-1 scale
        complexity_cost = complexity * 0.2
        
        total_cost = (base_cost + vuln_cost + complexity_cost) * urgency_multiplier
        return min(1.0, total_cost)
    
    def _recommend_migration_strategy(self, quantum_vulns: List[QuantumVulnerability], 
                                    agility: CryptoAgility, 
                                    constraints: Dict[str, Any]) -> MigrationStrategy:
        """Recommend optimal migration strategy."""
        
        # Check for immediate threats
        immediate_threats = [v for v in quantum_vulns if v.threat_level == QuantumThreatLevel.IMMEDIATE]
        near_term_threats = [v for v in quantum_vulns if v.threat_level == QuantumThreatLevel.NEAR_TERM]
        
        if immediate_threats:
            return MigrationStrategy.IMMEDIATE_REPLACEMENT
        
        if near_term_threats and agility in [CryptoAgility.DYNAMIC, CryptoAgility.ADAPTIVE]:
            return MigrationStrategy.HYBRID_COEXISTENCE
        
        if len(quantum_vulns) > 5 or constraints.get('high_availability_required', False):
            return MigrationStrategy.STAGED_ROLLOUT
        
        return MigrationStrategy.GRADUAL_TRANSITION
    
    def _generate_timeline_recommendations(self, quantum_vulns: List[QuantumVulnerability]) -> Dict[str, str]:
        """Generate timeline-specific recommendations."""
        recommendations = {}
        
        current_year = datetime.now().year
        
        # Immediate actions (0-6 months)
        immediate_actions = []
        if any(v.threat_level == QuantumThreatLevel.IMMEDIATE for v in quantum_vulns):
            immediate_actions.append("Begin emergency PQC deployment")
        if any(v.threat_level == QuantumThreatLevel.NEAR_TERM for v in quantum_vulns):
            immediate_actions.append("Initiate PQC migration project")
        else:
            immediate_actions.append("Establish PQC evaluation team")
        
        recommendations["immediate"] = "; ".join(immediate_actions)
        
        # Short term (6-24 months)
        recommendations["short_term"] = "Complete PQC algorithm selection and begin pilot deployments"
        
        # Medium term (2-5 years)
        recommendations["medium_term"] = "Execute phased PQC rollout with hybrid cryptography support"
        
        # Long term (5+ years)
        recommendations["long_term"] = "Achieve full PQC migration and establish quantum-safe security posture"
        
        return recommendations
    
    def _check_compliance_status(self, quantum_vulns: List[QuantumVulnerability], 
                               constraints: Dict[str, Any]) -> Dict[str, bool]:
        """Check compliance with various standards."""
        compliance = {}
        
        # NIST compliance
        nist_compliant = not any(v.threat_level in [QuantumThreatLevel.IMMEDIATE, QuantumThreatLevel.NEAR_TERM] for v in quantum_vulns)
        compliance['NIST_PQC'] = nist_compliant
        
        # CNSA 2.0 compliance
        cnsa_compliant = not any(v.algorithm in [CryptoAlgorithm.RSA_1024, CryptoAlgorithm.RSA_2048] for v in quantum_vulns)
        compliance['CNSA_2_0'] = cnsa_compliant
        
        # Industry specific compliance
        if constraints.get('financial_sector', False):
            compliance['Financial_Regulatory'] = nist_compliant
        
        if constraints.get('government_sector', False):
            compliance['Government_Standards'] = cnsa_compliant
        
        if constraints.get('healthcare_sector', False):
            compliance['HIPAA_PQC'] = nist_compliant  # Future requirement
        
        return compliance
    
    @track_performance("migration_plan_generation")
    def generate_migration_plan(self, assessment: ResilienceAssessment, 
                              system_info: Dict[str, Any]) -> MigrationPlan:
        """Generate detailed PQC migration plan."""
        
        plan_id = f"pqc_migration_{int(time.time())}"
        
        # Extract current algorithms
        current_algorithms = [v.algorithm for v in assessment.quantum_vulnerabilities]
        
        # Select recommended PQC algorithms
        recommended_algorithms = []
        for vuln in assessment.quantum_vulnerabilities:
            if vuln.algorithm.value.startswith('RSA') or vuln.algorithm.value.startswith('ECDSA'):
                # Signature replacement
                suitable = self.algorithm_db.find_suitable_algorithms({
                    'algorithm_type': 'signature',
                    'min_security_level': 2,
                    'max_memory_kb': system_info.get('max_memory_kb', 32)
                })
                if suitable:
                    recommended_algorithms.append(suitable[0].name)
            else:
                # KEM replacement
                suitable = self.algorithm_db.find_suitable_algorithms({
                    'algorithm_type': 'kem',
                    'min_security_level': 1,
                    'max_memory_kb': system_info.get('max_memory_kb', 32)
                })
                if suitable:
                    recommended_algorithms.append(suitable[0].name)
        
        # Generate migration phases
        phases = self._generate_migration_phases(assessment, system_info)
        
        # Estimate timeline
        timeline = self._estimate_migration_timeline(assessment, phases)
        
        # Calculate resource requirements
        resources = self._calculate_resource_requirements(recommended_algorithms, system_info)
        
        # Risk assessment
        risks = self._assess_migration_risks(assessment, system_info)
        
        # Testing requirements
        testing = self._generate_testing_requirements(recommended_algorithms)
        
        # Rollback procedures
        rollback = self._generate_rollback_procedures(assessment.recommended_strategy)
        
        # Success criteria
        success_criteria = self._generate_success_criteria(assessment)
        
        return MigrationPlan(
            plan_id=plan_id,
            target_system=system_info.get('system_name', 'Unknown System'),
            current_algorithms=current_algorithms,
            recommended_algorithms=list(set(recommended_algorithms)),
            migration_phases=phases,
            estimated_timeline=timeline,
            resource_requirements=resources,
            risk_assessment=risks,
            testing_requirements=testing,
            rollback_procedures=rollback,
            success_criteria=success_criteria
        )
    
    def _generate_migration_phases(self, assessment: ResilienceAssessment, 
                                 system_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed migration phases."""
        phases = []
        
        # Phase 1: Preparation and Planning
        phases.append({
            'phase': 1,
            'name': 'Preparation and Planning',
            'duration_weeks': 8,
            'activities': [
                'Establish PQC migration team',
                'Complete detailed crypto inventory',
                'Select PQC algorithms',
                'Set up development and testing environments',
                'Create migration timeline and budget'
            ],
            'deliverables': ['Migration Plan', 'Testing Framework', 'Risk Assessment'],
            'resources_required': {'developers': 2, 'architects': 1, 'testers': 1}
        })
        
        # Phase 2: Algorithm Implementation
        phases.append({
            'phase': 2,
            'name': 'Algorithm Implementation',
            'duration_weeks': 12,
            'activities': [
                'Implement selected PQC algorithms',
                'Create crypto abstraction layer',
                'Develop configuration management',
                'Build automated testing suite'
            ],
            'deliverables': ['PQC Library', 'Integration Guide', 'Test Suite'],
            'resources_required': {'developers': 4, 'crypto_experts': 1, 'testers': 2}
        })
        
        # Phase 3: Integration and Testing
        phases.append({
            'phase': 3,
            'name': 'Integration and Testing',
            'duration_weeks': 16,
            'activities': [
                'Integrate PQC algorithms into system',
                'Conduct functional testing',
                'Perform security testing',
                'Execute performance testing',
                'Validate interoperability'
            ],
            'deliverables': ['Integrated System', 'Test Reports', 'Performance Analysis'],
            'resources_required': {'developers': 3, 'testers': 3, 'security_analysts': 2}
        })
        
        # Phase 4: Deployment Planning
        strategy_duration = {
            MigrationStrategy.IMMEDIATE_REPLACEMENT: 4,
            MigrationStrategy.GRADUAL_TRANSITION: 8,
            MigrationStrategy.HYBRID_COEXISTENCE: 12,
            MigrationStrategy.STAGED_ROLLOUT: 20
        }
        
        duration = strategy_duration.get(assessment.recommended_strategy, 12)
        
        phases.append({
            'phase': 4,
            'name': 'Deployment and Rollout',
            'duration_weeks': duration,
            'activities': [
                'Deploy to staging environment',
                'Conduct user acceptance testing',
                'Execute phased production rollout',
                'Monitor system performance',
                'Provide user training'
            ],
            'deliverables': ['Production System', 'Monitoring Dashboard', 'User Documentation'],
            'resources_required': {'devops': 2, 'support': 3, 'trainers': 1}
        })
        
        # Phase 5: Monitoring and Optimization
        phases.append({
            'phase': 5,
            'name': 'Monitoring and Optimization',
            'duration_weeks': 8,
            'activities': [
                'Monitor system performance',
                'Optimize configurations',
                'Address issues and bugs',
                'Update documentation',
                'Plan for future enhancements'
            ],
            'deliverables': ['Optimized System', 'Lessons Learned', 'Future Roadmap'],
            'resources_required': {'developers': 2, 'analysts': 1, 'support': 2}
        })
        
        return phases
    
    def _estimate_migration_timeline(self, assessment: ResilienceAssessment, 
                                   phases: List[Dict[str, Any]]) -> timedelta:
        """Estimate total migration timeline."""
        total_weeks = sum(phase['duration_weeks'] for phase in phases)
        
        # Adjust for urgency
        if assessment.quantum_vulnerabilities:
            max_threat = max(v.threat_level for v in assessment.quantum_vulnerabilities)
            if max_threat == QuantumThreatLevel.IMMEDIATE:
                total_weeks *= 0.7  # Accelerate timeline
            elif max_threat == QuantumThreatLevel.NEAR_TERM:
                total_weeks *= 0.85
        
        return timedelta(weeks=total_weeks)
    
    def _calculate_resource_requirements(self, algorithms: List[str], 
                                       system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource requirements for migration."""
        return {
            'team_size': 8,  # Average team size
            'budget_estimate': system_info.get('system_complexity', 0.5) * 500000,  # USD
            'hardware_requirements': {
                'development_servers': 3,
                'testing_devices': 10,
                'monitoring_tools': 2
            },
            'software_licenses': ['Crypto libraries', 'Testing frameworks', 'Monitoring tools'],
            'training_budget': 50000,  # USD for team training
            'external_consulting': 100000  # USD for crypto expertise
        }
    
    def _assess_migration_risks(self, assessment: ResilienceAssessment, 
                              system_info: Dict[str, Any]) -> Dict[str, float]:
        """Assess migration risks (0-1 scale)."""
        return {
            'technical_integration': 0.3,  # Risk of technical integration issues
            'performance_impact': 0.2,     # Risk of performance degradation
            'security_vulnerabilities': 0.1,  # Risk of introducing new vulnerabilities
            'timeline_delays': 0.4,        # Risk of project delays
            'budget_overrun': 0.3,         # Risk of budget overruns
            'compatibility_issues': 0.25,  # Risk of compatibility problems
            'user_adoption': 0.15,         # Risk of user resistance
            'regulatory_compliance': 0.1   # Risk of compliance issues
        }
    
    def _generate_testing_requirements(self, algorithms: List[str]) -> List[str]:
        """Generate testing requirements for migration."""
        return [
            'Functional correctness testing',
            'Performance benchmark testing', 
            'Security vulnerability testing',
            'Interoperability testing',
            'Side-channel attack testing',
            'Fault injection testing',
            'Load and stress testing',
            'Regression testing',
            'User acceptance testing',
            'Compliance validation testing'
        ]
    
    def _generate_rollback_procedures(self, strategy: MigrationStrategy) -> List[str]:
        """Generate rollback procedures."""
        base_procedures = [
            'Maintain hybrid crypto support during transition',
            'Implement feature flags for algorithm selection',
            'Create automated rollback scripts',
            'Establish monitoring and alerting systems',
            'Document rollback decision criteria'
        ]
        
        strategy_specific = {
            MigrationStrategy.IMMEDIATE_REPLACEMENT: [
                'Prepare emergency rollback to classical crypto',
                'Maintain full system backups'
            ],
            MigrationStrategy.GRADUAL_TRANSITION: [
                'Roll back individual components incrementally',
                'Maintain parallel classical/PQC systems'
            ],
            MigrationStrategy.HYBRID_COEXISTENCE: [
                'Adjust algorithm selection policies',
                'Gradually reduce classical crypto usage'
            ],
            MigrationStrategy.STAGED_ROLLOUT: [
                'Roll back deployment stages in reverse order',
                'Maintain staging environment mirrors'
            ]
        }
        
        return base_procedures + strategy_specific.get(strategy, [])
    
    def _generate_success_criteria(self, assessment: ResilienceAssessment) -> List[str]:
        """Generate success criteria for migration."""
        return [
            f'Achieve quantum resilience score > 0.9 (current: {assessment.overall_resilience_score:.2f})',
            'Zero critical quantum vulnerabilities remaining',
            'System performance within 10% of baseline',
            'All security tests passing',
            'User acceptance rate > 90%',
            'Zero critical security incidents',
            'Compliance with all relevant standards',
            'Migration completed within budget and timeline',
            'Documentation complete and up-to-date',
            'Team trained and ready for ongoing maintenance'
        ]


# Global quantum resilience analyzer instance
quantum_resilience = QuantumResilienceAnalyzer()