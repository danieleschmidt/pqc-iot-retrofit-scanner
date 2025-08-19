#!/usr/bin/env python3
"""Global-First Implementation - I18n, Compliance & Multi-Region Support.

Enterprise-grade global deployment capabilities:
- Multi-language support (en, es, fr, de, ja, zh)
- Regional compliance (GDPR, CCPA, PDPA)
- Cross-platform compatibility
- Multi-region deployment architecture
- Cultural adaptation for different markets
- Timezone-aware operations
- Currency and measurement localization
"""

import sys
import os
import json
import time
import locale
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import hashlib
import uuid

# Add source path
sys.path.insert(0, 'src')

from pqc_iot_retrofit.scanner import FirmwareScanner, CryptoVulnerability, RiskLevel
from scalable_generation3_analyzer import ScalableFirmwareAnalyzer, ScalabilityConfig


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"  
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class ComplianceRegion(Enum):
    """Regional compliance frameworks."""
    EU_GDPR = "eu_gdpr"      # European Union - GDPR
    US_CCPA = "us_ccpa"      # California - CCPA
    SINGAPORE_PDPA = "sg_pdpa"  # Singapore - PDPA
    CANADA_PIPEDA = "ca_pipeda"  # Canada - PIPEDA
    AUSTRALIA_APA = "au_apa"     # Australia - Privacy Act
    JAPAN_APPI = "jp_appi"       # Japan - Act on Protection of Personal Information


class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    ASIA_PACIFIC_1 = "ap-southeast-1"
    ASIA_PACIFIC_2 = "ap-northeast-1"


@dataclass
class LocalizationConfig:
    """Localization and cultural adaptation settings."""
    language: SupportedLanguage
    region: str  # ISO 3166-1 alpha-2 country code
    timezone: str  # IANA timezone
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "1,234.56"
    currency: str = "USD"
    measurement_system: str = "metric"  # or "imperial"


@dataclass
class ComplianceConfig:
    """Regional compliance requirements."""
    region: ComplianceRegion
    data_retention_days: int
    encryption_required: bool
    audit_logging: bool
    consent_required: bool
    right_to_deletion: bool
    data_portability: bool
    breach_notification_hours: int
    specific_requirements: Dict[str, Any] = field(default_factory=dict)


class InternationalizationManager:
    """Manages multi-language support and cultural adaptation."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        
        # Load translation dictionaries
        self.translations = self._load_translations()
        
        # Initialize localization settings
        self.localization_configs = self._initialize_localization_configs()
        
        print(f"🌍 Internationalization Manager initialized")
        print(f"   Default Language: {default_language.value}")
        print(f"   Supported Languages: {', '.join([lang.value for lang in SupportedLanguage])}")
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries for all supported languages."""
        
        translations = {
            "en": {
                "firmware_analysis": "Firmware Analysis",
                "vulnerabilities_found": "Vulnerabilities Found",
                "risk_level": "Risk Level",
                "critical": "Critical",
                "high": "High", 
                "medium": "Medium",
                "low": "Low",
                "analysis_complete": "Analysis Complete",
                "patch_generation": "Patch Generation",
                "security_scan": "Security Scan",
                "performance_test": "Performance Test",
                "quality_gates": "Quality Gates",
                "deployment_ready": "Ready for Deployment",
                "error_occurred": "An error occurred",
                "file_not_found": "File not found",
                "invalid_architecture": "Invalid architecture",
                "memory_constraints": "Memory Constraints",
                "recommendations": "Recommendations",
                "replace_with_pqc": "Replace with post-quantum cryptography",
                "immediate_action_required": "Immediate action required",
                "test_in_isolation": "Test in isolated environment",
            },
            "es": {
                "firmware_analysis": "Análisis de Firmware",
                "vulnerabilities_found": "Vulnerabilidades Encontradas",
                "risk_level": "Nivel de Riesgo",
                "critical": "Crítico",
                "high": "Alto",
                "medium": "Medio", 
                "low": "Bajo",
                "analysis_complete": "Análisis Completo",
                "patch_generation": "Generación de Parches",
                "security_scan": "Escaneo de Seguridad",
                "performance_test": "Prueba de Rendimiento",
                "quality_gates": "Puertas de Calidad",
                "deployment_ready": "Listo para Despliegue",
                "error_occurred": "Ocurrió un error",
                "file_not_found": "Archivo no encontrado",
                "invalid_architecture": "Arquitectura inválida",
                "memory_constraints": "Restricciones de Memoria",
                "recommendations": "Recomendaciones",
                "replace_with_pqc": "Reemplazar con criptografía post-cuántica",
                "immediate_action_required": "Se requiere acción inmediata",
                "test_in_isolation": "Probar en entorno aislado",
            },
            "fr": {
                "firmware_analysis": "Analyse de Firmware",
                "vulnerabilities_found": "Vulnérabilités Trouvées",
                "risk_level": "Niveau de Risque",
                "critical": "Critique",
                "high": "Élevé",
                "medium": "Moyen",
                "low": "Faible",
                "analysis_complete": "Analyse Terminée",
                "patch_generation": "Génération de Correctifs",
                "security_scan": "Analyse de Sécurité",
                "performance_test": "Test de Performance",
                "quality_gates": "Portes de Qualité",
                "deployment_ready": "Prêt pour le Déploiement",
                "error_occurred": "Une erreur s'est produite",
                "file_not_found": "Fichier non trouvé",
                "invalid_architecture": "Architecture invalide",
                "memory_constraints": "Contraintes Mémoire",
                "recommendations": "Recommandations",
                "replace_with_pqc": "Remplacer par la cryptographie post-quantique",
                "immediate_action_required": "Action immédiate requise",
                "test_in_isolation": "Tester en environnement isolé",
            },
            "de": {
                "firmware_analysis": "Firmware-Analyse",
                "vulnerabilities_found": "Gefundene Schwachstellen",
                "risk_level": "Risikostufe",
                "critical": "Kritisch",
                "high": "Hoch",
                "medium": "Mittel",
                "low": "Niedrig",
                "analysis_complete": "Analyse Abgeschlossen",
                "patch_generation": "Patch-Generierung",
                "security_scan": "Sicherheitsscan",
                "performance_test": "Leistungstest",
                "quality_gates": "Qualitätstüren",
                "deployment_ready": "Bereit für Bereitstellung",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "file_not_found": "Datei nicht gefunden",
                "invalid_architecture": "Ungültige Architektur",
                "memory_constraints": "Speicherbeschränkungen",
                "recommendations": "Empfehlungen",
                "replace_with_pqc": "Mit Post-Quanten-Kryptographie ersetzen",
                "immediate_action_required": "Sofortige Maßnahme erforderlich",
                "test_in_isolation": "In isolierter Umgebung testen",
            },
            "ja": {
                "firmware_analysis": "ファームウェア解析",
                "vulnerabilities_found": "発見された脆弱性",
                "risk_level": "リスクレベル",
                "critical": "クリティカル",
                "high": "高",
                "medium": "中",
                "low": "低",
                "analysis_complete": "解析完了",
                "patch_generation": "パッチ生成",
                "security_scan": "セキュリティスキャン",
                "performance_test": "性能テスト",
                "quality_gates": "品質ゲート",
                "deployment_ready": "デプロイ準備完了",
                "error_occurred": "エラーが発生しました",
                "file_not_found": "ファイルが見つかりません",
                "invalid_architecture": "無効なアーキテクチャ",
                "memory_constraints": "メモリ制約",
                "recommendations": "推奨事項",
                "replace_with_pqc": "ポスト量子暗号に置き換える",
                "immediate_action_required": "即座の対応が必要",
                "test_in_isolation": "隔離環境でテスト",
            },
            "zh": {
                "firmware_analysis": "固件分析",
                "vulnerabilities_found": "发现的漏洞",
                "risk_level": "风险级别",
                "critical": "关键",
                "high": "高",
                "medium": "中",
                "low": "低",
                "analysis_complete": "分析完成",
                "patch_generation": "补丁生成",
                "security_scan": "安全扫描",
                "performance_test": "性能测试",
                "quality_gates": "质量门",
                "deployment_ready": "准备部署",
                "error_occurred": "发生错误",
                "file_not_found": "文件未找到",
                "invalid_architecture": "无效架构",
                "memory_constraints": "内存约束",
                "recommendations": "建议",
                "replace_with_pqc": "替换为后量子密码学",
                "immediate_action_required": "需要立即行动",
                "test_in_isolation": "在隔离环境中测试",
            }
        }
        
        return translations
    
    def _initialize_localization_configs(self) -> Dict[SupportedLanguage, LocalizationConfig]:
        """Initialize localization configurations for each supported language."""
        
        configs = {
            SupportedLanguage.ENGLISH: LocalizationConfig(
                language=SupportedLanguage.ENGLISH,
                region="US",
                timezone="America/New_York",
                date_format="%m/%d/%Y",
                time_format="%I:%M:%S %p",
                number_format="1,234.56",
                currency="USD",
                measurement_system="imperial"
            ),
            SupportedLanguage.SPANISH: LocalizationConfig(
                language=SupportedLanguage.SPANISH,
                region="ES",
                timezone="Europe/Madrid",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="1.234,56",
                currency="EUR",
                measurement_system="metric"
            ),
            SupportedLanguage.FRENCH: LocalizationConfig(
                language=SupportedLanguage.FRENCH,
                region="FR",
                timezone="Europe/Paris",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="1 234,56",
                currency="EUR",
                measurement_system="metric"
            ),
            SupportedLanguage.GERMAN: LocalizationConfig(
                language=SupportedLanguage.GERMAN,
                region="DE",
                timezone="Europe/Berlin",
                date_format="%d.%m.%Y",
                time_format="%H:%M:%S",
                number_format="1.234,56",
                currency="EUR",
                measurement_system="metric"
            ),
            SupportedLanguage.JAPANESE: LocalizationConfig(
                language=SupportedLanguage.JAPANESE,
                region="JP",
                timezone="Asia/Tokyo",
                date_format="%Y年%m月%d日",
                time_format="%H時%M分%S秒",
                number_format="1,234.56",
                currency="JPY",
                measurement_system="metric"
            ),
            SupportedLanguage.CHINESE: LocalizationConfig(
                language=SupportedLanguage.CHINESE,
                region="CN",
                timezone="Asia/Shanghai",
                date_format="%Y年%m月%d日",
                time_format="%H:%M:%S",
                number_format="1,234.56",
                currency="CNY",
                measurement_system="metric"
            )
        }
        
        return configs
    
    def set_language(self, language: SupportedLanguage):
        """Set the current language for localization."""
        self.current_language = language
        print(f"🌐 Language set to: {language.value}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key to the current language with optional formatting."""
        
        translation = self.translations.get(
            self.current_language.value, 
            self.translations[self.default_language.value]
        ).get(key, key)
        
        # Apply formatting if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError):
                pass  # Return unformatted translation if formatting fails
        
        return translation
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to current localization."""
        config = self.localization_configs[self.current_language]
        return dt.strftime(f"{config.date_format} {config.time_format}")
    
    def format_number(self, number: float) -> str:
        """Format number according to current localization."""
        config = self.localization_configs[self.current_language]
        
        if config.number_format == "1,234.56":
            return f"{number:,.2f}"
        elif config.number_format == "1.234,56":
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif config.number_format == "1 234,56":
            return f"{number:,.2f}".replace(",", " ").replace(".", ",")
        else:
            return f"{number:.2f}"


class ComplianceManager:
    """Manages regional compliance requirements and data governance."""
    
    def __init__(self):
        self.compliance_configs = self._initialize_compliance_configs()
        
        print("🔒 Compliance Manager initialized")
        print(f"   Supported Regions: {', '.join([region.value for region in ComplianceRegion])}")
    
    def _initialize_compliance_configs(self) -> Dict[ComplianceRegion, ComplianceConfig]:
        """Initialize compliance configurations for different regions."""
        
        configs = {
            ComplianceRegion.EU_GDPR: ComplianceConfig(
                region=ComplianceRegion.EU_GDPR,
                data_retention_days=365,  # Default, varies by purpose
                encryption_required=True,
                audit_logging=True,
                consent_required=True,
                right_to_deletion=True,
                data_portability=True,
                breach_notification_hours=72,
                specific_requirements={
                    "lawful_basis": True,
                    "data_protection_officer": False,  # Depends on organization size
                    "impact_assessment": True,
                    "data_minimization": True,
                    "purpose_limitation": True
                }
            ),
            ComplianceRegion.US_CCPA: ComplianceConfig(
                region=ComplianceRegion.US_CCPA,
                data_retention_days=365,
                encryption_required=False,  # Recommended but not required
                audit_logging=True,
                consent_required=False,  # Opt-out model
                right_to_deletion=True,
                data_portability=True,
                breach_notification_hours=0,  # No specific timeframe
                specific_requirements={
                    "sale_opt_out": True,
                    "non_discrimination": True,
                    "third_party_disclosure": True,
                    "consumer_request_verification": True
                }
            ),
            ComplianceRegion.SINGAPORE_PDPA: ComplianceConfig(
                region=ComplianceRegion.SINGAPORE_PDPA,
                data_retention_days=365,
                encryption_required=True,
                audit_logging=True,
                consent_required=True,
                right_to_deletion=False,  # Limited
                data_portability=False,
                breach_notification_hours=72,
                specific_requirements={
                    "data_protection_officer": True,
                    "consent_withdrawal": True,
                    "purpose_limitation": True,
                    "data_accuracy": True
                }
            ),
            ComplianceRegion.CANADA_PIPEDA: ComplianceConfig(
                region=ComplianceRegion.CANADA_PIPEDA,
                data_retention_days=365,
                encryption_required=True,
                audit_logging=True,
                consent_required=True,
                right_to_deletion=False,
                data_portability=False,
                breach_notification_hours=72,
                specific_requirements={
                    "meaningful_consent": True,
                    "privacy_by_design": True,
                    "accountability": True
                }
            ),
            ComplianceRegion.AUSTRALIA_APA: ComplianceConfig(
                region=ComplianceRegion.AUSTRALIA_APA,
                data_retention_days=365,
                encryption_required=False,
                audit_logging=True,
                consent_required=False,  # Varies by purpose
                right_to_deletion=False,
                data_portability=False,
                breach_notification_hours=72,
                specific_requirements={
                    "notifiable_data_breaches": True,
                    "privacy_policy": True,
                    "collection_limitation": True
                }
            ),
            ComplianceRegion.JAPAN_APPI: ComplianceConfig(
                region=ComplianceRegion.JAPAN_APPI,
                data_retention_days=365,
                encryption_required=True,
                audit_logging=True,
                consent_required=True,
                right_to_deletion=True,
                data_portability=False,
                breach_notification_hours=0,  # No specific timeframe
                specific_requirements={
                    "purpose_specification": True,
                    "use_limitation": True,
                    "data_quality": True,
                    "organizational_measures": True
                }
            )
        }
        
        return configs
    
    def get_compliance_requirements(self, region: ComplianceRegion) -> ComplianceConfig:
        """Get compliance requirements for a specific region."""
        return self.compliance_configs[region]
    
    def validate_compliance(self, region: ComplianceRegion, data_practices: Dict[str, Any]) -> Dict[str, Any]:
        """Validate current practices against regional compliance requirements."""
        
        config = self.compliance_configs[region]
        compliance_report = {
            "region": region.value,
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "score": 100.0
        }
        
        # Check encryption requirement
        if config.encryption_required and not data_practices.get("encryption_enabled", False):
            compliance_report["violations"].append("Encryption required but not enabled")
            compliance_report["compliant"] = False
            compliance_report["score"] -= 20
        
        # Check audit logging
        if config.audit_logging and not data_practices.get("audit_logging", False):
            compliance_report["violations"].append("Audit logging required but not enabled")
            compliance_report["compliant"] = False
            compliance_report["score"] -= 15
        
        # Check data retention
        actual_retention = data_practices.get("data_retention_days", 0)
        if actual_retention > config.data_retention_days:
            compliance_report["violations"].append(
                f"Data retention {actual_retention} days exceeds limit {config.data_retention_days} days"
            )
            compliance_report["compliant"] = False
            compliance_report["score"] -= 10
        
        # Check consent requirements
        if config.consent_required and not data_practices.get("consent_obtained", False):
            compliance_report["violations"].append("User consent required but not obtained")
            compliance_report["compliant"] = False
            compliance_report["score"] -= 25
        
        # Check breach notification capability
        if config.breach_notification_hours > 0 and not data_practices.get("breach_notification_process", False):
            compliance_report["recommendations"].append(
                f"Implement breach notification process within {config.breach_notification_hours} hours"
            )
            compliance_report["score"] -= 5
        
        # Check specific regional requirements
        for requirement, required in config.specific_requirements.items():
            if required and not data_practices.get(requirement, False):
                compliance_report["recommendations"].append(
                    f"Consider implementing {requirement.replace('_', ' ')}"
                )
                compliance_report["score"] -= 3
        
        return compliance_report


class GlobalFirmwareAnalyzer:
    """Global-first firmware analyzer with multi-region and i18n support."""
    
    def __init__(self, language: SupportedLanguage = SupportedLanguage.ENGLISH,
                 compliance_region: ComplianceRegion = ComplianceRegion.EU_GDPR,
                 deployment_region: DeploymentRegion = DeploymentRegion.US_EAST_1):
        
        self.i18n = InternationalizationManager(language)
        self.compliance = ComplianceManager()
        self.deployment_region = deployment_region
        self.compliance_region = compliance_region
        
        # Base analyzer
        self.analyzer = ScalableFirmwareAnalyzer("cortex-m4")
        
        print(f"🌍 Global Firmware Analyzer initialized")
        print(f"   Language: {language.value}")
        print(f"   Compliance: {compliance_region.value}")
        print(f"   Deployment: {deployment_region.value}")
    
    def analyze_firmware_global(self, firmware_path: str) -> Dict[str, Any]:
        """Perform globally-compliant firmware analysis."""
        
        print(f"\n🔍 {self.i18n.translate('firmware_analysis')}: {Path(firmware_path).name}")
        
        # Perform analysis
        result = self.analyzer.analyze_firmware(firmware_path)
        
        if not result:
            return {
                "status": "error",
                "message": self.i18n.translate("error_occurred"),
                "timestamp": self._get_localized_timestamp()
            }
        
        # Translate results
        localized_result = {
            "status": "success",
            "analysis_id": str(uuid.uuid4()),
            "timestamp": self._get_localized_timestamp(),
            "language": self.i18n.current_language.value,
            "compliance_region": self.compliance_region.value,
            "deployment_region": self.deployment_region.value,
            
            # Localized analysis results
            "firmware_analysis": {
                "file_path": firmware_path,
                "architecture": result.architecture,
                "vulnerabilities_found": len(result.vulnerabilities),
                "risk_score": self.i18n.format_number(result.risk_score),
                "status": result.status.value
            },
            
            # Translated vulnerability details
            "vulnerabilities": self._translate_vulnerabilities(result.vulnerabilities),
            
            # Localized recommendations
            "recommendations": self._translate_recommendations(result.recommendations),
            
            # Compliance information
            "compliance": self._generate_compliance_report(),
            
            # Regional metadata
            "metadata": {
                "timezone": self.i18n.localization_configs[self.i18n.current_language].timezone,
                "currency": self.i18n.localization_configs[self.i18n.current_language].currency,
                "measurement_system": self.i18n.localization_configs[self.i18n.current_language].measurement_system,
                "data_retention_days": self.compliance.compliance_configs[self.compliance_region].data_retention_days
            }
        }
        
        return localized_result
    
    def _translate_vulnerabilities(self, vulnerabilities: List[CryptoVulnerability]) -> List[Dict[str, Any]]:
        """Translate vulnerability information to current language."""
        
        translated_vulns = []
        
        for vuln in vulnerabilities:
            translated_vuln = {
                "function_name": vuln.function_name,
                "algorithm": vuln.algorithm.value,
                "address": f"0x{vuln.address:08x}",
                "risk_level": self.i18n.translate(vuln.risk_level.value.lower()),
                "description": vuln.description,  # Could be translated with more sophisticated system
                "mitigation": self._translate_mitigation(vuln.mitigation),
                "memory_impact": {
                    "stack_usage": f"{vuln.stack_usage} bytes",
                    "available_stack": f"{vuln.available_stack} bytes"
                }
            }
            translated_vulns.append(translated_vuln)
        
        return translated_vulns
    
    def _translate_mitigation(self, mitigation: str) -> str:
        """Translate mitigation recommendations."""
        
        # Simple translation mapping - in production this would be more sophisticated
        translation_map = {
            "Replace": self.i18n.translate("replace_with_pqc"),
            "Immediate": self.i18n.translate("immediate_action_required"),
            "Test": self.i18n.translate("test_in_isolation")
        }
        
        translated = mitigation
        for en_term, translated_term in translation_map.items():
            if en_term in mitigation:
                translated = translated.replace(en_term, translated_term)
        
        return translated
    
    def _translate_recommendations(self, recommendations: List[str]) -> List[str]:
        """Translate recommendations to current language."""
        
        translated_recs = []
        
        for rec in recommendations:
            # Simple pattern-based translation
            if "RSA" in rec:
                translated_recs.append(self.i18n.translate("replace_with_pqc") + " (RSA → Dilithium)")
            elif "ECC" in rec or "ECDSA" in rec or "ECDH" in rec:
                translated_recs.append(self.i18n.translate("replace_with_pqc") + " (ECC → Kyber + Dilithium)")
            elif "memory" in rec.lower():
                translated_recs.append(f"{self.i18n.translate('memory_constraints')}: {rec}")
            elif "test" in rec.lower():
                translated_recs.append(self.i18n.translate("test_in_isolation"))
            else:
                translated_recs.append(rec)  # Keep original if no translation pattern
        
        return translated_recs
    
    def _generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for current region."""
        
        # Simulate current data practices
        current_practices = {
            "encryption_enabled": True,
            "audit_logging": True,
            "data_retention_days": 365,
            "consent_obtained": True,
            "breach_notification_process": True,
            "privacy_policy": True
        }
        
        compliance_report = self.compliance.validate_compliance(
            self.compliance_region, current_practices
        )
        
        return compliance_report
    
    def _get_localized_timestamp(self) -> str:
        """Get current timestamp in localized format."""
        now = datetime.now(timezone.utc)
        return self.i18n.format_datetime(now)
    
    def set_language(self, language: SupportedLanguage):
        """Change the analysis language."""
        self.i18n.set_language(language)
    
    def set_compliance_region(self, region: ComplianceRegion):
        """Change the compliance region."""
        self.compliance_region = region
        print(f"🔒 Compliance region set to: {region.value}")
    
    def set_deployment_region(self, region: DeploymentRegion):
        """Change the deployment region."""
        self.deployment_region = region
        print(f"🌐 Deployment region set to: {region.value}")


def demonstrate_global_implementation():
    """Demonstrate global-first implementation with i18n and compliance."""
    
    print("🌍 PQC IoT Retrofit Scanner - Global-First Implementation Demo")
    print("=" * 70)
    
    # Create test firmware
    test_firmware = Path("global_test_firmware.bin")
    test_firmware.write_bytes(b"GLOBAL_FIRMWARE" + b"RSA_SIGNATURE_2048" + b"ECDSA_P256" + b"\x00" * 1024)
    
    try:
        # Test different language and region combinations
        test_scenarios = [
            {
                "name": "US English (CCPA)",
                "language": SupportedLanguage.ENGLISH,
                "compliance": ComplianceRegion.US_CCPA,
                "deployment": DeploymentRegion.US_WEST_2
            },
            {
                "name": "European French (GDPR)",
                "language": SupportedLanguage.FRENCH,
                "compliance": ComplianceRegion.EU_GDPR,
                "deployment": DeploymentRegion.EU_WEST_1
            },
            {
                "name": "Japanese (APPI)",
                "language": SupportedLanguage.JAPANESE,
                "compliance": ComplianceRegion.JAPAN_APPI,
                "deployment": DeploymentRegion.ASIA_PACIFIC_2
            },
            {
                "name": "Chinese (Singapore PDPA)",
                "language": SupportedLanguage.CHINESE,
                "compliance": ComplianceRegion.SINGAPORE_PDPA,
                "deployment": DeploymentRegion.ASIA_PACIFIC_1
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print("-" * 50)
            
            # Initialize global analyzer for this scenario
            global_analyzer = GlobalFirmwareAnalyzer(
                language=scenario["language"],
                compliance_region=scenario["compliance"],
                deployment_region=scenario["deployment"]
            )
            
            # Perform analysis
            result = global_analyzer.analyze_firmware_global(str(test_firmware))
            
            # Display key results
            print(f"   📊 {global_analyzer.i18n.translate('vulnerabilities_found')}: {result['firmware_analysis']['vulnerabilities_found']}")
            print(f"   📊 {global_analyzer.i18n.translate('risk_level')}: {result['firmware_analysis']['risk_score']}")
            print(f"   🔒 Compliance Score: {result['compliance']['score']:.1f}%")
            print(f"   🌐 Timezone: {result['metadata']['timezone']}")
            print(f"   💱 Currency: {result['metadata']['currency']}")
            
            # Show translated recommendations
            if result["recommendations"]:
                print(f"   💡 {global_analyzer.i18n.translate('recommendations')}:")
                for rec in result["recommendations"][:2]:
                    print(f"      • {rec}")
            
            # Show compliance status
            if not result["compliance"]["compliant"]:
                print("   ⚠️ Compliance Issues:")
                for violation in result["compliance"]["violations"]:
                    print(f"      ❌ {violation}")
        
        # Demonstrate language switching
        print(f"\n📋 Language Switching Demo")
        print("-" * 50)
        
        analyzer = GlobalFirmwareAnalyzer()
        
        for lang in [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH, SupportedLanguage.GERMAN]:
            analyzer.set_language(lang)
            print(f"   {lang.value}: {analyzer.i18n.translate('analysis_complete')} | {analyzer.i18n.translate('critical')} | {analyzer.i18n.translate('recommendations')}")
        
        # Demonstrate compliance comparison
        print(f"\n📋 Regional Compliance Comparison")
        print("-" * 50)
        
        compliance_manager = ComplianceManager()
        
        sample_practices = {
            "encryption_enabled": True,
            "audit_logging": False,  # Violation
            "consent_obtained": False,  # Potential violation
            "data_retention_days": 400  # Potential violation
        }
        
        for region in [ComplianceRegion.EU_GDPR, ComplianceRegion.US_CCPA, ComplianceRegion.SINGAPORE_PDPA]:
            compliance_result = compliance_manager.validate_compliance(region, sample_practices)
            status_icon = "✅" if compliance_result["compliant"] else "❌"
            print(f"   {status_icon} {region.value}: {compliance_result['score']:.1f}% ({len(compliance_result['violations'])} violations)")
    
    finally:
        # Cleanup
        test_firmware.unlink(missing_ok=True)
    
    print(f"\n🎉 Global-first implementation demonstration complete!")
    print(f"   Languages Supported: {len(SupportedLanguage)}")
    print(f"   Compliance Regions: {len(ComplianceRegion)}")
    print(f"   Deployment Regions: {len(DeploymentRegion)}")


if __name__ == "__main__":
    demonstrate_global_implementation()