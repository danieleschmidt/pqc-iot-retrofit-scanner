"""
Internationalization and localization support for PQC IoT Retrofit Scanner.

This module provides:
- Multi-language support for CLI and API messages
- Locale-specific formatting for dates, numbers, and currencies
- Regional compliance and regulatory information
- Cultural adaptations for different markets
"""

import os
import json
import gettext
import locale
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import logging

from .monitoring import metrics_collector


class SupportedLanguage(Enum):
    """Supported languages with ISO 639-1 codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    KOREAN = "ko"


class SupportedRegion(Enum):
    """Supported regions with regulatory compliance information."""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    LATIN_AMERICA = "latam"
    MIDDLE_EAST_AFRICA = "mea"


@dataclass
class RegionConfig:
    """Configuration for specific regions."""
    region_code: str
    name: str
    primary_languages: List[str]
    regulatory_standards: List[str]
    compliance_requirements: Dict[str, Any]
    date_format: str
    number_format: str
    currency_symbol: str
    timezone_offset: int  # Hours from UTC


# Regional configurations
REGION_CONFIGS = {
    SupportedRegion.NORTH_AMERICA: RegionConfig(
        region_code="na",
        name="North America",
        primary_languages=["en", "es", "fr"],
        regulatory_standards=["NIST", "FCC", "ISED"],
        compliance_requirements={
            "data_retention": 90,  # days
            "encryption_standards": ["AES-256", "RSA-2048", "ECDSA-P256"],
            "audit_logging": True,
            "vulnerability_disclosure": 30  # days
        },
        date_format="%m/%d/%Y",
        number_format="en_US",
        currency_symbol="$",
        timezone_offset=-5  # EST
    ),
    
    SupportedRegion.EUROPE: RegionConfig(
        region_code="eu",
        name="Europe",
        primary_languages=["en", "de", "fr", "es", "it"],
        regulatory_standards=["GDPR", "ETSI", "CE", "ISO_27001"],
        compliance_requirements={
            "data_retention": 30,  # days (GDPR minimization)
            "encryption_standards": ["AES-256", "RSA-2048"],
            "audit_logging": True,
            "right_to_be_forgotten": True,
            "data_protection_officer": True,
            "vulnerability_disclosure": 45  # days
        },
        date_format="%d/%m/%Y",
        number_format="de_DE",
        currency_symbol="€",
        timezone_offset=1  # CET
    ),
    
    SupportedRegion.ASIA_PACIFIC: RegionConfig(
        region_code="apac",
        name="Asia Pacific",
        primary_languages=["en", "ja", "zh", "ko"],
        regulatory_standards=["JIS", "GB", "ACMA", "TTA"],
        compliance_requirements={
            "data_retention": 180,  # days
            "encryption_standards": ["AES-256", "SM4", "RSA-2048"],
            "audit_logging": True,
            "local_data_storage": True,
            "vulnerability_disclosure": 15  # days
        },
        date_format="%Y/%m/%d",
        number_format="ja_JP",
        currency_symbol="¥",
        timezone_offset=9  # JST
    ),
    
    SupportedRegion.LATIN_AMERICA: RegionConfig(
        region_code="latam",
        name="Latin America", 
        primary_languages=["es", "pt", "en"],
        regulatory_standards=["ANATEL", "CITEL", "SUBTEL"],
        compliance_requirements={
            "data_retention": 365,  # days
            "encryption_standards": ["AES-256", "RSA-2048"],
            "audit_logging": True,
            "local_language_support": True,
            "vulnerability_disclosure": 60  # days
        },
        date_format="%d/%m/%Y",
        number_format="es_MX",
        currency_symbol="$",
        timezone_offset=-3  # BRT
    )
}


class I18nManager:
    """Internationalization manager for multi-language support."""
    
    def __init__(self, default_language: str = "en", default_region: str = "na"):
        self.default_language = default_language
        self.default_region = default_region
        self.current_language = default_language
        self.current_region = default_region
        
        # Translation catalogs
        self.translations: Dict[str, Dict[str, str]] = {}
        self.regional_configs: Dict[str, RegionConfig] = {
            region.value: config for region, config in REGION_CONFIGS.items()
        }
        
        # Initialize locales directory
        self.locale_dir = Path(__file__).parent / "locales"
        self.locale_dir.mkdir(exist_ok=True)
        
        # Load translations
        self._load_all_translations()
        
        # Set system locale
        self._set_system_locale()
    
    def _load_all_translations(self):
        """Load translation files for all supported languages."""
        # Create default translations if they don't exist
        self._create_default_translations()
        
        for language in SupportedLanguage:
            lang_code = language.value
            translation_file = self.locale_dir / f"{lang_code}.json"
            
            try:
                if translation_file.exists():
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                else:
                    # Use English as fallback
                    self.translations[lang_code] = self.translations.get('en', {})
                    
            except Exception as e:
                logging.warning(f"Failed to load translation for {lang_code}: {e}")
                self.translations[lang_code] = {}
    
    def _create_default_translations(self):
        """Create default English translations."""
        default_translations = {
            # CLI Messages
            "cli.scan.start": "Starting firmware scan...",
            "cli.scan.complete": "Scan completed. Found {count} vulnerabilities.",
            "cli.scan.error": "Scan failed: {error}",
            "cli.patch.start": "Generating PQC patches...",
            "cli.patch.complete": "Generated {count} patches successfully.",
            "cli.patch.error": "Patch generation failed: {error}",
            
            # Vulnerability Messages
            "vuln.rsa_weak": "Weak RSA key detected (size: {size} bits)",
            "vuln.dh_weak": "Weak Diffie-Hellman parameters detected",
            "vuln.ecc_weak": "Weak elliptic curve detected: {curve}",
            "vuln.hash_weak": "Weak hash algorithm detected: {algorithm}",
            "vuln.cipher_weak": "Weak cipher detected: {cipher}",
            
            # Risk Levels
            "risk.critical": "Critical",
            "risk.high": "High",
            "risk.medium": "Medium",
            "risk.low": "Low",
            "risk.info": "Informational",
            
            # PQC Algorithms
            "pqc.kyber512": "Kyber-512 (NIST Level 1)",
            "pqc.kyber768": "Kyber-768 (NIST Level 3)", 
            "pqc.kyber1024": "Kyber-1024 (NIST Level 5)",
            "pqc.dilithium2": "Dilithium-2 (NIST Level 1)",
            "pqc.dilithium3": "Dilithium-3 (NIST Level 3)",
            "pqc.dilithium5": "Dilithium-5 (NIST Level 5)",
            
            # Error Messages
            "error.file_not_found": "File not found: {filepath}",
            "error.permission_denied": "Permission denied: {filepath}",
            "error.invalid_architecture": "Unsupported architecture: {arch}",
            "error.memory_constraint": "Memory constraint exceeded: {constraint}",
            "error.network_error": "Network error: {details}",
            
            # Progress Messages
            "progress.analyzing": "Analyzing firmware...",
            "progress.disassembling": "Disassembling binary...",
            "progress.detecting_crypto": "Detecting cryptographic functions...",
            "progress.generating_patches": "Generating PQC implementations...",
            "progress.optimizing": "Optimizing for target architecture...",
            
            # Compliance Messages
            "compliance.nist_pqc": "NIST Post-Quantum Cryptography Standard",
            "compliance.gdpr_ready": "GDPR Compliant Data Processing",
            "compliance.iot_baseline": "IoT Cybersecurity Baseline Compliant",
            "compliance.fips_140": "FIPS 140-2 Level 2 Compliant",
            
            # Success Messages
            "success.scan_complete": "Firmware scan completed successfully",
            "success.patches_generated": "PQC patches generated successfully",
            "success.deployment_ready": "Deployment package ready",
            "success.compliance_validated": "Regulatory compliance validated",
            
            # Status Messages
            "status.healthy": "System healthy",
            "status.degraded": "System performance degraded",
            "status.unhealthy": "System experiencing issues",
            "status.maintenance": "System under maintenance",
            
            # Regional Compliance
            "regional.na.standards": "North American Standards (NIST, FCC)",
            "regional.eu.standards": "European Standards (GDPR, ETSI, CE)",
            "regional.apac.standards": "Asia-Pacific Standards (JIS, GB, ACMA)",
            "regional.latam.standards": "Latin American Standards (ANATEL, CITEL)"
        }
        
        # Save English translations
        en_file = self.locale_dir / "en.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        self.translations['en'] = default_translations
    
    def set_language(self, language_code: str):
        """Set the current language."""
        if language_code in [lang.value for lang in SupportedLanguage]:
            self.current_language = language_code
            self._set_system_locale()
            metrics_collector.record_metric("i18n.language_changed", 1, "events", 
                                          tags={"language": language_code})
        else:
            raise ValueError(f"Unsupported language: {language_code}")
    
    def set_region(self, region_code: str):
        """Set the current region."""
        if region_code in self.regional_configs:
            self.current_region = region_code
            metrics_collector.record_metric("i18n.region_changed", 1, "events",
                                          tags={"region": region_code})
        else:
            raise ValueError(f"Unsupported region: {region_code}")
    
    def _set_system_locale(self):
        """Set system locale based on current language and region."""
        try:
            # Map language/region to locale
            locale_map = {
                "en": "en_US.UTF-8",
                "es": "es_ES.UTF-8", 
                "fr": "fr_FR.UTF-8",
                "de": "de_DE.UTF-8",
                "ja": "ja_JP.UTF-8",
                "zh": "zh_CN.UTF-8",
                "pt": "pt_BR.UTF-8",
                "it": "it_IT.UTF-8",
                "ru": "ru_RU.UTF-8",
                "ko": "ko_KR.UTF-8"
            }
            
            system_locale = locale_map.get(self.current_language, "en_US.UTF-8")
            locale.setlocale(locale.LC_ALL, system_locale)
            
        except locale.Error:
            # Fallback to C locale
            locale.setlocale(locale.LC_ALL, "C")
            logging.warning(f"Failed to set locale for {self.current_language}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a message key to the current language."""
        # Get translation for current language
        lang_translations = self.translations.get(self.current_language, {})
        
        # Fall back to English if translation not found
        if key not in lang_translations and self.current_language != 'en':
            lang_translations = self.translations.get('en', {})
        
        # Get translated text
        text = lang_translations.get(key, key)  # Use key as fallback
        
        # Format with provided arguments
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError) as e:
            logging.warning(f"Translation formatting error for key '{key}': {e}")
            return text
    
    def get_regional_config(self, region_code: Optional[str] = None) -> RegionConfig:
        """Get regional configuration."""
        region = region_code or self.current_region
        return self.regional_configs.get(region, REGION_CONFIGS[SupportedRegion.NORTH_AMERICA])
    
    def format_date(self, date_obj, region_code: Optional[str] = None) -> str:
        """Format date according to regional preferences."""
        config = self.get_regional_config(region_code)
        return date_obj.strftime(config.date_format)
    
    def format_number(self, number: Union[int, float], region_code: Optional[str] = None) -> str:
        """Format number according to regional preferences."""
        config = self.get_regional_config(region_code)
        try:
            # Use locale-specific formatting
            if config.number_format == "en_US":
                return f"{number:,}"  # US format: 1,234.56
            elif config.number_format == "de_DE":
                return f"{number:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')  # DE format: 1.234,56
            else:
                return str(number)
        except Exception:
            return str(number)
    
    def format_currency(self, amount: float, region_code: Optional[str] = None) -> str:
        """Format currency according to regional preferences."""
        config = self.get_regional_config(region_code)
        formatted_number = self.format_number(amount, region_code)
        return f"{config.currency_symbol}{formatted_number}"
    
    def get_compliance_requirements(self, region_code: Optional[str] = None) -> Dict[str, Any]:
        """Get compliance requirements for a region."""
        config = self.get_regional_config(region_code)
        return config.compliance_requirements
    
    def get_regulatory_standards(self, region_code: Optional[str] = None) -> List[str]:
        """Get regulatory standards for a region."""
        config = self.get_regional_config(region_code)
        return config.regulatory_standards
    
    def validate_regional_compliance(self, scan_results: Dict[str, Any], 
                                   region_code: Optional[str] = None) -> Dict[str, Any]:
        """Validate scan results against regional compliance requirements."""
        config = self.get_regional_config(region_code)
        compliance_results = {
            "region": config.name,
            "standards": config.regulatory_standards,
            "compliant": True,
            "violations": [],
            "recommendations": []
        }
        
        requirements = config.compliance_requirements
        
        # Check encryption standards compliance
        if "encryption_standards" in requirements:
            allowed_algorithms = requirements["encryption_standards"]
            found_algorithms = scan_results.get("crypto_algorithms", [])
            
            for algorithm in found_algorithms:
                if algorithm not in allowed_algorithms:
                    compliance_results["violations"].append({
                        "type": "encryption_standard",
                        "details": self.translate("compliance.weak_algorithm", 
                                                algorithm=algorithm, 
                                                region=config.name)
                    })
                    compliance_results["compliant"] = False
        
        # Check data retention requirements
        if "data_retention" in requirements:
            max_retention = requirements["data_retention"]
            compliance_results["recommendations"].append({
                "type": "data_retention",
                "details": self.translate("compliance.data_retention", 
                                        days=max_retention, 
                                        region=config.name)
            })
        
        # Check audit logging requirements
        if requirements.get("audit_logging", False):
            if not scan_results.get("audit_logging_enabled", False):
                compliance_results["violations"].append({
                    "type": "audit_logging",
                    "details": self.translate("compliance.audit_logging_required",
                                            region=config.name)
                })
                compliance_results["compliant"] = False
        
        return compliance_results
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages."""
        return [
            {
                "code": lang.value,
                "name": self.translate(f"language.{lang.value}"),
                "native_name": self._get_native_language_name(lang.value)
            }
            for lang in SupportedLanguage
        ]
    
    def get_supported_regions(self) -> List[Dict[str, Any]]:
        """Get list of supported regions."""
        return [
            {
                "code": region_code,
                "name": config.name,
                "languages": config.primary_languages,
                "standards": config.regulatory_standards
            }
            for region_code, config in self.regional_configs.items()
        ]
    
    def _get_native_language_name(self, language_code: str) -> str:
        """Get native name of language."""
        native_names = {
            "en": "English",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "ja": "日本語",
            "zh": "中文",
            "pt": "Português",
            "it": "Italiano",
            "ru": "Русский",
            "ko": "한국어"
        }
        return native_names.get(language_code, language_code)


# Global I18n manager instance
i18n_manager = I18nManager()

# Convenience functions
def translate(key: str, **kwargs) -> str:
    """Translate a message key."""
    return i18n_manager.translate(key, **kwargs)

def set_language(language_code: str):
    """Set the current language."""
    i18n_manager.set_language(language_code)

def set_region(region_code: str):
    """Set the current region."""
    i18n_manager.set_region(region_code)

def get_compliance_requirements(region_code: Optional[str] = None) -> Dict[str, Any]:
    """Get compliance requirements for current or specified region."""
    return i18n_manager.get_compliance_requirements(region_code)

def validate_compliance(scan_results: Dict[str, Any], 
                       region_code: Optional[str] = None) -> Dict[str, Any]:
    """Validate scan results against regional compliance."""
    return i18n_manager.validate_regional_compliance(scan_results, region_code)

# Translation decorator for functions
def localized(message_key: str):
    """Decorator to automatically translate function error messages."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                localized_message = translate(message_key, error=str(e))
                # Replace original exception message with localized version
                e.args = (localized_message,) + e.args[1:]
                raise
        return wrapper
    return decorator