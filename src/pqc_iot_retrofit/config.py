"""Configuration management for PQC IoT Retrofit Scanner."""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "pqc_scanner.db"
    enable_wal_mode: bool = True
    connection_timeout: int = 30
    max_connections: int = 10
    backup_interval_hours: int = 24


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    directory: str = ".cache"
    max_memory_items: int = 1000
    default_ttl_minutes: int = 60
    cleanup_interval_hours: int = 6
    max_file_cache_size_mb: int = 500


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    max_parallel_threads: int = 4
    max_analysis_time_seconds: int = 300
    enable_deep_analysis: bool = True
    cache_disassembly: bool = True
    cache_scan_results: bool = True
    max_firmware_size_mb: int = 100


@dataclass
class PatchConfig:
    """Patch generation configuration."""
    default_security_level: int = 2
    default_optimization: str = "balanced"
    enable_hybrid_patches: bool = True
    verify_patches: bool = True
    max_patch_size_mb: int = 10


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_file_hash_verification: bool = True
    quarantine_suspicious_files: bool = True
    max_file_scan_size_mb: int = 50
    allowed_architectures: List[str] = field(default_factory=lambda: [
        'cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7',
        'esp32', 'riscv32', 'avr'
    ])
    blocked_file_extensions: List[str] = field(default_factory=lambda: [
        '.exe', '.scr', '.bat', '.cmd', '.ps1'
    ])


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enable_metrics: bool = True
    metrics_export_interval_seconds: int = 60
    enable_performance_profiling: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    enable_sentry: bool = False
    sentry_dsn: Optional[str] = None


@dataclass
class PQCConfig:
    """Main configuration class."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    data_directory: str = "data"
    temp_directory: str = "tmp"
    enable_experimental_features: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PQCConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if 'database' in data:
            data['database'] = DatabaseConfig(**data['database'])
        if 'cache' in data:
            data['cache'] = CacheConfig(**data['cache'])
        if 'analysis' in data:
            data['analysis'] = AnalysisConfig(**data['analysis'])
        if 'patch' in data:
            data['patch'] = PatchConfig(**data['patch'])
        if 'security' in data:
            data['security'] = SecurityConfig(**data['security'])
        if 'monitoring' in data:
            if 'log_level' in data['monitoring']:
                data['monitoring']['log_level'] = LogLevel(data['monitoring']['log_level'])
            data['monitoring'] = MonitoringConfig(**data['monitoring'])
        
        return cls(**data)
    
    def save(self, config_path: str) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to configuration file
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = self.to_dict()
        
        # Convert enums to strings for JSON serialization
        if 'monitoring' in config_data and 'log_level' in config_data['monitoring']:
            config_data['monitoring']['log_level'] = config_data['monitoring']['log_level'].value
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    @classmethod
    def load(cls, config_path: str) -> 'PQCConfig':
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration instance
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            return cls.from_dict(config_data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def validate(self) -> List[str]:
        """Validate configuration settings.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate database configuration
        if not self.database.path:
            errors.append("Database path cannot be empty")
        
        if self.database.connection_timeout <= 0:
            errors.append("Database connection timeout must be positive")
        
        # Validate cache configuration
        if self.cache.max_memory_items <= 0:
            errors.append("Cache max memory items must be positive")
        
        if self.cache.default_ttl_minutes <= 0:
            errors.append("Cache default TTL must be positive")
        
        # Validate analysis configuration
        if self.analysis.max_parallel_threads <= 0:
            errors.append("Max parallel threads must be positive")
        
        if self.analysis.max_analysis_time_seconds <= 0:
            errors.append("Max analysis time must be positive")
        
        # Validate patch configuration
        if not 1 <= self.patch.default_security_level <= 5:
            errors.append("Default security level must be between 1 and 5")
        
        if self.patch.default_optimization not in ['size', 'speed', 'balanced', 'memory']:
            errors.append("Invalid default optimization level")
        
        # Validate security configuration
        if self.security.max_file_scan_size_mb <= 0:
            errors.append("Max file scan size must be positive")
        
        # Validate directories exist or can be created
        for directory in [self.data_directory, self.temp_directory, self.cache.directory]:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
        
        return errors
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.monitoring.log_level.value)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup file handler if specified
        handlers = [console_handler]
        if self.monitoring.log_file:
            log_file = Path(self.monitoring.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
        
        # Set specific logger levels
        logging.getLogger('pqc_iot_retrofit').setLevel(log_level)
        
        # Suppress verbose third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        logger.info(f"Logging configured: level={self.monitoring.log_level.value}")
    
    def get_data_path(self, *path_components: str) -> Path:
        """Get path within data directory.
        
        Args:
            *path_components: Path components
            
        Returns:
            Full path within data directory
        """
        return Path(self.data_directory) / Path(*path_components)
    
    def get_cache_path(self, *path_components: str) -> Path:
        """Get path within cache directory.
        
        Args:
            *path_components: Path components
            
        Returns:
            Full path within cache directory
        """
        return Path(self.cache.directory) / Path(*path_components)
    
    def get_temp_path(self, *path_components: str) -> Path:
        """Get path within temp directory.
        
        Args:
            *path_components: Path components
            
        Returns:
            Full path within temp directory
        """
        return Path(self.temp_directory) / Path(*path_components)


class ConfigManager:
    """Configuration manager with environment variable support."""
    
    DEFAULT_CONFIG_PATHS = [
        'pqc_config.json',
        '~/.pqc_config.json',
        '/etc/pqc/config.json'
    ]
    
    ENV_PREFIX = 'PQC_'
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config = None
    
    def load_config(self, config_path: Optional[str] = None) -> PQCConfig:
        """Load configuration from file and environment variables.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Configuration instance
        """
        # Load base configuration from file
        if config_path:
            config = PQCConfig.load(config_path)
        else:
            # Try default paths
            config = None
            for default_path in self.DEFAULT_CONFIG_PATHS:
                expanded_path = Path(default_path).expanduser()
                if expanded_path.exists():
                    config = PQCConfig.load(str(expanded_path))
                    break
            
            if config is None:
                config = PQCConfig()
        
        # Override with environment variables
        self._apply_env_overrides(config)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error("Configuration validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")
        
        # Setup logging
        config.setup_logging()
        
        self._config = config
        return config
    
    def _apply_env_overrides(self, config: PQCConfig) -> None:
        """Apply environment variable overrides.
        
        Args:
            config: Configuration to modify
        """
        env_mappings = {
            f'{self.ENV_PREFIX}DATABASE_PATH': ('database.path', str),
            f'{self.ENV_PREFIX}CACHE_ENABLED': ('cache.enabled', lambda x: x.lower() == 'true'),
            f'{self.ENV_PREFIX}CACHE_DIRECTORY': ('cache.directory', str),
            f'{self.ENV_PREFIX}MAX_THREADS': ('analysis.max_parallel_threads', int),
            f'{self.ENV_PREFIX}LOG_LEVEL': ('monitoring.log_level', lambda x: LogLevel(x.upper())),
            f'{self.ENV_PREFIX}LOG_FILE': ('monitoring.log_file', str),
            f'{self.ENV_PREFIX}DATA_DIR': ('data_directory', str),
            f'{self.ENV_PREFIX}TEMP_DIR': ('temp_directory', str),
            f'{self.ENV_PREFIX}DEBUG': ('enable_experimental_features', lambda x: x.lower() == 'true'),
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    self._set_nested_attr(config, config_path, converted_value)
                    logger.debug(f"Applied environment override: {env_var}={env_value}")
                except Exception as e:
                    logger.warning(f"Failed to apply environment override {env_var}={env_value}: {e}")
    
    def _set_nested_attr(self, obj: Any, path: str, value: Any) -> None:
        """Set nested attribute using dot notation.
        
        Args:
            obj: Object to modify
            path: Dot-separated attribute path
            value: Value to set
        """
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def get_config(self) -> PQCConfig:
        """Get current configuration.
        
        Returns:
            Current configuration instance
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self, config_path: Optional[str] = None) -> PQCConfig:
        """Reload configuration.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Reloaded configuration instance
        """
        logger.info("Reloading configuration")
        self._config = None
        return self.load_config(config_path)
    
    def create_default_config(self, output_path: str) -> None:
        """Create default configuration file.
        
        Args:
            output_path: Path to save default configuration
        """
        default_config = PQCConfig()
        default_config.save(output_path)
        logger.info(f"Default configuration created at {output_path}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> PQCConfig:
    """Get global configuration instance.
    
    Returns:
        Global configuration instance
    """
    return config_manager.get_config()


def load_config(config_path: Optional[str] = None) -> PQCConfig:
    """Load configuration from file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration instance
    """
    return config_manager.load_config(config_path)