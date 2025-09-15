"""
CuliFeed Configuration System
============================

Comprehensive configuration management with YAML files, environment variables,
validation, and type safety using Pydantic models.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings

from ..utils.exceptions import ConfigurationError, ErrorCode
from ..utils.validators import validate_environment_variable, validate_file_path


class AIProvider(str, Enum):
    """Available AI providers."""
    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI = "openai"


class LogLevel(str, Enum):
    """Available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProcessingSettings(BaseModel):
    """Processing pipeline configuration."""
    daily_run_hour: int = Field(default=8, ge=0, le=23, description="Hour of day to run processing (0-23)")
    max_articles_per_topic: int = Field(default=5, ge=1, le=20, description="Maximum articles to deliver per topic")
    ai_provider: AIProvider = Field(default=AIProvider.GEMINI, description="Primary AI provider")
    batch_size: int = Field(default=10, ge=1, le=50, description="Articles to process in one batch")
    parallel_feeds: int = Field(default=5, ge=1, le=20, description="Concurrent feed fetches")
    cache_embeddings: bool = Field(default=True, description="Cache article embeddings")
    max_content_length: int = Field(default=2000, ge=500, le=10000, description="Max content length for AI processing")
    
    @field_validator('daily_run_hour')
    @classmethod
    def validate_hour(cls, v):
        """Ensure hour is valid."""
        if not (0 <= v <= 23):
            raise ValueError("daily_run_hour must be between 0 and 23")
        return v


class LimitsSettings(BaseModel):
    """Cost control and rate limiting settings."""
    max_daily_api_calls: int = Field(default=950, ge=10, description="Maximum AI API calls per day")
    fallback_to_groq: bool = Field(default=True, description="Use Groq when primary API exhausted")
    fallback_to_keywords: bool = Field(default=True, description="Use keyword-only when all APIs exhausted") 
    enable_usage_alerts: bool = Field(default=True, description="Enable usage monitoring alerts")
    alert_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="Alert when usage exceeds threshold")
    max_feed_errors: int = Field(default=10, ge=1, le=100, description="Max errors before disabling feed")
    request_timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")


class DatabaseSettings(BaseModel):
    """Database configuration."""
    path: str = Field(default="data/culifeed.db", description="SQLite database file path")
    pool_size: int = Field(default=5, ge=1, le=20, description="Connection pool size")
    cleanup_days: int = Field(default=7, ge=1, le=365, description="Days to keep old articles")
    auto_vacuum: bool = Field(default=True, description="Enable automatic database maintenance")
    backup_enabled: bool = Field(default=True, description="Enable automatic backups")
    backup_interval_hours: int = Field(default=24, ge=1, le=168, description="Hours between backups")


class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Global log level")
    file_path: Optional[str] = Field(default="logs/culifeed.log", description="Log file path")
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Max log file size in MB")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of log backup files")
    structured_logging: bool = Field(default=False, description="Use structured JSON logging")
    console_logging: bool = Field(default=True, description="Enable console logging")


class TelegramSettings(BaseModel):
    """Telegram bot configuration."""
    bot_token: str = Field(..., description="Telegram bot token")
    admin_user_id: Optional[str] = Field(default=None, description="Admin user ID for management commands")
    webhook_url: Optional[AnyHttpUrl] = Field(default=None, description="Webhook URL for updates")
    webhook_secret: Optional[str] = Field(default=None, description="Webhook secret token")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retries for failed messages")
    
    @field_validator('bot_token')
    @classmethod
    def validate_bot_token(cls, v):
        """Validate bot token format."""
        if not v or not isinstance(v, str):
            raise ValueError("Bot token is required")
        
        # Basic format check for Telegram bot tokens
        # Allow test tokens for development
        if v.endswith('_test'):
            return v
            
        if not v.count(':') == 1 or len(v) < 20:
            raise ValueError("Invalid bot token format")
        
        return v


class AISettings(BaseModel):
    """AI providers configuration."""
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    gemini_model: str = Field(default="gemini-2.5-flash-lite", description="Gemini model to use")
    groq_model: str = Field(default="llama3-8b-8192", description="Groq model to use")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="AI temperature setting")
    max_tokens: int = Field(default=500, ge=50, le=2000, description="Maximum tokens per response")
    
    def get_primary_api_key(self, provider: AIProvider) -> Optional[str]:
        """Get API key for specified provider."""
        if provider == AIProvider.GEMINI:
            return self.gemini_api_key
        elif provider == AIProvider.GROQ:
            return self.groq_api_key
        elif provider == AIProvider.OPENAI:
            return self.openai_api_key
        return None
    
    def validate_provider_key(self, provider: AIProvider) -> bool:
        """Check if API key is available for provider."""
        return bool(self.get_primary_api_key(provider))


class UserSettings(BaseModel):
    """User preferences."""
    timezone: str = Field(default="UTC", description="User timezone")
    admin_user_id: Optional[str] = Field(default=None, description="Admin user ID")


class CuliFeedSettings(BaseSettings):
    """Main application settings."""
    
    # Core settings sections
    user: UserSettings = Field(default_factory=UserSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings) 
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    telegram: TelegramSettings
    ai: AISettings = Field(default_factory=AISettings)
    
    # Application metadata
    app_name: str = Field(default="CuliFeed", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_nested_delimiter": "__",
        "env_prefix": "CULIFEED_"
    }
    
    def validate_configuration(self) -> None:
        """Validate complete configuration."""
        errors = []
        
        # Validate AI provider setup
        primary_provider = self.processing.ai_provider
        if not self.ai.validate_provider_key(primary_provider):
            errors.append(f"Missing API key for primary AI provider: {primary_provider}")
        
        # Validate database path
        try:
            db_path = Path(self.database.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Invalid database path: {e}")
        
        # Validate log path if specified
        if self.logging.file_path:
            try:
                log_path = Path(self.logging.file_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Invalid log file path: {e}")
        
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}",
                error_code=ErrorCode.CONFIG_INVALID
            )
    
    def get_ai_fallback_providers(self) -> List[AIProvider]:
        """Get list of available fallback AI providers."""
        providers = []
        
        # Always try primary provider first
        if self.ai.validate_provider_key(self.processing.ai_provider):
            providers.append(self.processing.ai_provider)
        
        # Add other available providers as fallbacks
        for provider in AIProvider:
            if provider != self.processing.ai_provider and self.ai.validate_provider_key(provider):
                providers.append(provider)
        
        return providers
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and os.getenv("ENV", "development").lower() == "production"
    
    def get_effective_log_level(self) -> str:
        """Get effective log level considering debug mode."""
        if self.debug:
            return "DEBUG"
        return self.logging.level.value


def load_settings(config_path: Optional[str] = None) -> CuliFeedSettings:
    """Load settings from YAML file and environment variables.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Loaded and validated settings
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Load environment variables from .env file first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Find config file
    if config_path:
        config_file = Path(config_path)
    else:
        # Look for config file in standard locations
        possible_paths = [
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path("culifeed/config.yaml"),
        ]
        
        config_file = None
        for path in possible_paths:
            if path.exists():
                config_file = path
                break
    
    # Load YAML configuration
    yaml_data = {}
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
                
            # Substitute environment variables in YAML content
            yaml_content = _substitute_env_vars(yaml_content)
            
            yaml_data = yaml.safe_load(yaml_content) or {}
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config file {config_file}: {e}",
                error_code=ErrorCode.CONFIG_PARSE_ERROR
            )
    
    # Override with environment variables and create settings
    try:
        # Merge YAML data with environment variables
        settings = CuliFeedSettings(**yaml_data)
        
        # Validate the complete configuration
        settings.validate_configuration()
        
        return settings
        
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Failed to initialize settings: {e}",
            error_code=ErrorCode.CONFIG_INVALID
        )


def _substitute_env_vars(yaml_content: str) -> str:
    """Substitute environment variables in YAML content.
    
    Args:
        yaml_content: YAML content with ${VAR_NAME} placeholders
        
    Returns:
        YAML content with environment variables substituted
    """
    import re
    import os
    
    def replacer(match):
        var_name = match.group(1)
        env_value = os.getenv(var_name)
        if env_value is None:
            # Keep the placeholder if environment variable is not set
            return match.group(0)
        return env_value
    
    # Replace ${VAR_NAME} patterns with environment variable values
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replacer, yaml_content)


def create_example_config() -> str:
    """Create example configuration file content.
    
    Returns:
        YAML configuration template
    """
    return """# CuliFeed Configuration
# Edit this file to customize your CuliFeed installation

# User Settings
user:
  timezone: "UTC"
  admin_user_id: "${TELEGRAM_ADMIN_ID}"  # Optional: for admin commands

# Processing Settings
processing:
  daily_run_hour: 8                    # Hour of day to run processing (0-23)
  ai_provider: "gemini"                # Primary AI provider: gemini, groq, openai
  max_articles_per_topic: 5            # Maximum articles per topic per day
  batch_size: 10                       # Articles to process in one AI request
  parallel_feeds: 5                    # Concurrent RSS feed fetches
  max_content_length: 2000             # Max content length for AI analysis

# Cost Controls and Limits
limits:
  max_daily_api_calls: 950             # Stay under Gemini 1000 RPD free tier
  fallback_to_groq: true               # Use Groq when primary API exhausted
  fallback_to_keywords: true           # Use keyword-only when all APIs exhausted
  enable_usage_alerts: true            # Monitor free tier usage
  alert_threshold: 0.8                 # Alert at 80% of limits
  max_feed_errors: 10                  # Max errors before disabling feed
  request_timeout: 30                  # Request timeout in seconds

# Database Settings
database:
  path: "data/culifeed.db"             # SQLite database file location
  pool_size: 5                         # Connection pool size
  cleanup_days: 7                      # Days to keep old articles
  auto_vacuum: true                    # Automatic database maintenance
  backup_enabled: true                 # Enable automatic backups

# Logging Configuration
logging:
  level: "INFO"                        # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file_path: "logs/culifeed.log"       # Log file location
  max_file_size_mb: 10                 # Max size before rotation
  backup_count: 5                      # Number of backup files
  structured_logging: false            # Use JSON structured logging
  console_logging: true                # Enable console output

# Telegram Bot Settings (Environment variables required)
telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"   # Required: Get from @BotFather
  admin_user_id: "${TELEGRAM_ADMIN_ID}" # Optional: Admin user ID
  webhook_url: null                    # Optional: Webhook URL for updates
  max_retries: 3                       # Max retries for failed messages

# AI Provider Settings (Environment variables required)
ai:
  gemini_api_key: "${GEMINI_API_KEY}"  # Google Gemini API key (recommended)
  groq_api_key: "${GROQ_API_KEY}"      # Groq API key (fallback)
  openai_api_key: "${OPENAI_API_KEY}"  # OpenAI API key (optional)
  
  # Model Configuration
  gemini_model: "gemini-2.5-flash-lite"  # Gemini model
  groq_model: "llama3-8b-8192"           # Groq model
  openai_model: "gpt-4o-mini"            # OpenAI model
  
  temperature: 0.1                       # AI temperature (0.0-2.0)
  max_tokens: 500                        # Maximum tokens per response

# Application Settings
app_name: "CuliFeed"
version: "1.0.0"
debug: false
"""


# Global settings instance
_settings: Optional[CuliFeedSettings] = None


def get_settings(reload: bool = False) -> CuliFeedSettings:
    """Get global settings instance (singleton pattern).
    
    Args:
        reload: Force reload of settings
        
    Returns:
        Global settings instance
    """
    global _settings
    
    if _settings is None or reload:
        _settings = load_settings()
    
    return _settings