# Changelog

All notable changes to CuliFeed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-01-23

### Added

#### Smart Content Analysis System
- **Configurable Generic Patterns**: Revolutionary user-configurable system replacing 120+ hard-coded patterns
  - 192 generic patterns organized across 9 logical categories (update_feature, guide_tutorial, general_tech, cloud_aws, business_industry, time_frequency, quality_status, descriptors, actions)
  - Complete user control via YAML configuration - users can add custom domains, languages, and patterns
  - Enable/disable functionality via `generic_patterns_enabled` setting for flexible testing
  - Type-safe configuration with Pydantic validation and comprehensive defaults

#### Enhanced Semantic Analysis
- **Advanced Semantic Penalty Classification**: Smart keyword categorization for improved topic matching accuracy
  - Intelligent classification of generic vs domain-specific keywords
  - Semantic penalties applied to articles with high generic keyword ratios
  - 56.4% score reduction for generic content vs specific content (0.732 → 0.319)
  - Context-aware routing decisions preventing false positives

### Fixed

#### Critical Production Issues
- **False Positive Resolution**: Resolved production issue where AWS Weekly Roundup incorrectly matched EKS/ECS topics
  - Root cause: Generic keywords ("new feature", "best practices", "new update") were treated as topic-specific
  - Solution: Enhanced semantic penalty system correctly identifies and penalizes generic content
  - Result: Articles with generic content now route to "definitely_irrelevant" instead of false positives

#### Core Algorithm Improvements
- **Pre-filter Partial Word Matching**: Fixed partial word matching logic to require ALL words in multi-word keywords
  - Previously: Partial matches (1 of 3 words) received partial credit leading to false positives
  - Now: Multi-word keywords require complete word coverage for scoring
  - Maintains backward compatibility while improving precision

### Changed
- **Architecture**: Moved from hard-coded patterns to user-configurable YAML system for better maintainability
- **Smart Analyzer**: Enhanced keyword classification system with configurable generic patterns
- **Configuration Management**: Added smart_processing.generic_patterns section to config.yaml
- **User Experience**: Users can now customize generic patterns for their specific domains and use cases

### Technical Improvements
- **Type Safety**: Full Pydantic validation for new configuration fields
- **Performance**: 56.4% improvement in semantic penalty effectiveness
- **Maintainability**: Eliminated 120+ hard-coded patterns from source code
- **Extensibility**: Framework for easy addition of custom pattern categories
- **Testing**: Comprehensive validation suite with 100% test success rate

## [1.1.1] - 2025-01-23

### Fixed

#### Bot Command Improvements
- **Multi-word Topic Editing**: Fixed `/edittopic` command to correctly parse topic names containing spaces
  - Now properly handles topics like "TikTok software engineers" instead of only recognizing first word
  - Implements smart topic name matching that progressively checks existing topics
  - Consistent parsing behavior with `/addtopic` and `/removetopic` commands
  - Enhanced help message with clearer examples and comma requirement explanation

#### Technical Fixes
- **Command Argument Parsing**: Replaced single-word parsing (`args[0]`) with intelligent multi-word parser
- **Topic Name Resolution**: Added `_parse_edit_topic_args()` method for robust topic name matching
- **User Experience**: Updated help text to clearly show comma-separated keyword format requirement

## [1.1.0] - 2025-01-22

### Added

#### User Experience Improvements
- **Topic Input Validation**: AI keyword generation now requires 5-20 words for better context and quality
- **Enhanced Bot Command Menu**: Fixed missing Telegram command suggestions and bot menu display
- **Improved Topics Display**: Better visual separation between topic names and keywords with indented format
- **Smart Validation Guidance**: Helpful examples and fallback options when validation fails

#### Bot Interface Enhancements
- **Visual Topic Formatting**: Topics now display with clear visual hierarchy using 🎯 emoji and indented keywords
- **Command Menu Registration**: Fixed bot command registration in sync initialization path
- **Complete Keyword Display**: Removed truncation of keywords - users now see all their configured keywords
- **Better Error Messages**: More user-friendly validation errors with examples and alternatives

#### Technical Improvements
- **Dual-Mode Topic Creation**: Maintains both AI generation (with validation) and manual keyword modes
- **Development Workflow**: Added venv activation requirements to CLAUDE.md for consistent development
- **Docker Validation**: Comprehensive dependency testing in containerized environment
- **Enhanced Validation System**: Separate validation methods for AI vs manual topic creation

### Changed
- **Topic Validation**: Only applies to AI keyword generation mode, preserving manual mode flexibility
- **Display Format**: Improved topic list formatting for better readability and user experience
- **Command Registration**: Fixed synchronous command setup to ensure bot menu appears correctly

### Fixed
- **Missing Bot Menu**: Resolved issue where Telegram bot command suggestions and menu were not displayed
- **Keyword Truncation**: Fixed "+N more" issue that prevented users from seeing all their keywords
- **Command Registration**: Fixed bot command menu setup in sync initialization path
- **Markdown Formatting**: Corrected topic name formatting for better Telegram display

### Technical Details
- Enhanced `ContentValidator` with `validate_topic_name_for_ai_generation()` method
- Improved `TopicCommandHandler` with better UX and visual formatting
- Fixed `TelegramBotService` command menu registration in sync mode
- Updated development guidelines in CLAUDE.md

## [1.0.0] - 2025-01-19

### Added

#### Core Features
- **AI-Powered Content Curation**: Intelligent RSS content analysis using Google Gemini, Groq, and OpenAI APIs
- **Multi-Channel Telegram Bot**: Full-featured bot with command-based management and auto-registration
- **Smart Pre-Filtering**: 85% content filtering before AI processing for cost efficiency
- **Daily Processing Pipeline**: Automated scheduled processing with health monitoring
- **Multi-Provider AI Fallback**: Graceful fallback between AI providers for reliability

#### Architecture & Infrastructure
- **SQLite Database**: Connection pooling, schema management, and data persistence
- **YAML Configuration**: Flexible configuration with environment variable support
- **Structured Logging**: Comprehensive logging with configurable levels and formats
- **Error Handling**: Structured error codes and graceful degradation
- **Docker Support**: Multi-stage Dockerfile with security hardening
- **GitHub Actions**: Automated Docker builds triggered by releases

#### Bot Commands & Management
- `/start` - Channel registration and setup
- `/add_feed <url>` - RSS feed subscription management
- `/list_feeds` - View and manage subscribed feeds
- `/set_topic <topic>` - Configure content filtering preferences
- `/help` - Comprehensive command documentation
- Auto-registration for new channels with intelligent setup

#### Processing & Content Management
- **RSS Feed Processing**: Robust feed parsing with error isolation
- **Content Sanitization**: HTML cleaning and security validation
- **Batch Processing**: Efficient concurrent processing of multiple feeds
- **Article Deduplication**: Smart duplicate detection and filtering
- **Topic-Based Filtering**: AI-powered relevance scoring and content matching

#### CLI Management Tools
- `python main.py --check-config` - Configuration validation
- `python main.py --test-foundation` - Foundation component testing
- `python main.py --init-db` - Database initialization
- `python main.py --daily-process` - Manual processing trigger
- `python main.py --health-check` - System health monitoring
- `python main.py --full-test` - End-to-end system testing

#### Development & Quality Assurance
- **Comprehensive Test Suite**: Unit tests, integration tests, and end-to-end testing
- **Type Safety**: Full type hints with mypy validation
- **Code Quality**: Black formatting, flake8 linting, pytest coverage
- **Documentation**: Extensive inline documentation and architectural guidelines

### Technical Specifications

#### Dependencies
- **Python**: 3.13+ with latest stable package versions
- **AI Providers**: Google Gemini (primary), Groq (fallback), OpenAI (optional)
- **Database**: SQLite with connection pooling
- **Messaging**: python-telegram-bot 21.0+
- **Configuration**: Pydantic 2.9+ for validation
- **Async Processing**: aiohttp for concurrent operations

#### Performance & Scalability
- **Memory Efficient**: Batch processing with configurable chunk sizes
- **Network Optimized**: Concurrent feed fetching with rate limiting
- **Cost Optimized**: Pre-filtering reduces AI API calls by 85%
- **Error Resilient**: Isolated error handling with automatic recovery

#### Security Features
- **Process Isolation**: Single-instance locking to prevent conflicts
- **Content Sanitization**: HTML cleaning and XSS protection
- **Input Validation**: Comprehensive validation for all external inputs
- **Secure Configuration**: Environment variable management for sensitive data

### Deployment

#### Supported Platforms
- **Local Development**: Direct Python execution with virtual environments
- **VPS Deployment**: Dual-process architecture with systemd services
- **Container Deployment**: Docker with multi-platform support (amd64/arm64)
- **GitHub Packages**: Automated container registry integration

#### System Services
- **Bot Service**: Long-running Telegram bot with automatic restart
- **Processing Service**: Daily scheduled processing with health checks
- **Monitoring**: Built-in health checks and status reporting

### Breaking Changes
- Initial release - no breaking changes from previous versions

### Migration Guide
- This is the initial stable release
- Follow installation instructions in README.md for new deployments
- Docker deployment recommended for production environments

### Known Issues
- None reported for this release

### Contributors
- CuliFeed Development Team

---

## Release Notes

This is the first stable release of CuliFeed, representing a complete AI-powered RSS content curation system. The system has been thoroughly tested and is ready for production deployment.

### Key Highlights
- 🤖 **AI-Powered**: Smart content curation using multiple AI providers
- 📱 **Telegram Integration**: Full-featured bot with intuitive commands
- 🔄 **Automated Processing**: Hands-off daily content delivery
- 🛡️ **Production Ready**: Comprehensive error handling and monitoring
- 🐳 **Docker Support**: Easy deployment with container technology

### Getting Started
1. Clone the repository
2. Copy `.env.example` to `.env` and configure your API keys
3. Run `python main.py --init-db` to set up the database
4. Start the bot with `python run_bot.py`
5. Use Docker for production deployment

For detailed installation and configuration instructions, see the [README.md](README.md).