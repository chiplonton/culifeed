# CuliFeed - Smart Content Curation System

## Project Overview

CuliFeed is an AI-powered content curation system that eliminates manual content filtering by providing intelligent topic matching and automated summarization for RSS feeds.

### Problem Statement

Traditional RSS monitoring approaches often face challenges:
- **Basic keyword filtering**: Can be noisy (irrelevant matches) or too narrow (missed relevant content)
- **Manual time investment**: Users must read and filter content manually
- **Limited topic understanding**: Difficulty distinguishing between nuanced topics like "AWS Lambda performance optimization" vs "AWS Lambda cost analysis"

### Solution

CuliFeed provides a production-ready Python application with:
- **Semantic topic matching**: AI-powered content relevance assessment using Gemini/Groq/OpenAI
- **Confidence-based filtering**: Multi-stage pipeline with 85% pre-filtering + AI validation
- **Automated content processing**: Full RSS feed parsing, deduplication, and cleaning
- **Telegram integration**: Complete bot interface with auto-registration and daily digests
- **Robust architecture**: Service-oriented design with comprehensive error handling and testing

### Key Benefits

- ✅ **Zero manual filtering**: AI handles relevance assessment with multi-provider support
- ✅ **Cost-effective**: Optimized for free tiers (Gemini free tier + 85% pre-filtering)
- ✅ **Production ready**: Comprehensive test suite, error handling, and monitoring
- ✅ **Easy deployment**: Single Python service with SQLite database
- ✅ **Familiar interface**: Telegram bot with intuitive commands
- ✅ **Daily digest**: Organized by topics, delivered automatically
- ✅ **Extensible architecture**: Clean codebase with comprehensive documentation

## Quick Start

### Prerequisites
- Python 3.11+ installed
- Telegram Bot Token (from @BotFather)
- API keys for AI providers (Gemini/Groq/OpenAI)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd culifeed

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and bot token

# Initialize the database
python main.py init-db

# Start the bot service
python run_bot.py
```

### Configuration

1. **Set up your bot token** in `.env`:
   ```bash
   CULIFEED_TELEGRAM__BOT_TOKEN=your_bot_token_here
   ```

2. **Add AI provider keys** (at least one required):
   ```bash
   CULIFEED_AI__GEMINI_API_KEY=your_gemini_key
   CULIFEED_AI__GROQ_API_KEY=your_groq_key
   ```

3. **Add bot to your Telegram group/channel** (auto-registers)
4. **Configure topics**: `/add_topic "AWS Lambda performance"`
5. **Add feeds**: `/add_feed "https://example.com/feed.xml"`
6. **Receive daily digest automatically!**

## Project Structure

```
culifeed/
├── culifeed/                    # Main package
│   ├── ai/                      # AI provider integrations
│   ├── bot/                     # Telegram bot implementation
│   ├── config/                  # Configuration management
│   ├── database/                # Database models and connections
│   ├── delivery/                # Message formatting and delivery
│   ├── ingestion/               # RSS feed parsing and content cleaning
│   ├── processing/              # Content processing pipeline
│   ├── storage/                 # Repository pattern implementations
│   └── utils/                   # Shared utilities and exceptions
├── tests/                       # Comprehensive test suite
├── deployment/                  # Deployment configurations
├── docs/                        # Additional documentation
├── main.py                      # CLI interface
├── run_bot.py                   # Bot service runner
└── requirements.txt             # Python dependencies
```

## Documentation

- [CLAUDE.md](./CLAUDE.md) - Architectural guidelines and development practices
- [Deployment Guide](./deployment/) - Setup and configuration instructions
- [Test Documentation](./tests/CLAUDE.md) - Testing practices and patterns

## Configuration

CuliFeed uses environment variables with YAML configuration fallbacks. Key settings:

### Required Environment Variables

```bash
# Telegram Bot (required)
CULIFEED_TELEGRAM__BOT_TOKEN=your_bot_token_from_botfather

# Admin User (recommended)
CULIFEED_USER__ADMIN_USER_ID=your_telegram_user_id

# AI Provider (at least one required)
CULIFEED_AI__GEMINI_API_KEY=your_gemini_api_key
CULIFEED_AI__GROQ_API_KEY=your_groq_api_key
CULIFEED_AI__OPENAI_API_KEY=your_openai_api_key
```

### Optional Configuration

```bash
# Processing Limits
CULIFEED_LIMITS__MAX_DAILY_API_CALLS=950
CULIFEED_PROCESSING__BATCH_SIZE=10
CULIFEED_PROCESSING__PARALLEL_FEEDS=5

# Database
CULIFEED_DATABASE__PATH=data/culifeed.db
CULIFEED_DATABASE__POOL_SIZE=5

# Logging
CULIFEED_LOGGING__LEVEL=INFO
CULIFEED_LOGGING__CONSOLE_LOGGING=true
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=culifeed --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/ -v`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Available Commands

### CLI Commands (main.py)

```bash
python main.py --help                    # Show all available commands
python main.py check-config              # Validate configuration
python main.py init-db                   # Initialize database schema
python main.py test-foundation           # Test core components
python main.py cleanup                   # Clean up old data
```

### Bot Service

```bash
python run_bot.py                        # Start Telegram bot service
```

### Telegram Bot Commands

- `/start` - Initialize the bot and register channel
- `/help` - Show available commands
- `/add_topic <name> <keywords>` - Add content topic
- `/add_feed <url>` - Add RSS feed to monitor
- `/list_topics` - Show configured topics
- `/list_feeds` - Show monitored feeds
- `/status` - Show system status

## Project Status

✅ **Phase**: Fully Implemented and Operational
🚀 **Status**: Production ready with comprehensive testing suite
📊 **Features**: Complete RSS processing pipeline, AI-powered filtering, Telegram integration