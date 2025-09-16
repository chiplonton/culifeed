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


## Debug Logging & Troubleshooting

### Enable Debug Logging

Debug logging is essential for troubleshooting bot issues and monitoring system behavior.

#### Method 1: Environment Variables (Recommended)

```bash
# Enable debug logging in .env file
CULIFEED_LOGGING__LEVEL=DEBUG
CULIFEED_LOGGING__CONSOLE_LOGGING=true
CULIFEED_DEBUG=true
```

#### Method 2: Runtime Override

```bash
# Override logging level when starting the bot
CULIFEED_LOGGING__LEVEL=DEBUG python run_bot.py

# Enable all debug features
CULIFEED_DEBUG=true CULIFEED_LOGGING__LEVEL=DEBUG python run_bot.py

# Use the dedicated debug runner (recommended)
python debug_bot.py
```

#### Method 3: Python Logging Override

```bash
# Set Python's root logger to debug
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import run_bot
run_bot.main()
"
```

### Log Levels Explained

| Level | Description | Use Case |
|-------|-------------|----------|
| `DEBUG` | Detailed diagnostic info | Development, troubleshooting |
| `INFO` | General operational messages | Production monitoring |
| `WARNING` | Warning messages | Production issues |
| `ERROR` | Error messages only | Critical issues only |

### Debug Output Examples

#### Normal Operation (INFO level)
```
[INFO] culifeed.bot_runner:main:37 - Starting CuliFeed Telegram Bot...
[INFO] culifeed.bot_runner:main:49 - Bot initialized successfully. Starting polling...
🤖 CuliFeed Bot is running! Press Ctrl+C to stop.
```

#### Debug Mode (DEBUG level)
```bash
# Start with debug logging
CULIFEED_LOGGING__LEVEL=DEBUG python run_bot.py
```

Expected debug output:
```
[DEBUG] culifeed.config.settings - Loading configuration from environment
[DEBUG] culifeed.database.connection - Connecting to database: data/culifeed.db
[DEBUG] culifeed.bot.telegram_bot - Initializing Telegram bot service
[DEBUG] culifeed.bot.telegram_bot - Creating bot application with token: ***
[INFO]  culifeed.bot_runner:main:37 - Starting CuliFeed Telegram Bot...
[DEBUG] culifeed.bot.telegram_bot - Registering command handlers
[DEBUG] culifeed.bot.telegram_bot - Added handler: start
[DEBUG] culifeed.bot.telegram_bot - Added handler: help
[DEBUG] culifeed.bot.telegram_bot - All command handlers registered
[INFO]  culifeed.bot_runner:main:49 - Bot initialized successfully. Starting polling...
🤖 CuliFeed Bot is running! Press Ctrl+C to stop.
```

### Component-Specific Debug Logging

Enable debug logging for specific components:

```bash
# Database operations
CULIFEED_LOGGING__LEVEL=DEBUG python -c "
import logging
logging.getLogger('culifeed.database').setLevel(logging.DEBUG)
# ... run your code
"

# Bot message handling
CULIFEED_LOGGING__LEVEL=DEBUG python -c "
import logging
logging.getLogger('culifeed.bot').setLevel(logging.DEBUG)
# ... run your code
"

# AI provider calls
CULIFEED_LOGGING__LEVEL=DEBUG python -c "
import logging
logging.getLogger('culifeed.ai').setLevel(logging.DEBUG)
# ... run your code
"
```

### Common Troubleshooting Scenarios

#### 1. Bot Not Starting

```bash
# Check configuration
python main.py check-config

# Start with maximum debug info
CULIFEED_LOGGING__LEVEL=DEBUG CULIFEED_DEBUG=true python run_bot.py
```

Common issues:
- Invalid bot token: `TELEGRAM_BOT_TOKEN` not set or incorrect
- Database issues: Check `CULIFEED_DATABASE__PATH` permissions
- Missing dependencies: Run `pip install -r requirements.txt`

#### 2. Bot Not Responding to Commands

```bash
# Enable telegram-bot library debug logging
CULIFEED_LOGGING__LEVEL=DEBUG python -c "
import logging
logging.getLogger('telegram').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.INFO)  # HTTP requests
exec(open('run_bot.py').read())
"
```

Check for:
- Bot not added to channel as admin
- Commands not registered properly
- Network connectivity issues

#### 3. Database Connection Issues

```bash
# Test database connection
python -c "
from culifeed.database.connection import get_db_manager
from culifeed.config.settings import get_settings
import logging
logging.basicConfig(level=logging.DEBUG)

settings = get_settings()
db = get_db_manager(settings.database.path)
with db.get_connection() as conn:
    result = conn.execute('SELECT COUNT(*) FROM channels').fetchone()
    print(f'Channels in database: {result[0]}')
"
```

#### 4. AI Provider Issues

```bash
# Test AI providers with debug logging
CULIFEED_LOGGING__LEVEL=DEBUG python -c "
from culifeed.ai.providers.gemini_provider import GeminiProvider
import logging
logging.basicConfig(level=logging.DEBUG)

# Test your configured provider
provider = GeminiProvider('your-api-key')
print('AI provider initialized successfully')
"
```

### Log File Output

Enable file logging for persistent debug information:

```bash
# Enable file logging
CULIFEED_LOGGING__FILE_PATH=logs/culifeed.log
CULIFEED_LOGGING__LEVEL=DEBUG
CULIFEED_LOGGING__CONSOLE_LOGGING=true

# Create logs directory
mkdir -p logs

# Start bot with file logging
python run_bot.py
```

Monitor logs in real-time:
```bash
# Follow log file
tail -f logs/culifeed.log

# Filter for specific components
tail -f logs/culifeed.log | grep "telegram_bot"

# Search for errors
tail -f logs/culifeed.log | grep "ERROR"
```

### Performance Debug Mode

Enable performance monitoring and detailed timing:

```bash
# Enable performance debugging
CULIFEED_DEBUG=true
CULIFEED_LOGGING__LEVEL=DEBUG
CULIFEED_PERFORMANCE__ENABLE_MONITORING=true

python run_bot.py
```

This will show:
- Request processing times
- Database query duration
- AI provider response times
- Memory usage statistics

### Debug Commands

Test individual components:

```bash
# Test bot initialization only
python -c "
from culifeed.bot.telegram_bot import TelegramBotService
import asyncio
import logging
logging.basicConfig(level=logging.DEBUG)

async def test():
    bot = TelegramBotService()
    print('Bot service created successfully')

asyncio.run(test())
"

# Test database initialization
python main.py init-db --verbose

# Test configuration loading
python main.py check-config --debug
```

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