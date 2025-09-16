#!/usr/bin/env python3
"""
CuliFeed Bot Debug Runner
========================

Enhanced bot runner with comprehensive debug logging and troubleshooting helpers.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_debug_logging():
    """Set up comprehensive debug logging."""
    # Set up root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Enable debug for CuliFeed components
    logging.getLogger('culifeed').setLevel(logging.DEBUG)

    # Enable debug for Telegram bot library (but limit HTTP noise)
    logging.getLogger('telegram').setLevel(logging.DEBUG)
    logging.getLogger('httpx').setLevel(logging.INFO)  # Reduce HTTP request noise
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    print("🔍 Debug logging enabled for all CuliFeed components")

def check_environment():
    """Check and display environment configuration."""
    print("\n📋 Environment Configuration:")

    required_vars = [
        'CULIFEED_TELEGRAM__BOT_TOKEN',
        'CULIFEED_LOGGING__LEVEL',
    ]

    optional_vars = [
        'CULIFEED_DEBUG',
        'CULIFEED_LOGGING__CONSOLE_LOGGING',
        'CULIFEED_LOGGING__FILE_PATH',
        'CULIFEED_USER__ADMIN_USER_ID',
        'CULIFEED_AI__GEMINI_API_KEY',
        'CULIFEED_AI__GROQ_API_KEY',
        'CULIFEED_AI__OPENAI_API_KEY',
    ]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive tokens
            if 'TOKEN' in var or 'KEY' in var:
                display_value = f"***{value[-4:]}" if len(value) > 4 else "***"
            else:
                display_value = value
            print(f"  ✅ {var}={display_value}")
        else:
            print(f"  ❌ {var}=NOT_SET")

    print(f"\n📋 Optional Configuration:")
    for var in optional_vars:
        value = os.getenv(var, "not_set")
        if 'KEY' in var or 'TOKEN' in var:
            display_value = f"***{value[-4:]}" if len(value) > 4 and value != "not_set" else value
        else:
            display_value = value
        print(f"  • {var}={display_value}")

def test_imports():
    """Test that all required modules can be imported."""
    print("\n🔧 Testing Module Imports:")

    try:
        from culifeed.bot.telegram_bot import TelegramBotService
        print("  ✅ TelegramBotService")
    except ImportError as e:
        print(f"  ❌ TelegramBotService: {e}")
        return False

    try:
        from culifeed.config.settings import get_settings
        settings = get_settings()
        print("  ✅ Settings configuration")
    except Exception as e:
        print(f"  ❌ Settings configuration: {e}")
        return False

    try:
        from culifeed.database.connection import get_db_manager
        db = get_db_manager(settings.database.path)
        print("  ✅ Database connection")
    except Exception as e:
        print(f"  ❌ Database connection: {e}")
        return False

    return True

def main():
    """Main debug runner."""
    print("🔍 CuliFeed Bot Debug Runner")
    print("=" * 40)

    # Override environment variables for debug mode
    os.environ['CULIFEED_LOGGING__LEVEL'] = 'DEBUG'
    os.environ['CULIFEED_LOGGING__CONSOLE_LOGGING'] = 'true'
    os.environ['CULIFEED_DEBUG'] = 'true'

    # Set up debug logging
    setup_debug_logging()

    # Check environment
    check_environment()

    # Test imports
    if not test_imports():
        print("\n❌ Module import tests failed. Check your installation.")
        sys.exit(1)

    print(f"\n🚀 Starting bot in debug mode...")
    print(f"   Press Ctrl+C to stop")
    print("=" * 40)

    try:
        # Import and run the main function
        from culifeed.bot.telegram_bot import TelegramBotService

        # Create and run bot service
        bot_service = TelegramBotService()
        print("🤖 Bot service created, starting polling...")
        bot_service.run()

    except KeyboardInterrupt:
        print("\n👋 Debug session stopped by user")
    except Exception as e:
        logging.exception("Fatal error in debug mode")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()