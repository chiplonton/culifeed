#!/usr/bin/env python3
"""
CuliFeed Bot Runner
==================

Main entry point for running the CuliFeed Telegram bot service.
Handles initialization, startup, and graceful shutdown.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from culifeed.bot.telegram_bot import TelegramBotService
from culifeed.config.settings import get_settings
from culifeed.utils.logging import setup_logger
from culifeed.utils.telegram_conflict_detector import check_bot_availability_sync, handle_telegram_conflict


def main():
    """Main entry point for the bot service."""
    # Load settings first to get bot token
    settings = get_settings()

    # Check if bot token is available (detect conflicts early)
    print("🔍 Checking Telegram bot availability...")
    is_available, error_msg = check_bot_availability_sync(settings.telegram.bot_token)

    if not is_available:
        handle_telegram_conflict(error_msg)
        sys.exit(1)

    print("✅ Telegram bot token is available")

    # Setup logging
    setup_logger(
        name="culifeed.bot_runner",
        level=settings.logging.level.value,
        log_file=settings.logging.file_path,
        console=settings.logging.console_logging
    )

    logger = logging.getLogger('culifeed.bot_runner')
    logger.info("Starting CuliFeed Telegram Bot...")

    # Validate required settings
    if not settings.telegram.bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is required")
        sys.exit(1)

    try:
        # Create bot service
        bot_service = TelegramBotService()
        
        # Initialize and start the bot using run_polling which handles everything
        logger.info("Bot initialized successfully. Starting polling...")
        print("🤖 CuliFeed Bot is running! Press Ctrl+C to stop.")
        
        # This will block until interrupted
        bot_service.run()
        
    except KeyboardInterrupt:
        print("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Failed to start bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()