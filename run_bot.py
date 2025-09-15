#!/usr/bin/env python3
"""
CuliFeed Bot Runner
==================

Main entry point for running the CuliFeed Telegram bot service.
Handles initialization, startup, and graceful shutdown.
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from culifeed.bot.telegram_bot import TelegramBotService
from culifeed.config.settings import get_settings
from culifeed.utils.logging import setup_logger


def setup_signal_handlers(bot_service: TelegramBotService):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        del frame  # Unused parameter
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        asyncio.create_task(shutdown(bot_service))

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def shutdown(bot_service: TelegramBotService):
    """Gracefully shutdown the bot service."""
    try:
        await bot_service.stop()
        print("Bot service stopped successfully")
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        # Cancel all running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    """Main entry point for the bot service."""
    logger = None
    bot_service = None

    try:
        # Load settings
        settings = get_settings()

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

        # Create and initialize bot service
        bot_service = TelegramBotService()
        await bot_service.initialize()

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(bot_service)

        logger.info("Bot initialized successfully. Starting polling...")
        print("🤖 CuliFeed Bot is running! Press Ctrl+C to stop.")

        # Start the bot (this will run indefinitely)
        await bot_service.start_polling()

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        if logger:
            logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        if bot_service:
            await shutdown(bot_service)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        sys.exit(1)