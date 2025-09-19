"""
Telegram Bot Conflict Detection
===============================

Utilities to detect if a Telegram bot token is already in use by another instance.
"""

import asyncio
import logging
from typing import Optional

from telegram import Bot
from telegram.error import Conflict, InvalidToken, NetworkError

logger = logging.getLogger(__name__)


async def check_telegram_bot_conflict(bot_token: str) -> tuple[bool, Optional[str]]:
    """
    Check if a Telegram bot token is already in use by another instance.
    
    Args:
        bot_token: Telegram bot token to check
        
    Returns:
        Tuple of (is_available, error_message)
        - is_available: True if token is available for use
        - error_message: Error description if not available
    """
    if not bot_token or bot_token.startswith("${"):
        return False, "Bot token is not configured"
    
    try:
        # Create a bot instance
        bot = Bot(token=bot_token)
        
        # Try to get bot info - this will fail if another instance is polling
        await bot.get_me()
        
        # Try to get updates (this is where conflicts are detected)
        await bot.get_updates(limit=1, timeout=1)
        
        return True, None
        
    except Conflict as e:
        return False, str(e)
    except InvalidToken:
        return False, "Invalid bot token"
    except NetworkError as e:
        return False, f"Network error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"
    finally:
        # Always close the bot session
        try:
            await bot.close()
        except:
            pass


def check_bot_availability_sync(bot_token: str) -> tuple[bool, Optional[str]]:
    """
    Synchronous wrapper for checking bot availability.
    
    Args:
        bot_token: Telegram bot token to check
        
    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        return asyncio.run(check_telegram_bot_conflict(bot_token))
    except Exception as e:
        return False, f"Error checking bot availability: {e}"


def handle_telegram_conflict(error_message: str) -> None:
    """
    Handle Telegram bot conflict with user-friendly messaging.
    
    Args:
        error_message: Error message from conflict detection
    """
    print("❌ Telegram Bot Conflict Detected!")
    print("━" * 60)
    print("Another instance is already using this Telegram bot token.")
    print()
    print("Possible causes:")
    print("  🐳 Docker container running the bot")
    print("  🖥️  Another terminal/process running the bot")
    print("  ⚙️  Systemd service running the bot")
    print("  ☁️  Bot running on another server")
    print()
    print("To resolve:")
    print("  1. Stop Docker containers: docker ps | grep culifeed")
    print("     Then: docker stop <container_id>")
    print("  2. Check running processes: ps aux | grep culifeed")
    print("     Then: kill <process_id>")
    print("  3. Check systemd services: systemctl status culifeed*")
    print("     Then: systemctl stop culifeed-bot")
    print()
    print(f"Technical details: {error_message}")
    print("━" * 60)
    print("Only ONE instance can use the same Telegram bot token at a time.")