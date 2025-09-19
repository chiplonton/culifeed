"""
Process Lock Utilities
======================

Utilities for preventing multiple instances of the same service from running.
"""

import os
import sys
import fcntl
import logging
import hashlib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ProcessLock:
    """File-based process lock to prevent multiple instances."""
    
    def __init__(self, lock_name: str, lock_dir: Optional[str] = None):
        """
        Initialize process lock.
        
        Args:
            lock_name: Name for the lock file (without extension)
            lock_dir: Directory for lock files (defaults to /tmp or system temp)
        """
        if lock_dir is None:
            lock_dir = "/tmp" if os.name == "posix" else os.environ.get("TEMP", ".")
        
        self.lock_file = Path(lock_dir) / f"{lock_name}.lock"
        self.lock_fd: Optional[int] = None
        self.acquired = False
    
    def acquire(self) -> bool:
        """
        Acquire the lock.
        
        Returns:
            True if lock was acquired successfully, False if already locked
        """
        try:
            # Create lock file if it doesn't exist
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open lock file
            self.lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write PID to lock file
            os.write(self.lock_fd, f"{os.getpid()}\n".encode())
            os.fsync(self.lock_fd)
            
            self.acquired = True
            logger.info(f"Process lock acquired: {self.lock_file}")
            return True
            
        except (OSError, IOError) as e:
            # Lock is already held by another process
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                except:
                    pass
                self.lock_fd = None
            
            # Try to read PID from existing lock file
            existing_pid = self._get_lock_holder_pid()
            if existing_pid:
                logger.warning(f"Process lock already held by PID {existing_pid}: {self.lock_file}")
            else:
                logger.warning(f"Process lock unavailable: {self.lock_file}")
            
            return False
    
    def release(self) -> None:
        """Release the lock."""
        if self.lock_fd is not None and self.acquired:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                self.lock_file.unlink(missing_ok=True)
                logger.info(f"Process lock released: {self.lock_file}")
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")
            finally:
                self.lock_fd = None
                self.acquired = False
    
    def _get_lock_holder_pid(self) -> Optional[int]:
        """Get PID of the process holding the lock."""
        try:
            if self.lock_file.exists():
                content = self.lock_file.read_text().strip()
                return int(content)
        except (ValueError, OSError):
            pass
        return None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError(f"Could not acquire process lock: {self.lock_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def ensure_single_telegram_bot(bot_token: str) -> ProcessLock:
    """
    Ensure only one Telegram bot instance is running with the same token.
    
    Args:
        bot_token: Telegram bot token to create unique lock for
    
    Returns:
        ProcessLock instance if successful
    
    Raises:
        SystemExit: If another bot instance with same token is running
    """
    # Create a hash of the bot token for the lock name (for security)
    token_hash = hashlib.sha256(bot_token.encode()).hexdigest()[:16]
    lock_name = f"culifeed-bot-{token_hash}"
    
    lock = ProcessLock(lock_name)
    
    if not lock.acquire():
        existing_pid = lock._get_lock_holder_pid()
        
        print("❌ Telegram Bot Conflict Detected!")
        print("━" * 50)
        print("Another CuliFeed bot instance is already running with the same Telegram bot token.")
        print()
        print("This could be:")
        print("  • Another terminal/process running the bot")
        print("  • A Docker container running the bot")
        print("  • A systemd service running the bot")
        print()
        
        if existing_pid:
            print(f"Conflicting process PID: {existing_pid}")
            print(f"To stop the local process: kill {existing_pid}")
        
        print("To stop Docker containers: docker ps | grep culifeed")
        print("Then: docker stop <container_id>")
        print()
        print("Only one bot instance can use the same Telegram token at a time.")
        print("━" * 50)
        
        sys.exit(1)
    
    return lock


def ensure_single_instance(service_name: str) -> ProcessLock:
    """
    Ensure only one instance of a service is running.
    
    Args:
        service_name: Name of the service (e.g., 'culifeed-bot')
    
    Returns:
        ProcessLock instance if successful
    
    Raises:
        SystemExit: If another instance is already running
    """
    lock = ProcessLock(f"culifeed-{service_name}")
    
    if not lock.acquire():
        existing_pid = lock._get_lock_holder_pid()
        if existing_pid:
            print(f"❌ Another {service_name} instance is already running (PID: {existing_pid})")
            print(f"   To stop it: kill {existing_pid}")
        else:
            print(f"❌ Another {service_name} instance is already running")
        
        sys.exit(1)
    
    return lock