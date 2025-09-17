"""Channel repository for database operations."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from ..database.connection import DatabaseConnection
from ..database.models import Channel
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import DatabaseError, ErrorCode


class ChannelRepository:
    """Repository for Channel CRUD operations with database abstraction."""
    
    def __init__(self, db_connection: DatabaseConnection):
        """Initialize repository with database connection.
        
        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.logger = get_logger_for_component('channel_repository')
    
    def get_all_active_channels(self) -> List[Dict[str, Any]]:
        """Get all active channels.
        
        Returns:
            List of channel dictionaries
        """
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM channels 
                    WHERE active = ? 
                    ORDER BY created_at
                """, (True,)).fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting active channels: {e}")
            raise DatabaseError(
                message="Failed to get active channels",
                error_code=ErrorCode.DATABASE_QUERY_ERROR,
                context={'operation': 'get_all_active_channels'}
            ) from e
    
    def get_channel_by_id(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get channel by chat ID.
        
        Args:
            chat_id: Channel chat ID
            
        Returns:
            Channel dictionary or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                row = conn.execute("""
                    SELECT * FROM channels WHERE chat_id = ?
                """, (chat_id,)).fetchone()
                
                return dict(row) if row else None
                
        except Exception as e:
            self.logger.error(f"Error getting channel {chat_id}: {e}")
            raise DatabaseError(
                message=f"Failed to get channel {chat_id}",
                error_code=ErrorCode.DATABASE_QUERY_ERROR,
                context={'chat_id': chat_id, 'operation': 'get_channel_by_id'}
            ) from e
    
    def create_channel(self, chat_id: str, chat_title: str, **kwargs) -> bool:
        """Create a new channel.
        
        Args:
            chat_id: Channel chat ID
            chat_title: Channel title
            **kwargs: Additional channel properties
            
        Returns:
            True if created successfully
        """
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO channels (
                        chat_id, chat_title, active, created_at
                    ) VALUES (?, ?, ?, ?)
                """, (
                    chat_id,
                    chat_title,
                    kwargs.get('active', True),
                    kwargs.get('created_at', datetime.now(timezone.utc))
                ))
                conn.commit()
                
                self.logger.info(f"Created channel {chat_id}: {chat_title}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating channel {chat_id}: {e}")
            raise DatabaseError(
                message=f"Failed to create channel {chat_id}",
                error_code=ErrorCode.DATABASE_INSERT_ERROR,
                context={'chat_id': chat_id, 'chat_title': chat_title}
            ) from e
    
    def update_last_delivery(self, chat_id: str) -> bool:
        """Update last delivery time for a channel.
        
        Args:
            chat_id: Channel chat ID
            
        Returns:
            True if updated successfully
        """
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    UPDATE channels 
                    SET last_delivery_at = ?
                    WHERE chat_id = ?
                """, (datetime.now(timezone.utc), chat_id))
                conn.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating last delivery for {chat_id}: {e}")
            raise DatabaseError(
                message=f"Failed to update last delivery for {chat_id}",
                error_code=ErrorCode.DATABASE_UPDATE_ERROR,
                context={'chat_id': chat_id}
            ) from e