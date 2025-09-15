"""
Article Repository
==================

Repository pattern implementation for Article CRUD operations with proper
error handling and data access abstraction.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from ..database.models import Article
from ..database.connection import DatabaseConnection
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import DatabaseError, ErrorCode


class ArticleRepository:
    """Repository for Article CRUD operations with database abstraction."""
    
    def __init__(self, db_connection: DatabaseConnection):
        """Initialize article repository.
        
        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.logger = get_logger_for_component("article_repository")
        self.settings = get_settings()
    
    def create_article(self, article: Article) -> str:
        """Create a new article.
        
        Args:
            article: Article model to create
            
        Returns:
            Created article ID
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO articles (id, title, url, content, published_at, 
                                        source_feed, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        article.id, article.title, str(article.url), article.content,
                        article.published_at, article.source_feed, article.content_hash,
                        article.created_at
                    )
                )
                conn.commit()
                
            self.logger.debug(f"Created article: {article.id}")
            return article.id
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to create article: {e}",
                error_code=ErrorCode.DATABASE_ERROR
            ) from e
    
    def create_articles_batch(self, articles: List[Article]) -> int:
        """Create multiple articles in a single transaction.
        
        Args:
            articles: List of Article models to create
            
        Returns:
            Number of articles created
            
        Raises:
            DatabaseError: If batch creation fails
        """
        if not articles:
            return 0
        
        try:
            with self.db.transaction() as conn:
                article_data = [
                    (
                        article.id, article.title, str(article.url), article.content,
                        article.published_at, article.source_feed, article.content_hash,
                        article.created_at
                    )
                    for article in articles
                ]
                
                cursor = conn.executemany(
                    """
                    INSERT OR REPLACE INTO articles 
                    (id, title, url, content, published_at, source_feed, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    article_data
                )
                
                created_count = cursor.rowcount
                
            self.logger.info(f"Batch created {created_count} articles")
            return created_count
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to batch create articles: {e}",
                error_code=ErrorCode.DATABASE_TRANSACTION
            ) from e
    
    def get_article(self, article_id: str) -> Optional[Article]:
        """Get article by ID.
        
        Args:
            article_id: Article ID to retrieve
            
        Returns:
            Article model or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM articles WHERE id = ?", 
                    (article_id,)
                ).fetchone()
                
                if row:
                    return Article(**dict(row))
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get article {article_id}: {e}")
            return None
    
    def get_articles_by_feed(self, feed_url: str, limit: int = 100) -> List[Article]:
        """Get articles from a specific feed.
        
        Args:
            feed_url: Source feed URL
            limit: Maximum number of articles to return
            
        Returns:
            List of Article models
        """
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM articles 
                    WHERE source_feed = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                    """,
                    (feed_url, limit)
                ).fetchall()
                
                return [Article(**dict(row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get articles for feed {feed_url}: {e}")
            return []
    
    def get_recent_articles(self, hours: int = 24, limit: int = 100) -> List[Article]:
        """Get recent articles within specified time window.
        
        Args:
            hours: Hours back to search
            limit: Maximum number of articles to return
            
        Returns:
            List of recent Article models
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM articles 
                    WHERE created_at >= ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                    """,
                    (cutoff_time, limit)
                ).fetchall()
                
                return [Article(**dict(row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get recent articles: {e}")
            return []
    
    def find_by_content_hash(self, content_hash: str) -> Optional[Article]:
        """Find article by content hash for deduplication.
        
        Args:
            content_hash: Content hash to search for
            
        Returns:
            Article model or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM articles WHERE content_hash = ?",
                    (content_hash,)
                ).fetchone()
                
                if row:
                    return Article(**dict(row))
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to find article by hash: {e}")
            return None
    
    def check_duplicates(self, articles: List[Article]) -> List[Article]:
        """Filter out duplicate articles based on content hash.
        
        Args:
            articles: List of articles to check
            
        Returns:
            List of unique articles (no duplicates)
        """
        if not articles:
            return []
        
        try:
            # Get all content hashes from input articles
            content_hashes = [article.content_hash for article in articles]
            placeholders = ','.join('?' * len(content_hashes))
            
            with self.db.get_connection() as conn:
                existing_hashes = conn.execute(
                    f"SELECT content_hash FROM articles WHERE content_hash IN ({placeholders})",
                    content_hashes
                ).fetchall()
                
            existing_hash_set = {row['content_hash'] for row in existing_hashes}
            
            # Filter out articles with existing hashes
            unique_articles = [
                article for article in articles 
                if article.content_hash not in existing_hash_set
            ]
            
            self.logger.debug(
                f"Deduplication: {len(articles)} -> {len(unique_articles)} "
                f"({len(articles) - len(unique_articles)} duplicates removed)"
            )
            
            return unique_articles
            
        except Exception as e:
            self.logger.error(f"Failed to check duplicates: {e}")
            return articles  # Return all if check fails
    
    def update_article(self, article_id: str, updates: Dict[str, Any]) -> bool:
        """Update article fields.
        
        Args:
            article_id: Article ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        if not updates:
            return True
        
        try:
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for field, value in updates.items():
                if field in ['title', 'url', 'content', 'published_at', 'source_feed']:
                    set_clauses.append(f"{field} = ?")
                    values.append(value)
            
            if not set_clauses:
                return True
            
            values.append(article_id)  # For WHERE clause
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    f"UPDATE articles SET {', '.join(set_clauses)} WHERE id = ?",
                    values
                )
                conn.commit()
                
                success = cursor.rowcount > 0
                if success:
                    self.logger.debug(f"Updated article: {article_id}")
                
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to update article {article_id}: {e}")
            return False
    
    def delete_article(self, article_id: str) -> bool:
        """Delete article by ID.
        
        Args:
            article_id: Article ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM articles WHERE id = ?",
                    (article_id,)
                )
                conn.commit()
                
                success = cursor.rowcount > 0
                if success:
                    self.logger.debug(f"Deleted article: {article_id}")
                
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to delete article {article_id}: {e}")
            return False
    
    def delete_old_articles(self, days_to_keep: int = None) -> int:
        """Delete articles older than specified days.
        
        Args:
            days_to_keep: Days of articles to keep (default from config)
            
        Returns:
            Number of articles deleted
        """
        if days_to_keep is None:
            days_to_keep = self.settings.database.cleanup_days
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM articles WHERE created_at < ?",
                    (cutoff_date,)
                )
                conn.commit()
                
                deleted_count = cursor.rowcount
                self.logger.info(f"Deleted {deleted_count} articles older than {days_to_keep} days")
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to delete old articles: {e}")
            return 0
    
    def get_article_count(self) -> int:
        """Get total number of articles.
        
        Returns:
            Total article count
        """
        try:
            with self.db.get_connection() as conn:
                result = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            self.logger.error(f"Failed to get article count: {e}")
            return 0
    
    def get_feed_article_stats(self) -> Dict[str, int]:
        """Get article count per feed.
        
        Returns:
            Dictionary mapping feed URLs to article counts
        """
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT source_feed, COUNT(*) as count 
                    FROM articles 
                    GROUP BY source_feed 
                    ORDER BY count DESC
                    """
                ).fetchall()
                
                return {row['source_feed']: row['count'] for row in rows}
                
        except Exception as e:
            self.logger.error(f"Failed to get feed stats: {e}")
            return {}