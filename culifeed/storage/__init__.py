"""
CuliFeed Storage Layer
=====================

Repository pattern implementations for data access abstraction.

This module provides:
- Article repository for article CRUD operations
- Topic repository for topic management
- Database abstraction with proper error handling
- Standardized data access patterns
"""

from .article_repository import ArticleRepository
from .topic_repository import TopicRepository

__all__ = [
    "ArticleRepository",
    "TopicRepository",
]
