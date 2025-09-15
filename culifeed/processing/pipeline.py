"""
Processing Pipeline Orchestrator
===============================

Orchestrates the complete content processing workflow from RSS fetching
to pre-filtering, coordinating all processing components.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from contextlib import asynccontextmanager

from ..database.models import Article, Topic, Feed, ProcessingStats
from ..database.connection import DatabaseConnection
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import ProcessingError, ErrorCode

from .feed_fetcher import FeedFetcher, FetchResult
from .feed_manager import FeedManager
from .article_processor import ArticleProcessor, DeduplicationStats
from .pre_filter import PreFilterEngine, FilterResult


@dataclass
class PipelineResult:
    """Result of complete pipeline processing."""
    channel_id: str
    total_feeds_processed: int
    successful_feed_fetches: int
    total_articles_fetched: int
    unique_articles_after_dedup: int
    articles_passed_prefilter: int
    articles_ready_for_ai: int
    processing_time_seconds: float
    feed_fetch_time_seconds: float
    deduplication_stats: Optional[DeduplicationStats]
    topic_matches: Dict[str, int]
    errors: List[str]
    
    @property
    def efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        return {
            'feed_success_rate': (self.successful_feed_fetches / self.total_feeds_processed) * 100 if self.total_feeds_processed > 0 else 0.0,
            'deduplication_rate': self.deduplication_stats.deduplication_rate if self.deduplication_stats else 0.0,
            'prefilter_reduction': ((self.unique_articles_after_dedup - self.articles_passed_prefilter) / self.unique_articles_after_dedup) * 100 if self.unique_articles_after_dedup > 0 else 0.0,
            'overall_reduction': ((self.total_articles_fetched - self.articles_ready_for_ai) / self.total_articles_fetched) * 100 if self.total_articles_fetched > 0 else 0.0,
            'articles_per_second': self.total_articles_fetched / self.processing_time_seconds if self.processing_time_seconds > 0 else 0.0
        }


class ProcessingPipeline:
    """Complete content processing pipeline orchestrator."""
    
    def __init__(self, db_connection: DatabaseConnection):
        """Initialize processing pipeline.
        
        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.settings = get_settings()
        self.logger = get_logger_for_component("pipeline")
        
        # Initialize components
        self.feed_fetcher = FeedFetcher(
            max_concurrent=self.settings.processing.parallel_feeds,
            timeout=self.settings.limits.request_timeout
        )
        self.feed_manager = FeedManager(db_connection)
        self.article_processor = ArticleProcessor(db_connection)
        self.pre_filter = PreFilterEngine()
    
    async def process_channel(self, chat_id: str, max_articles_per_topic: int = None) -> PipelineResult:
        """Process all feeds for a single channel.
        
        Args:
            chat_id: Channel chat ID to process
            max_articles_per_topic: Maximum articles per topic (default from config)
            
        Returns:
            PipelineResult with processing statistics
        """
        if max_articles_per_topic is None:
            max_articles_per_topic = self.settings.processing.max_articles_per_topic
        
        start_time = datetime.now(timezone.utc)
        errors = []
        
        self.logger.info(f"Starting pipeline processing for channel {chat_id}")
        
        try:
            # Step 1: Get active feeds for channel
            feeds = self.feed_manager.get_feeds_for_channel(chat_id, active_only=True)
            if not feeds:
                self.logger.warning(f"No active feeds found for channel {chat_id}")
                return self._create_empty_result(chat_id, errors)
            
            feed_urls = [str(feed.url) for feed in feeds]
            self.logger.info(f"Processing {len(feeds)} feeds for channel {chat_id}")
            
            # Step 2: Fetch RSS feeds concurrently
            fetch_start_time = datetime.now(timezone.utc)
            fetch_results = await self.feed_fetcher.fetch_feeds_batch(feed_urls)
            fetch_duration = (datetime.now(timezone.utc) - fetch_start_time).total_seconds()
            
            # Update feed statuses
            self._update_feed_statuses(feeds, fetch_results)
            
            # Step 3: Collect all articles
            all_articles = []
            successful_fetches = 0
            
            for result in fetch_results:
                if result.success and result.articles:
                    all_articles.extend(result.articles)
                    successful_fetches += 1
                elif not result.success:
                    errors.append(f"Feed fetch failed: {result.feed_url} - {result.error}")
            
            self.logger.info(f"Collected {len(all_articles)} articles from {successful_fetches} feeds")
            
            if not all_articles:
                self.logger.warning(f"No articles collected for channel {chat_id}")
                return self._create_result(
                    chat_id, len(feeds), successful_fetches, 0, 0, 0, 0,
                    (datetime.now(timezone.utc) - start_time).total_seconds(),
                    fetch_duration, None, {}, errors
                )
            
            # Step 4: Process articles (normalize and deduplicate)
            unique_articles, dedup_stats = self.article_processor.process_articles(
                all_articles, check_database=True
            )
            
            self.logger.info(f"After deduplication: {len(unique_articles)} unique articles")
            
            # Step 5: Get topics for channel
            topics = self._get_channel_topics(chat_id)
            if not topics:
                self.logger.warning(f"No active topics found for channel {chat_id}")
                return self._create_result(
                    chat_id, len(feeds), successful_fetches, len(all_articles),
                    len(unique_articles), 0, 0,
                    (datetime.now(timezone.utc) - start_time).total_seconds(),
                    fetch_duration, dedup_stats, {}, errors
                )
            
            # Step 6: Pre-filter articles
            filter_results = self.pre_filter.filter_articles(unique_articles, topics)
            passed_articles = [r.article for r in filter_results if r.passed_filter]
            
            # Count topic matches
            topic_matches = {}
            for result in filter_results:
                for topic in result.matched_topics:
                    topic_matches[topic] = topic_matches.get(topic, 0) + 1
            
            self.logger.info(f"After pre-filtering: {len(passed_articles)} articles ready for AI")
            
            # Step 7: Store articles and prepare for AI processing
            ai_ready_articles = self._prepare_for_ai_processing(
                passed_articles, filter_results, chat_id, max_articles_per_topic
            )
            
            # Step 8: Calculate final metrics
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = self._create_result(
                chat_id, len(feeds), successful_fetches, len(all_articles),
                len(unique_articles), len(passed_articles), len(ai_ready_articles),
                total_processing_time, fetch_duration, dedup_stats, topic_matches, errors
            )
            
            self.logger.info(
                f"Pipeline complete for channel {chat_id}: "
                f"{len(ai_ready_articles)} articles ready for AI processing "
                f"in {total_processing_time:.2f}s"
            )
            
            # Log efficiency metrics
            metrics = result.efficiency_metrics
            self.logger.info(
                f"Efficiency metrics: "
                f"Feed success {metrics['feed_success_rate']:.1f}%, "
                f"Dedup {metrics['deduplication_rate']:.1f}%, "
                f"Pre-filter reduction {metrics['prefilter_reduction']:.1f}%, "
                f"Overall reduction {metrics['overall_reduction']:.1f}%"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            return self._create_result(
                chat_id, 0, 0, 0, 0, 0, 0, total_time, 0, None, {}, errors
            )
    
    def _update_feed_statuses(self, feeds: List[Feed], fetch_results: List[FetchResult]) -> None:
        """Update feed statuses based on fetch results.
        
        Args:
            feeds: List of feeds
            fetch_results: List of fetch results
        """
        # Create mapping of URL to feed
        feed_map = {str(feed.url): feed for feed in feeds}
        
        for result in fetch_results:
            feed = feed_map.get(result.feed_url)
            if feed:
                self.feed_manager.update_feed_status(feed.id, result)
    
    def _get_channel_topics(self, chat_id: str) -> List[Topic]:
        """Get active topics for a channel.
        
        Args:
            chat_id: Channel chat ID
            
        Returns:
            List of active Topic models
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM topics WHERE chat_id = ? AND active = ? ORDER BY created_at",
                (chat_id, True)
            ).fetchall()
            
            topics = []
            for row in rows:
                topic_data = dict(row)
                # Parse JSON fields
                if isinstance(topic_data.get('keywords'), str):
                    import json
                    topic_data['keywords'] = json.loads(topic_data['keywords'])
                if isinstance(topic_data.get('exclude_keywords'), str):
                    import json
                    topic_data['exclude_keywords'] = json.loads(topic_data['exclude_keywords'])
                
                topics.append(Topic(**topic_data))
            
            return topics
    
    def _prepare_for_ai_processing(self, articles: List[Article], filter_results: List[FilterResult], 
                                 chat_id: str, max_per_topic: int) -> List[Article]:
        """Prepare articles for AI processing with topic-based limiting.
        
        Args:
            articles: Filtered articles
            filter_results: Pre-filter results
            chat_id: Channel chat ID
            max_per_topic: Maximum articles per topic
            
        Returns:
            List of articles ready for AI processing
        """
        # Group articles by their best matching topic
        topic_articles = {}
        result_map = {r.article.id: r for r in filter_results if r.passed_filter}
        
        for article in articles:
            result = result_map.get(article.id)
            if result and result.best_match_topic:
                topic = result.best_match_topic
                if topic not in topic_articles:
                    topic_articles[topic] = []
                topic_articles[topic].append((article, result.best_match_score))
        
        # Select top articles per topic
        ai_ready_articles = []
        for topic, topic_article_scores in topic_articles.items():
            # Sort by relevance score (descending)
            topic_article_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N articles for this topic
            selected = topic_article_scores[:max_per_topic]
            ai_ready_articles.extend([article for article, score in selected])
            
            self.logger.debug(
                f"Selected {len(selected)} articles for topic '{topic}' "
                f"(scores: {[f'{score:.3f}' for _, score in selected[:3]]}"
                f"{'...' if len(selected) > 3 else ''})"
            )
        
        # Store articles in database for AI processing
        self._store_articles_for_processing(ai_ready_articles)
        
        return ai_ready_articles
    
    def _store_articles_for_processing(self, articles: List[Article]) -> None:
        """Store articles in database for later AI processing.
        
        Args:
            articles: Articles to store
        """
        if not articles:
            return
        
        with self.db.get_connection() as conn:
            for article in articles:
                # Insert or update article
                conn.execute("""
                    INSERT OR REPLACE INTO articles 
                    (id, title, url, content, published_at, source_feed, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.id, article.title, str(article.url), article.content,
                    article.published_at, article.source_feed, article.content_hash,
                    article.created_at
                ))
            
            conn.commit()
        
        self.logger.info(f"Stored {len(articles)} articles for AI processing")
    
    def _create_empty_result(self, chat_id: str, errors: List[str]) -> PipelineResult:
        """Create empty pipeline result."""
        return self._create_result(chat_id, 0, 0, 0, 0, 0, 0, 0.0, 0.0, None, {}, errors)
    
    def _create_result(self, chat_id: str, total_feeds: int, successful_feeds: int,
                      total_articles: int, unique_articles: int, passed_filter: int,
                      ai_ready: int, processing_time: float, fetch_time: float,
                      dedup_stats: Optional[DeduplicationStats], topic_matches: Dict[str, int],
                      errors: List[str]) -> PipelineResult:
        """Create pipeline result object."""
        return PipelineResult(
            channel_id=chat_id,
            total_feeds_processed=total_feeds,
            successful_feed_fetches=successful_feeds,
            total_articles_fetched=total_articles,
            unique_articles_after_dedup=unique_articles,
            articles_passed_prefilter=passed_filter,
            articles_ready_for_ai=ai_ready,
            processing_time_seconds=processing_time,
            feed_fetch_time_seconds=fetch_time,
            deduplication_stats=dedup_stats,
            topic_matches=topic_matches,
            errors=errors
        )
    
    async def process_multiple_channels(self, chat_ids: List[str]) -> List[PipelineResult]:
        """Process multiple channels concurrently.
        
        Args:
            chat_ids: List of channel chat IDs to process
            
        Returns:
            List of PipelineResult objects
        """
        if not chat_ids:
            return []
        
        self.logger.info(f"Starting multi-channel processing for {len(chat_ids)} channels")
        
        # Process channels concurrently with semaphore to limit concurrency
        max_concurrent = min(len(chat_ids), 3)  # Limit to avoid overwhelming system
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(chat_id: str) -> PipelineResult:
            async with semaphore:
                return await self.process_channel(chat_id)
        
        # Execute all channel processing tasks
        tasks = [process_with_semaphore(chat_id) for chat_id in chat_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and convert to PipelineResult
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Channel {chat_ids[i]} processing failed: {result}",
                    exc_info=result
                )
                final_results.append(self._create_empty_result(
                    chat_ids[i], [f"Processing exception: {result}"]
                ))
            else:
                final_results.append(result)
        
        # Log summary
        total_articles = sum(r.articles_ready_for_ai for r in final_results)
        total_time = max(r.processing_time_seconds for r in final_results) if final_results else 0
        
        self.logger.info(
            f"Multi-channel processing complete: {len(chat_ids)} channels, "
            f"{total_articles} articles ready for AI in {total_time:.2f}s"
        )
        
        return final_results
    
    async def run_daily_processing(self) -> ProcessingStats:
        """Run daily processing for all active channels.
        
        Returns:
            ProcessingStats with comprehensive statistics
        """
        start_time = datetime.now(timezone.utc)
        self.logger.info("Starting daily processing for all channels")
        
        # Get all active channels
        with self.db.get_connection() as conn:
            channel_rows = conn.execute("SELECT DISTINCT chat_id FROM feeds WHERE active = ?", (True,)).fetchall()
            chat_ids = [row['chat_id'] for row in channel_rows]
        
        if not chat_ids:
            self.logger.warning("No active channels found for daily processing")
            return ProcessingStats()
        
        # Process all channels
        results = await self.process_multiple_channels(chat_ids)
        
        # Aggregate statistics
        total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        stats = ProcessingStats(
            total_articles=sum(r.total_articles_fetched for r in results),
            pre_filtered_articles=sum(r.articles_passed_prefilter for r in results),
            ai_processed_articles=sum(r.articles_ready_for_ai for r in results),
            delivered_articles=0,  # This will be updated by AI processing phase
            processing_time_seconds=total_processing_time,
            api_calls_used=0,  # This will be updated by AI processing phase
            estimated_cost=0.0,  # This will be updated by AI processing phase
            channels_processed=len(chat_ids),
            topics_matched=sum(len(r.topic_matches) for r in results)
        )
        
        self.logger.info(
            f"Daily processing complete: {stats.channels_processed} channels, "
            f"{stats.total_articles} articles fetched, "
            f"{stats.ai_processed_articles} ready for AI processing "
            f"({stats.pre_filter_reduction_percent:.1f}% pre-filter reduction)"
        )
        
        return stats