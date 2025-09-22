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

# AI Integration
from ..ai.ai_manager import AIManager
from ..ai.providers.base import AIResult


@dataclass
class PipelineResult:
    """Result of complete pipeline processing with comprehensive metrics."""
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
    
    # Enhanced AI Processing Metrics
    ai_requests_sent: int = 0
    ai_requests_successful: int = 0
    ai_requests_failed: int = 0
    ai_provider_breakdown: Dict[str, Dict[str, int]] = None  # provider -> {requests, successes, failures}
    ai_processing_time_seconds: float = 0.0
    articles_processed_by_ai: int = 0
    articles_ai_relevant: int = 0  # Articles with relevance score >= threshold
    
    # Delivery Metrics
    articles_sent_to_telegram: int = 0
    telegram_messages_sent: int = 0
    telegram_delivery_failures: int = 0
    delivery_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Initialize nested dictionaries if None."""
        if self.ai_provider_breakdown is None:
            self.ai_provider_breakdown = {}
    
    @property
    def efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        return {
            'feed_success_rate': (self.successful_feed_fetches / self.total_feeds_processed) * 100 if self.total_feeds_processed > 0 else 0.0,
            'deduplication_rate': self.deduplication_stats.deduplication_rate if self.deduplication_stats else 0.0,
            'prefilter_reduction': ((self.unique_articles_after_dedup - self.articles_passed_prefilter) / self.unique_articles_after_dedup) * 100 if self.unique_articles_after_dedup > 0 else 0.0,
            'overall_reduction': ((self.total_articles_fetched - self.articles_ready_for_ai) / self.total_articles_fetched) * 100 if self.total_articles_fetched > 0 else 0.0,
            'articles_per_second': self.total_articles_fetched / self.processing_time_seconds if self.processing_time_seconds > 0 else 0.0,
            'ai_success_rate': (self.ai_requests_successful / self.ai_requests_sent) * 100 if self.ai_requests_sent > 0 else 0.0,
            'ai_relevance_rate': (self.articles_ai_relevant / self.articles_processed_by_ai) * 100 if self.articles_processed_by_ai > 0 else 0.0,
            'delivery_success_rate': ((self.articles_sent_to_telegram - self.telegram_delivery_failures) / self.articles_sent_to_telegram) * 100 if self.articles_sent_to_telegram > 0 else 100.0
        }
    
    @property 
    def ai_provider_summary(self) -> Dict[str, str]:
        """Get AI provider usage summary."""
        if not self.ai_provider_breakdown:
            return {}
        
        summary = {}
        for provider, stats in self.ai_provider_breakdown.items():
            requests = stats.get('requests', 0)
            successes = stats.get('successes', 0)
            success_rate = (successes / requests * 100) if requests > 0 else 0
            summary[provider] = f"{requests} requests ({success_rate:.1f}% success)"
        
        return summary


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
        
        # Initialize components with settings
        self.feed_fetcher = FeedFetcher(
            max_concurrent=self.settings.processing.parallel_feeds,
            timeout=self.settings.limits.request_timeout
        )
        self.feed_manager = FeedManager(db_connection)
        self.article_processor = ArticleProcessor(db_connection, settings=self.settings)  # Pass settings
        self.pre_filter = PreFilterEngine(self.settings)  # Pass settings for configurable thresholds
        
        # AI Integration - Initialize AI Manager
        self.ai_manager = AIManager()
        
        # Smart Processing - Initialize Smart Keyword Analyzer
        if self.settings.smart_processing.enabled:
            from .smart_analyzer import SmartKeywordAnalyzer
            self.smart_analyzer = SmartKeywordAnalyzer()
        else:
            self.smart_analyzer = None
    
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
            # Don't check database to avoid filtering out articles we just stored
            unique_articles, dedup_stats = self.article_processor.process_articles(
                all_articles, check_database=False
            )
            
            self.logger.info(f"After deduplication: {len(unique_articles)} unique articles")
            
            # Step 5: Store unique articles in database (before AI processing)
            # This ensures articles are preserved even if AI processing fails
            if unique_articles:
                self._store_articles_basic(unique_articles, chat_id)
                self.logger.info(f"Stored {len(unique_articles)} articles in database")

            # Step 6: Get topics for channel
            topics = self._get_channel_topics(chat_id)
            if not topics:
                self.logger.warning(f"No active topics found for channel {chat_id}")
                return self._create_result(
                    chat_id, len(feeds), successful_fetches, len(all_articles),
                    len(unique_articles), 0, 0,
                    (datetime.now(timezone.utc) - start_time).total_seconds(),
                    fetch_duration, dedup_stats, {}, errors
                )
            
            # Step 7: Get unprocessed articles from database for AI analysis
            # This includes articles just stored and any previously stored but unprocessed articles
            unprocessed_articles = self._get_unprocessed_articles(chat_id)
            self.logger.info(f"Found {len(unprocessed_articles)} unprocessed articles in database")
            
            # Step 8: Pre-filter articles
            filter_results = self.pre_filter.filter_articles(unprocessed_articles, topics)
            passed_articles = [r.article for r in filter_results if r.passed_filter]
            
            # Count topic matches
            topic_matches = {}
            for result in filter_results:
                for topic in result.matched_topics:
                    topic_matches[topic] = topic_matches.get(topic, 0) + 1
            
            self.logger.info(f"After pre-filtering: {len(passed_articles)} articles ready for AI")
            
            # Step 9: AI Analysis and Processing
            # Only pass filter results that correspond to articles that passed pre-filter
            passed_filter_results = [r for r in filter_results if r.passed_filter]
            ai_processed_articles, ai_metrics = await self._ai_analysis_and_processing(
                passed_articles, passed_filter_results, topics, max_articles_per_topic
            )
            
            # Step 10: Calculate final metrics
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = self._create_result(
                chat_id, len(feeds), successful_fetches, len(all_articles),
                len(unique_articles), len(passed_articles), len(ai_processed_articles),
                total_processing_time, fetch_duration, dedup_stats, topic_matches, errors,
                ai_requests=ai_metrics["ai_requests_sent"],
                ai_successes=ai_metrics["ai_requests_successful"],
                ai_failures=ai_metrics["ai_requests_failed"],
                ai_provider_breakdown=ai_metrics["ai_provider_breakdown"],
                articles_processed_by_ai=len(ai_processed_articles),
                articles_ai_relevant=ai_metrics["articles_ai_relevant"]  # NEW: Pass the count
            )
            
            self.logger.info(
                f"Pipeline complete for channel {chat_id}: "
                f"{len(ai_processed_articles)} articles ready for AI processing "
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
    
    async def _ai_analysis_and_processing(self, articles: List[Article], filter_results: List[FilterResult],
                                       topics: List[Topic], max_per_topic: int) -> tuple[List[Article], dict]:
        """Perform AI analysis and processing on filtered articles.

        Args:
            articles: List of articles to analyze
            filter_results: Pre-filtering results for articles
            topics: Topics for relevance analysis
            max_per_topic: Maximum articles per topic

        Returns:
            Tuple of (processed articles, AI metrics dict including articles_ai_relevant count)
        """
        if not articles or not topics:
            self.logger.info("No articles or topics for AI analysis")
            return [], {"ai_requests_sent": 0, "ai_requests_successful": 0, "ai_requests_failed": 0, "ai_provider_breakdown": {}, "articles_ai_relevant": 0}
        
        self.logger.info(f"Starting AI analysis for {len(articles)} articles across {len(topics)} topics")
        
        ai_processed_articles = []
        processing_results = []  # Track article-topic relationships

        # Initialize AI metrics tracking
        ai_requests_sent = 0
        ai_requests_successful = 0
        ai_requests_failed = 0
        ai_provider_breakdown = {}  # Format: {provider: {'requests': X, 'successes': Y, 'failures': Z}}
        articles_ai_relevant = 0  # Count articles that meet AI relevance threshold
        smart_routing_stats = {"ai_skipped": 0, "confident_relevant": 0, "confident_irrelevant": 0}
        
        # Group articles by topic for processing
        for topic in topics:
            topic_name = topic.name
            self.logger.debug(f"Processing AI analysis for topic: {topic_name}")
            
            # Get articles that passed pre-filtering for this topic
            topic_articles = []
            for article, filter_result in zip(articles, filter_results):
                if filter_result.passed_filter and topic_name in filter_result.matched_topics:
                    topic_articles.append(article)
            if not topic_articles:
                self.logger.debug(f"No articles passed pre-filtering for topic '{topic_name}'")
                continue
            
            # Limit articles per topic and sort by publication date
            topic_articles = sorted(topic_articles, key=lambda a: a.published_at, reverse=True)[:max_per_topic]
            
            # Process articles with smart routing
            for article in topic_articles:
                try:
                    # Smart routing: check if we can skip AI processing
                    should_skip_ai, routing_reason, smart_result = await self._smart_routing_decision(article, topic)
                    
                    if should_skip_ai and smart_result:
                        # Handle confident routing decisions
                        if smart_result.routing_decision == "high_confidence":
                            smart_routing_stats["confident_relevant"] += 1
                            smart_routing_stats["ai_skipped"] += 1
                            
                            # Treat as relevant article
                            articles_ai_relevant += 1
                            article.ai_relevance_score = smart_result.relevance_score
                            article.ai_confidence = smart_result.confidence_level
                            article.ai_provider = "smart_routing_confident"
                            article.ai_reasoning = f"Smart routing (confident): {routing_reason}"
                            
                            if article not in ai_processed_articles:
                                ai_processed_articles.append(article)
                            
                            processing_results.append({
                                'article_id': article.id,
                                'chat_id': topic.chat_id,
                                'topic_name': topic_name,
                                'ai_relevance_score': smart_result.relevance_score,
                                'confidence_score': smart_result.confidence_level,
                                'summary': None  # No AI summary for smart routing
                            })
                            
                            self.logger.debug(f"Smart routing ACCEPT: {article.title[:50]}... (score={smart_result.relevance_score:.3f})")
                            continue
                            
                        elif smart_result.routing_decision == "low_confidence":
                            smart_routing_stats["confident_irrelevant"] += 1
                            smart_routing_stats["ai_skipped"] += 1
                            
                            # Skip article (confident it's irrelevant)
                            self.logger.debug(f"Smart routing REJECT: {article.title[:50]}... (score={smart_result.relevance_score:.3f})")
                            continue
                    
                    # Standard AI processing for uncertain cases
                    ai_requests_sent += 1
                    ai_result = await self.ai_manager.analyze_relevance(article, topic)

                    # Track AI provider breakdown
                    if ai_result.provider:
                        if ai_result.provider not in ai_provider_breakdown:
                            ai_provider_breakdown[ai_result.provider] = {'requests': 0, 'successes': 0, 'failures': 0}
                        ai_provider_breakdown[ai_result.provider]['requests'] += 1

                    # Track success/failure
                    if ai_result.success:
                        ai_requests_successful += 1
                        if ai_result.provider:
                            ai_provider_breakdown[ai_result.provider]['successes'] += 1
                    else:
                        ai_requests_failed += 1
                        if ai_result.provider:
                            ai_provider_breakdown[ai_result.provider]['failures'] += 1
                    
                    if ai_result.success and ai_result.relevance_score >= self.settings.processing.ai_relevance_threshold:
                        # Article met main AI relevance threshold
                        articles_ai_relevant += 1
                        
                        # Generate summary if relevance is high enough
                        if ai_result.relevance_score >= self.settings.processing.ai_summary_threshold:
                            try:
                                summary_result = await self.ai_manager.generate_summary(article)
                                if summary_result and hasattr(summary_result, 'summary') and summary_result.summary:
                                    article.summary = summary_result.summary
                                else:
                                    article.summary = None
                            except Exception as e:
                                self.logger.warning(f"Summary generation failed for article {article.id}: {e}")
                                article.summary = None
                        
                        # Store AI analysis results in article
                        article.ai_relevance_score = ai_result.relevance_score
                        article.ai_confidence = ai_result.confidence
                        article.ai_provider = ai_result.provider
                        article.ai_reasoning = ai_result.reasoning

                        # Add to processed articles if not already added
                        if article not in ai_processed_articles:
                            ai_processed_articles.append(article)
                        
                        # Record the topic-article relationship
                        processing_results.append({
                            'article_id': article.id,
                            'chat_id': topic.chat_id,
                            'topic_name': topic_name,
                            'ai_relevance_score': ai_result.relevance_score,
                            'confidence_score': ai_result.confidence,
                            'summary': article.summary
                        })
                        
                        self.logger.debug(
                            f"AI processed article '{article.title}' for topic '{topic_name}': "
                            f"relevance={ai_result.relevance_score:.3f}, "
                            f"confidence={ai_result.confidence:.3f}, "
                            f"provider={ai_result.provider}"
                        )
                    else:
                        self.logger.debug(
                            f"Article '{article.title}' rejected by AI: "
                            f"relevance={ai_result.relevance_score:.3f} < threshold={self.settings.processing.ai_relevance_threshold}"
                        )
                        
                except Exception as e:
                    self.logger.error(f"AI processing failed for article {article.id}: {e}")

                    # Use hybrid fallback: keyword-based analysis
                    try:
                        hybrid_result = self.ai_manager._keyword_fallback_analysis(article, topic)
                        
                        if hybrid_result.success:
                            article.ai_relevance_score = hybrid_result.relevance_score
                            article.ai_confidence = min(hybrid_result.confidence, self.settings.filtering.fallback_confidence_cap)
                            article.ai_provider = "keyword_backup"
                            article.ai_reasoning = f"Hybrid fallback: {hybrid_result.reasoning}"

                            # Only include if it meets a minimum threshold
                            if hybrid_result.relevance_score >= self.settings.filtering.fallback_relevance_threshold:
                                if article not in ai_processed_articles:
                                    ai_processed_articles.append(article)

                                processing_results.append({
                                    'article_id': article.id,
                                    'chat_id': topic.chat_id,
                                    'topic_name': topic_name,
                                    'ai_relevance_score': hybrid_result.relevance_score,
                                    'confidence_score': article.ai_confidence,
                                    'summary': None
                                })

                                self.logger.info(
                                    f"Used keyword fallback for article '{article.title}': "
                                    f"score={hybrid_result.relevance_score:.3f}, "
                                    f"confidence={article.ai_confidence:.3f}"
                                )
                            else:
                                self.logger.debug(
                                    f"Keyword fallback score too low for article '{article.title}': "
                                    f"{hybrid_result.relevance_score:.3f} < {self.settings.filtering.fallback_relevance_threshold}"
                                )
                        else:
                            self.logger.warning(f"Keyword fallback also failed for article {article.id}")

                    except Exception as fallback_error:
                        self.logger.error(f"Hybrid fallback failed for article {article.id}: {fallback_error}")
            
            self.logger.info(
                f"AI processed {len([a for a in ai_processed_articles if any(proc['article_id'] == a.id for proc in processing_results)])} "
                f"articles for topic '{topic_name}'"
            )
        
        # Store processed articles and their topic relationships in database
        if ai_processed_articles:
            self._store_articles_for_processing(ai_processed_articles)
        
        if processing_results:
            self._store_processing_results(processing_results)
        
        self.logger.info(f"AI analysis complete: {len(ai_processed_articles)} articles ready for delivery")
        self.logger.info(f"AI metrics: {ai_requests_sent} requests sent, {ai_requests_successful} successful, {ai_requests_failed} failed")
        self.logger.info(f"Smart routing: {smart_routing_stats['ai_skipped']} AI requests skipped ({smart_routing_stats['confident_relevant']} relevant, {smart_routing_stats['confident_irrelevant']} irrelevant)")
        self.logger.info(f"Articles meeting AI relevance threshold (>= {self.settings.processing.ai_relevance_threshold}): {articles_ai_relevant}")

        # Prepare AI metrics with routing stats
        ai_metrics = {
            "ai_requests_sent": ai_requests_sent,
            "ai_requests_successful": ai_requests_successful,
            "ai_requests_failed": ai_requests_failed,
            "ai_provider_breakdown": ai_provider_breakdown,
            "articles_ai_relevant": articles_ai_relevant,
            "smart_routing_stats": smart_routing_stats
        }

        return ai_processed_articles, ai_metrics

    async def _smart_routing_decision(self, article: Article, topic: Topic) -> tuple[bool, str, Optional]:
        """Make smart routing decision to potentially skip AI processing.
        
        Args:
            article: Article to analyze
            topic: Topic for relevance analysis
            
        Returns:
            Tuple of (should_skip_ai, reasoning, smart_result)
        """
        # Skip smart routing if disabled
        if not self.settings.smart_processing.enabled or not self.smart_analyzer:
            return False, "Smart processing disabled", None
        
        try:
            # Perform smart keyword analysis
            smart_result = self.smart_analyzer.analyze_article_confidence(article, topic)
            
            # Check routing decision
            if smart_result.routing_decision == "high_confidence":
                if smart_result.relevance_score >= self.settings.smart_processing.definitely_relevant_threshold:
                    return True, f"High confidence relevant (score={smart_result.relevance_score:.3f}, confidence={smart_result.confidence_level:.3f})", smart_result
            
            elif smart_result.routing_decision == "low_confidence":
                if smart_result.relevance_score <= self.settings.smart_processing.definitely_irrelevant_threshold:
                    return True, f"High confidence irrelevant (score={smart_result.relevance_score:.3f}, confidence={smart_result.confidence_level:.3f})", smart_result
            
            # Default to uncertain - needs AI processing
            return False, f"Uncertain case (score={smart_result.relevance_score:.3f}, confidence={smart_result.confidence_level:.3f})", smart_result
            
        except Exception as e:
            self.logger.warning(f"Smart routing analysis failed for article {article.id}: {e}")
            return False, "Smart routing failed - fallback to AI", None
    
    def _store_articles_for_processing(self, articles: List[Article]) -> None:
        """Store articles in database with AI analysis results.
        
        Args:
            articles: Articles to store with AI analysis
        """
        if not articles:
            return
        
        with self.db.get_connection() as conn:
            for article in articles:
                # Insert or update article with AI fields
                conn.execute("""
                    INSERT OR REPLACE INTO articles 
                    (id, title, url, content, published_at, source_feed, content_hash, created_at,
                     summary, ai_relevance_score, ai_confidence, ai_provider, ai_reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.id, article.title, str(article.url), article.content,
                    article.published_at, article.source_feed, article.content_hash,
                    article.created_at, article.summary, article.ai_relevance_score,
                    article.ai_confidence, article.ai_provider, article.ai_reasoning
                ))
            
            conn.commit()
        
        self.logger.info(f"Stored {len(articles)} articles with AI analysis results")
    
    def _store_articles_basic(self, articles: List[Article], chat_id: str) -> None:
        """Store articles in database without AI analysis results.
        
        Args:
            articles: Articles to store (before AI processing)
            chat_id: Channel chat ID for context
        """
        if not articles:
            return
        
        with self.db.get_connection() as conn:
            for article in articles:
                # Insert or update article with basic fields only
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
        
        self.logger.info(f"Stored {len(articles)} articles in database for chat {chat_id}")
    
    def _get_unprocessed_articles(self, chat_id: str) -> List[Article]:
        """Get articles from database that haven't been processed with AI yet.
        
        Args:
            chat_id: Channel chat ID
            
        Returns:
            List of articles without AI processing results
        """
        with self.db.get_connection() as conn:
            # Get articles from feeds belonging to this chat that don't have AI scores
            rows = conn.execute("""
                SELECT a.* FROM articles a
                JOIN feeds f ON a.source_feed = f.url
                WHERE f.chat_id = ? 
                AND a.ai_relevance_score IS NULL
                AND datetime(a.created_at) >= datetime('now', '-2 days')
                ORDER BY a.published_at DESC
            """, (chat_id,)).fetchall()
            
            articles = []
            for row in rows:
                article_data = dict(row)
                article = Article(**article_data)
                articles.append(article)
            
            return articles
    
    def _store_processing_results(self, processing_results: List[dict]) -> None:
        """Store processing results with topic-article relationships.
        
        Args:
            processing_results: List of processing result dictionaries
        """
        if not processing_results:
            return
        
        with self.db.get_connection() as conn:
            for result in processing_results:
                # Insert or replace processing result
                conn.execute("""
                    INSERT OR REPLACE INTO processing_results 
                    (article_id, chat_id, topic_name, ai_relevance_score, confidence_score, 
                     summary, processed_at, delivered)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 0)
                """, (
                    result['article_id'],
                    result['chat_id'], 
                    result['topic_name'],
                    result['ai_relevance_score'],
                    result['confidence_score'],
                    result.get('summary')
                ))
            
            conn.commit()
        
        self.logger.info(f"Stored {len(processing_results)} processing results with topic relationships")
    
    def _create_empty_result(self, chat_id: str, errors: List[str]) -> PipelineResult:
        """Create empty pipeline result."""
        return self._create_result(chat_id, 0, 0, 0, 0, 0, 0, 0.0, 0.0, None, {}, errors)
    
    def _create_result(self, chat_id: str, total_feeds: int, successful_feeds: int,
                      total_articles: int, unique_articles: int, passed_filter: int,
                      ai_ready: int, processing_time: float, fetch_time: float,
                      dedup_stats: Optional[DeduplicationStats], topic_matches: Dict[str, int],
                      errors: List[str], ai_requests: int = 0, ai_successes: int = 0,
                      ai_failures: int = 0, ai_provider_breakdown: Dict = None,
                      ai_processing_time: float = 0.0, articles_processed_by_ai: int = 0,
                      articles_ai_relevant: int = 0) -> PipelineResult:
        """Create pipeline result object with comprehensive metrics."""
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
            errors=errors,
            ai_requests_sent=ai_requests,
            ai_requests_successful=ai_successes,
            ai_requests_failed=ai_failures,
            ai_provider_breakdown=ai_provider_breakdown or {},
            ai_processing_time_seconds=ai_processing_time,
            articles_processed_by_ai=articles_processed_by_ai,
            articles_ai_relevant=articles_ai_relevant
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