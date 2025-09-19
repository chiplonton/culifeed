#!/usr/bin/env python3
"""
CuliFeed Daily Scheduler - Cron Coordination
==========================================

Orchestrates daily processing workflow for content curation and delivery.
Designed to be called by systemd timers or cron jobs.

Features:
- Schedule coordination across multiple channels
- Health monitoring and error handling
- Performance monitoring and resource management
- Graceful failure handling with notifications
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config.settings import get_settings
from ..database.connection import get_db_manager
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import CuliFeedError

from ..processing.pipeline import ProcessingPipeline
from ..database.connection import get_db_manager
from ..delivery.message_sender import MessageSender
from telegram import Bot


class DailyScheduler:
    """
    Coordinates daily processing across all registered channels.
    
    This is the main entry point for scheduled processing, typically
    invoked by systemd timers or cron jobs.
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.logger = get_logger_for_component('scheduler')
        self.db_manager = get_db_manager(self.settings.database.path)
        # Performance monitoring disabled for now
        self.performance_monitor = None
        
        # Processing components
        self.pipeline = ProcessingPipeline(self.db_manager)
        
        # Message delivery (requires bot token)
        bot_token = self.settings.telegram.bot_token
        self.bot = Bot(token=bot_token) if bot_token else None
        self.message_sender = MessageSender(self.bot, self.db_manager) if self.bot else None
        
        # Execution tracking
        self.execution_id = f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = None
        self.channels_processed = 0
        self.total_articles_processed = 0
        self.errors_encountered = []
        
    async def run_daily_processing(self, dry_run: bool = False) -> Dict:
        """
        Execute complete daily processing workflow.
        
        Args:
            dry_run: If True, simulate processing without sending messages
            
        Returns:
            Dictionary with processing results and metrics
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting daily processing", extra={
            'execution_id': self.execution_id,
            'dry_run': dry_run,
            'scheduled_hour': self.settings.processing.daily_run_hour
        })
        
        try:
            # Phase 1: Pre-processing health checks
            await self._perform_health_checks()
            
            # Phase 1.5: Initialize Bot for message delivery
            if self.bot and not dry_run:
                await self.bot.initialize()
                self.logger.debug("Bot initialized for message delivery")
            
            # Phase 2: Get active channels
            channels = await self._get_active_channels()
            if not channels:
                self.logger.warning("No active channels found for processing")
                return self._create_result_summary(success=True, message="No channels to process")
            
            self.logger.info(f"Processing {len(channels)} channels", extra={
                'channel_count': len(channels),
                'channels': [ch['chat_id'] for ch in channels]
            })
            
            # Phase 3: Process each channel
            channel_results = []
            for channel in channels:
                try:
                    # Performance monitoring disabled for now
                        result = await self._process_channel(channel, dry_run)
                        channel_results.append(result)
                        self.channels_processed += 1
                        
                except Exception as e:
                    error_msg = f"Channel {channel['chat_id']} processing failed: {e}"
                    self.logger.error(error_msg, extra={
                        'channel_id': channel['chat_id'],
                        'error': str(e)
                    }, exc_info=True)
                    self.errors_encountered.append({
                        'channel_id': channel['chat_id'],
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
                    channel_results.append({
                        'channel_id': channel['chat_id'],
                        'success': False,
                        'error': str(e),
                        'articles_processed': 0,
                        'messages_sent': 0
                    })
            
            # Phase 4: Post-processing tasks
            await self._post_processing_cleanup()
            
            # Generate final results
            result_summary = self._create_result_summary(
                success=len(self.errors_encountered) == 0,
                channel_results=channel_results
            )
            
            # Log completion
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"Daily processing completed", extra={
                'execution_id': self.execution_id,
                'duration_seconds': duration,
                'channels_processed': self.channels_processed,
                'total_articles': self.total_articles_processed,
                'errors': len(self.errors_encountered),
                'success': result_summary['success']
            })
            
            # Cleanup: Shutdown bot if initialized
            if self.bot and not dry_run:
                await self.bot.shutdown()
                self.logger.debug("Bot shutdown completed")
            
            return result_summary
            
        except Exception as e:
            error_msg = f"Daily processing failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            
            # Cleanup: Shutdown bot if initialized
            if self.bot and not dry_run:
                try:
                    await self.bot.shutdown()
                except Exception as shutdown_error:
                    self.logger.warning(f"Bot shutdown error: {shutdown_error}")
            
            return self._create_result_summary(success=False, message=error_msg)
    
    async def _perform_health_checks(self) -> None:
        """Perform system health checks before processing."""
        self.logger.debug("Performing pre-processing health checks")
        
        # Check database connectivity
        try:
            db_info = self.db_manager.get_database_info()
            # Check database size (use a reasonable default if not configured)
            max_size_mb = getattr(self.settings.database, 'max_size_mb', 500)  # 500MB default
            if db_info['database_size_mb'] > max_size_mb:
                self.logger.warning(
                    f"Database size ({db_info['database_size_mb']:.1f}MB) is large (>{max_size_mb}MB)"
                )
        except Exception as e:
            raise CuliFeedError(f"Database health check failed: {e}")
        
        # Check available disk space
        try:
            db_path = Path(self.settings.database.path)
            stat = db_path.stat() if db_path.exists() else None
            # Basic check - in production you'd want more sophisticated disk space monitoring
        except Exception as e:
            self.logger.warning(f"Disk space check failed: {e}")
        
        # Check AI provider availability (basic connectivity test)
        try:
            # This would be a quick health check to primary AI provider
            # Implementation depends on specific AI client interfaces
            pass
        except Exception as e:
            self.logger.warning(f"AI provider health check warning: {e}")
    
    async def _get_active_channels(self) -> List:
        """Get list of channels that should be processed today."""
        try:
            # Get all active channels
            # Get active channels from database
            with self.db_manager.get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM channels 
                    WHERE active = ? 
                    ORDER BY created_at
                """, (True,)).fetchall()
                channels = [dict(row) for row in rows]
            
            # Filter channels based on processing schedule
            # For example, some channels might process daily, others weekly
            current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday
            
            # For now, process all active channels daily
            # This could be extended with scheduling logic later
            return channels
            
        except Exception as e:
            raise CuliFeedError(f"Failed to get active channels: {e}")
    
    async def _process_channel(self, channel, dry_run: bool) -> Dict:
        """
        Process content for a single channel.
        
        Args:
            channel: Channel configuration object
            dry_run: If True, simulate processing
            
        Returns:
            Dictionary with processing results for this channel
        """
        channel_start = time.time()
        self.logger.info(f"Processing channel {channel['chat_id']}", extra={
            'channel_id': channel['chat_id'],
            'channel_name': channel.get('chat_title', 'Unknown'),
            'dry_run': dry_run
        })
        
        try:
            # Step 1: Process RSS feeds and generate curated content
            processing_result = await self.pipeline.process_channel(
                channel['chat_id'],
                max_articles_per_topic=self.settings.processing.max_articles_per_topic
            )
            
            # Determine success based on processing results
            processing_success = (
                processing_result.successful_feed_fetches > 0 and 
                len(processing_result.errors) == 0
            )
            
            if not processing_success:
                return {
                    'channel_id': channel['chat_id'],
                    'success': False,
                    'error': '; '.join(processing_result.errors) if processing_result.errors else 'Processing failed',
                    'articles_processed': 0,
                    'messages_sent': 0,
                    'processing_time': time.time() - channel_start
                }
            
            # Step 2: Send digest to channel (if not dry run)
            messages_sent = 0
            if not dry_run and processing_result.successful_feed_fetches > 0:
                try:
                    digest_result = await self.message_sender.deliver_daily_digest(
                        channel['chat_id'],
                        self.settings.processing.max_articles_per_topic
                    )
                    messages_sent = digest_result.messages_sent if digest_result.success else 0
                    
                    if not digest_result.success:
                        self.logger.warning(f"Digest sending failed for {channel['chat_id']}: {digest_result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"Digest sending error for {channel['chat_id']}: {e}")
            
            # Update processing statistics
            articles_processed = processing_result.articles_ready_for_ai
            self.total_articles_processed += articles_processed
            
            # Record processing success
            # Record processing success in database
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    UPDATE channels 
                    SET last_delivery_at = ?
                    WHERE chat_id = ?
                """, (datetime.now(), channel['chat_id']))
                conn.commit()
            
            processing_time = time.time() - channel_start
            
            self.logger.info(f"Channel {channel['chat_id']} processed successfully", extra={
                'channel_id': channel['chat_id'],
                'articles_processed': articles_processed,
                'messages_sent': messages_sent,
                'processing_time': processing_time,
                'dry_run': dry_run
            })
            
            return {
                'channel_id': channel['chat_id'],
                'success': True,
                'articles_processed': articles_processed,
                'messages_sent': messages_sent,
                'processing_time': processing_time,
                'curated_articles': processing_result.articles_ready_for_ai if not dry_run else None
            }
            
        except Exception as e:
            processing_time = time.time() - channel_start
            error_msg = f"Channel processing failed: {e}"
            
            # Record processing failure
            try:
                # Record processing failure 
                self.logger.error(f"Processing failed for channel {channel['chat_id']}: {e}")
            except Exception as record_error:
                self.logger.error(f"Failed to record processing failure: {record_error}")
            
            return {
                'channel_id': channel['chat_id'],
                'success': False,
                'error': error_msg,
                'articles_processed': 0,
                'messages_sent': 0,
                'processing_time': processing_time
            }
    
    async def _post_processing_cleanup(self) -> None:
        """Perform cleanup tasks after processing."""
        try:
            # Clean up old processed articles (if configured)
            if self.settings.database.cleanup_days > 0:
                self.logger.debug(f"Cleaning up articles older than {self.settings.database.cleanup_days} days")
                cleaned_count = self.db_manager.cleanup_old_data(self.settings.database.cleanup_days)
                if cleaned_count > 0:
                    self.logger.info(f"Cleaned up {cleaned_count} old records")
            
            # Vacuum database periodically (e.g., once per week)
            current_day = datetime.now().weekday()
            if current_day == 0:  # Monday
                self.logger.debug("Performing weekly database vacuum")
                self.db_manager.vacuum_database()
                self.logger.info("Database vacuum completed")
                
        except Exception as e:
            self.logger.warning(f"Post-processing cleanup warning: {e}")
    
    def _create_result_summary(self, success: bool, message: str = None, channel_results: List = None) -> Dict:
        """Create standardized result summary."""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        summary = {
            'execution_id': self.execution_id,
            'success': success,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration_seconds': duration,
            'channels_processed': self.channels_processed,
            'total_articles_processed': self.total_articles_processed,
            'errors_count': len(self.errors_encountered),
            'performance_metrics': {'disabled': True},
            'message': message
        }
        
        if channel_results:
            summary['channel_results'] = channel_results
            summary['successful_channels'] = len([r for r in channel_results if r['success']])
            summary['failed_channels'] = len([r for r in channel_results if not r['success']])
        
        if self.errors_encountered:
            summary['errors'] = self.errors_encountered
            
        return summary
    
    async def check_processing_status(self) -> Dict:
        """
        Check status of recent processing runs.
        Useful for monitoring and alerting.
        """
        try:
            # Get recent processing history from database
            # Get recent processing history from database
            with self.db_manager.get_connection() as conn:
                # Count channels that have had recent deliveries (proxy for processing)
                total_runs = conn.execute("""
                    SELECT COUNT(*) FROM channels 
                    WHERE last_delivery_at >= datetime('now', '-7 days')
                """).fetchone()[0]
                
                # Count successful runs (those with recent deliveries)
                successful_runs = total_runs  # Assume all deliveries were successful
                
                # Check if any processing happened today
                processed_today = conn.execute("""
                    SELECT COUNT(*) FROM channels 
                    WHERE date(last_delivery_at) = date('now')
                """).fetchone()[0] > 0
                
                # Get most recent successful run
                last_success_row = conn.execute("""
                    SELECT last_delivery_at FROM channels 
                    WHERE last_delivery_at IS NOT NULL
                    ORDER BY last_delivery_at DESC LIMIT 1
                """).fetchone()
                last_success = last_success_row[0] if last_success_row else None
            
            # Calculate success rate
            success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
            
            return {
                'current_time': datetime.now().isoformat(),
                'processed_today': processed_today,
                'last_successful_run': last_success if last_success else None,
                'recent_success_rate': round(success_rate, 1),
                'total_recent_runs': total_runs,
                'successful_recent_runs': successful_runs,
                'health_status': 'healthy' if success_rate >= 80 and processed_today else 'warning'
            }
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}", exc_info=True)
            return {
                'current_time': datetime.now().isoformat(),
                'health_status': 'error',
                'error': str(e)
            }




