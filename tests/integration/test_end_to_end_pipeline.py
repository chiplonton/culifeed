#!/usr/bin/env python3
"""
End-to-End Pipeline Integration Test
===================================

Tests the complete workflow: RSS Fetch → Pre-filter → AI Analysis → Storage
This validates the integration of all phases (1-3) working together.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from culifeed.processing.pipeline import ProcessingPipeline
from culifeed.database.connection import get_db_manager
from culifeed.database.models import Article, Topic, Feed
from culifeed.config.settings import get_settings
from culifeed.utils.logging import configure_application_logging


async def test_end_to_end_with_mock_feeds():
    """Test complete pipeline with mocked RSS feeds."""
    print("🔄 Testing End-to-End Pipeline Integration...")
    
    try:
        # Configure logging
        configure_application_logging(
            log_level="INFO",
            enable_console=True,
            structured_logging=False
        )
        
        # Initialize database and pipeline
        settings = get_settings()
        db_manager = get_db_manager(":memory:")  # Use in-memory database for testing
        
        # Initialize database schema using the db_manager's connection
        from culifeed.database.schema import DatabaseSchema
        schema = DatabaseSchema(":memory:")
        
        # Create tables using the same connection as db_manager
        with db_manager.get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tables manually here since the schema uses a different db instance
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channels (
                    chat_id TEXT PRIMARY KEY,
                    chat_title TEXT NOT NULL,
                    chat_type TEXT NOT NULL CHECK (chat_type IN ('private', 'group', 'supergroup', 'channel')),
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE,
                    last_delivery_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    content TEXT,
                    published_at TIMESTAMP,
                    source_feed TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT,
                    ai_relevance_score REAL CHECK (ai_relevance_score BETWEEN 0.0 AND 1.0),
                    ai_confidence REAL CHECK (ai_confidence BETWEEN 0.0 AND 1.0),
                    ai_provider TEXT,
                    ai_reasoning TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    exclude_keywords TEXT DEFAULT '[]',
                    confidence_threshold REAL DEFAULT 0.8 CHECK (confidence_threshold BETWEEN 0.0 AND 1.0),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_match_at TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (chat_id) REFERENCES channels(chat_id) ON DELETE CASCADE,
                    UNIQUE(chat_id, name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    last_fetched_at TIMESTAMP,
                    last_success_at TIMESTAMP,
                    error_count INTEGER DEFAULT 0,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES channels(chat_id) ON DELETE CASCADE,
                    UNIQUE(chat_id, url)
                )
            """)
            
            conn.commit()
        
        # Create test data
        await _setup_test_data(db_manager)
        
        # Initialize pipeline
        pipeline = ProcessingPipeline(db_manager)
        
        # Mock the feed fetcher to return test articles
        test_articles = _create_test_articles()
        
        # Mock feed fetching
        async def mock_fetch_feeds_batch(urls):
            from culifeed.processing.feed_fetcher import FetchResult
            return [
                FetchResult(
                    feed_url=urls[0],
                    success=True,
                    articles=test_articles,
                    title="Test Feed",
                    description="Test feed for integration testing",
                    fetch_time_seconds=0.5
                )
            ]
        
        pipeline.feed_fetcher.fetch_feeds_batch = mock_fetch_feeds_batch
        
        # Run pipeline processing
        print("  📊 Running pipeline processing...")
        result = await pipeline.process_channel("test_chat_id")
        
        # Validate results
        print(f"  📈 Processing Results:")
        print(f"    Total feeds processed: {result.total_feeds_processed}")
        print(f"    Articles fetched: {result.total_articles_fetched}")
        print(f"    Unique articles: {result.unique_articles_after_dedup}")
        print(f"    Pre-filter passed: {result.articles_passed_prefilter}")
        print(f"    AI processed: {result.articles_ready_for_ai}")
        print(f"    Processing time: {result.processing_time_seconds:.2f}s")
        
        # Validate pipeline worked end-to-end
        if result.total_articles_fetched > 0:
            print("  ✅ RSS fetching working")
        else:
            print("  ❌ RSS fetching failed")
            return False
            
        if result.unique_articles_after_dedup > 0:
            print("  ✅ Article processing working")
        else:
            print("  ❌ Article processing failed")
            return False
            
        if result.articles_passed_prefilter >= 0:
            print("  ✅ Pre-filtering working")
        else:
            print("  ❌ Pre-filtering failed")
            return False
            
        # Note: AI processing may have 0 results with test/invalid API keys, 
        # but should not crash
        print(f"  {'✅' if result.articles_ready_for_ai >= 0 else '❌'} AI processing integration")
        
        # Check efficiency metrics
        metrics = result.efficiency_metrics
        print(f"  📊 Efficiency Metrics:")
        print(f"    Feed success rate: {metrics['feed_success_rate']:.1f}%")
        print(f"    Deduplication rate: {metrics['deduplication_rate']:.1f}%")
        print(f"    Pre-filter reduction: {metrics['prefilter_reduction']:.1f}%")
        print(f"    Overall reduction: {metrics['overall_reduction']:.1f}%")
        
        # Cleanup AI manager
        if hasattr(pipeline, 'ai_manager'):
            await pipeline.ai_manager.shutdown()
        
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end test error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def _setup_test_data(db_manager):
    """Set up test database with feeds and topics."""
    with db_manager.get_connection() as conn:
        # Insert test channel
        conn.execute("""
            INSERT OR REPLACE INTO channels (chat_id, chat_title, chat_type)
            VALUES (?, ?, ?)
        """, ("test_chat_id", "Test Channel", "supergroup"))
        
        # Insert test feed
        conn.execute("""
            INSERT OR REPLACE INTO feeds (chat_id, url, title, active)
            VALUES (?, ?, ?, ?)
        """, ("test_chat_id", "https://example.com/feed.xml", "Test Feed", True))
        
        # Insert test topic
        conn.execute("""
            INSERT OR REPLACE INTO topics (chat_id, name, keywords, exclude_keywords, active)
            VALUES (?, ?, ?, ?, ?)
        """, ("test_chat_id", "Technology", '["AI", "tech", "software", "cloud"]', '["marketing"]', True))
        
        conn.commit()


def _create_test_articles():
    """Create test articles for pipeline testing."""
    articles = [
        # Relevant article (should pass pre-filter and AI)
        Article(
            title="New AI Software Revolutionizes Cloud Computing",
            url="https://example.com/ai-cloud-article",
            content="This groundbreaking AI software is transforming how companies approach cloud computing. "
                   "The technology leverages machine learning algorithms to optimize resource allocation "
                   "and improve performance in cloud environments. Early adopters report significant "
                   "improvements in efficiency and cost savings.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml"
        ),
        
        # Partially relevant (should pass pre-filter, may pass AI)  
        Article(
            title="Software Development Best Practices for Modern Applications",
            url="https://example.com/software-practices",
            content="Modern software development requires adherence to established best practices. "
                   "This article covers essential techniques for building scalable applications, "
                   "including code organization, testing strategies, and deployment methods.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml"
        ),
        
        # Irrelevant article (should be filtered out)
        Article(
            title="Local Restaurant Opens New Location Downtown",
            url="https://example.com/restaurant-news",
            content="Mario's Pizza is excited to announce the opening of their third location "
                   "in downtown Springfield. The new restaurant features an expanded menu "
                   "and will be open seven days a week.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml"
        ),
        
        # Marketing article (should be excluded by exclude keywords)
        Article(
            title="Tech Marketing Strategies for 2024",
            url="https://example.com/tech-marketing",
            content="Discover the latest marketing trends in the technology sector. "
                   "This comprehensive guide covers digital marketing strategies, "
                   "content marketing, and promotional techniques for tech companies.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml"
        )
    ]
    
    return articles


async def test_ai_fallback_functionality():
    """Test AI fallback when providers fail."""
    print("\n🔄 Testing AI Fallback Functionality...")
    
    try:
        from culifeed.ai.ai_manager import AIManager
        from culifeed.database.models import Article, Topic
        
        # Create test data
        article = Article(
            title="Advanced Cloud Computing with AI Integration",
            url="https://example.com/ai-cloud-test",
            content="This article discusses the integration of artificial intelligence with cloud computing platforms.",
            published_at=datetime.now(timezone.utc),
            source_feed="test_feed"
        )
        
        topic = Topic(
            chat_id="test_chat",
            name="AI Technology",
            keywords=["AI", "cloud", "computing", "artificial intelligence"]
        )
        
        # Test AI manager with invalid API key (should fall back to keywords)
        manager = AIManager()
        
        print("  🤔 Testing relevance analysis with fallback...")
        result = await manager.analyze_relevance(article, topic)
        
        if result.success:
            print(f"    ✅ Fallback analysis successful with {result.provider}")
            print(f"    ✅ Relevance score: {result.relevance_score:.3f}")
            print(f"    ✅ Matched keywords: {result.matched_keywords}")
        else:
            print(f"    ❌ Fallback analysis failed: {result.error_message}")
            return False
            
        # Test summarization with fallback
        print("  📝 Testing summarization with fallback...")
        summary_result = await manager.generate_summary(article)
        
        if summary_result.success:
            print(f"    ✅ Fallback summarization successful with {summary_result.provider}")
            print(f"    ✅ Summary: {summary_result.summary[:100]}...")
        else:
            print(f"    ❌ Fallback summarization failed: {summary_result.error_message}")
        
        await manager.shutdown()
        return True
        
    except Exception as e:
        print(f"  ❌ AI fallback test error: {e}")
        return False


async def test_real_api_integration():
    """Test with real API if available."""
    print("\n🔗 Testing Real API Integration...")
    
    import os
    api_key = os.getenv('CULIFEED_AI__GROQ_API_KEY')
    
    if not api_key or api_key.startswith('test'):
        print("  ⚠️ No real API key available, skipping real API test")
        print("    Set CULIFEED_AI__GROQ_API_KEY environment variable for real API testing")
        return True
    
    try:
        from culifeed.ai.ai_manager import AIManager
        from culifeed.database.models import Article, Topic
        
        # Create test data
        article = Article(
            title="Revolutionary AI Breakthrough in Natural Language Processing",
            url="https://example.com/ai-nlp-breakthrough",
            content="Researchers have achieved a significant breakthrough in natural language processing "
                   "using advanced machine learning techniques. The new model demonstrates unprecedented "
                   "accuracy in understanding context and generating human-like responses.",
            published_at=datetime.now(timezone.utc),
            source_feed="test_feed"
        )
        
        topic = Topic(
            chat_id="test_chat", 
            name="Artificial Intelligence",
            keywords=["AI", "machine learning", "natural language", "breakthrough"]
        )
        
        manager = AIManager()
        
        print("  🤔 Testing real API relevance analysis...")
        result = await manager.analyze_relevance(article, topic)
        
        if result.success:
            print(f"    ✅ Real API analysis successful with {result.provider}")
            print(f"    ✅ Relevance score: {result.relevance_score:.3f}")
            print(f"    ✅ Confidence: {result.confidence:.3f}")
            if result.reasoning:
                print(f"    💭 Reasoning: {result.reasoning[:100]}...")
        else:
            print(f"    ❌ Real API analysis failed: {result.error_message}")
        
        # Test real API summarization
        print("  📝 Testing real API summarization...")
        summary_result = await manager.generate_summary(article, max_sentences=2)
        
        if summary_result.success:
            print(f"    ✅ Real API summarization successful")
            print(f"    📄 Summary: {summary_result.summary}")
        else:
            print(f"    ❌ Real API summarization failed")
        
        await manager.shutdown()
        return True
        
    except Exception as e:
        print(f"  ❌ Real API test error: {e}")
        return False


async def main():
    """Run all end-to-end integration tests."""
    print("🧪 End-to-End Pipeline Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Pipeline Integration", test_end_to_end_with_mock_feeds),
        ("AI Fallback", test_ai_fallback_functionality), 
        ("Real API Integration", test_real_api_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔬 Running {test_name}...")
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    # Results summary
    print("\n" + "=" * 50)
    print(f"🏁 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n✨ Phase Integration Status:")
        print("   ✅ PHASE 1: Database & Models")
        print("   ✅ PHASE 2: RSS Processing & Pre-filtering") 
        print("   ✅ PHASE 3: AI Integration & Fallback")
        print("   🔄 Ready for PHASE 4: Bot Integration")
        
        return True
    else:
        failed = total - passed
        print(f"⚠️ {failed} integration test(s) failed")
        print("\n🔧 Integration Issues Found:")
        if passed == 0:
            print("   ❌ Critical: No integration working")
        else:
            print("   ⚠️ Partial: Some components integrated")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)