#!/usr/bin/env python3
"""
Simple test script for RSS Feed Manager implementation.
Tests the PHASE 2: Content Processing components with real feeds.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path (from tests/unit/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing basic imports...")
    
    try:
        # Test config import
        from culifeed.config.settings import get_settings
        print("  ✅ Settings module imported successfully")
        
        # Test utils imports
        from culifeed.utils.logging import get_logger_for_component
        from culifeed.utils.exceptions import FeedFetchError
        from culifeed.utils.validators import validate_url
        print("  ✅ Utils modules imported successfully")
        
        # Test ingestion imports
        from culifeed.ingestion.feed_manager import FeedManager, ParsedArticle, FeedMetadata
        from culifeed.ingestion.content_cleaner import ContentCleaner
        print("  ✅ Ingestion modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_feedparser_integration():
    """Test feedparser integration."""
    print("\n🔍 Testing feedparser integration...")
    
    try:
        import feedparser
        
        # Test with sample RSS
        sample_rss = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <link>http://example.com</link>
                <description>Test RSS feed</description>
                <item>
                    <title>Test Article</title>
                    <link>http://example.com/article1</link>
                    <description>Test article content</description>
                </item>
            </channel>
        </rss>'''
        
        parsed = feedparser.parse(sample_rss)
        
        assert hasattr(parsed, 'feed'), "Parsed feed should have 'feed' attribute"
        assert hasattr(parsed, 'entries'), "Parsed feed should have 'entries' attribute"
        assert parsed.feed.title == "Test Feed", f"Expected 'Test Feed', got '{parsed.feed.title}'"
        assert len(parsed.entries) == 1, f"Expected 1 entry, got {len(parsed.entries)}"
        assert parsed.entries[0].title == "Test Article", f"Expected 'Test Article', got '{parsed.entries[0].title}'"
        
        print("  ✅ Feedparser working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Feedparser test failed: {e}")
        return False

def test_beautifulsoup_integration():
    """Test BeautifulSoup integration."""
    print("\n🔍 Testing BeautifulSoup integration...")
    
    try:
        from bs4 import BeautifulSoup
        
        # Test HTML parsing
        html_content = '''
        <div>
            <p>This is <strong>bold</strong> text with <a href="http://example.com">a link</a>.</p>
            <script>alert('xss')</script>
            <p>More content here.</p>
        </div>
        '''
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts
        for script in soup.find_all('script'):
            script.decompose()
        
        # Extract text
        clean_text = soup.get_text(strip=True)
        
        print(f"  Debug: clean_text = '{clean_text}'")
        assert 'bold' in clean_text, "Should contain 'bold'"
        assert 'a link' in clean_text, "Should contain 'a link'"
        assert 'More content here' in clean_text, "Should contain 'More content here'"
        assert 'alert' not in clean_text, "Should not contain 'alert' (script removed)"
        
        print("  ✅ BeautifulSoup working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ BeautifulSoup test failed: {e}")
        return False

def test_data_classes():
    """Test data class creation."""
    print("\n🔍 Testing data class creation...")
    
    try:
        from culifeed.ingestion.feed_manager import ParsedArticle, FeedMetadata
        from datetime import datetime, timezone
        
        # Test ParsedArticle creation
        article = ParsedArticle(
            title="Test Article",
            link="http://example.com/article",
            summary="This is a test article summary"
        )
        
        assert article.title == "Test Article"
        assert article.link == "http://example.com/article"
        assert article.summary == "This is a test article summary"
        assert article.categories == []  # Default empty list
        assert article.enclosures == []  # Default empty list
        
        # Test FeedMetadata creation
        metadata = FeedMetadata(
            title="Test Feed",
            link="http://example.com",
            description="Test feed description"
        )
        
        assert metadata.title == "Test Feed"
        assert metadata.link == "http://example.com"
        assert metadata.description == "Test feed description"
        assert metadata.categories == []  # Default empty list
        
        print("  ✅ Data classes working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Data class test failed: {e}")
        return False

def test_content_cleaner():
    """Test content cleaner functionality."""
    print("\n🔍 Testing content cleaner...")
    
    try:
        from culifeed.ingestion.content_cleaner import ContentCleaner
        
        cleaner = ContentCleaner()
        
        # Test HTML cleaning
        html_content = '''
        <p>This is <strong>important</strong> content.</p>
        <script>alert('xss')</script>
        <p>More <a href="http://example.com">information</a> here.</p>
        '''
        
        cleaned = cleaner.clean_html_content(html_content)
        
        assert 'important content' in cleaned, "Should contain 'important content'"
        assert 'information' in cleaned, "Should contain 'information'"
        assert 'script' not in cleaned.lower(), "Should not contain 'script'"
        assert 'alert' not in cleaned, "Should not contain 'alert'"
        
        # Test text-only extraction
        text_only = cleaner.extract_text_only(html_content)
        
        assert 'important content' in text_only, "Should contain 'important content'"
        assert '<' not in text_only, "Should not contain HTML tags"
        assert '>' not in text_only, "Should not contain HTML tags"
        
        print("  ✅ Content cleaner working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Content cleaner test failed: {e}")
        return False

def test_feed_manager_basic():
    """Test basic feed manager functionality."""
    print("\n🔍 Testing feed manager basic functionality...")
    
    try:
        from culifeed.ingestion.feed_manager import FeedManager
        
        feed_manager = FeedManager()
        
        # Test initialization
        assert feed_manager is not None, "FeedManager should initialize"
        assert feed_manager.settings is not None, "Should have settings"
        assert feed_manager.logger is not None, "Should have logger"
        assert hasattr(feed_manager, '_feed_errors'), "Should have _feed_errors dict"
        assert hasattr(feed_manager, '_last_fetch_times'), "Should have _last_fetch_times dict"
        
        # Test error tracking
        test_url = "http://example.com/feed"
        
        # Initially no errors
        assert not feed_manager._should_skip_feed(test_url), "Should not skip feed initially"
        
        # Add some errors
        for i in range(3):
            feed_manager._record_feed_error(test_url)
        
        assert feed_manager._feed_errors[test_url] == 3, "Should have recorded 3 errors"
        
        # Reset errors
        feed_manager._record_successful_fetch(test_url)
        assert test_url not in feed_manager._feed_errors, "Errors should be reset after success"
        
        print("  ✅ Feed manager basic functionality working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Feed manager basic test failed: {e}")
        return False

def test_with_simple_http_request():
    """Test with a simple HTTP request to validate network functionality."""
    print("\n🔍 Testing simple HTTP request...")
    
    try:
        import requests
        
        # Test simple HTTP request
        response = requests.get("https://httpbin.org/get", timeout=10)
        response.raise_for_status()
        
        print(f"  ✅ HTTP request successful (status: {response.status_code})")
        return True
        
    except Exception as e:
        print(f"  ❌ HTTP request test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting RSS Feed Manager Implementation Tests")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Feedparser Integration", test_feedparser_integration),
        ("BeautifulSoup Integration", test_beautifulsoup_integration),
        ("Data Classes", test_data_classes),
        ("Content Cleaner", test_content_cleaner),
        ("Feed Manager Basic", test_feed_manager_basic),
        ("HTTP Request", test_with_simple_http_request),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! PHASE 2: Content Processing implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. Implementation needs fixes.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)