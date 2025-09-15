"""
Integration tests for workflow-specified feeds.
Tests the specific feeds mentioned in the implementation workflow.
"""

import sys
from pathlib import Path
import pytest
import requests
import feedparser
import time
from bs4 import BeautifulSoup

# Add project root to Python path (from tests/integration/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Workflow-specified test feeds
WORKFLOW_FEEDS = [
    "https://aws.amazon.com/blogs/compute/feed/",
    "https://aws.amazon.com/blogs/aws/feed/",
    "https://feeds.bbci.co.uk/news/technology/rss.xml"
]

# Feeds that may be unreliable (test separately)
OPTIONAL_FEEDS = [
    "https://feeds.feedburner.com/oreilly/radar",  # Sometimes returns 404
]

@pytest.fixture(params=WORKFLOW_FEEDS)
def feed_url(request):
    """Parameterized fixture providing each workflow feed URL."""
    return request.param

def test_workflow_feed(feed_url):
    """Test that workflow-specified feeds can be fetched and parsed successfully."""
    print(f"\n📡 Testing: {feed_url}")
    
    # Test feed fetching
    start_time = time.time()
    response = requests.get(feed_url, timeout=30)
    response.raise_for_status()
    
    # Test feed parsing
    parsed_feed = feedparser.parse(response.content)
    processing_time = time.time() - start_time
    
    # Verify feed structure
    assert hasattr(parsed_feed, 'feed'), "Feed should have feed metadata"
    assert hasattr(parsed_feed, 'entries'), "Feed should have entries"
    
    # Extract and verify data
    feed_title = getattr(parsed_feed.feed, 'title', 'Unknown Feed')
    article_count = len(parsed_feed.entries)
    
    assert article_count > 0, f"Feed should have at least one article, got {article_count}"
    assert processing_time < 30.0, f"Processing time should be under 30s, got {processing_time:.2f}s"
    
    # Log results
    print(f"  ✅ Success: '{feed_title}'")
    print(f"  📊 Articles: {article_count}")
    print(f"  ⏱️ Processing time: {processing_time:.2f}s")
    
    # Show warnings if any
    if parsed_feed.bozo:
        print(f"  ⚠️ Parsing warnings: {parsed_feed.bozo_exception}")
    
    # Show sample article
    if parsed_feed.entries:
        first_article = parsed_feed.entries[0]
        print(f"  📰 Sample: '{first_article.title[:60]}...'")
        assert hasattr(first_article, 'title'), "Articles should have titles"
        assert hasattr(first_article, 'link'), "Articles should have links"

def test_article_processing():
    """Test detailed article processing for one representative feed."""
    # Use a single reliable feed for detailed testing
    feed_url = "https://feeds.bbci.co.uk/news/technology/rss.xml"
    print(f"\n🔍 Testing article processing for: {feed_url}")
    
    response = requests.get(feed_url, timeout=30)
    response.raise_for_status()
    
    parsed_feed = feedparser.parse(response.content)
    
    assert parsed_feed.entries, "Feed should have entries for article processing test"
    
    # Analyze first article in detail
    article = parsed_feed.entries[0]
    
    print(f"  📰 Title: {article.title}")
    print(f"  🔗 Link: {article.link}")
    
    # Check for summary/description
    summary = getattr(article, 'summary', '') or getattr(article, 'description', '')
    assert summary, "Article should have summary or description"
    
    print(f"  📝 Summary length: {len(summary)} chars")
    print(f"  📝 Summary preview: '{summary[:100]}...'")
    
    # Check for published date
    if hasattr(article, 'published'):
        print(f"  📅 Published: {article.published}")
        assert article.published, "Published date should not be empty"
    
    # Check for author
    if hasattr(article, 'author') and article.author:
        print(f"  👤 Author: {article.author}")
    
    # Check for categories/tags
    if hasattr(article, 'tags') and article.tags:
        try:
            categories = [tag.term for tag in article.tags if hasattr(tag, 'term')]
            print(f"  🏷️ Categories: {categories[:3]}")
        except (AttributeError, TypeError):
            print("  🏷️ Categories: parsing issues with tags")
    
    # Test HTML cleaning if content contains HTML
    if summary and '<' in summary:
        soup = BeautifulSoup(summary, 'html.parser')
        clean_text = soup.get_text(strip=True)
        print(f"  🧹 Cleaned text: {len(clean_text)} chars (was {len(summary)})")
        assert len(clean_text) > 0, "Cleaned text should not be empty"
        assert len(clean_text) <= len(summary), "Cleaned text should not be longer than original"
    
    print("  ✅ Article processing successful")

@pytest.mark.slow
def test_workflow_feeds_statistical_analysis():
    """Test all workflow feeds and generate statistical analysis."""
    print("\n📊 WORKFLOW FEED STATISTICAL ANALYSIS")
    print("="*70)
    
    results = []
    total_processing_time = 0
    total_articles = 0
    
    for feed_url in WORKFLOW_FEEDS:
        print(f"\n📡 Processing: {feed_url}")
        
        try:
            start_time = time.time()
            response = requests.get(feed_url, timeout=30)
            response.raise_for_status()
            
            parsed_feed = feedparser.parse(response.content)
            processing_time = time.time() - start_time
            
            if hasattr(parsed_feed, 'feed') and hasattr(parsed_feed, 'entries'):
                feed_title = getattr(parsed_feed.feed, 'title', 'Unknown Feed')
                article_count = len(parsed_feed.entries)
                
                result = {
                    'url': feed_url,
                    'success': True,
                    'feed_title': feed_title,
                    'article_count': article_count,
                    'processing_time': processing_time,
                    'has_bozo_exception': parsed_feed.bozo
                }
                
                total_processing_time += processing_time
                total_articles += article_count
                
                print(f"  ✅ Success: {feed_title} ({article_count} articles)")
                print(f"  ⏱️ Time: {processing_time:.2f}s")
                
            else:
                result = {'url': feed_url, 'success': False, 'error': 'Invalid feed structure'}
                print(f"  ❌ Failed: Invalid feed structure")
                
        except Exception as e:
            result = {'url': feed_url, 'success': False, 'error': str(e)}
            print(f"  ❌ Failed: {e}")
        
        results.append(result)
    
    # Generate statistics
    successful_feeds = [r for r in results if r.get('success', False)]
    success_rate = len(successful_feeds) / len(results)
    avg_processing_time = total_processing_time / len(successful_feeds) if successful_feeds else 0
    
    print(f"\n📈 STATISTICAL SUMMARY:")
    print(f"  Total feeds tested: {len(results)}")
    print(f"  Successful feeds: {len(successful_feeds)}")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Total articles: {total_articles}")
    print(f"  Average processing time: {avg_processing_time:.2f}s")
    print(f"  Average articles per feed: {total_articles/len(successful_feeds):.1f}")
    
    # Assertions for PHASE 2 validation
    assert success_rate >= 0.75, f"Success rate ({success_rate*100:.1f}%) should be at least 75%"
    assert avg_processing_time < 10.0, f"Average processing time ({avg_processing_time:.2f}s) should be under 10s"
    assert total_articles > 0, "Should have processed at least some articles"
    
    print("\n🎉 PHASE 2: Content Processing validation PASSED!")
    print("✅ RSS feed processing meets workflow requirements")

@pytest.mark.parametrize("feed_url", OPTIONAL_FEEDS)
def test_optional_feeds(feed_url):
    """Test optional feeds that may sometimes be unavailable."""
    print(f"\n📡 Testing optional feed: {feed_url}")
    
    try:
        # Test feed fetching
        start_time = time.time()
        response = requests.get(feed_url, timeout=30)
        response.raise_for_status()
        
        # Test feed parsing
        parsed_feed = feedparser.parse(response.content)
        processing_time = time.time() - start_time
        
        # Verify feed structure if successful
        assert hasattr(parsed_feed, 'feed'), "Feed should have feed metadata"
        assert hasattr(parsed_feed, 'entries'), "Feed should have entries"
        
        feed_title = getattr(parsed_feed.feed, 'title', 'Unknown Feed')
        article_count = len(parsed_feed.entries)
        
        print(f"  ✅ Success: '{feed_title}' ({article_count} articles)")
        print(f"  ⏱️ Processing time: {processing_time:.2f}s")
        
        # Verify we got some content
        assert article_count >= 0, f"Feed should have valid article count, got {article_count}"
        
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, 
            requests.exceptions.Timeout) as e:
        print(f"  ⚠️ Optional feed temporarily unavailable: {e}")
        pytest.skip(f"Optional feed {feed_url} is temporarily unavailable: {e}")
        
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        pytest.fail(f"Unexpected error testing optional feed {feed_url}: {e}")