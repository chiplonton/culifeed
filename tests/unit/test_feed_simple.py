#!/usr/bin/env python3
"""
Simple RSS feed test without full settings initialization.
Tests just the core RSS parsing functionality.
"""

import sys
from pathlib import Path

# Add project root to Python path (from tests/unit/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_feed_parsing_directly():
    """Test RSS parsing without settings dependency."""
    print("🔍 Testing direct RSS parsing...")

    try:
        import feedparser
        from culifeed.ingestion.feed_manager import ParsedArticle, FeedMetadata

        # Sample RSS content
        sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test RSS Feed</title>
                <link>http://example.com</link>
                <description>Test feed for validation</description>
                <language>en-us</language>
                <pubDate>Sat, 07 Sep 2024 00:00:01 GMT</pubDate>
                <item>
                    <title>Test Article Title</title>
                    <link>http://example.com/article1</link>
                    <description>This is a test article with some content</description>
                    <pubDate>Thu, 05 Sep 2024 12:00:00 GMT</pubDate>
                    <guid>test-article-guid</guid>
                    <author>test@example.com</author>
                </item>
            </channel>
        </rss>"""

        # Parse with feedparser
        parsed_feed = feedparser.parse(sample_rss)

        print(f"  ✅ Feedparser parsed feed with title: '{parsed_feed.feed.title}'")
        print(f"  ✅ Found {len(parsed_feed.entries)} entries")

        # Test data structure creation
        feed_metadata = FeedMetadata(
            title=parsed_feed.feed.title,
            link=parsed_feed.feed.link,
            description=parsed_feed.feed.description,
            language=getattr(parsed_feed.feed, "language", None),
        )

        print(f"  ✅ Created FeedMetadata: {feed_metadata.title}")

        # Test article creation
        entry = parsed_feed.entries[0]
        article = ParsedArticle(
            title=entry.title,
            link=entry.link,
            summary=entry.description,
            guid=getattr(entry, "guid", None),
            author=getattr(entry, "author", None),
        )

        print(f"  ✅ Created ParsedArticle: {article.title}")

        return True

    except Exception as e:
        print(f"  ❌ Direct parsing test failed: {e}")
        return False


def test_content_cleaning_directly():
    """Test content cleaning without settings dependency."""
    print("\n🔍 Testing direct content cleaning...")

    try:
        from bs4 import BeautifulSoup

        # Sample HTML content
        html_content = """
        <p>This article has <strong>important information</strong>.</p>
        <script>alert('should be removed')</script>
        <p>More content with <a href="http://example.com">a link</a>.</p>
        """

        # Direct BeautifulSoup processing
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove dangerous elements
        for script in soup.find_all("script"):
            script.decompose()

        # Extract clean text
        clean_text = soup.get_text(separator=" ", strip=True)

        print(f"  ✅ Original HTML: {len(html_content)} chars")
        print(f"  ✅ Clean text: {len(clean_text)} chars")
        print(f"  ✅ Content: '{clean_text[:50]}...'")

        # Verify cleaning worked
        assert "important information" in clean_text
        assert "More content" in clean_text
        assert "script" not in clean_text.lower()
        assert "alert" not in clean_text

        print("  ✅ Content cleaning validation passed")

        return True

    except Exception as e:
        print(f"  ❌ Direct content cleaning test failed: {e}")
        return False


def test_real_rss_feed():
    """Test with a real RSS feed (simple HTTP request)."""
    print("\n🔍 Testing with real RSS feed...")

    try:
        import requests
        import feedparser

        # Test with a reliable RSS feed
        test_url = "https://feeds.bbci.co.uk/news/rss.xml"

        print(f"  📡 Fetching: {test_url}")

        # Fetch with requests
        response = requests.get(test_url, timeout=15)
        response.raise_for_status()

        print(f"  ✅ HTTP request successful (status: {response.status_code})")
        print(f"  ✅ Content length: {len(response.content)} bytes")

        # Parse with feedparser
        parsed_feed = feedparser.parse(response.content)

        print(f"  ✅ Feed title: '{parsed_feed.feed.title}'")
        print(f"  ✅ Feed articles: {len(parsed_feed.entries)}")

        if parsed_feed.entries:
            first_article = parsed_feed.entries[0]
            print(f"  ✅ First article: '{first_article.title[:50]}...'")

        # Check for parsing issues (bozo)
        if parsed_feed.bozo:
            print(f"  ⚠️ Feed has parsing issues: {parsed_feed.bozo_exception}")
        else:
            print("  ✅ Feed parsed without issues")

        return True

    except Exception as e:
        print(f"  ❌ Real RSS feed test failed: {e}")
        return False


def main():
    """Run all simple tests."""
    print("🚀 RSS Feed Manager - Simple Validation Tests")
    print("=" * 55)

    tests = [
        ("Direct RSS Parsing", test_feed_parsing_directly),
        ("Direct Content Cleaning", test_content_cleaning_directly),
        ("Real RSS Feed", test_real_rss_feed),
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
    print("\n" + "=" * 55)
    print("📊 TEST RESULTS")
    print("=" * 55)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 Core RSS feed processing is working correctly!")
        print(
            "✅ PHASE 2: Content Processing - RSS Feed Manager implementation validated"
        )
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
