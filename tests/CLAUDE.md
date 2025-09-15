# CuliFeed Testing Guidelines

**Purpose**: Specialized testing practices and patterns for CuliFeed RSS feed processing system

## Test Architecture Decision Tree

### When to Use Each Test Type

**🔬 Unit Tests (`tests/unit/`)**
- Testing isolated components and functions
- Pydantic model validation
- Database operations with mocked external dependencies
- Fast execution requirements (<5s total)
- **Example**: `test_database_models.py`, `test_feed_manager.py`

**🌐 Integration Tests (`tests/integration/`)**  
- End-to-end workflows with external dependencies
- Real RSS feed processing
- Network request testing
- Performance measurement and statistical analysis
- **Example**: `test_real_feeds.py`, `test_workflow_feeds.py`

**⚡ Simple Validation Tests**
- Quick functionality checks without full system setup
- Dependency-free core logic validation  
- CI/CD pipeline smoke tests
- **Example**: `test_feed_simple.py`, `test_feed_implementation.py`

## Established Testing Patterns

### 🏗️ Fixture Usage Patterns

**Database Testing**
```python
@pytest.fixture
def test_database():
    """Create temporary database with schema."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    schema = DatabaseSchema(db_path)
    schema.create_tables()
    
    yield db_path
    
    # Always cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass
```

**Sample Data Strategy**
- Use `sample_*()` fixtures for consistent test data
- Include edge cases: malformed XML, missing fields, encoding issues
- Maintain both RSS and Atom format samples

### 🎭 Mock Strategy Guidelines

**Mock External APIs When**:
- Testing error handling and edge cases
- Ensuring deterministic test results
- Testing network failure scenarios
- Unit testing isolated components

**Use Real APIs When**:
- Integration testing workflows
- Validating actual feed parsing behavior
- Performance measurement
- End-to-end validation

**Example Mock Pattern**:
```python
@patch('requests.get')
def test_feed_network_error(self, mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
    
    with pytest.raises(FeedFetchError):
        fetch_single_feed("https://example.com/feed.xml")
```

### ⚡ Async Testing Standards

**Use `asyncio.run()` for Integration Tests**:
```python
def test_concurrent_feeds(self):
    feeds = ["https://feed1.com", "https://feed2.com"]
    
    async def run_test():
        results = await fetch_feeds_batch(feeds)
        return results
    
    results = asyncio.run(run_test())
    assert len(results) == 2
```

**Mock Async Dependencies**:
```python
@patch('aiohttp.ClientSession.get', new_callable=AsyncMock)
async def test_async_feed_fetch(self, mock_get):
    mock_response = AsyncMock()
    mock_response.text = AsyncMock(return_value=SAMPLE_RSS_FEED)
    mock_get.return_value.__aenter__.return_value = mock_response
```

## RSS Feed-Specific Testing

### 📡 Feed Format Coverage

**Always Test Both Formats**:
- RSS 2.0 with standard elements
- Atom 1.0 with namespace handling
- Malformed XML resilience
- Character encoding edge cases

**Required Test Data**:
```python
SAMPLE_RSS_FEED = '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Test Feed</title>
        <item>
            <title>Article Title</title>
            <description><![CDATA[HTML content]]></description>
            <pubDate>Thu, 05 Sep 2024 12:00:00 GMT</pubDate>
        </item>
    </channel>
</rss>'''
```

### 🛡️ Security Testing Requirements

**HTML Content Cleaning**:
```python
def test_xss_prevention(self):
    malicious_content = '<script>alert("xss")</script><p>Safe content</p>'
    cleaner = ContentCleaner()
    
    cleaned = cleaner.clean_html_content(malicious_content)
    
    assert 'script' not in cleaned.lower()
    assert 'Safe content' in cleaned
```

**URL Validation**:
```python
def test_url_security(self):
    dangerous_urls = [
        'javascript:alert(1)',
        'data:text/html,<script>alert(1)</script>',
        'file:///etc/passwd'
    ]
    
    for url in dangerous_urls:
        with pytest.raises(ValidationError):
            URLValidator.validate_feed_url(url)
```

## Quality Standards

### 📋 Test Documentation Requirements

**Test Class Docstrings**:
```python
class TestFeedManager:
    """Test RSS feed parsing and content extraction.
    
    Covers:
    - Feed metadata extraction
    - Article parsing with HTML cleaning
    - Error handling for malformed feeds
    - Performance measurement
    """
```

**Test Method Docstrings**:
```python
def test_feed_parsing_with_encoding_issues(self):
    """Test feed parsing with various character encodings.
    
    Validates proper handling of UTF-8, Latin-1, and Windows-1252
    encodings commonly found in RSS feeds.
    """
```

### ⏱️ Performance Testing Integration

**Always Measure Critical Paths**:
```python
def test_feed_processing_performance(self):
    start_time = time.time()
    
    feed_metadata, articles = fetch_single_feed(test_url)
    
    processing_time = time.time() - start_time
    
    assert processing_time < 5.0  # 5 second max
    assert len(articles) > 0
    self.logger.info(f"Processed {len(articles)} articles in {processing_time:.2f}s")
```

### 📊 Statistical Analysis Patterns

**For Integration Tests with Real Feeds**:
```python
def generate_test_report(self, results):
    total_feeds = len(results['successful']) + len(results['failed'])
    success_rate = len(results['successful']) / total_feeds * 100
    
    avg_processing_time = sum(results['times']) / len(results['times'])
    
    assert success_rate >= 75.0  # Minimum acceptable success rate
    assert avg_processing_time < 3.0  # Maximum acceptable processing time
```

## Error Testing Excellence

### 🚨 Comprehensive Error Scenarios

**Network Errors**:
```python
PROBLEMATIC_FEEDS = [
    "https://nonexistent-domain-12345.com/feed.xml",  # DNS failure
    "https://httpbin.org/status/404",  # 404 Not Found
    "https://httpbin.org/status/500",  # Server Error
    "https://httpbin.org/delay/30",    # Timeout
]
```

**Malformed Content**:
```python
MALFORMED_FEEDS = [
    "<?xml version='1.0'?><rss><channel><title>No closing tags",
    "Not XML content at all",
    "<rss version='2.0'><channel></rss>",  # Invalid structure
]
```

### 🔄 Exception Testing Pattern

```python
def test_exception_details(self):
    with pytest.raises(FeedFetchError) as exc_info:
        fetch_single_feed("https://invalid-url")
    
    error = exc_info.value
    assert error.error_code == ErrorCode.FEED_NETWORK_ERROR
    assert "invalid-url" in error.context['url']
    assert error.recoverable is True
```

## Cleanup and Resource Management

### 🧹 Automatic Cleanup Patterns

**Database Cleanup**:
```python
@pytest.fixture
def clean_database(test_database):
    yield test_database
    
    # Clean up test data
    with sqlite3.connect(test_database) as conn:
        conn.execute("DELETE FROM articles")
        conn.execute("DELETE FROM feeds")
        conn.commit()
```

**Temporary File Management**:
```python
def test_with_temp_files(self, tmp_path):
    test_file = tmp_path / "test_feed.xml"
    test_file.write_text(SAMPLE_RSS_FEED)
    
    # Test logic here
    
    # Automatic cleanup via tmp_path fixture
```

## Test Environment Configuration

### 🔧 Environment Setup

**Required Environment Variables** (set in `conftest.py`):
```python
os.environ['CULIFEED_TELEGRAM__BOT_TOKEN'] = 'test-token'
os.environ['CULIFEED_AI__GEMINI_API_KEY'] = 'test-key'
os.environ['CULIFEED_DEBUG'] = 'true'
```

### 🎯 Test Execution Guidelines

**Run Test Subsets**:
```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests (slow, external dependencies)  
pytest tests/integration/ -v

# Specific test patterns
pytest tests/ -k "feed_manager" -v
```

**Performance Testing**:
```bash
# With timing output
pytest tests/integration/ -v --durations=10
```

## Decision Matrix Summary

| Scenario | Test Type | Use Mocks | Real APIs | Complexity |
|----------|-----------|-----------|-----------|------------|
| Model validation | Unit | ✅ | ❌ | Simple |
| Feed parsing logic | Unit | ✅ | ❌ | Medium |
| Error handling | Unit | ✅ | ❌ | Medium |
| End-to-end workflow | Integration | ❌ | ✅ | High |
| Performance validation | Integration | ❌ | ✅ | High |
| Quick smoke test | Simple | Minimal | ❌ | Low |

---

**Remember**: The goal is maintainable, comprehensive testing that provides confidence in RSS feed processing reliability while being efficient to run and modify.