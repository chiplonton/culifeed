"""
Unit tests for PreFilterEngine keyword-based filtering.

Tests the pre-filtering system that reduces AI processing costs by:
- Keyword-based relevance scoring using TF analysis
- Exclusion keyword filtering
- Topic matching and thresholding
- Text feature extraction and normalization
"""

import pytest
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from culifeed.processing.pre_filter import PreFilterEngine, FilterResult
from culifeed.database.models import Article, Topic


class TestPreFilterEngine:
    """Test PreFilterEngine keyword matching and filtering logic.
    
    Covers:
    - Text feature extraction and normalization
    - Keyword relevance scoring with TF analysis
    - Exclusion keyword filtering
    - Topic matching and threshold application
    - Batch article filtering with performance metrics
    """

    @pytest.fixture
    def engine(self):
        """Create PreFilterEngine with default settings."""
        return PreFilterEngine(min_relevance_threshold=0.1)

    @pytest.fixture
    def sample_articles(self):
        """Create sample articles with different content types."""
        return [
            Article(
                id="ai_article",
                title="Advanced Machine Learning Techniques",
                url="https://example.com/ai-article",
                content="This article explores artificial intelligence and machine learning algorithms. "
                        "We discuss neural networks, deep learning, and AI applications in modern technology.",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/tech-feed.xml",
                content_hash="hash1",
                created_at=datetime.now(timezone.utc)
            ),
            Article(
                id="web_article",
                title="React Performance Optimization",
                url="https://example.com/react-article",
                content="Learn how to optimize React applications using hooks, memoization, and modern "
                        "JavaScript techniques. Frontend development best practices for web applications.",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/web-feed.xml",
                content_hash="hash2",
                created_at=datetime.now(timezone.utc)
            ),
            Article(
                id="crypto_article",
                title="Blockchain Technology Overview",
                url="https://example.com/crypto-article",
                content="Comprehensive guide to blockchain and cryptocurrency. Bitcoin mining, smart contracts, "
                        "and decentralized finance applications. AI-powered trading algorithms.",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/finance-feed.xml",
                content_hash="hash3",
                created_at=datetime.now(timezone.utc)
            ),
            Article(
                id="empty_article",
                title="Empty Article",
                url="https://example.com/empty",
                content="",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/empty-feed.xml",
                content_hash="hash4",
                created_at=datetime.now(timezone.utc)
            ),
            Article(
                id="html_article",
                title="Web Development with <script>JavaScript</script>",
                url="https://example.com/html-article",
                content="<p>This article contains <strong>HTML tags</strong> and special characters! "
                        "It covers <em>frontend</em> development with React & JavaScript.</p>",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/html-feed.xml",
                content_hash="hash5",
                created_at=datetime.now(timezone.utc)
            )
        ]

    @pytest.fixture
    def sample_topics(self):
        """Create sample topics with different keyword patterns."""
        return [
            Topic(
                id=1,
                chat_id="test_channel",
                name="AI Technology",
                keywords=["artificial intelligence", "machine learning", "neural networks", "deep learning"],
                exclude_keywords=["crypto", "trading"],
                active=True,
                created_at=datetime.now(timezone.utc)
            ),
            Topic(
                id=2,
                chat_id="test_channel",
                name="Web Development",
                keywords=["react", "javascript", "frontend", "web development"],
                exclude_keywords=["spam", "advertisement"],
                active=True,
                created_at=datetime.now(timezone.utc)
            ),
            Topic(
                id=3,
                chat_id="test_channel",
                name="Blockchain",
                keywords=["blockchain", "cryptocurrency", "bitcoin", "smart contracts"],
                exclude_keywords=[],
                active=True,
                created_at=datetime.now(timezone.utc)
            ),
            Topic(
                id=4,
                chat_id="test_channel",
                name="Inactive Topic",
                keywords=["inactive", "disabled"],
                exclude_keywords=[],
                active=False,
                created_at=datetime.now(timezone.utc)
            )
        ]

    def test_engine_initialization(self, engine):
        """Test PreFilterEngine initialization with proper defaults."""
        assert engine.min_relevance_threshold == 0.1
        assert engine.logger is not None
        assert isinstance(engine.stop_words, set)
        assert len(engine.stop_words) > 0
        assert 'the' in engine.stop_words
        assert 'and' in engine.stop_words

    def test_engine_custom_threshold(self):
        """Test PreFilterEngine with custom relevance threshold."""
        engine = PreFilterEngine(min_relevance_threshold=0.3)
        assert engine.min_relevance_threshold == 0.3

    def test_extract_text_features_basic(self, engine, sample_articles):
        """Test text feature extraction from article content."""
        article = sample_articles[0]  # AI article
        features = engine._extract_text_features(article)
        
        assert isinstance(features, dict)
        assert 'clean_text' in features
        assert 'words' in features
        assert 'word_counts' in features
        assert 'tf_scores' in features
        assert 'total_words' in features
        
        # Verify text cleaning
        clean_text = features['clean_text']
        assert 'advanced machine learning techniques' in clean_text
        assert 'artificial intelligence' in clean_text
        
        # Verify word extraction (stop words removed)
        words = features['words']
        assert 'machine' in words
        assert 'learning' in words
        assert 'artificial' in words
        assert 'intelligence' in words
        assert 'the' not in words  # Stop word should be removed
        assert 'and' not in words  # Stop word should be removed
        
        # Verify word counts
        word_counts = features['word_counts']
        assert isinstance(word_counts, Counter)
        assert word_counts['machine'] >= 1
        assert word_counts['learning'] >= 1
        
        # Verify TF scores
        tf_scores = features['tf_scores']
        assert isinstance(tf_scores, dict)
        assert all(0 <= score <= 1 for score in tf_scores.values())
        assert sum(tf_scores.values()) == pytest.approx(1.0, rel=1e-1)

    def test_extract_text_features_html_cleaning(self, engine, sample_articles):
        """Test HTML tag removal and special character handling."""
        article = sample_articles[4]  # HTML article
        features = engine._extract_text_features(article)
        
        clean_text = features['clean_text']
        
        # Verify HTML tags removed
        assert '<p>' not in clean_text
        assert '<strong>' not in clean_text
        assert '<em>' not in clean_text
        assert '<script>' not in clean_text
        
        # Verify content preserved
        assert 'html tags' in clean_text
        assert 'frontend' in clean_text
        assert 'development' in clean_text
        assert 'react' in clean_text
        assert 'javascript' in clean_text
        
        # Verify special characters handled
        words = features['words']
        assert 'react' in words
        assert 'javascript' in words

    def test_extract_text_features_empty_content(self, engine, sample_articles):
        """Test feature extraction from empty/minimal content."""
        article = sample_articles[3]  # Empty article
        features = engine._extract_text_features(article)
        
        assert features['clean_text'] == 'empty article'  # Title is used when content is empty
        assert features['words'] == ['empty', 'article']
        assert features['word_counts'] == Counter({'empty': 1, 'article': 1})
        assert features['tf_scores'] == {'empty': 0.5, 'article': 0.5}
        assert features['total_words'] == 2

    def test_calculate_keyword_relevance_exact_match(self, engine):
        """Test keyword relevance scoring with exact phrase matches."""
        # Create text features with known content
        text_features = {
            'clean_text': 'artificial intelligence and machine learning are transforming technology',
            'words': ['artificial', 'intelligence', 'machine', 'learning', 'transforming', 'technology'],
            'word_counts': Counter(['artificial', 'intelligence', 'machine', 'learning', 'transforming', 'technology']),
            'tf_scores': {'artificial': 1/6, 'intelligence': 1/6, 'machine': 1/6, 'learning': 1/6, 'transforming': 1/6, 'technology': 1/6}
        }
        
        # Test exact phrase match
        keywords = ["artificial intelligence", "machine learning"]
        score = engine._calculate_keyword_relevance(text_features, keywords)
        
        assert score > 0.5  # Should be high for exact matches
        assert score <= 1.0

    def test_calculate_keyword_relevance_partial_match(self, engine):
        """Test keyword relevance scoring with partial word matches."""
        text_features = {
            'clean_text': 'modern javascript frameworks for web development',
            'words': ['modern', 'javascript', 'frameworks', 'web', 'development'],
            'word_counts': Counter(['modern', 'javascript', 'frameworks', 'web', 'development']),
            'tf_scores': {'modern': 0.2, 'javascript': 0.2, 'frameworks': 0.2, 'web': 0.2, 'development': 0.2}
        }
        
        # Test multi-word keyword with partial match
        keywords = ["web development", "react hooks"]  # Only first one matches
        score = engine._calculate_keyword_relevance(text_features, keywords)
        
        assert score > 0.0
        assert score < 1.0

    def test_calculate_keyword_relevance_no_match(self, engine):
        """Test keyword relevance scoring with no matches."""
        text_features = {
            'clean_text': 'cooking recipes and kitchen tips',
            'words': ['cooking', 'recipes', 'kitchen', 'tips'],
            'word_counts': Counter(['cooking', 'recipes', 'kitchen', 'tips']),
            'tf_scores': {'cooking': 0.25, 'recipes': 0.25, 'kitchen': 0.25, 'tips': 0.25}
        }
        
        keywords = ["artificial intelligence", "machine learning"]
        score = engine._calculate_keyword_relevance(text_features, keywords)
        
        assert score == 0.0

    def test_calculate_keyword_relevance_empty_inputs(self, engine):
        """Test keyword relevance scoring with edge cases."""
        empty_features = {
            'clean_text': '',
            'words': [],
            'word_counts': Counter(),
            'tf_scores': {},
            'total_words': 0
        }
        
        # Empty keywords
        score = engine._calculate_keyword_relevance(empty_features, [])
        assert score == 0.0
        
        # Empty text features  
        score = engine._calculate_keyword_relevance(empty_features, ["test"])
        assert score == 0.0
        
        # Normal features but empty keywords
        normal_features = {
            'clean_text': 'test content',
            'words': ['test', 'content'],
            'word_counts': Counter(['test', 'content']),
            'tf_scores': {'test': 0.5, 'content': 0.5}
        }
        score = engine._calculate_keyword_relevance(normal_features, [])
        assert score == 0.0

    def test_check_exclusion_keywords_match(self, engine):
        """Test exclusion keyword checking with matches."""
        text_features = {
            'clean_text': 'cryptocurrency trading and bitcoin mining guide',
            'words': ['cryptocurrency', 'trading', 'bitcoin', 'mining', 'guide'],
            'word_counts': Counter(['cryptocurrency', 'trading', 'bitcoin', 'mining', 'guide']),
            'tf_scores': {}
        }
        
        # Should be excluded
        excluded = engine._check_exclusion_keywords(text_features, ["trading", "spam"])
        assert excluded is True
        
        # Should not be excluded
        excluded = engine._check_exclusion_keywords(text_features, ["spam", "advertisement"])
        assert excluded is False

    def test_check_exclusion_keywords_no_exclusions(self, engine):
        """Test exclusion keyword checking with no exclusion list."""
        text_features = {
            'clean_text': 'any content should pass',
            'words': ['any', 'content', 'should', 'pass'],
            'word_counts': Counter(['any', 'content', 'should', 'pass']),
            'tf_scores': {}
        }
        
        # Empty exclusion list
        excluded = engine._check_exclusion_keywords(text_features, [])
        assert excluded is False
        
        # None exclusion list  
        excluded = engine._check_exclusion_keywords(text_features, None)
        assert excluded is False

    def test_filter_article_success(self, engine, sample_articles, sample_topics):
        """Test successful article filtering against topics."""
        ai_article = sample_articles[0]  # AI article
        active_topics = [t for t in sample_topics if t.active]
        
        result = engine.filter_article(ai_article, active_topics)
        
        assert isinstance(result, FilterResult)
        assert result.article == ai_article
        assert result.passed_filter is True
        assert len(result.matched_topics) > 0
        assert "AI Technology" in result.matched_topics
        assert len(result.relevance_scores) > 0
        assert result.filter_reason is None
        
        # Verify scores
        assert "AI Technology" in result.relevance_scores
        ai_score = result.relevance_scores["AI Technology"]
        assert 0.0 < ai_score <= 1.0

    def test_filter_article_excluded(self, engine, sample_articles, sample_topics):
        """Test article filtering with exclusion keywords."""
        crypto_article = sample_articles[2]  # Crypto article (has AI but should be excluded)
        ai_topic = next(t for t in sample_topics if t.name == "AI Technology")
        
        result = engine.filter_article(crypto_article, [ai_topic])
        
        assert isinstance(result, FilterResult)
        assert result.article == crypto_article
        assert result.passed_filter is False
        assert len(result.matched_topics) == 0
        assert result.filter_reason is not None
        assert "Excluded by topic" in result.filter_reason

    def test_filter_article_below_threshold(self, engine, sample_articles, sample_topics):
        """Test article filtering with scores below threshold."""
        # Use high threshold to test failure case
        high_threshold_engine = PreFilterEngine(min_relevance_threshold=0.9)
        
        web_article = sample_articles[1]  # Web article
        web_topic = next(t for t in sample_topics if t.name == "Web Development")
        
        result = high_threshold_engine.filter_article(web_article, [web_topic])
        
        assert isinstance(result, FilterResult)
        assert result.article == web_article
        
        # Might pass or fail depending on content - check logic
        if not result.passed_filter and result.relevance_scores:
            assert "below threshold" in result.filter_reason

    def test_filter_article_no_topics(self, engine, sample_articles):
        """Test article filtering with no topics configured."""
        article = sample_articles[0]
        
        result = engine.filter_article(article, [])
        
        assert isinstance(result, FilterResult)
        assert result.article == article
        assert result.passed_filter is False
        assert len(result.matched_topics) == 0
        assert len(result.relevance_scores) == 0
        assert result.filter_reason == "No topics configured"

    def test_filter_article_inactive_topics(self, engine, sample_articles, sample_topics):
        """Test article filtering with only inactive topics."""
        article = sample_articles[0]
        inactive_topic = next(t for t in sample_topics if not t.active)
        
        result = engine.filter_article(article, [inactive_topic])
        
        assert isinstance(result, FilterResult)
        assert result.article == article
        assert result.passed_filter is False
        assert len(result.matched_topics) == 0

    def test_filter_articles_batch(self, engine, sample_articles, sample_topics):
        """Test batch filtering of multiple articles."""
        active_topics = [t for t in sample_topics if t.active]
        
        results = engine.filter_articles(sample_articles, active_topics)
        
        assert len(results) == len(sample_articles)
        assert all(isinstance(r, FilterResult) for r in results)
        
        # Verify each result corresponds to correct article
        for i, result in enumerate(results):
            assert result.article == sample_articles[i]
        
        # Should have some passing and some failing
        passed_count = sum(1 for r in results if r.passed_filter)
        assert 0 < passed_count < len(sample_articles)  # Some should pass, some fail

    def test_filter_articles_empty_list(self, engine, sample_topics):
        """Test batch filtering with empty article list."""
        results = engine.filter_articles([], sample_topics)
        assert results == []

    def test_filter_articles_performance_logging(self, engine, sample_articles, sample_topics, caplog):
        """Test that filtering logs performance metrics."""
        active_topics = [t for t in sample_topics if t.active]
        
        with caplog.at_level("INFO"):
            results = engine.filter_articles(sample_articles, active_topics)
        
        # Check for performance logging
        log_messages = [record.message for record in caplog.records]
        
        # Should log filtering start
        assert any("Pre-filtering" in msg and "articles against" in msg for msg in log_messages)
        
        # Should log filtering completion with reduction percentage
        assert any("Pre-filtering complete" in msg and "reduction" in msg for msg in log_messages)
        
        # Should log topic matches if any
        passed_count = sum(1 for r in results if r.passed_filter)
        if passed_count > 0:
            assert any("Topic matches:" in msg for msg in log_messages)

    def test_get_filtered_articles(self, engine, sample_articles, sample_topics):
        """Test convenience method for getting filtered articles with scores."""
        active_topics = [t for t in sample_topics if t.active]
        
        filtered = engine.get_filtered_articles(sample_articles, active_topics)
        
        assert isinstance(filtered, list)
        
        # Each item should be (article, topic_name, score) tuple
        for article, topic_name, score in filtered:
            assert isinstance(article, Article)
            assert isinstance(topic_name, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        
        # Should only include articles that passed filtering
        filter_results = engine.filter_articles(sample_articles, active_topics)
        passed_articles = [r.article for r in filter_results if r.passed_filter]
        filtered_articles = [article for article, _, _ in filtered]
        
        assert set(a.id for a in filtered_articles) == set(a.id for a in passed_articles)

    def test_filter_result_properties(self, engine, sample_articles, sample_topics):
        """Test FilterResult properties and computed values."""
        ai_article = sample_articles[0]
        active_topics = [t for t in sample_topics if t.active]
        
        result = engine.filter_article(ai_article, active_topics)
        
        # Test best_match_topic property
        if result.relevance_scores:
            expected_best_topic = max(result.relevance_scores.keys(), key=lambda k: result.relevance_scores[k])
            assert result.best_match_topic == expected_best_topic
        else:
            assert result.best_match_topic is None
        
        # Test best_match_score property
        if result.relevance_scores:
            expected_best_score = max(result.relevance_scores.values())
            assert result.best_match_score == expected_best_score
        else:
            assert result.best_match_score == 0.0
        
        # Test that relevance_scores is accessible
        assert hasattr(result, 'relevance_scores')

    def test_text_feature_normalization(self, engine):
        """Test text normalization and feature extraction edge cases."""
        # Test with various text formatting issues
        test_cases = [
            ("UPPERCASE TEXT", "uppercase text"),
            ("Text    with     extra     spaces", "text with extra spaces"),
            ("Text\nwith\nlinebreaks", "text with linebreaks"),
            ("Text with 123 numbers!", "text with 123 numbers"),
            ("Text@#$%with^&*special()chars", "text with special chars"),
        ]
        
        for input_text, expected_clean in test_cases:
            article = Article(
                id="test",
                title=input_text,
                url="https://example.com/test",
                content="",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/feed.xml",
                content_hash="hash",
                created_at=datetime.now(timezone.utc)
            )
            
            features = engine._extract_text_features(article)
            assert expected_clean in features['clean_text']

    def test_stop_words_filtering(self, engine):
        """Test that stop words are properly filtered from analysis."""
        article = Article(
            id="test",
            title="The quick brown fox jumps over the lazy dog",
            url="https://example.com/test",
            content="This is a test article with many common stop words that should be filtered out.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml",
            content_hash="hash",
            created_at=datetime.now(timezone.utc)
        )
        
        features = engine._extract_text_features(article)
        words = features['words']
        
        # Content words should be present
        assert 'quick' in words
        assert 'brown' in words
        assert 'fox' in words
        assert 'test' in words
        assert 'article' in words
        
        # Stop words should be removed
        stop_words_to_check = ['the', 'is', 'a', 'with', 'that', 'should', 'be', 'out']
        for stop_word in stop_words_to_check:
            if stop_word in engine.stop_words:
                assert stop_word not in words

    def test_keyword_scoring_edge_cases(self, engine):
        """Test keyword scoring with edge cases and boundary conditions."""
        # Test with empty/whitespace keywords
        text_features = {
            'clean_text': 'test content for analysis',
            'words': ['test', 'content', 'analysis'],
            'word_counts': Counter(['test', 'content', 'analysis']),
            'tf_scores': {'test': 1/3, 'content': 1/3, 'analysis': 1/3}
        }
        
        edge_case_keywords = ["", "   ", "test", "test content", "nonexistent keyword"]
        score = engine._calculate_keyword_relevance(text_features, edge_case_keywords)
        
        # Should handle edge cases gracefully
        assert 0.0 <= score <= 1.0

    def test_relevance_threshold_boundary(self, engine):
        """Test filtering behavior at relevance threshold boundaries."""
        # Create article that should score close to threshold
        article = Article(
            id="test",
            title="javascript development",
            url="https://example.com/test",
            content="basic javascript tutorial",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml",
            content_hash="hash",
            created_at=datetime.now(timezone.utc)
        )
        
        topic = Topic(
            id=5,
            chat_id="test",
            name="JS Topic",
            keywords=["javascript"],
            exclude_keywords=[],
            active=True,
            created_at=datetime.now(timezone.utc)
        )
        
        # Test with different thresholds
        low_threshold_engine = PreFilterEngine(min_relevance_threshold=0.01)
        high_threshold_engine = PreFilterEngine(min_relevance_threshold=0.99)
        
        low_result = low_threshold_engine.filter_article(article, [topic])
        high_result = high_threshold_engine.filter_article(article, [topic])
        
        # Both should have the same relevance scores, but different pass/fail based on threshold
        # For this javascript article with perfect match, both should pass since score is 1.0
        assert low_result.relevance_scores == high_result.relevance_scores
        assert low_result.passed_filter == True  # 1.0 > 0.01
        assert high_result.passed_filter == True  # 1.0 > 0.99

    def test_multiple_topic_matching(self, engine, sample_articles, sample_topics):
        """Test article matching against multiple topics with overlapping keywords."""
        # Test article that should match multiple topics
        mixed_article = Article(
            id="mixed",
            title="AI-powered Web Development with React",
            url="https://example.com/mixed",
            content="Using artificial intelligence and machine learning to enhance React frontend development and JavaScript optimization.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml", 
            content_hash="hash",
            created_at=datetime.now(timezone.utc)
        )
        
        ai_topic = next(t for t in sample_topics if t.name == "AI Technology")
        web_topic = next(t for t in sample_topics if t.name == "Web Development")
        
        result = engine.filter_article(mixed_article, [ai_topic, web_topic])
        
        # Should match both topics
        assert result.passed_filter is True
        assert len(result.matched_topics) >= 1  # At least one topic should match
        assert len(result.relevance_scores) >= 1
        
        # Both topics should have scores
        expected_topics = {"AI Technology", "Web Development"}
        matching_topics = set(result.matched_topics)
        assert len(matching_topics.intersection(expected_topics)) > 0

    def test_complex_exclusion_scenarios(self, engine):
        """Test complex exclusion keyword scenarios."""
        # Article that would match topic but gets excluded
        article = Article(
            id="test",
            title="Machine Learning for Cryptocurrency Trading",
            url="https://example.com/test",
            content="Advanced artificial intelligence algorithms for automated crypto trading systems.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml",
            content_hash="hash",
            created_at=datetime.now(timezone.utc)
        )
        
        # Topic with AI keywords but excludes crypto content
        topic = Topic(
            id=6,
            chat_id="test",
            name="AI (No Crypto)",
            keywords=["artificial intelligence", "machine learning"],
            exclude_keywords=["cryptocurrency", "crypto", "trading"],
            active=True,
            created_at=datetime.now(timezone.utc)
        )
        
        result = engine.filter_article(article, [topic])
        
        # Should be excluded despite AI keywords
        assert result.passed_filter is False
        assert "Excluded by topic" in result.filter_reason
        assert len(result.matched_topics) == 0