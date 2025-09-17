"""
Article Processing and Deduplication
===================================

Article content normalization, deduplication, and quality scoring
before AI processing.
"""

import re
import hashlib
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, parse_qs
import html

from ..database.models import Article
from ..database.connection import DatabaseConnection
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import ProcessingError, ErrorCode
from ..utils.validators import ContentValidator


@dataclass
class ProcessingResult:
    """Result of article processing."""
    article: Article
    is_duplicate: bool
    duplicate_of: Optional[str] = None
    quality_score: float = 0.0
    content_issues: List[str] = None
    normalized: bool = False
    
    def __post_init__(self):
        if self.content_issues is None:
            self.content_issues = []


@dataclass
class DeduplicationStats:
    """Statistics from deduplication process."""
    total_articles: int
    unique_articles: int
    duplicates_found: int
    duplicates_by_hash: int
    duplicates_by_url: int
    duplicates_by_content: int
    
    @property
    def deduplication_rate(self) -> float:
        """Percentage of articles that were duplicates."""
        if self.total_articles == 0:
            return 0.0
        return (self.duplicates_found / self.total_articles) * 100


class ArticleProcessor:
    """Article content processing and deduplication."""
    
    def __init__(self, db_connection: DatabaseConnection, max_content_length: int = 2000):
        """Initialize article processor.
        
        Args:
            db_connection: Database connection manager
            max_content_length: Maximum content length for processing
        """
        self.db = db_connection
        self.logger = get_logger_for_component("article_processor")
        
        # Content quality thresholds
        self.min_title_length = 10
        self.min_content_length = 50
        self.max_content_length = max_content_length
    
    def normalize_content(self, article: Article) -> Article:
        """Normalize article content for consistent processing.
        
        Args:
            article: Article to normalize
            
        Returns:
            Article with normalized content
        """
        # Create a copy to avoid modifying original
        normalized_article = Article(
            id=article.id,
            title=article.title,
            url=article.url,
            content=article.content,
            published_at=article.published_at,
            source_feed=article.source_feed,
            content_hash=article.content_hash,
            created_at=article.created_at
        )
        
        # Normalize title
        if normalized_article.title:
            # Decode HTML entities
            normalized_article.title = html.unescape(normalized_article.title)
            # Clean whitespace
            normalized_article.title = re.sub(r'\s+', ' ', normalized_article.title).strip()
            # Remove common prefixes/suffixes
            normalized_article.title = self._clean_title(normalized_article.title)
        
        # Normalize content
        if normalized_article.content:
            # Decode HTML entities
            normalized_article.content = html.unescape(normalized_article.content)
            # Remove HTML tags
            normalized_article.content = re.sub(r'<[^>]+>', ' ', normalized_article.content)
            # Clean whitespace and normalize line breaks
            normalized_article.content = re.sub(r'\s+', ' ', normalized_article.content).strip()
            # Truncate to max length
            if len(normalized_article.content) > self.max_content_length:
                normalized_article.content = normalized_article.content[:self.max_content_length] + "..."
        
        # Normalize URL (remove tracking parameters)
        normalized_article.url = self._normalize_url(str(normalized_article.url))
        
        # Regenerate content hash with normalized content
        content_for_hash = f"{normalized_article.title}|{normalized_article.url}".encode('utf-8')
        normalized_article.content_hash = hashlib.sha256(content_for_hash).hexdigest()
        
        return normalized_article
    
    def _clean_title(self, title: str) -> str:
        """Clean common title prefixes and suffixes.

        Args:
            title: Original title

        Returns:
            Cleaned title
        """
        # Common patterns to remove
        patterns_to_remove = [
            r'^\[.*?\]\s*',  # [Category] prefix
            r'\s*\|\s*.*$',  # | Site name suffix
            r'^\w+:\s*',     # Category: prefix
        ]

        cleaned = title
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned).strip()

        # More selective dash pattern - only remove if it looks like a site name
        # (common site names after dash, not technical terms)
        dash_site_patterns = [
            r'\s*-\s*(TechCrunch|Wired|Ars Technica|The Verge|Engadget|Gizmodo|Mashable|ZDNet|CNET|VentureBeat|ReadWrite|TechRadar).*$',
            r'\s*-\s*(AWS|Amazon Web Services|Microsoft|Google|Apple|IBM|Oracle|Red Hat|VMware).*$',
            r'\s*-\s*(Blog|News|Website|Site|Portal|Magazine|Daily|Weekly|Times|Post|Herald).*$'
        ]

        # Only apply dash removal for very specific site name patterns
        for pattern in dash_site_patterns:
            if len(cleaned) > 40:  # Only for longer titles
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        return cleaned if cleaned else title
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing tracking parameters.
        
        Args:
            url: Original URL
            
        Returns:
            Normalized URL
        """
        try:
            parsed = urlparse(url)
            
            # Remove common tracking parameters
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'ref', 'referer', 'source', 'campaign', 'medium',
                'fbclid', 'gclid', 'dclid', '_ga', '_gl'
            }
            
            if parsed.query:
                query_params = parse_qs(parsed.query)
                clean_params = {k: v for k, v in query_params.items() if k not in tracking_params}
                
                # Rebuild query string
                if clean_params:
                    from urllib.parse import urlencode
                    clean_query = urlencode(clean_params, doseq=True)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{clean_query}"
                else:
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                # Add fragment if present
                if parsed.fragment:
                    clean_url += f"#{parsed.fragment}"
                
                return clean_url
            
            return url
            
        except Exception:
            # If URL parsing fails, return original
            return url
    
    def calculate_quality_score(self, article: Article) -> float:
        """Calculate content quality score for an article.
        
        Args:
            article: Article to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Title quality (0.3 weight)
        if article.title:
            title_len = len(article.title)
            if title_len >= self.min_title_length:
                title_score = min(title_len / 100.0, 1.0)  # Normalize to 100 chars
                score += title_score * 0.3
        
        # Content quality (0.5 weight)
        if article.content:
            content_len = len(article.content)
            if content_len >= self.min_content_length:
                content_score = min(content_len / 1000.0, 1.0)  # Normalize to 1000 chars
                score += content_score * 0.5
        
        # Publication date (0.1 weight)
        if article.published_at:
            # Newer articles get higher scores
            age_hours = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
            # Give full points for articles less than 24 hours old
            recency_score = max(0.0, 1.0 - (age_hours / (24 * 7)))  # Decay over a week
            score += recency_score * 0.1
        
        # URL quality (0.1 weight)
        if article.url:
            url_str = str(article.url)
            # Penalize URLs with too many parameters (often low quality)
            param_count = url_str.count('&') + url_str.count('?')
            url_score = max(0.5, 1.0 - (param_count * 0.1))
            score += url_score * 0.1
        
        return min(score, 1.0)
    
    def find_duplicates_in_batch(self, articles: List[Article]) -> List[ProcessingResult]:
        """Find duplicates within a batch of articles.
        
        Args:
            articles: List of articles to check for duplicates
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        seen_hashes = set()
        seen_urls = set()
        
        for article in articles:
            is_duplicate = False
            duplicate_of = None
            content_issues = []
            
            # Check for hash duplicates (exact content)
            if article.content_hash in seen_hashes:
                is_duplicate = True
                duplicate_of = article.content_hash
                content_issues.append("Duplicate content hash")
            else:
                seen_hashes.add(article.content_hash)
            
            # Check for URL duplicates
            url_str = str(article.url)
            if url_str in seen_urls:
                is_duplicate = True
                duplicate_of = url_str
                content_issues.append("Duplicate URL")
            else:
                seen_urls.add(url_str)
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(article)
            
            # Check for quality issues
            if article.title and len(article.title) < self.min_title_length:
                content_issues.append("Title too short")
            
            if article.content and len(article.content) < self.min_content_length:
                content_issues.append("Content too short")
            
            results.append(ProcessingResult(
                article=article,
                is_duplicate=is_duplicate,
                duplicate_of=duplicate_of,
                quality_score=quality_score,
                content_issues=content_issues,
                normalized=True
            ))
        
        return results
    
    def find_duplicates_in_database(self, articles: List[Article], days_lookback: int = 7) -> List[ProcessingResult]:
        """Find duplicates against existing database articles.
        
        Args:
            articles: List of articles to check
            days_lookback: Days to look back for duplicates
            
        Returns:
            List of ProcessingResult objects
        """
        if not articles:
            return []
        
        # Get recent articles from database for comparison
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_lookback)
        
        with self.db.get_connection() as conn:
            # Get existing article hashes and URLs
            existing_data = conn.execute("""
                SELECT content_hash, url FROM articles 
                WHERE created_at >= ?
            """, (cutoff_date,)).fetchall()
            
            existing_hashes = {row['content_hash'] for row in existing_data}
            existing_urls = {row['url'] for row in existing_data}
        
        results = []
        for article in articles:
            is_duplicate = False
            duplicate_of = None
            content_issues = []
            
            # Check against database
            if article.content_hash in existing_hashes:
                is_duplicate = True
                duplicate_of = f"db_hash:{article.content_hash}"
                content_issues.append("Duplicate in database (content)")
            
            url_str = str(article.url)
            if url_str in existing_urls:
                is_duplicate = True
                duplicate_of = f"db_url:{url_str}"
                content_issues.append("Duplicate in database (URL)")
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(article)
            
            results.append(ProcessingResult(
                article=article,
                is_duplicate=is_duplicate,
                duplicate_of=duplicate_of,
                quality_score=quality_score,
                content_issues=content_issues,
                normalized=True
            ))
        
        return results
    
    def process_articles(self, articles: List[Article], check_database: bool = True) -> Tuple[List[Article], DeduplicationStats]:
        """Process articles with normalization and deduplication.
        
        Args:
            articles: List of articles to process
            check_database: Whether to check for duplicates in database
            
        Returns:
            Tuple of (unique_articles, deduplication_stats)
        """
        if not articles:
            return [], DeduplicationStats(0, 0, 0, 0, 0, 0)
        
        self.logger.info(f"Processing {len(articles)} articles for deduplication")
        
        # Step 1: Normalize content
        normalized_articles = [self.normalize_content(article) for article in articles]
        
        # Step 2: Find duplicates within batch
        batch_results = self.find_duplicates_in_batch(normalized_articles)
        
        # Step 3: Check against database if requested
        if check_database:
            db_results = self.find_duplicates_in_database(normalized_articles)
            
            # Merge results (database duplicates take precedence)
            final_results = []
            for i, article in enumerate(normalized_articles):
                batch_result = batch_results[i]
                db_result = db_results[i]
                
                if db_result.is_duplicate:
                    final_results.append(db_result)
                else:
                    final_results.append(batch_result)
        else:
            final_results = batch_results
        
        # Step 4: Extract unique articles
        unique_articles = []
        duplicates_by_hash = 0
        duplicates_by_url = 0
        duplicates_by_content = 0
        
        for result in final_results:
            if not result.is_duplicate:
                unique_articles.append(result.article)
            else:
                # Count duplicate types
                if "content hash" in ' '.join(result.content_issues).lower():
                    duplicates_by_hash += 1
                elif "url" in ' '.join(result.content_issues).lower():
                    duplicates_by_url += 1
                else:
                    duplicates_by_content += 1
        
        # Step 5: Generate statistics
        total_duplicates = len(articles) - len(unique_articles)
        stats = DeduplicationStats(
            total_articles=len(articles),
            unique_articles=len(unique_articles),
            duplicates_found=total_duplicates,
            duplicates_by_hash=duplicates_by_hash,
            duplicates_by_url=duplicates_by_url,
            duplicates_by_content=duplicates_by_content
        )
        
        # Log results
        self.logger.info(
            f"Article processing complete: {len(unique_articles)}/{len(articles)} unique "
            f"({stats.deduplication_rate:.1f}% duplicates removed)"
        )
        
        if total_duplicates > 0:
            self.logger.info(
                f"Duplicate breakdown: {duplicates_by_hash} hash, "
                f"{duplicates_by_url} URL, {duplicates_by_content} content"
            )
        
        return unique_articles, stats
    
    def get_processing_summary(self, results: List[ProcessingResult]) -> Dict[str, any]:
        """Generate processing summary statistics.
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        total = len(results)
        duplicates = sum(1 for r in results if r.is_duplicate)
        unique = total - duplicates
        
        # Quality statistics
        quality_scores = [r.quality_score for r in results if not r.is_duplicate]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Issue statistics
        all_issues = []
        for result in results:
            all_issues.extend(result.content_issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'total_articles': total,
            'unique_articles': unique,
            'duplicate_articles': duplicates,
            'deduplication_rate': (duplicates / total) * 100 if total > 0 else 0.0,
            'average_quality_score': avg_quality,
            'common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
        }