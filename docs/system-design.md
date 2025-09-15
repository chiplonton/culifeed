# System Design Document

## Overview

CuliFeed implements an all-local VPS architecture with AI API integration that optimizes for cost efficiency while maintaining high content quality through AI-powered semantic analysis.

## Core Architecture

### Local VPS Processing Strategy

```
📊 Content Volume Reduction Pipeline
RSS Feeds (100+ articles/day)
    ↓ Local Pre-filtering (keyword matching)
Filtered Content (~15 articles/day)  
    ↓ AI Processing (semantic analysis)
High-Confidence Content (~8 articles/day)
    ↓ Grouping & Summarization
Telegram Delivery (~3 articles per topic)
```

### Component Design

#### 1. **Content Ingestion Engine**
- **RSS Parser**: feedparser library for robust feed handling
- **Source Management**: YAML configuration for easy feed management
- **Deduplication**: Content hash-based duplicate detection
- **Error Handling**: Graceful failure with retry logic

#### 2. **Local Pre-filtering System**
- **Keyword Matching**: Multiple strategies per topic
  - Exact phrase matching
  - Fuzzy matching for variations
  - Exclusion keywords to reduce noise
- **Content Cleaning**: Strip ads, navigation, focus on article content
- **Relevance Scoring**: Basic scoring algorithm for initial filtering

#### 3. **AI Processing Pipeline**
- **Semantic Analysis**: Vector embeddings for topic similarity
- **Relevance Scoring**: LLM-based confidence assessment (0-1 scale)
- **Content Summarization**: Key points extraction (2-3 sentences)
- **Quality Assessment**: Authority, depth, and technical accuracy scoring

#### 4. **Telegram Integration**
- **Bot Commands**: Topic and feed management interface
- **Message Formatting**: Structured delivery with title, summary, link
- **Delivery Scheduling**: Daily digest with topic-based grouping
- **User Interaction**: Simple command set for configuration

## Data Models

### Topic Definition
```yaml
topics:
  - name: "AWS Lambda Performance Optimization"
    keywords: ["lambda", "performance", "optimization", "cold start"]
    exclude_keywords: ["pricing", "cost", "comparison"]
    confidence_threshold: 0.8
    
  - name: "Leadership in Tech"
    keywords: ["leadership", "management", "team", "culture"]
    exclude_keywords: ["hiring", "recruitment"]
    confidence_threshold: 0.7
```

### Article Processing Record
```python
{
    "id": "uuid",
    "title": "string", 
    "url": "string",
    "content": "string",
    "published": "datetime",
    "source_feed": "string",
    "pre_filter_score": "float",
    "ai_relevance_score": "float", 
    "confidence_score": "float",
    "summary": "string",
    "matched_topics": ["string"],
    "processed_at": "datetime"
}
```

## Processing Algorithms

### Pre-filtering Algorithm
```python
def calculate_pre_filter_score(article, topic):
    score = 0.0
    
    # Keyword matching (weighted)
    for keyword in topic.keywords:
        if keyword.lower() in article.content.lower():
            score += keyword.weight
    
    # Exclusion penalty
    for exclude in topic.exclude_keywords:
        if exclude.lower() in article.content.lower():
            score -= 0.3
    
    # Content quality signals
    score += length_score(article.content)
    score += source_authority_score(article.source)
    
    return min(score, 1.0)
```

### AI Relevance Assessment
```python
def assess_relevance(article, topic, gemini_model):
    prompt = f"""
    Topic: {topic.name}
    Article Title: {article.title}
    Article Content: {article.content[:2000]}
    
    Rate relevance (0-1) and provide 2-sentence summary.
    Focus on: Does this specifically help someone interested in {topic.name}?
    
    Format response as:
    RELEVANCE: 0.85
    SUMMARY: Article discusses Lambda cold start optimization techniques using provisioned concurrency. Provides specific performance benchmarks and cost analysis.
    """
    
    response = gemini_model.generate_content(prompt)
    return parse_gemini_response(response.text)
```

## Technology Stack

### Core Dependencies
```
feedparser==6.0.10      # RSS/Atom parsing
requests==2.31.0         # HTTP client  
python-telegram-bot==20.0  # Telegram integration
google-generativeai==0.3.0  # Gemini API client
openai==1.0.0           # Fallback LLM providers
pyyaml==6.0             # Configuration management
sqlite3                 # Built-in database
schedule==1.2.0         # Job scheduling
```

### AI Service Options
- **Primary**: Google Gemini 2.5 Flash-Lite (free tier: 1,000 requests/day)
- **Secondary**: Gemini 2.5 Flash (free tier: 250 requests/day)
- **Fallback**: Groq (free tier: 100 requests/day)
- **Premium**: OpenAI GPT-4 (paid, for enhanced accuracy if needed)

## Security & Privacy

### Data Handling
- **Local Storage**: All sensitive data stored locally in SQLite
- **API Keys**: Environment variables, never committed to repo
- **Content Retention**: Articles purged after 7 days
- **User Privacy**: No tracking, minimal data collection

### Security Measures
- **Input Validation**: Sanitize all RSS feed URLs and user inputs
- **Rate Limiting**: Prevent API abuse and cost overruns
- **Error Boundaries**: Graceful degradation on service failures
- **Backup Strategy**: Configuration and topic data backup

## Performance Considerations

### Optimization Strategies
- **Batch Processing**: Daily scheduling reduces overhead
- **Caching**: Store embeddings and avoid recomputation
- **Parallel Processing**: Concurrent feed fetching and analysis
- **Memory Management**: Process articles in chunks for large feeds

### Scalability Limits
- **Articles/day**: ~1000 (with current cost targets)
- **Topics**: ~20 (before performance degradation)
- **RSS Feeds**: ~100 (practical limit for daily processing)
- **Response Time**: Target <5 minutes for daily processing

## Monitoring & Maintenance

### Health Metrics
- Daily processing success rate
- AI API usage and costs
- Content delivery success rate
- User engagement with delivered content

### Maintenance Tasks
- Weekly: Review AI usage costs
- Monthly: Analyze content relevance feedback
- Quarterly: Update topic definitions and thresholds
- As needed: Add new RSS sources

## Future Enhancements

### Short-term (Months 1-3)
- Source recommendation engine
- Improved summarization quality
- Mobile-responsive configuration interface

### Long-term (Months 6+)
- Multi-language content support
- Advanced topic modeling
- Integration with other platforms (Slack, Discord)
- Machine learning-based preference learning