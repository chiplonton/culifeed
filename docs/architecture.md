# CuliFeed Architecture

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RSS Sources   │───▶│  CuliFeed Core  │───▶│ Telegram Bot    │
│                 │    │                 │    │                 │
│ • Tech blogs    │    │ ┌─────────────┐ │    │ • Daily digest  │
│ • AWS updates   │    │ │Pre-filtering│ │    │ • Topic msgs    │
│ • Dev forums    │    │ │   Engine    │ │    │ • Bot commands  │
│ • Newsletters   │    │ └─────────────┘ │    │                 │
└─────────────────┘    │ ┌─────────────┐ │    └─────────────────┘
                       │ │ AI Analysis │ │    
                       │ │   Pipeline  │ │    
                       │ └─────────────┘ │    
                       │ ┌─────────────┐ │    
                       │ │  SQLite DB  │ │    
                       │ └─────────────┘ │    
                       └─────────────────┘    
```

## Component Architecture

### 1. Content Ingestion Layer

#### RSS Feed Manager
```python
class FeedManager:
    - fetch_feeds(): Retrieve new articles from all configured feeds
    - parse_content(): Extract clean article content
    - detect_duplicates(): Hash-based deduplication
    - store_articles(): Persist to SQLite with metadata
```

#### Data Models
```python
@dataclass
class Article:
    id: str
    title: str
    url: str
    content: str
    published_at: datetime
    source_feed: str
    hash: str  # For deduplication
    
@dataclass  
class Topic:
    name: str
    keywords: List[str]
    exclude_keywords: List[str]
    confidence_threshold: float
    chat_id: str  # Channel/group where topic was created
```

### 2. Processing Pipeline

#### Pre-filtering Engine
```python
class PreFilterEngine:
    - calculate_keyword_score(): Basic relevance scoring
    - apply_exclusion_rules(): Remove obvious non-matches
    - rank_by_potential(): Sort by likelihood of relevance
    - filter_top_candidates(): Select top 15% for AI processing
```

#### AI Analysis Pipeline
```python
class AIProcessor:
    - analyze_relevance(): Semantic topic matching
    - generate_summary(): Extract key points and value
    - calculate_confidence(): Assess certainty of match
    - format_for_delivery(): Structure for Telegram
```

### 3. Delivery Layer

#### Telegram Bot Interface
```python  
class TelegramBot:
    - send_daily_digest(): Deliver organized content
    - handle_topic_management(): Add/remove topics
    - handle_feed_management(): Add/remove RSS sources  
    - handle_configuration(): Adjust settings
```

#### Message Formatter
```python
class MessageFormatter:
    - format_topic_digest(): Group articles by topic
    - create_article_summary(): Title + summary + link format
    - apply_message_limits(): Respect Telegram limits
    - handle_delivery_errors(): Retry logic
```

## Data Flow

### Daily Processing Cycle

```
1. INGESTION (Morning)
   ├─ Fetch RSS feeds (parallel)
   ├─ Parse and clean content  
   ├─ Deduplicate articles
   └─ Store in SQLite

2. PRE-FILTERING (Mid-morning)
   ├─ Apply keyword matching per topic
   ├─ Calculate basic relevance scores
   ├─ Filter to top candidates (~85% reduction)
   └─ Queue for AI processing

3. AI PROCESSING (Late morning)
   ├─ Semantic relevance analysis
   ├─ Generate article summaries
   ├─ Calculate confidence scores
   └─ Filter by confidence threshold

4. DELIVERY (Evening)
   ├─ Group articles by topic
   ├─ Format for Telegram delivery
   ├─ Send separate message per topic
   └─ Log delivery status
```

### Auto-Registration Flow

```
Bot Added to Group/Channel
     ↓
Extract chat_id, chat_title, chat_type
     ↓
Store in channels table
     ↓
Send welcome message with setup instructions
     ↓
Ready for topic/feed management per channel
```

### Topic Management Flow

```
User Command (per channel) → Bot Parser → Database Update → Immediate Confirmation
     ↓
/add_topic "AWS ECS Container Optimization"
     ↓  
Bot validates → Saves to database (with chat_id) → "Topic added successfully"
     ↓
Next daily run includes new topic for this channel
```

## Database Schema

### SQLite Tables

#### articles
```sql
CREATE TABLE articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    content TEXT,
    published_at TIMESTAMP,
    source_feed TEXT,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### processing_results  
```sql
CREATE TABLE processing_results (
    article_id TEXT,
    chat_id TEXT,
    topic_name TEXT,
    pre_filter_score REAL,
    ai_relevance_score REAL,
    confidence_score REAL,
    summary TEXT,
    processed_at TIMESTAMP,
    delivered BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (article_id, chat_id, topic_name),
    FOREIGN KEY (chat_id) REFERENCES channels(chat_id)
);
```

#### channels (new for auto-registration)
```sql
CREATE TABLE channels (
    chat_id TEXT PRIMARY KEY,
    chat_title TEXT,
    chat_type TEXT,  -- group, supergroup, channel
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    last_delivery_at TIMESTAMP
);
```

#### topics  
```sql
CREATE TABLE topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    name TEXT NOT NULL,
    keywords TEXT,  -- JSON array
    exclude_keywords TEXT,  -- JSON array
    confidence_threshold REAL DEFAULT 0.8,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_match_at TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES channels(chat_id),
    UNIQUE(chat_id, name)
);
```

#### feeds
```sql
CREATE TABLE feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    last_fetched_at TIMESTAMP,
    last_success_at TIMESTAMP,
    error_count INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (chat_id) REFERENCES channels(chat_id),
    UNIQUE(chat_id, url)
);
```

## Deployment Architecture

### VPS/Local Server (Single Machine)
```bash
# Service Architecture
Your VPS/Server
├─ Telegram Bot (systemd service, always running)
├─ Daily Processor (cron job at 8 AM)
├─ SQLite Database (local file storage)
└─ Configuration (YAML + environment variables)

# Process Management
systemctl start culifeed-bot    # Bot service
crontab -e                      # Daily processing
# 0 8 * * * cd /opt/culifeed && python daily_processor.py
```

### Service Components
```python
# telegram_bot.py - Always running service
class TelegramBotService:
    - handle_commands(): Real-time command processing
    - manage_configuration(): Topic/feed management
    - send_status_updates(): Health monitoring
    
# daily_processor.py - Scheduled batch job  
class DailyProcessor:
    - fetch_content(): RSS scraping
    - analyze_articles(): AI processing
    - deliver_digest(): Telegram delivery
```

## Error Handling Strategy

### Graceful Degradation
- **API Failures**: Fall back to keyword-only filtering
- **Feed Errors**: Skip failed feeds, continue processing others
- **Telegram Failures**: Store messages for retry
- **AI Service Outage**: Queue articles for next successful run

### Recovery Mechanisms
- **Exponential Backoff**: For transient API failures
- **Circuit Breaker**: Disable failing feeds temporarily  
- **Dead Letter Queue**: Manual review of failed processing
- **Health Monitoring**: Daily success/failure reporting

## Configuration Management

### config.yaml Structure
```yaml
# User Configuration (no chat_id needed - auto-registered)
user:
  admin_user_id: "123456789"  # Optional: for admin commands
  timezone: "UTC"
  
# Processing Settings
processing:
  daily_run_time: "08:00"
  max_articles_per_topic: 5
  ai_provider: "gemini"  # gemini|groq|openai
  
# Cost Controls (Free Tier Management)
limits:
  max_daily_api_calls: 950  # Under Gemini 1000 RPD limit
  fallback_to_groq: true    # If Gemini limit reached
  fallback_to_keywords: true # If all APIs exhausted

# Topics and feeds are managed per-channel via Telegram bot
# No static configuration needed - stored in database per chat_id
```

## Security Architecture

### API Key Management
- Environment variables for all secrets
- No hardcoded credentials
- Separate dev/prod configurations

### Data Protection
- Local SQLite storage (no cloud data exposure)
- Content purged after 7 days
- No personal information collection
- Telegram bot token rotation support

### Input Validation
- URL validation for RSS feeds
- Command sanitization for Telegram inputs
- Content size limits to prevent abuse
- Rate limiting on bot commands

## Monitoring & Observability

### Key Metrics
- **Processing Success Rate**: Daily completion percentage
- **Cost Tracking**: API usage and spend monitoring
- **Content Quality**: User feedback on relevance
- **Performance**: Processing time per article

### Logging Strategy
```python
# Structured logging for operational visibility
logger.info("Processing started", extra={
    "articles_count": len(articles),
    "topics_count": len(topics),
    "timestamp": datetime.utcnow()
})
```

### Alerting
- Daily processing failures
- API rate limit approaches (>80% of free tier)
- Gemini quota warnings (approaching 1,000 RPD)
- Feed parsing errors (>50% failure rate)