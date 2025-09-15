# API Specifications

## Auto-Registration Flow

### Bot Added to Group/Channel
When the CuliFeed bot is added to any Telegram group or channel:

```
1. Bot receives "my_chat_member" update
2. Extracts chat_id, chat_title, chat_type
3. Stores registration in database
4. Sends welcome message with setup instructions
```

**Auto-registration Message**:
```
🤖 CuliFeed Bot Registered Successfully!

📊 Channel: {chat_title}
🆔 ID: {chat_id}
📋 Type: {chat_type}

🚀 Quick Setup:
/add_topic "Your first topic"
/add_feed "https://example.com/feed.xml"
/help

💡 Each channel can have independent topics and feeds
```

### Multi-Channel Support
- Each group/channel maintains separate topics and feeds
- Bot automatically detects which channel commands come from
- Cross-channel topic sharing available via `/share_topic` command

## Telegram Bot Commands

### Topic Management

#### `/add_topic <topic_name>`
Add a new topic for content curation.

**Usage**: `/add_topic "AWS Lambda performance optimization"`

**Parameters**:
- `topic_name`: Descriptive topic name (required)

**Response**: 
```
✅ Topic added: "AWS Lambda performance optimization"
📊 Will be included in next daily processing cycle
🎯 Suggested keywords: lambda, performance, optimization, cold start
```

**Error Handling**:
- Duplicate topic: "Topic already exists"
- Invalid format: "Please use: /add_topic 'topic name'"

#### `/remove_topic <topic_name>`
Remove an existing topic.

**Usage**: `/remove_topic "leadership"`

**Response**:
```
✅ Topic removed: "leadership"  
📊 Will be excluded from next processing cycle
```

#### `/list_topics`
Display all configured topics with statistics.

**Response**:
```
📋 Your Topics (3):

1. 🔧 AWS Lambda performance optimization
   └─ Last match: 2 articles yesterday
   
2. 🏗️ AWS ECS container management  
   └─ Last match: 1 article yesterday
   
3. 👥 Leadership in tech
   └─ Last match: 3 articles yesterday
   
📊 Total articles delivered yesterday: 6
```

### Feed Management

#### `/add_feed <feed_url>`
Add a new RSS/Atom feed source.

**Usage**: `/add_feed https://aws.amazon.com/blogs/compute/feed/`

**Response**:
```
✅ Feed added: AWS Compute Blog
📡 URL: https://aws.amazon.com/blogs/compute/feed/
🔍 Will be included in next daily scan
```

**Validation**:
- URL format check
- Feed accessibility test
- Duplicate prevention

#### `/list_feeds`
Show all configured feeds with health status.

**Response**:
```
📡 RSS Feeds (5):

✅ AWS Compute Blog
   └─ Last update: 6 hours ago (3 new articles)
   
⚠️  Hacker News
   └─ Last update: Failed 2 hours ago
   
✅ Martin Fowler's Blog  
   └─ Last update: 1 day ago (0 new articles)
```

#### `/remove_feed <feed_url_or_name>`
Remove a feed from monitoring.

### Configuration Commands

#### `/set_confidence <topic_name> <threshold>`
Adjust confidence threshold for a specific topic.

**Usage**: `/set_confidence "AWS Lambda performance" 0.9`

**Response**: 
```
🎯 Confidence threshold updated
📊 Topic: AWS Lambda performance  
📈 New threshold: 0.9 (was 0.8)
💡 Higher threshold = fewer but more relevant articles
```

#### `/adjust_sensitivity <high|medium|low>`
Global sensitivity adjustment across all topics.

**Response**:
```
⚙️ Sensitivity set to: HIGH
📊 Effect: More strict filtering, fewer articles
🎯 Confidence thresholds adjusted: +0.1 for all topics
```

### Channel Management Commands

#### `/register`
Manually register current channel (auto-registration also works).

**Response**:
```
✅ Channel registered successfully!
📊 Channel: Technical Discussion Group
🆔 ID: -1001234567890
📋 Ready for topic and feed configuration
```

#### `/list_channels` (Admin only)
Show all registered channels and their activity.

**Response**:
```
📡 Registered Channels (3):

✅ Personal Feed
   ├─ ID: 123456789 (private chat)
   ├─ Topics: 5 active
   └─ Last delivery: 2 hours ago

✅ Tech Team Group  
   ├─ ID: -1001234567890 (group)
   ├─ Topics: 3 active
   └─ Last delivery: 6 hours ago

⚠️  Archive Channel
   ├─ ID: -1009876543210 (channel) 
   ├─ Topics: 0 active
   └─ Last delivery: 3 days ago
```

### Status & Control Commands

#### `/status`
Show system health and recent activity for current channel.

**Response**:
```
🤖 CuliFeed Status - {Channel Name}

📊 Last Processing: Today 8:00 AM
├─ Articles scanned: 127
├─ Articles filtered: 23  
├─ Articles delivered: 8
└─ Processing time: 3m 42s

🎯 This Channel:
├─ Topics: 4 active
├─ Feeds: 12 active
└─ Articles delivered: 3

💰 Total System Usage:
├─ Gemini API calls: 847/1000 daily
├─ Groq fallback: 23/100 daily  
├─ Estimated cost: $0.00 (free tier)
└─ Rate limit headroom: 15%
```

#### `/help`
Display available commands and usage examples.

#### `/pause`
Temporarily pause daily processing.

#### `/resume` 
Resume daily processing.

## Internal APIs

### Core Processing Functions

#### `process_daily_content()`
Main processing pipeline execution.

**Returns**:
```json
{
    "success": true,
    "articles_processed": 127,
    "articles_delivered": 8,
    "topics_matched": 4,
    "processing_time_seconds": 222,
    "api_calls_used": 23,
    "estimated_cost": 0.00
}
```

#### `analyze_article_relevance(article, topic)`
AI-powered relevance analysis for single article.

**Parameters**:
```json
{
    "article": {
        "title": "string",
        "content": "string", 
        "url": "string"
    },
    "topic": {
        "name": "string",
        "keywords": ["string"],
        "exclude_keywords": ["string"]
    }
}
```

**Returns**:
```json
{
    "relevance_score": 0.85,
    "confidence_score": 0.92,
    "summary": "Article discusses Lambda cold start optimization techniques...",
    "reasoning": "Strong match for performance optimization topic",
    "processing_cost": 0.00
}
```

### Configuration API

#### `get_user_config()`
Retrieve current user configuration.

#### `update_topic(topic_name, config)`
Modify topic settings programmatically.

#### `add_feed(feed_url, validation=True)`
Add new RSS feed with optional validation.

## Webhook Interfaces

### Telegram Webhook (Optional)
For real-time command processing instead of polling.

**Endpoint**: `POST /telegram/webhook`

**Payload**: Standard Telegram webhook format

### Health Check Endpoint
**Endpoint**: `GET /health`

**Response**:
```json
{
    "status": "healthy",
    "last_processing": "2024-01-15T08:00:00Z",
    "next_processing": "2024-01-16T08:00:00Z",
    "uptime_seconds": 86400
}
```

## Rate Limits & Quotas

### AI API Limits
- **Groq Free Tier**: 100 requests/day
- **Cost Protection**: Automatic cutoff at $5/month
- **Fallback Strategy**: Keyword-only filtering when limits reached

### Telegram API Limits  
- **Bot Messages**: 30 messages/second
- **Group Messages**: 20 messages/minute
- **Command Processing**: 1 command/second per user

### RSS Feed Limits
- **Fetch Frequency**: Once per day per feed
- **Timeout**: 30 seconds per feed
- **Retry Logic**: 3 attempts with exponential backoff
- **Maximum Feeds**: 100 per user

## Error Codes & Responses

### Bot Command Errors
```
E001: "Invalid topic name format"
E002: "Topic already exists"  
E003: "Topic not found"
E004: "Invalid RSS feed URL"
E005: "Feed already configured"
E006: "Maximum topics reached (20)"
E007: "Maximum feeds reached (100)"
E008: "Configuration save failed"
```

### Processing Errors
```
P001: "RSS feed fetch timeout"
P002: "AI API quota exceeded"
P003: "Content parsing failed"
P004: "Database connection error"
P005: "Telegram delivery failed"
```

### Recovery Actions
```python
error_handlers = {
    "E001-E008": "Send usage help message",
    "P001": "Skip feed, continue processing", 
    "P002": "Switch to keyword-only mode",
    "P003": "Log and skip article",
    "P004": "Retry with exponential backoff",
    "P005": "Queue for retry delivery"
}
```

## Integration Specifications

### External Services

#### Gemini API Integration
```python
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Generate content
response = model.generate_content(
    analysis_prompt,
    generation_config=genai.types.GenerationConfig(
        temperature=0.1,
        max_output_tokens=500
    )
)
```

### Groq API Integration (Fallback)
```python
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3-8b-8192", 
    "messages": [{"role": "user", "content": analysis_prompt}],
    "temperature": 0.1,
    "max_tokens": 500
}
```

#### Telegram Bot API
```python
import telegram
bot = telegram.Bot(token=TELEGRAM_TOKEN)
bot.send_message(
    chat_id=USER_CHAT_ID,
    text=formatted_message,
    parse_mode='Markdown',
    disable_web_page_preview=True
)
```

### Configuration File Format
```yaml
# config.yaml
user:
  admin_user_id: "${TELEGRAM_ADMIN_ID}"  # Optional: for admin commands
  timezone: "UTC"
  
# Note: Topics and feeds are managed per-channel via Telegram bot
# No static configuration needed - use /add_topic and /add_feed commands
# Each channel maintains independent topics and feeds in database
    
processing:
  ai_provider: "gemini"  # Primary: gemini, fallback: groq
  daily_run_hour: 8
  max_articles_per_topic: 5
```