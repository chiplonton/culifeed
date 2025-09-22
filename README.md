# CuliFeed - AI-Powered RSS Content Curation

**Smart content filtering that learns what you care about**

CuliFeed is an AI-powered Telegram bot that monitors RSS feeds and delivers only the content relevant to your interests. No more manual filtering - the AI learns your topics and curates content automatically.

## ✨ **Key Features**

- 🤖 **Smart AI Filtering** - Uses Gemini/Groq/OpenAI to understand content relevance
- 📱 **Telegram Integration** - Easy setup and daily digest delivery 
- 🎯 **Topic Matching** - Define interests, get personalized content
- 💰 **Cost Effective** - Optimized for free AI provider tiers
- 🔄 **Automated Processing** - Daily content curation and delivery
- 🗂️ **Multi-Feed Support** - Monitor multiple RSS sources

---

## 🚀 **Quick Start (5 Minutes)**

### Step 1: Get API Keys
- **Telegram Bot Token**: Message [@BotFather](https://t.me/BotFather) → `/newbot`
- **AI Provider Key**: [Gemini (Free)](https://makersuite.google.com/app/apikey) or [Groq (Free)](https://console.groq.com/keys)

### Step 2: Deploy with Docker
```bash
# Create environment file
cat > .env << 'EOF'
CULIFEED_TELEGRAM__BOT_TOKEN=your_bot_token_here
CULIFEED_AI__GEMINI_API_KEY=your_gemini_api_key_here
EOF

# Start CuliFeed services
docker run -d \
  --name culifeed \
  --env-file .env \
  -v culifeed_data:/app/data \
  --restart unless-stopped \
  culifeed:latest
```

### Step 3: Configure Bot in Telegram
1. **Add bot to your channel/group** (auto-registers)
2. **Initialize**: `/start`
3. **Add topics**: `/addtopic Machine Learning` (AI generates keywords) or `/addtopic Cloud, AWS, Azure` (manual keywords) 
4. **Add RSS feeds**: `/addfeed https://news.ycombinator.com/rss`
5. **Check setup**: `/status`
6. **Get daily digest automatically!** 📬

### Step 4: Daily Management
- **View topics**: `/topics` | **Edit**: `/edittopic AI` | **Remove**: `/removetopic AI`
- **View feeds**: `/feeds` | **Test**: `/testfeed URL` | **Remove**: `/removefeed URL`  
- **Preview content**: `/preview` | **Settings**: `/settings`

**✅ You're all set! Daily digests will be delivered automatically.**

---

## 🔧 **Configuration**

### Required Environment Variables
```bash
# Telegram bot token from @BotFather
CULIFEED_TELEGRAM__BOT_TOKEN=your_bot_token

# At least one AI provider (Gemini recommended - free tier)
CULIFEED_AI__GEMINI_API_KEY=your_gemini_key
CULIFEED_AI__GROQ_API_KEY=your_groq_key
CULIFEED_AI__OPENAI_API_KEY=your_openai_key
```

### Optional Settings
```bash
# When to run daily processing (0-23)
CULIFEED_PROCESSING__DAILY_RUN_HOUR=8

# Max articles per topic in daily digest
CULIFEED_PROCESSING__MAX_ARTICLES_PER_TOPIC=5

# Log level for debugging
CULIFEED_LOGGING__LEVEL=INFO
```

---

## 🤖 **Telegram Bot Commands**

### 📋 **Basic Commands**
| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Initialize bot for your channel | `/start` |
| `/help` | Show all available commands | `/help` |
| `/status` | Show channel statistics | `/status` |

### 🎯 **Topic Management**  
| Command | Description | Example |
|---------|-------------|---------|
| `/topics` | List your configured topics | `/topics` |
| `/addtopic` | Add topic with AI keywords or manual | `/addtopic Machine Learning` or `/addtopic Cloud, AWS, Azure` |
| `/removetopic` | Remove a topic | `/removetopic AI` |
| `/edittopic` | Edit existing topic | `/edittopic AI new, keywords, here` |

### 📡 **Feed Management**
| Command | Description | Example |
|---------|-------------|---------|
| `/feeds` | List your RSS feeds | `/feeds` |
| `/addfeed` | Add RSS feed to monitor | `/addfeed https://news.ycombinator.com/rss` |
| `/removefeed` | Remove RSS feed | `/removefeed https://example.com/feed` |
| `/testfeed` | Test feed connectivity | `/testfeed https://example.com/feed` |

### ⚙️ **Content & Settings**
| Command | Description | Example |
|---------|-------------|---------|
| `/preview` | Preview latest curated content | `/preview` |
| `/settings` | Show channel settings | `/settings` |

---

## 🐳 **Docker Deployment**

### Development
```bash
# Build image
docker build -t culifeed .

# Run with environment file
docker run -d --name culifeed --env-file .env -v culifeed_data:/app/data culifeed
```

### Production
```bash
# Production deployment with resource limits
docker run -d \
  --name culifeed-prod \
  --env-file .env \
  --volume culifeed_data:/app/data \
  --volume culifeed_logs:/app/logs \
  --restart unless-stopped \
  --memory 512m \
  --cpus 0.5 \
  culifeed:latest
```

### Service Management
```bash
# Check service status
docker exec culifeed supervisorctl status

# View logs
docker logs culifeed

# Restart services
docker exec culifeed supervisorctl restart culifeed-bot
docker exec culifeed supervisorctl restart culifeed-daily
```

---

## 💻 **Manual Installation**

If you prefer running without Docker:

```bash
# Clone and setup
git clone https://github.com/your-username/culifeed.git
cd culifeed
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python main.py init-db

# Run services (in separate terminals)
python run_bot.py                        # Terminal 1: Bot
python run_daily_scheduler.py --service  # Terminal 2: Scheduler
```

---

## 🛠️ **Management Commands**

```bash
# Configuration and health
python main.py check-config              # Validate setup
python main.py health-check              # Check system status
python main.py init-db                   # Initialize database

# Testing and processing
python main.py test-foundation           # Test core components
python main.py daily-process --dry-run   # Test processing pipeline
python main.py show-feeds                # List all feeds

# Manual operations
python run_daily_scheduler.py --check-status  # Check processing status
python run_daily_scheduler.py --dry-run       # Test daily processing
```

---

## 🐛 **Troubleshooting**

### Bot Not Starting?
```bash
# Check configuration
python main.py check-config

# Enable debug logging
CULIFEED_LOGGING__LEVEL=DEBUG python run_bot.py
```

### No Content Being Delivered?
1. **Check topics**: `/topics` - Make sure you have topics configured
2. **Check feeds**: `/feeds` - Verify RSS feeds are working
3. **Check processing**: `docker exec culifeed python run_daily_scheduler.py --check-status`

### Docker Issues?
```bash
# Check container status
docker logs culifeed

# Check services inside container
docker exec culifeed supervisorctl status

# Restart container
docker restart culifeed
```

---

## 📊 **Architecture**

```
┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │    │ Daily Scheduler │
│    (run_bot)    │    │ (run_scheduler) │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 │
    ┌─────────────────────────────┐
    │     CuliFeed Core           │
    │  ┌─────────────────────────┐│
    │  │ RSS Feeds → AI Filter   ││  
    │  │ Content → Topics Match  ││
    │  │ Digest → Telegram       ││
    │  └─────────────────────────┘│
    └─────────────────────────────┘
                 │
         ┌─────────────────┐
         │  SQLite Database │
         └─────────────────┘
```

**Two Services:**
- **Bot Service**: Handles Telegram commands and user interaction
- **Daily Scheduler**: Processes RSS feeds and delivers daily digests

---

## 📝 **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Add tests: `pytest tests/ -v`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push and create Pull Request

---

## 📄 **License**

GNU Affero General Public License v3 (AGPL v3) - see [LICENSE](LICENSE) file for details.

---

## 🎯 **Project Status**

✅ **Production Ready** - Fully implemented with comprehensive testing  
🚀 **Actively Maintained** - Regular updates and improvements  
📊 **Battle Tested** - Running in production environments