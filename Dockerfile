# CuliFeed Multi-Service Docker Container
# Runs both Telegram bot and daily processing scheduler

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash culifeed && \
    mkdir -p /app/logs && \
    chown -R culifeed:culifeed /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chown -R culifeed:culifeed /app

# Copy Docker configuration files
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create required directories
RUN mkdir -p /app/data /app/logs && \
    chown -R culifeed:culifeed /app/data /app/logs

# Switch to non-root user
USER culifeed

# Expose port for health checks (optional)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Entry point
ENTRYPOINT ["/entrypoint.sh"]