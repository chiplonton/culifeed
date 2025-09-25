#!/bin/bash
set -e

# CuliFeed Docker Entrypoint
# Initializes database and starts all services

echo "Starting CuliFeed Docker Container..."
echo "Working directory: $(pwd)"
echo "Current user: $(whoami)"

# Create logs directory if it doesn't exist
mkdir -p /app/logs

# Initialize database schema
echo "Initializing database..."
python -c "
import sys
import os
sys.path.insert(0, '/app')
from culifeed.database.schema import DatabaseSchema
from culifeed.config.settings import get_settings
settings = get_settings()
schema = DatabaseSchema(settings.database.path)
schema.create_tables()
print('Database schema initialized successfully')
"

# Verify database setup
echo "Verifying database setup..."
python -c "
import sys
sys.path.insert(0, '/app')
from culifeed.database.schema import DatabaseSchema
from culifeed.config.settings import get_settings
settings = get_settings()
schema = DatabaseSchema(settings.database.path)
if schema.verify_schema():
    print('Database verification successful')
else:
    print('Database verification failed')
    sys.exit(1)
"

# Start supervisor to manage all services
echo "Starting supervisor with all services..."
exec /usr/local/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf