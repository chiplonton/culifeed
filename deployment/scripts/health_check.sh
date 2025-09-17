#!/bin/bash
#
# CuliFeed Health Check Script
# ===========================
#
# Comprehensive health monitoring for CuliFeed production deployment.
# Checks system health, service status, database integrity, and performance metrics.
#
# Usage:
#   bash deployment/scripts/health_check.sh [OPTIONS]
#
# Options:
#   --quiet              Suppress normal output (errors still shown)
#   --json               Output results in JSON format
#   --detailed           Include detailed performance metrics
#   --check-config       Validate configuration files
#   --check-database     Verify database integrity
#   --check-services     Check systemd services status
#   --check-resources    Monitor system resource usage
#   --alert-webhook URL  Send alerts to webhook URL
#   --help               Show this help message

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly INSTALL_DIR="/opt/culifeed"
readonly CONFIG_DIR="/etc/culifeed"
readonly LOG_DIR="/var/log/culifeed"
readonly DATA_DIR="/var/lib/culifeed"
readonly DEPLOYMENT_USER="culifeed"

# Health check thresholds
readonly MEMORY_THRESHOLD=80    # Percentage
readonly DISK_THRESHOLD=85      # Percentage
readonly CPU_THRESHOLD=90       # Percentage
readonly DB_SIZE_THRESHOLD=500  # MB

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Global flags
QUIET_MODE=false
JSON_OUTPUT=false
DETAILED_MODE=false
CHECK_CONFIG=true
CHECK_DATABASE=true
CHECK_SERVICES=true
CHECK_RESOURCES=true
ALERT_WEBHOOK=""

# Health status tracking
declare -A HEALTH_STATUS
OVERALL_STATUS="healthy"
TOTAL_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Logging functions
log_info() {
    if [[ "$QUIET_MODE" == false ]] && [[ "$JSON_OUTPUT" == false ]]; then
        echo -e "${BLUE}[INFO]${NC} $*"
    fi
}

log_success() {
    if [[ "$QUIET_MODE" == false ]] && [[ "$JSON_OUTPUT" == false ]]; then
        echo -e "${GREEN}[SUCCESS]${NC} $*"
    fi
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
    ((WARNING_CHECKS++))
    if [[ "$OVERALL_STATUS" == "healthy" ]]; then
        OVERALL_STATUS="warning"
    fi
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    ((FAILED_CHECKS++))
    OVERALL_STATUS="critical"
}

# Health check functions
check_system_resources() {
    log_info "Checking system resources..."
    
    local memory_usage disk_usage cpu_load
    
    # Memory usage
    memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    if (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
        log_error "High memory usage: ${memory_usage}%"
        HEALTH_STATUS["memory"]="critical"
    elif (( $(echo "$memory_usage > 60" | bc -l) )); then
        log_warning "Elevated memory usage: ${memory_usage}%"
        HEALTH_STATUS["memory"]="warning"
    else
        log_success "Memory usage: ${memory_usage}%"
        HEALTH_STATUS["memory"]="healthy"
    fi
    
    # Disk usage
    disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if (( disk_usage > DISK_THRESHOLD )); then
        log_error "High disk usage: ${disk_usage}%"
        HEALTH_STATUS["disk"]="critical"
    elif (( disk_usage > 70 )); then
        log_warning "Elevated disk usage: ${disk_usage}%"
        HEALTH_STATUS["disk"]="warning"
    else
        log_success "Disk usage: ${disk_usage}%"
        HEALTH_STATUS["disk"]="healthy"
    fi
    
    # CPU load (5-minute average)
    cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $2}' | sed 's/,//')
    cpu_cores=$(nproc)
    cpu_percentage=$(echo "scale=1; $cpu_load * 100 / $cpu_cores" | bc)
    
    if (( $(echo "$cpu_percentage > $CPU_THRESHOLD" | bc -l) )); then
        log_error "High CPU load: ${cpu_percentage}% (${cpu_load} on ${cpu_cores} cores)"
        HEALTH_STATUS["cpu"]="critical"
    elif (( $(echo "$cpu_percentage > 70" | bc -l) )); then
        log_warning "Elevated CPU load: ${cpu_percentage}% (${cpu_load} on ${cpu_cores} cores)"
        HEALTH_STATUS["cpu"]="warning"
    else
        log_success "CPU load: ${cpu_percentage}% (${cpu_load} on ${cpu_cores} cores)"
        HEALTH_STATUS["cpu"]="healthy"
    fi
    
    ((TOTAL_CHECKS += 3))
}

check_service_status() {
    log_info "Checking service status..."
    
    local services=("culifeed-bot.service" "culifeed-processor.timer")
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            log_success "Service $service is running"
            HEALTH_STATUS["service_$service"]="healthy"
        else
            log_error "Service $service is not running"
            HEALTH_STATUS["service_$service"]="critical"
        fi
        
        # Check if service is enabled
        if systemctl is-enabled --quiet "$service"; then
            log_success "Service $service is enabled"
        else
            log_warning "Service $service is not enabled"
        fi
        
        ((TOTAL_CHECKS += 2))
    done
    
    # Check service logs for recent errors
    for service in "${services[@]}"; do
        local error_count
        error_count=$(journalctl -u "$service" --since "1 hour ago" --grep="ERROR\|CRITICAL" --no-pager -q | wc -l)
        
        if (( error_count > 10 )); then
            log_error "Service $service has $error_count errors in the last hour"
            HEALTH_STATUS["service_${service}_errors"]="critical"
        elif (( error_count > 5 )); then
            log_warning "Service $service has $error_count errors in the last hour"
            HEALTH_STATUS["service_${service}_errors"]="warning"
        else
            log_success "Service $service has $error_count errors in the last hour"
            HEALTH_STATUS["service_${service}_errors"]="healthy"
        fi
        
        ((TOTAL_CHECKS++))
    done
}

check_database_health() {
    log_info "Checking database health..."
    
    local db_path="$DATA_DIR/culifeed.db"
    
    # Check if database file exists
    if [[ ! -f "$db_path" ]]; then
        log_error "Database file not found: $db_path"
        HEALTH_STATUS["database_exists"]="critical"
        ((TOTAL_CHECKS++))
        return
    fi
    
    log_success "Database file exists"
    HEALTH_STATUS["database_exists"]="healthy"
    
    # Check database size
    local db_size_mb
    db_size_mb=$(du -m "$db_path" | cut -f1)
    
    if (( db_size_mb > DB_SIZE_THRESHOLD )); then
        log_warning "Database size: ${db_size_mb}MB (consider cleanup)"
        HEALTH_STATUS["database_size"]="warning"
    else
        log_success "Database size: ${db_size_mb}MB"
        HEALTH_STATUS["database_size"]="healthy"
    fi
    
    # Test database connectivity
    if sudo -u "$DEPLOYMENT_USER" sqlite3 "$db_path" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" >/dev/null 2>&1; then
        log_success "Database connectivity test passed"
        HEALTH_STATUS["database_connectivity"]="healthy"
    else
        log_error "Database connectivity test failed"
        HEALTH_STATUS["database_connectivity"]="critical"
    fi
    
    # Check for database corruption
    if sudo -u "$DEPLOYMENT_USER" sqlite3 "$db_path" "PRAGMA integrity_check;" | grep -q "ok"; then
        log_success "Database integrity check passed"
        HEALTH_STATUS["database_integrity"]="healthy"
    else
        log_error "Database integrity check failed - database may be corrupted"
        HEALTH_STATUS["database_integrity"]="critical"
    fi
    
    ((TOTAL_CHECKS += 4))
}

check_configuration() {
    log_info "Checking configuration..."
    
    # Check main config file
    if [[ -f "$CONFIG_DIR/config.yaml" ]]; then
        log_success "Main configuration file exists"
        HEALTH_STATUS["config_yaml"]="healthy"
    else
        log_error "Main configuration file missing: $CONFIG_DIR/config.yaml"
        HEALTH_STATUS["config_yaml"]="critical"
    fi
    
    # Check environment file
    if [[ -f "$CONFIG_DIR/.env" ]]; then
        log_success "Environment file exists"
        
        # Check for placeholder values
        if grep -q "your_.*_here" "$CONFIG_DIR/.env"; then
            log_warning "Environment file contains placeholder values"
            HEALTH_STATUS["env_file"]="warning"
        else
            HEALTH_STATUS["env_file"]="healthy"
        fi
    else
        log_error "Environment file missing: $CONFIG_DIR/.env"
        HEALTH_STATUS["env_file"]="critical"
    fi
    
    # Check file permissions
    local config_perms env_perms
    config_perms=$(stat -c "%a" "$CONFIG_DIR/config.yaml" 2>/dev/null || echo "000")
    env_perms=$(stat -c "%a" "$CONFIG_DIR/.env" 2>/dev/null || echo "000")
    
    if [[ "$config_perms" == "644" ]]; then
        log_success "Configuration file permissions correct"
        HEALTH_STATUS["config_permissions"]="healthy"
    else
        log_warning "Configuration file permissions: $config_perms (should be 644)"
        HEALTH_STATUS["config_permissions"]="warning"
    fi
    
    if [[ "$env_perms" == "600" ]]; then
        log_success "Environment file permissions correct"
        HEALTH_STATUS["env_permissions"]="healthy"
    else
        log_warning "Environment file permissions: $env_perms (should be 600)"
        HEALTH_STATUS["env_permissions"]="warning"
    fi
    
    ((TOTAL_CHECKS += 4))
}

check_application_health() {
    log_info "Checking application health..."
    
    # Run application health check
    if sudo -u "$DEPLOYMENT_USER" CULIFEED_CONFIG="$CONFIG_DIR/config.yaml" \
       "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/main.py" health-check >/dev/null 2>&1; then
        log_success "Application health check passed"
        HEALTH_STATUS["app_health"]="healthy"
    else
        log_warning "Application health check failed (may be due to configuration)"
        HEALTH_STATUS["app_health"]="warning"
    fi
    
    # Check log file accessibility
    if [[ -w "$LOG_DIR" ]]; then
        log_success "Log directory is writable"
        HEALTH_STATUS["log_directory"]="healthy"
    else
        log_error "Log directory is not writable: $LOG_DIR"
        HEALTH_STATUS["log_directory"]="critical"
    fi
    
    # Check recent processing activity
    local last_processing
    if [[ -f "$LOG_DIR/culifeed.log" ]]; then
        last_processing=$(grep "Daily processing completed" "$LOG_DIR/culifeed.log" | tail -1 | awk '{print $1, $2}' || echo "")
        if [[ -n "$last_processing" ]]; then
            log_success "Last processing completed: $last_processing"
            HEALTH_STATUS["recent_processing"]="healthy"
        else
            log_warning "No recent processing activity found in logs"
            HEALTH_STATUS["recent_processing"]="warning"
        fi
    else
        log_warning "Log file not found: $LOG_DIR/culifeed.log"
        HEALTH_STATUS["recent_processing"]="warning"
    fi
    
    ((TOTAL_CHECKS += 3))
}

check_network_connectivity() {
    log_info "Checking network connectivity..."
    
    # Test external connectivity
    if curl -s --connect-timeout 5 https://www.google.com >/dev/null; then
        log_success "External network connectivity OK"
        HEALTH_STATUS["network_external"]="healthy"
    else
        log_error "External network connectivity failed"
        HEALTH_STATUS["network_external"]="critical"
    fi
    
    # Test Telegram API (if token is configured)
    if [[ -f "$CONFIG_DIR/.env" ]] && grep -q "TELEGRAM_BOT_TOKEN=" "$CONFIG_DIR/.env"; then
        local bot_token
        bot_token=$(grep "TELEGRAM_BOT_TOKEN=" "$CONFIG_DIR/.env" | cut -d'=' -f2)
        
        if [[ "$bot_token" != "your_bot_token_here" ]] && [[ -n "$bot_token" ]]; then
            if curl -s --connect-timeout 10 "https://api.telegram.org/bot$bot_token/getMe" | grep -q '"ok":true'; then
                log_success "Telegram API connectivity OK"
                HEALTH_STATUS["telegram_api"]="healthy"
            else
                log_error "Telegram API connectivity failed"
                HEALTH_STATUS["telegram_api"]="critical"
            fi
        else
            log_warning "Telegram bot token not configured"
            HEALTH_STATUS["telegram_api"]="warning"
        fi
    else
        log_warning "Telegram configuration not found"
        HEALTH_STATUS["telegram_api"]="warning"
    fi
    
    ((TOTAL_CHECKS += 2))
}

generate_json_report() {
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    echo "{"
    echo "  \"timestamp\": \"$timestamp\","
    echo "  \"overall_status\": \"$OVERALL_STATUS\","
    echo "  \"total_checks\": $TOTAL_CHECKS,"
    echo "  \"failed_checks\": $FAILED_CHECKS,"
    echo "  \"warning_checks\": $WARNING_CHECKS,"
    echo "  \"success_checks\": $((TOTAL_CHECKS - FAILED_CHECKS - WARNING_CHECKS)),"
    echo "  \"checks\": {"
    
    local first=true
    for key in "${!HEALTH_STATUS[@]}"; do
        if [[ "$first" == true ]]; then
            first=false
        else
            echo ","
        fi
        echo -n "    \"$key\": \"${HEALTH_STATUS[$key]}\""
    done
    
    echo ""
    echo "  }"
    echo "}"
}

send_alert() {
    local webhook_url="$1"
    local message="$2"
    
    if [[ -n "$webhook_url" ]]; then
        local payload
        payload=$(cat <<EOF
{
  "text": "CuliFeed Health Alert",
  "attachments": [
    {
      "color": "danger",
      "fields": [
        {
          "title": "Status",
          "value": "$OVERALL_STATUS",
          "short": true
        },
        {
          "title": "Failed Checks",
          "value": "$FAILED_CHECKS",
          "short": true
        },
        {
          "title": "Details",
          "value": "$message",
          "short": false
        }
      ]
    }
  ]
}
EOF
        )
        
        curl -X POST -H "Content-Type: application/json" -d "$payload" "$webhook_url" >/dev/null 2>&1 || true
    fi
}

show_summary() {
    if [[ "$JSON_OUTPUT" == true ]]; then
        generate_json_report
        return
    fi
    
    echo ""
    echo "=========================="
    echo "CuliFeed Health Check Summary"
    echo "=========================="
    echo "Overall Status: $OVERALL_STATUS"
    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed: $((TOTAL_CHECKS - FAILED_CHECKS - WARNING_CHECKS))"
    echo "Warnings: $WARNING_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo ""
    
    if [[ "$OVERALL_STATUS" == "healthy" ]]; then
        log_success "All systems are healthy! 🎉"
    elif [[ "$OVERALL_STATUS" == "warning" ]]; then
        log_warning "System has warnings but is operational ⚠️"
    else
        log_error "System has critical issues requiring attention! 🚨"
    fi
}

show_help() {
    cat <<EOF
CuliFeed Health Check Script
===========================

Usage: $0 [OPTIONS]

Options:
    --quiet              Suppress normal output (errors still shown)
    --json               Output results in JSON format
    --detailed           Include detailed performance metrics
    --check-config       Validate configuration files (default: enabled)
    --check-database     Verify database integrity (default: enabled)
    --check-services     Check systemd services status (default: enabled)
    --check-resources    Monitor system resource usage (default: enabled)
    --alert-webhook URL  Send alerts to webhook URL
    --help               Show this help message

Examples:
    # Basic health check
    $0
    
    # Quiet mode for monitoring scripts
    $0 --quiet
    
    # JSON output for programmatic use
    $0 --json
    
    # Send alerts to Slack webhook
    $0 --alert-webhook https://hooks.slack.com/services/...

EOF
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quiet)
                QUIET_MODE=true
                shift
                ;;
            --json)
                JSON_OUTPUT=true
                shift
                ;;
            --detailed)
                DETAILED_MODE=true
                shift
                ;;
            --check-config)
                CHECK_CONFIG=true
                shift
                ;;
            --check-database)
                CHECK_DATABASE=true
                shift
                ;;
            --check-services)
                CHECK_SERVICES=true
                shift
                ;;
            --check-resources)
                CHECK_RESOURCES=true
                shift
                ;;
            --alert-webhook)
                ALERT_WEBHOOK="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Initialize health status
    OVERALL_STATUS="healthy"
    TOTAL_CHECKS=0
    FAILED_CHECKS=0
    WARNING_CHECKS=0
    
    # Run health checks
    if [[ "$CHECK_RESOURCES" == true ]]; then
        check_system_resources
    fi
    
    if [[ "$CHECK_SERVICES" == true ]]; then
        check_service_status
    fi
    
    if [[ "$CHECK_DATABASE" == true ]]; then
        check_database_health
    fi
    
    if [[ "$CHECK_CONFIG" == true ]]; then
        check_configuration
    fi
    
    check_application_health
    check_network_connectivity
    
    # Show results
    show_summary
    
    # Send alerts if needed
    if [[ -n "$ALERT_WEBHOOK" ]] && [[ "$OVERALL_STATUS" != "healthy" ]]; then
        send_alert "$ALERT_WEBHOOK" "CuliFeed system health check failed: $FAILED_CHECKS failures, $WARNING_CHECKS warnings"
    fi
    
    # Exit with appropriate code
    case "$OVERALL_STATUS" in
        "healthy")
            exit 0
            ;;
        "warning")
            exit 1
            ;;
        "critical")
            exit 2
            ;;
        *)
            exit 3
            ;;
    esac
}

# Run main function
main "$@"