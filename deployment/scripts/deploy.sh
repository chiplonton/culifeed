#!/bin/bash
#
# CuliFeed Production Deployment Script
# ====================================
#
# Automated deployment script for CuliFeed RSS content curation system.
# Handles VPS setup, dependency installation, service configuration, and monitoring setup.
#
# Usage:
#   bash deployment/scripts/deploy.sh [OPTIONS]
#
# Options:
#   --target-vps          Deploy to production VPS
#   --dry-run            Simulate deployment without making changes
#   --skip-deps          Skip dependency installation
#   --skip-services      Skip systemd service setup
#   --skip-validation    Skip post-deployment validation
#   --backup             Create backup before deployment
#   --rollback           Rollback to previous deployment
#   --help               Show this help message

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
readonly DEPLOYMENT_USER="culifeed"
readonly DEPLOYMENT_GROUP="culifeed"
readonly INSTALL_DIR="/opt/culifeed"
readonly CONFIG_DIR="/etc/culifeed"
readonly LOG_DIR="/var/log/culifeed"
readonly DATA_DIR="/var/lib/culifeed"
readonly BACKUP_DIR="/var/backups/culifeed"
readonly SYSTEMD_DIR="/etc/systemd/system"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Global flags
DRY_RUN=false
TARGET_VPS=false
SKIP_DEPS=false
SKIP_SERVICES=false
SKIP_VALIDATION=false
CREATE_BACKUP=false
ROLLBACK=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_step() {
    echo -e "\n${BLUE}=== $* ===${NC}"
}

# Error handling
cleanup_on_error() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        log_error "Check logs above for details"
        
        if [[ "$CREATE_BACKUP" == true ]]; then
            log_info "Consider rollback: $0 --rollback"
        fi
    fi
    exit $exit_code
}

trap cleanup_on_error ERR

# Utility functions
run_cmd() {
    local cmd="$*"
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would execute: $cmd"
        return 0
    else
        log_info "Executing: $cmd"
        eval "$cmd"
    fi
}

check_root() {
    if [[ $EUID -ne 0 ]] && [[ "$DRY_RUN" == false ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_system_requirements() {
    log_step "Checking System Requirements"
    
    # Check OS
    if ! grep -q "Ubuntu\|Debian" /etc/os-release 2>/dev/null; then
        log_warning "This script is designed for Ubuntu/Debian systems"
    fi
    
    # Check available disk space (minimum 2GB)
    local available_space
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 2097152 ]]; then  # 2GB in KB
        log_error "Insufficient disk space. At least 2GB free space required"
        exit 1
    fi
    
    # Check available memory (minimum 512MB)
    local available_memory
    available_memory=$(free -m | awk 'NR==2 {print $7}')
    if [[ $available_memory -lt 512 ]]; then
        log_warning "Less than 512MB free memory available. Performance may be affected"
    fi
    
    log_success "System requirements check passed"
}

install_system_dependencies() {
    if [[ "$SKIP_DEPS" == true ]]; then
        log_info "Skipping dependency installation"
        return 0
    fi
    
    log_step "Installing System Dependencies"
    
    # Update package list
    run_cmd "apt-get update"
    
    # Install essential packages
    local packages=(
        "python3"
        "python3-pip"
        "python3-venv"
        "python3-dev"
        "git"
        "curl"
        "wget"
        "sqlite3"
        "supervisor"
        "nginx"
        "certbot"
        "python3-certbot-nginx"
        "logrotate"
        "fail2ban"
        "ufw"
    )
    
    for package in "${packages[@]}"; do
        run_cmd "apt-get install -y $package"
    done
    
    log_success "System dependencies installed"
}

create_deployment_user() {
    log_step "Creating Deployment User"
    
    if ! id "$DEPLOYMENT_USER" &>/dev/null; then
        run_cmd "useradd --system --create-home --shell /bin/bash --groups systemd-journal $DEPLOYMENT_USER"
        run_cmd "usermod -aG $DEPLOYMENT_GROUP $DEPLOYMENT_USER"
        log_success "Created user: $DEPLOYMENT_USER"
    else
        log_info "User $DEPLOYMENT_USER already exists"
    fi
}

create_directory_structure() {
    log_step "Creating Directory Structure"
    
    local directories=(
        "$INSTALL_DIR"
        "$CONFIG_DIR"
        "$LOG_DIR"
        "$DATA_DIR"
        "$BACKUP_DIR"
    )
    
    for dir in "${directories[@]}"; do
        run_cmd "mkdir -p $dir"
        run_cmd "chown $DEPLOYMENT_USER:$DEPLOYMENT_GROUP $dir"
        run_cmd "chmod 755 $dir"
    done
    
    # Set more restrictive permissions for sensitive directories
    run_cmd "chmod 750 $CONFIG_DIR $DATA_DIR"
    
    log_success "Directory structure created"
}

backup_existing_installation() {
    if [[ "$CREATE_BACKUP" != true ]]; then
        return 0
    fi
    
    log_step "Creating Backup"
    
    local backup_timestamp
    backup_timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_path="$BACKUP_DIR/backup_$backup_timestamp"
    
    run_cmd "mkdir -p $backup_path"
    
    # Backup application files
    if [[ -d "$INSTALL_DIR" ]]; then
        run_cmd "cp -r $INSTALL_DIR $backup_path/"
    fi
    
    # Backup configuration
    if [[ -d "$CONFIG_DIR" ]]; then
        run_cmd "cp -r $CONFIG_DIR $backup_path/"
    fi
    
    # Backup database
    if [[ -f "$DATA_DIR/culifeed.db" ]]; then
        run_cmd "cp $DATA_DIR/culifeed.db $backup_path/"
    fi
    
    # Create backup manifest
    cat > "$backup_path/manifest.txt" <<EOF
CuliFeed Backup Manifest
========================
Created: $(date)
Version: $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
Backup Path: $backup_path

Contents:
- Application files: $(ls -la "$backup_path" 2>/dev/null | wc -l) items
- Database: $(test -f "$backup_path/culifeed.db" && echo "included" || echo "not found")
- Configuration: $(test -d "$backup_path/$(basename "$CONFIG_DIR")" && echo "included" || echo "not found")
EOF
    
    log_success "Backup created at: $backup_path"
    echo "$backup_path" > "$BACKUP_DIR/latest_backup"
}

deploy_application_files() {
    log_step "Deploying Application Files"
    
    # Stop services before deployment
    run_cmd "systemctl stop culifeed-bot.service || true"
    run_cmd "systemctl stop culifeed-processor.timer || true"
    
    # Copy application files
    run_cmd "cp -r $PROJECT_ROOT/* $INSTALL_DIR/"
    run_cmd "chown -R $DEPLOYMENT_USER:$DEPLOYMENT_GROUP $INSTALL_DIR"
    
    # Create Python virtual environment
    run_cmd "sudo -u $DEPLOYMENT_USER python3 -m venv $INSTALL_DIR/venv"
    
    # Install Python dependencies
    run_cmd "sudo -u $DEPLOYMENT_USER $INSTALL_DIR/venv/bin/pip install --upgrade pip"
    run_cmd "sudo -u $DEPLOYMENT_USER $INSTALL_DIR/venv/bin/pip install -r $INSTALL_DIR/requirements.txt"
    
    # Make scripts executable
    run_cmd "chmod +x $INSTALL_DIR/main.py"
    run_cmd "chmod +x $INSTALL_DIR/run_bot.py"
    run_cmd "chmod +x $INSTALL_DIR/deployment/scripts/*.sh"
    
    log_success "Application files deployed"
}

setup_configuration() {
    log_step "Setting up Configuration"
    
    # Copy configuration files
    if [[ -f "$PROJECT_ROOT/config.yaml" ]]; then
        run_cmd "cp $PROJECT_ROOT/config.yaml $CONFIG_DIR/"
    else
        # Create default configuration
        run_cmd "sudo -u $DEPLOYMENT_USER python3 $INSTALL_DIR/main.py create-config"
        run_cmd "mv $INSTALL_DIR/config.yaml $CONFIG_DIR/"
    fi
    
    # Copy environment file template
    if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
        run_cmd "cp $PROJECT_ROOT/.env.example $CONFIG_DIR/.env.template"
    fi
    
    # Create production environment file if it doesn't exist
    if [[ ! -f "$CONFIG_DIR/.env" ]]; then
        cat > "$CONFIG_DIR/.env" <<EOF
# CuliFeed Production Environment Configuration
# ===========================================
# Copy this file and fill in your API keys and configurations

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here

# AI Provider API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
DATABASE_PATH=$DATA_DIR/culifeed.db

# Logging Configuration  
LOG_LEVEL=INFO
LOG_FILE=$LOG_DIR/culifeed.log

# Production Settings
ENVIRONMENT=production
DEBUG=false
EOF
        run_cmd "chown $DEPLOYMENT_USER:$DEPLOYMENT_GROUP $CONFIG_DIR/.env"
        run_cmd "chmod 600 $CONFIG_DIR/.env"
        
        log_warning "Created template environment file at $CONFIG_DIR/.env"
        log_warning "You must edit this file with your API keys before the services will work"
    fi
    
    log_success "Configuration setup completed"
}

setup_database() {
    log_step "Setting up Database"
    
    # Initialize database
    run_cmd "sudo -u $DEPLOYMENT_USER CULIFEED_CONFIG=$CONFIG_DIR/config.yaml $INSTALL_DIR/venv/bin/python $INSTALL_DIR/main.py init-db"
    
    # Set database permissions
    run_cmd "chmod 640 $DATA_DIR/culifeed.db"
    run_cmd "chown $DEPLOYMENT_USER:$DEPLOYMENT_GROUP $DATA_DIR/culifeed.db"
    
    log_success "Database setup completed"
}

install_systemd_services() {
    if [[ "$SKIP_SERVICES" == true ]]; then
        log_info "Skipping systemd service installation"
        return 0
    fi
    
    log_step "Installing systemd Services"
    
    # Copy service files from deployment directory
    run_cmd "cp $PROJECT_ROOT/deployment/systemd/*.service $SYSTEMD_DIR/"
    run_cmd "cp $PROJECT_ROOT/deployment/systemd/*.timer $SYSTEMD_DIR/"
    
    # Reload systemd
    run_cmd "systemctl daemon-reload"
    
    # Enable services
    run_cmd "systemctl enable culifeed-bot.service"
    run_cmd "systemctl enable culifeed-processor.timer"
    
    log_success "systemd services installed and enabled"
}

setup_logging() {
    log_step "Setting up Logging"
    
    # Create log rotation configuration
    cat > "/etc/logrotate.d/culifeed" <<EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $DEPLOYMENT_USER $DEPLOYMENT_GROUP
    postrotate
        systemctl reload culifeed-bot.service || true
    endscript
}
EOF
    
    # Create rsyslog configuration for structured logging
    cat > "/etc/rsyslog.d/50-culifeed.conf" <<EOF
# CuliFeed application logging
if \$programname == 'culifeed' then $LOG_DIR/culifeed.log
& stop
EOF
    
    run_cmd "systemctl restart rsyslog"
    
    log_success "Logging setup completed"
}

setup_monitoring() {
    log_step "Setting up Monitoring"
    
    # Install monitoring script
    run_cmd "cp $PROJECT_ROOT/deployment/scripts/health_check.sh /usr/local/bin/"
    run_cmd "chmod +x /usr/local/bin/health_check.sh"
    
    # Create monitoring cron job
    cat > "/etc/cron.d/culifeed-monitoring" <<EOF
# CuliFeed monitoring cron jobs
*/5 * * * * $DEPLOYMENT_USER /usr/local/bin/health_check.sh --quiet
0 6 * * * $DEPLOYMENT_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/main.py cleanup --dry-run=false
EOF
    
    log_success "Monitoring setup completed"
}

setup_security() {
    log_step "Setting up Security"
    
    # Configure UFW firewall
    run_cmd "ufw --force enable"
    run_cmd "ufw default deny incoming"
    run_cmd "ufw default allow outgoing"
    run_cmd "ufw allow ssh"
    run_cmd "ufw allow 80/tcp"   # HTTP for Let's Encrypt
    run_cmd "ufw allow 443/tcp"  # HTTPS
    
    # Configure fail2ban for SSH protection
    cat > "/etc/fail2ban/jail.local" <<EOF
[DEFAULT]
bantime = 1800
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
EOF
    
    run_cmd "systemctl restart fail2ban"
    
    log_success "Security setup completed"
}

start_services() {
    log_step "Starting Services"
    
    # Start bot service
    run_cmd "systemctl start culifeed-bot.service"
    
    # Start processor timer
    run_cmd "systemctl start culifeed-processor.timer"
    
    # Wait a moment for services to start
    sleep 3
    
    log_success "Services started"
}

validate_deployment() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        log_info "Skipping deployment validation"
        return 0
    fi
    
    log_step "Validating Deployment"
    
    local validation_errors=0
    
    # Check service status
    if ! systemctl is-active --quiet culifeed-bot.service; then
        log_error "Bot service is not running"
        ((validation_errors++))
    else
        log_success "Bot service is running"
    fi
    
    if ! systemctl is-active --quiet culifeed-processor.timer; then
        log_error "Processor timer is not running"
        ((validation_errors++))
    else
        log_success "Processor timer is active"
    fi
    
    # Check configuration
    if [[ ! -f "$CONFIG_DIR/.env" ]]; then
        log_error "Environment configuration not found"
        ((validation_errors++))
    elif grep -q "your_.*_here" "$CONFIG_DIR/.env"; then
        log_warning "Environment file contains placeholder values"
        log_warning "Update $CONFIG_DIR/.env with real API keys"
    else
        log_success "Environment configuration exists"
    fi
    
    # Check database
    if [[ ! -f "$DATA_DIR/culifeed.db" ]]; then
        log_error "Database not found"
        ((validation_errors++))
    else
        log_success "Database exists"
    fi
    
    # Check application health
    if run_cmd "sudo -u $DEPLOYMENT_USER CULIFEED_CONFIG=$CONFIG_DIR/config.yaml $INSTALL_DIR/venv/bin/python $INSTALL_DIR/main.py health-check"; then
        log_success "Application health check passed"
    else
        log_warning "Application health check failed (may be due to missing API keys)"
    fi
    
    # Summary
    if [[ $validation_errors -eq 0 ]]; then
        log_success "Deployment validation passed"
        return 0
    else
        log_error "Deployment validation failed with $validation_errors errors"
        return 1
    fi
}

rollback_deployment() {
    log_step "Rolling Back Deployment"
    
    local latest_backup
    if [[ -f "$BACKUP_DIR/latest_backup" ]]; then
        latest_backup=$(cat "$BACKUP_DIR/latest_backup")
    else
        log_error "No backup found for rollback"
        exit 1
    fi
    
    log_info "Rolling back to: $latest_backup"
    
    # Stop services
    run_cmd "systemctl stop culifeed-bot.service || true"
    run_cmd "systemctl stop culifeed-processor.timer || true"
    
    # Restore files
    if [[ -d "$latest_backup/$(basename "$INSTALL_DIR")" ]]; then
        run_cmd "rm -rf $INSTALL_DIR"
        run_cmd "cp -r $latest_backup/$(basename "$INSTALL_DIR") $INSTALL_DIR"
        run_cmd "chown -R $DEPLOYMENT_USER:$DEPLOYMENT_GROUP $INSTALL_DIR"
    fi
    
    # Restore configuration
    if [[ -d "$latest_backup/$(basename "$CONFIG_DIR")" ]]; then
        run_cmd "rm -rf $CONFIG_DIR"
        run_cmd "cp -r $latest_backup/$(basename "$CONFIG_DIR") $CONFIG_DIR"
        run_cmd "chown -R $DEPLOYMENT_USER:$DEPLOYMENT_GROUP $CONFIG_DIR"
    fi
    
    # Restore database
    if [[ -f "$latest_backup/culifeed.db" ]]; then
        run_cmd "cp $latest_backup/culifeed.db $DATA_DIR/"
        run_cmd "chown $DEPLOYMENT_USER:$DEPLOYMENT_GROUP $DATA_DIR/culifeed.db"
        run_cmd "chmod 640 $DATA_DIR/culifeed.db"
    fi
    
    # Start services
    run_cmd "systemctl start culifeed-bot.service"
    run_cmd "systemctl start culifeed-processor.timer"
    
    log_success "Rollback completed"
}

show_post_deployment_info() {
    log_step "Post-Deployment Information"
    
    echo ""
    echo "🎉 CuliFeed deployment completed!"
    echo ""
    echo "Next steps:"
    echo "1. Edit $CONFIG_DIR/.env with your API keys"
    echo "2. Restart services: systemctl restart culifeed-bot.service"
    echo "3. Check service status: systemctl status culifeed-bot.service"
    echo "4. View logs: journalctl -u culifeed-bot.service -f"
    echo ""
    echo "Management commands:"
    echo "- Health check: /usr/local/bin/health_check.sh"
    echo "- Manual processing: sudo -u $DEPLOYMENT_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/main.py daily-process"
    echo "- View feeds: sudo -u $DEPLOYMENT_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/main.py show-feeds"
    echo ""
    echo "Service files:"
    echo "- Application: $INSTALL_DIR"
    echo "- Configuration: $CONFIG_DIR"
    echo "- Logs: $LOG_DIR"
    echo "- Data: $DATA_DIR"
    echo ""
}

show_help() {
    cat <<EOF
CuliFeed Production Deployment Script
====================================

Usage: $0 [OPTIONS]

Options:
    --target-vps          Deploy to production VPS
    --dry-run            Simulate deployment without making changes
    --skip-deps          Skip dependency installation
    --skip-services      Skip systemd service setup
    --skip-validation    Skip post-deployment validation
    --backup             Create backup before deployment
    --rollback           Rollback to previous deployment
    --help               Show this help message

Examples:
    # Full production deployment with backup
    sudo $0 --target-vps --backup
    
    # Dry run to see what would be done
    sudo $0 --target-vps --dry-run
    
    # Deploy without installing system dependencies
    sudo $0 --target-vps --skip-deps
    
    # Rollback to previous deployment
    sudo $0 --rollback

EOF
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --target-vps)
                TARGET_VPS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-services)
                SKIP_SERVICES=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --backup)
                CREATE_BACKUP=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Handle rollback
    if [[ "$ROLLBACK" == true ]]; then
        check_root
        rollback_deployment
        exit 0
    fi
    
    # Require target specification for deployment
    if [[ "$TARGET_VPS" != true ]] && [[ "$DRY_RUN" != true ]]; then
        log_error "Must specify --target-vps for production deployment"
        log_info "Use --dry-run to test deployment without making changes"
        show_help
        exit 1
    fi
    
    # Pre-flight checks
    check_root
    check_system_requirements
    
    # Main deployment flow
    install_system_dependencies
    create_deployment_user
    create_directory_structure
    backup_existing_installation
    deploy_application_files
    setup_configuration
    setup_database
    install_systemd_services
    setup_logging
    setup_monitoring
    setup_security
    start_services
    validate_deployment
    
    # Show completion info
    if [[ "$DRY_RUN" != true ]]; then
        show_post_deployment_info
    else
        log_info "Dry run completed - no changes were made"
    fi
}

# Run main function
main "$@"