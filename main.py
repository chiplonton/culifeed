#!/usr/bin/env python3
"""
CuliFeed - Smart Content Curation System
========================================

Main application entry point with CLI interface for management and testing.

Usage:
    python main.py --help                    # Show all commands
    python main.py --check-config            # Validate configuration
    python main.py --test-foundation         # Test foundation components
    python main.py --init-db                 # Initialize database
    python main.py --test-feeds              # Test RSS feed connectivity
    python main.py --daily-process           # Run daily processing pipeline
    python main.py --start-bot               # Start Telegram bot service
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from culifeed.config.settings import get_settings, create_example_config
from culifeed.database.schema import DatabaseSchema
from culifeed.database.connection import get_db_manager
from culifeed.utils.logging import configure_application_logging, get_logger_for_component
from culifeed.utils.exceptions import CuliFeedError, handle_exception

console = Console()
logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--config', '-c', help='Configuration file path')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, debug):
    """CuliFeed - AI-powered RSS content curation system."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['debug'] = debug
    
    if ctx.invoked_subcommand is None:
        # Show help if no subcommand provided
        click.echo(ctx.get_help())


@cli.command()
@click.pass_context
def check_config(ctx):
    """Validate configuration file and environment variables."""
    console.print("[bold blue]🔧 Checking CuliFeed Configuration[/bold blue]")
    
    try:
        settings = get_settings()
        
        # Configuration validation table
        table = Table(title="Configuration Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green") 
        table.add_column("Details")
        
        # Check each component
        checks = [
            ("Database", _check_database_config, settings),
            ("Logging", _check_logging_config, settings),
            ("Telegram Bot", _check_telegram_config, settings),
            ("AI Providers", _check_ai_config, settings),
            ("Processing", _check_processing_config, settings),
        ]
        
        all_passed = True
        for name, check_func, config in checks:
            try:
                status, details = check_func(config)
                table.add_row(name, "✅ Valid" if status else "❌ Invalid", details)
                if not status:
                    all_passed = False
            except Exception as e:
                table.add_row(name, "❌ Error", str(e))
                all_passed = False
        
        console.print(table)
        
        if all_passed:
            console.print("[bold green]✅ All configuration checks passed![/bold green]")
            sys.exit(0)
        else:
            console.print("[bold red]❌ Configuration validation failed[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]❌ Configuration error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def test_foundation(ctx):
    """Test foundation components (database, logging, configuration)."""
    console.print("[bold blue]🧪 Testing CuliFeed Foundation[/bold blue]")
    
    try:
        # Initialize settings and logging
        settings = get_settings()
        configure_application_logging(
            log_level="DEBUG" if ctx.obj.get('debug') else settings.logging.level.value,
            log_file=settings.logging.file_path,
            enable_console=settings.logging.console_logging,
            structured_logging=settings.logging.structured_logging
        )
        
        logger = get_logger_for_component('foundation')
        
        tests = []
        
        # Test 1: Database Schema Creation
        console.print("\n[yellow]📊 Testing Database Schema...[/yellow]")
        try:
            schema = DatabaseSchema(settings.database.path)
            schema.create_tables()
            
            if schema.verify_schema():
                tests.append(("Database Schema", True, "All tables created and verified"))
                console.print("  ✅ Database schema created successfully")
            else:
                tests.append(("Database Schema", False, "Schema verification failed"))
                console.print("  ❌ Database schema verification failed")
        except Exception as e:
            tests.append(("Database Schema", False, str(e)))
            console.print(f"  ❌ Database error: {e}")
        
        # Test 2: Database Connection
        console.print("\n[yellow]🔌 Testing Database Connection...[/yellow]")
        try:
            db_manager = get_db_manager(settings.database.path)
            info = db_manager.get_database_info()
            tests.append(("Database Connection", True, f"Connected, {info['total_connections']} connections"))
            console.print(f"  ✅ Database connected - {info['database_size_mb']:.1f}MB")
        except Exception as e:
            tests.append(("Database Connection", False, str(e)))
            console.print(f"  ❌ Connection error: {e}")
        
        # Test 3: Logging System
        console.print("\n[yellow]📝 Testing Logging System...[/yellow]")
        try:
            logger.info("Foundation test log message")
            logger.debug("Debug level log message")
            logger.warning("Warning level log message")
            tests.append(("Logging System", True, f"Level: {settings.logging.level}"))
            console.print("  ✅ Logging system working")
        except Exception as e:
            tests.append(("Logging System", False, str(e)))
            console.print(f"  ❌ Logging error: {e}")
        
        # Test 4: Configuration Loading
        console.print("\n[yellow]⚙️ Testing Configuration...[/yellow]")
        try:
            fallback_providers = settings.get_ai_fallback_providers()
            effective_log_level = settings.get_effective_log_level()
            tests.append(("Configuration", True, f"AI providers: {len(fallback_providers)}"))
            console.print(f"  ✅ Configuration loaded - {len(fallback_providers)} AI providers available")
        except Exception as e:
            tests.append(("Configuration", False, str(e)))
            console.print(f"  ❌ Configuration error: {e}")
        
        # Results Summary
        console.print("\n[bold blue]📋 Foundation Test Results[/bold blue]")
        
        results_table = Table()
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Details")
        
        passed_count = 0
        for test_name, passed, details in tests:
            status = "✅ PASSED" if passed else "❌ FAILED"
            results_table.add_row(test_name, status, details)
            if passed:
                passed_count += 1
        
        console.print(results_table)
        
        if passed_count == len(tests):
            console.print(f"[bold green]🎉 All {len(tests)} foundation tests passed![/bold green]")
            sys.exit(0)
        else:
            console.print(f"[bold red]❌ {len(tests) - passed_count} out of {len(tests)} tests failed[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]❌ Foundation test error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def init_db(ctx):
    """Initialize database with schema."""
    console.print("[bold blue]🗄️ Initializing CuliFeed Database[/bold blue]")
    
    try:
        settings = get_settings()
        schema = DatabaseSchema(settings.database.path)
        
        # Create database directory if it doesn't exist
        Path(settings.database.path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        schema.create_tables()
        
        # Verify schema
        if schema.verify_schema():
            console.print("[bold green]✅ Database initialized successfully![/bold green]")
            
            # Show database info
            db_manager = get_db_manager(settings.database.path)
            info = db_manager.get_database_info()
            
            info_table = Table(title="Database Information")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="green")
            
            info_table.add_row("Database Path", settings.database.path)
            info_table.add_row("Size", f"{info['database_size_mb']:.2f} MB")
            info_table.add_row("Page Size", f"{info['page_size']} bytes")
            info_table.add_row("Connection Pool", f"{info['total_connections']} connections")
            
            console.print(info_table)
            
        else:
            console.print("[bold red]❌ Database schema verification failed[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]❌ Database initialization error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def create_config():
    """Create example configuration file."""
    config_path = Path("config.yaml")
    
    if config_path.exists():
        if not click.confirm(f"Config file {config_path} already exists. Overwrite?"):
            console.print("[yellow]Configuration creation cancelled[/yellow]")
            return
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(create_example_config())
        
        console.print(f"[bold green]✅ Configuration file created: {config_path}[/bold green]")
        console.print("[yellow]📝 Don't forget to:[/yellow]")
        console.print("  1. Copy .env.example to .env")
        console.print("  2. Fill in your API keys in .env")
        console.print("  3. Customize config.yaml as needed")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error creating config: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without deleting')
def cleanup(dry_run):
    """Clean up old data and optimize database."""
    console.print("[bold blue]🧹 CuliFeed Database Cleanup[/bold blue]")
    
    try:
        settings = get_settings()
        db_manager = get_db_manager(settings.database.path)
        
        if dry_run:
            console.print("[yellow]📋 Dry run mode - no changes will be made[/yellow]")
        
        # Get initial database info
        initial_info = db_manager.get_database_info()
        console.print(f"Initial database size: {initial_info['database_size_mb']:.2f} MB")
        
        if not dry_run:
            # Clean up old data
            deleted_count = db_manager.cleanup_old_data(settings.database.cleanup_days)
            console.print(f"Deleted {deleted_count} old records")
            
            # Vacuum database
            db_manager.vacuum_database()
            
            # Update statistics
            db_manager.analyze_database()
            
            # Get final database info
            final_info = db_manager.get_database_info()
            console.print(f"Final database size: {final_info['database_size_mb']:.2f} MB")
            
            space_saved = initial_info['database_size_mb'] - final_info['database_size_mb']
            console.print(f"[bold green]✅ Cleanup complete! Saved {space_saved:.2f} MB[/bold green]")
        else:
            console.print("[yellow]Use --cleanup (without --dry-run) to perform actual cleanup[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Cleanup error: {e}[/bold red]")
        sys.exit(1)


# Helper functions for configuration checks
def _check_database_config(settings) -> tuple[bool, str]:
    """Check database configuration."""
    try:
        db_path = Path(settings.database.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return True, f"Path: {settings.database.path}"
    except Exception as e:
        return False, str(e)


def _check_logging_config(settings) -> tuple[bool, str]:
    """Check logging configuration."""
    try:
        if settings.logging.file_path:
            log_path = Path(settings.logging.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return True, f"Level: {settings.logging.level}, Console: {settings.logging.console_logging}"
    except Exception as e:
        return False, str(e)


def _check_telegram_config(settings) -> tuple[bool, str]:
    """Check Telegram configuration."""
    try:
        if not settings.telegram.bot_token or settings.telegram.bot_token.startswith("${"):
            return False, "Bot token not set"
        if len(settings.telegram.bot_token.split(':')) != 2:
            return False, "Invalid bot token format"
        return True, "Bot token configured"
    except Exception as e:
        return False, str(e)


def _check_ai_config(settings) -> tuple[bool, str]:
    """Check AI providers configuration."""
    try:
        providers = settings.get_ai_fallback_providers()
        if not providers:
            return False, "No AI providers configured"
        primary = settings.processing.ai_provider
        return True, f"Primary: {primary}, Available: {len(providers)}"
    except Exception as e:
        return False, str(e)


def _check_processing_config(settings) -> tuple[bool, str]:
    """Check processing configuration."""
    try:
        return True, f"Hour: {settings.processing.daily_run_hour}, Max articles: {settings.processing.max_articles_per_topic}"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 CuliFeed interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Unexpected error: {e}[/bold red]")
        sys.exit(1)