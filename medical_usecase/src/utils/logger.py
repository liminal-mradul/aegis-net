import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import threading

# Lock for thread-safe console output
_console_lock = threading.Lock()

def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Create logger with console and rotating file handlers.
    
    Args:
        name: Logger name (e.g., 'aegis.node')
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler - only show INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Add filter to prevent background thread spam
    console_handler.addFilter(ConsoleFilter())
    
    # Format: [TIMESTAMP] LEVEL - Component - Message
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler - 10MB max, keep 5 backups
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'aegis.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

class ConsoleFilter(logging.Filter):
    """Filter to reduce console noise from background threads"""
    
    # Specific messages to suppress from console (still in file)
    SUPPRESS_PATTERNS = [
        'Removing unresponsive peer',
        'appears offline',
    ]
    
    # Logger names to limit console output
    SUPPRESS_LOGGERS = [
        'aegis.peer',  # Only show errors
    ]
    
    def filter(self, record):
        """
        Determine if log record should appear on console.
        Returns True to show, False to suppress.
        """
        # Check for specific patterns to suppress
        msg = record.getMessage()
        for pattern in self.SUPPRESS_PATTERNS:
            if pattern in msg:
                return False
        
        # For peer manager, only show errors and above
        for logger_name in self.SUPPRESS_LOGGERS:
            if record.name.startswith(logger_name):
                if record.levelno < logging.ERROR:
                    return False
        
        return True

class OperationLogger:
    """Context manager for logging timed operations"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            # Log with full context for debugging
            self.logger.error(
                f"Failed: {self.operation} after {duration:.3f}s - {exc_val}",
                exc_info=True
            )
            # Don't suppress exception
            return False
        else:
            self.logger.debug(f"Completed: {self.operation} in {duration:.3f}s")
            return True
