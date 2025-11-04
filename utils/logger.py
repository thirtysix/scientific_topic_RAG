"""
Structured logging system for the RAG pipeline
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def get_logger(name: str, 
               level: str = "INFO",
               log_file: Optional[str] = None,
               log_format: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        log_file: Optional file to log to
        log_format: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class PipelineLogger:
    """Enhanced logger for pipeline operations with structured logging"""
    
    def __init__(self, name: str, config = None):
        self.config = config
        self.logger = get_logger(
            name,
            level=config.logging.level if config else "INFO",
            log_file=config.logging.log_file if config else None,
            log_format=config.logging.log_format if config else None
        )
        
    def log_step_start(self, step_name: str, **kwargs):
        """Log the start of a pipeline step"""
        self.logger.info(f"🚀 Starting step: {step_name}")
        for key, value in kwargs.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_step_complete(self, step_name: str, duration: float, **kwargs):
        """Log the completion of a pipeline step"""
        self.logger.info(f"✅ Completed step: {step_name} ({duration:.2f}s)")
        for key, value in kwargs.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_step_error(self, step_name: str, error: Exception):
        """Log an error in a pipeline step"""
        self.logger.error(f"❌ Failed step: {step_name}")
        self.logger.error(f"  Error: {str(error)}")
    
    def log_stats(self, title: str, stats: dict):
        """Log statistics in a structured format"""
        self.logger.info(f"📊 {title}")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_progress(self, current: int, total: int, item_name: str = "items"):
        """Log progress information"""
        percentage = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"⏳ Progress: {current}/{total} {item_name} ({percentage:.1f}%)")
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)