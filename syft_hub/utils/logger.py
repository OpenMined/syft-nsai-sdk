import logging
from colorama import Fore, Style

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')

# Define custom colors
COLOR_MAP = {
    'ERROR': Fore.RED,
    'SUCCESS': Fore.GREEN,
    'INFO': Fore.BLUE,
    'DEBUG': Fore.YELLOW,
    'WARNING': Fore.YELLOW,
}

class ColoredFormatter(logging.Formatter):
    """A custom formatter that adds color to log messages."""

    def format(self, record):
        log_message = super().format(record)
        color = COLOR_MAP.get(record.levelname, Fore.RESET)
        return f"{color}{log_message}{Style.RESET_ALL}"

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

logging.Logger.success = success

def get_logger(name: str = "syft_hub"):
    """
    Returns a custom logger with a colored formatter.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if the logger already has one
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        
        # Create a console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        
        # Create the custom colored formatter
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger