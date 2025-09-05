# src/utils/spinner.py
import threading
import time
import sys
import asyncio


class Spinner:
    """Simple spinner for showing progress during long-running operations."""
    
    def __init__(self, message="Scanning", delay=0.1):
        self.spinner_symbols = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.delay = delay
        self.message = message
        self.running = False
        self.thread = None
    
    def spin(self):
        """Display the spinner."""
        while self.running:
            for symbol in self.spinner_symbols:
                if not self.running:
                    break
                sys.stdout.write(f'\r{symbol} {self.message}...')
                sys.stdout.flush()
                time.sleep(self.delay)
    
    def start(self):
        """Start the spinner."""
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self, success_message=None):
        """Stop the spinner."""
        self.running = False
        if self.thread:
            self.thread.join()
        
        # Clear the spinner line completely with more spaces
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        
        if success_message:
            print(f"✓ {success_message}")
            print()  # Add clean separation


class AsyncSpinner(Spinner):
    """Async-compatible spinner that works with asyncio operations."""
    
    def __init__(self, message="Processing", delay=0.1):
        super().__init__(message, delay)
    
    async def start_async(self):
        """Start the spinner in an async context."""
        self.start()
        # Give spinner a moment to start
        await asyncio.sleep(0.01)
    
    async def stop_async(self, success_message=None):
        """Stop the spinner in an async context."""
        self.stop(success_message)
        # Give spinner a moment to clean up
        await asyncio.sleep(0.01)


def run_with_spinner(func, spinner_message, success_message=None):
    """Run a function with a spinner display."""
    spinner = Spinner(spinner_message)
    
    try:
        # Start spinner to show progress
        spinner.start()
        
        # Let spinner run for a moment to show activity
        time.sleep(0.5)
        
        # STOP spinner before running function to prevent output collision
        spinner.stop()
        
        # Brief pause to ensure spinner stops completely
        time.sleep(0.1)
        
        # Now run the function - agent will have clean terminal
        result = func()
        
        # Show success message after completion
        if success_message:
            print(f"✓ {success_message}")
            print()
        
        return result
        
    except Exception as e:
        spinner.stop()
        print(f"✗ Error: {e}")
        return None
    except KeyboardInterrupt:
        spinner.stop()
        print("\n✗ Scan interrupted by user")
        sys.exit(1)