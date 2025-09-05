# cli/main.py  
"""
Main CLI entry point for SyftBox NSAI SDK
"""
import sys
from .commands import cli_main

def main():
    """Main entry point for the CLI."""
    cli_main()

if __name__ == "__main__":
    main()