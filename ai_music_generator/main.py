#!/usr/bin/env python3
"""
AI Music Generator - Main Application
A thesis project on "The Impact of Artificial Intelligence on the Music Industry"
"""

import argparse
import os
from ui.web_interface import start_web_server

def main():
    """Main entry point for the AI Music Generator application."""
    parser = argparse.ArgumentParser(description='AI Music Generator')
    parser.add_argument('--web', action='store_true', help='Start the web interface')
    parser.add_argument('--port', type=int, default=12000, help='Port for web interface')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    if args.web:
        start_web_server(port=args.port)
    else:
        print("Command-line interface not implemented yet. Use --web to start the web interface.")

if __name__ == "__main__":
    main()