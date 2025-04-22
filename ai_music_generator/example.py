#!/usr/bin/env python3
"""
Example script demonstrating how to use the AI Music Generator programmatically.
"""

import os
from core.music_generator import MusicGenerator

def main():
    """Generate music with different parameters and modes."""
    # Create a MusicGenerator instance
    generator = MusicGenerator()
    
    # Example 1: Generate a simple pop song with default parameters
    print("Generating a simple pop song...")
    mp3_path = generator.generate_music({
        'genre': 'Pop',
        'instruments': ['Piano'],
        'scale': 'C Major',
        'mood': 'Happy',
        'tempo': 'Medium',
        'length': 'Short',
        'complexity': 'Simple',
        'mode': 'hybrid'
    })
    print(f"Generated music saved to: {mp3_path}")
    
    # Example 2: Generate a complex jazz song with multiple instruments
    print("\nGenerating a complex jazz song...")
    mp3_path = generator.generate_music({
        'genre': 'Jazz',
        'instruments': ['Piano', 'Bass', 'Saxophone'],
        'scale': 'D Minor',
        'mood': 'Calm',
        'tempo': 'Slow',
        'length': 'Medium',
        'complexity': 'Complex',
        'mode': 'markov'
    })
    print(f"Generated music saved to: {mp3_path}")
    
    # Example 3: Generate an energetic rock song using rule-based approach
    print("\nGenerating an energetic rock song...")
    mp3_path = generator.generate_music({
        'genre': 'Rock',
        'instruments': ['Electric Guitar', 'Bass', 'Drums'],
        'scale': 'E Minor',
        'mood': 'Energetic',
        'tempo': 'Fast',
        'length': 'Medium',
        'complexity': 'Intermediate',
        'mode': 'rule'
    })
    print(f"Generated music saved to: {mp3_path}")
    
    print("\nAll examples completed. Check the 'output' directory for the generated MP3 files.")

if __name__ == "__main__":
    main()