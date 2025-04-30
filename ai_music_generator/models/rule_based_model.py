"""
Rule-based model for music generation using music theory principles.
"""

import random
from utils.music_theory import get_scale_notes, get_chord_notes, note_to_midi, midi_to_note

class RuleBasedModel:
    """
    Rule-based model for music generation.
    
    This model applies music theory rules to generate melodies and harmonies.
    """
    
    def __init__(self):
        # Define common chord progressions for different genres
        self.chord_progressions = {
            'Pop': [
                ['I', 'V', 'vi', 'IV'],  # Most common pop progression
                ['I', 'IV', 'V'],         # Simple pop progression
                ['vi', 'IV', 'I', 'V'],   # Pop progression starting on vi
            ],
            'Rock': [
                ['I', 'IV', 'V'],         # Classic rock progression
                ['I', 'V', 'IV'],         # Rock progression
                ['I', 'bVII', 'IV'],      # Rock progression with flat VII
                ['i', 'bVI', 'bVII'],     # Minor rock progression
            ],
            'Jazz': [
                ['ii', 'V', 'I'],         # Classic jazz progression
                ['I', 'vi', 'ii', 'V'],   # Jazz turnaround
                ['iii', 'VI', 'ii', 'V'],  # Jazz progression
            ],
            'Classical': [
                ['I', 'IV', 'V', 'I'],    # Classical cadence
                ['I', 'ii', 'V', 'I'],    # Classical progression
                ['vi', 'ii', 'V', 'I'],   # Classical progression starting on vi
            ],
            'Blues': [
                ['I', 'IV', 'I', 'V', 'IV', 'I'],  # 12-bar blues (simplified)
            ],
        }
        
        # Define rhythm patterns for different complexity levels
        self.rhythm_patterns = {
            'Simple': [
                [0.25, 0.25, 0.25, 0.25],  # Four quarter notes
                [0.5, 0.25, 0.25],         # Half note + two quarter notes
                [0.25, 0.25, 0.5],         # Two quarter notes + half note
            ],
            'Intermediate': [
                [0.125, 0.125, 0.25, 0.125, 0.125, 0.25],  # Eighth notes + quarter notes
                [0.25, 0.125, 0.125, 0.25, 0.25],          # Quarter + eighth + eighth + quarter + quarter
                [0.125, 0.125, 0.125, 0.125, 0.25, 0.25],  # Four eighth notes + two quarter notes
            ],
            'Complex': [
                [0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.25],  # Sixteenth + eighth + quarter
                [0.125, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.125],  # Mixed rhythm
                [0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.125, 0.25],  # Complex rhythm
            ],
        }
        
        # Define melodic patterns for different moods
        self.melodic_patterns = {
            'Happy': [
                [0, 2, 4, 7],      # Major triad with octave
                [0, 4, 7, 4, 0],    # Major triad up and down
                [0, 2, 4, 5, 7],    # Major scale fragment ascending
            ],
            'Sad': [
                [0, 3, 7, 10],     # Minor seventh chord
                [0, -2, -3, -5],    # Descending minor scale fragment
                [0, 3, 2, 0],       # Minor third then step down
            ],
            'Energetic': [
                [0, 7, 12, 7],     # Octave jump pattern
                [0, 4, 7, 11],      # Major seventh chord
                [0, 2, 4, 7, 9, 12],  # Major scale with 6th and octave
            ],
            'Calm': [
                [0, 5, 7, 12],     # Perfect fifth and octave
                [0, 2, 0, -3],      # Small intervals
                [0, 4, 7, 5, 4, 0],  # Gentle arpeggios
            ],
            'Neutral': [
                [0, 4, 7],         # Major triad
                [0, 3, 7],          # Minor triad
                [0, 2, 4, 5, 7, 9, 11, 12],  # Major scale
            ],
        }
    
    def _get_chord_progression_for_genre(self, genre):
        """
        Get a chord progression for the specified genre.
        
        Args:
            genre (str): Music genre
            
        Returns:
            list: List of chord symbols (e.g., ['I', 'IV', 'V'])
        """
        if genre in self.chord_progressions:
            return random.choice(self.chord_progressions[genre])
        else:
            # Default to pop progression if genre not found
            return random.choice(self.chord_progressions['Pop'])
    
    def _get_rhythm_pattern(self, complexity, tempo='Medium'):
        """
        Get a rhythm pattern for the specified complexity level and tempo.
        
        Args:
            complexity (str): Complexity level - Simple/Intermediate/Complex
            tempo (str): Tempo - Slow/Medium/Fast
            
        Returns:
            list: List of note durations
        """
        # Define tempo-specific rhythm patterns
        tempo_patterns = {
            'Slow': {
                'Simple': [
                    [0.5, 0.5],  # Two half notes
                    [0.75, 0.25],  # Dotted half + quarter
                    [0.5, 0.25, 0.25],  # Half + two quarters
                ],
                'Intermediate': [
                    [0.5, 0.25, 0.25, 0.5],  # Half + two quarters + half
                    [0.25, 0.5, 0.25, 0.5],  # Quarter + half + quarter + half
                    [0.5, 0.5, 0.25, 0.25],  # Two halves + two quarters
                ],
                'Complex': [
                    [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],  # More varied rhythm
                    [0.5, 0.125, 0.125, 0.25, 0.5, 0.25],  # Mix of durations
                    [0.375, 0.125, 0.25, 0.25, 0.5, 0.25],  # Complex rhythm
                ]
            },
            'Fast': {
                'Simple': [
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # Eight eighth notes
                    [0.125, 0.125, 0.25, 0.125, 0.125, 0.25],  # Faster rhythm
                    [0.25, 0.125, 0.125, 0.25, 0.25],  # Mix of quarters and eighths
                ],
                'Intermediate': [
                    [0.0625, 0.0625, 0.125, 0.125, 0.0625, 0.0625, 0.125, 0.125, 0.25],  # Very fast
                    [0.125, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # Fast with variations
                    [0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.125],  # Sixteenths and eighths
                ],
                'Complex': [
                    [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125],  # Very complex
                    [0.0625, 0.03125, 0.03125, 0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.125, 0.125, 0.125],  # Extremely varied
                    [0.125, 0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.125, 0.125],  # Fast complex rhythm
                ]
            }
        }
        
        # If we have a specific tempo pattern, use it
        if tempo in tempo_patterns and complexity in tempo_patterns[tempo]:
            return random.choice(tempo_patterns[tempo][complexity])
        
        # Otherwise fall back to the default patterns
        if complexity in self.rhythm_patterns:
            return random.choice(self.rhythm_patterns[complexity])
        else:
            # Default to simple rhythm if complexity not found
            return random.choice(self.rhythm_patterns['Simple'])
    
    def _get_melodic_pattern(self, mood):
        """
        Get a melodic pattern for the specified mood.
        
        Args:
            mood (str): Mood/theme
            
        Returns:
            list: List of relative note positions
        """
        if mood in self.melodic_patterns:
            return random.choice(self.melodic_patterns[mood])
        else:
            # Default to neutral mood if mood not found
            return random.choice(self.melodic_patterns['Neutral'])
    
    def _apply_melodic_pattern(self, pattern, scale_notes, start_note_idx):
        """
        Apply a melodic pattern starting from a specific note in the scale.
        
        Args:
            pattern (list): List of relative note positions
            scale_notes (list): List of notes in the scale
            start_note_idx (int): Index of the starting note in the scale
            
        Returns:
            list: List of notes
        """
        notes = []
        for offset in pattern:
            idx = (start_note_idx + offset) % len(scale_notes)
            octave_shift = (start_note_idx + offset) // len(scale_notes)
            note = scale_notes[idx]
            # Extract the note name and current octave
            if note[-1].isdigit():
                note_name = note[:-1]
                base_octave = int(note[-1])
            else:
                note_name = note
                base_octave = 4  # Default octave
            
            # Apply octave shift
            new_octave = base_octave + octave_shift
            notes.append(f"{note_name}{new_octave}")
        
        return notes
    
    def generate_composition(self, scale, chord_progression, num_bars, complexity='Simple', mood='Neutral', tempo='Medium'):
        """
        Generate a composition using rule-based approach.
        
        Args:
            scale (str): Musical scale/key (e.g., "C Major")
            chord_progression (list): List of chord symbols or chord names
            num_bars (int): Number of bars to generate
            complexity (str): Complexity level - Simple/Intermediate/Complex
            mood (str): Mood/theme
            tempo (str): Tempo - Slow/Medium/Fast
            
        Returns:
            tuple: (melody, harmony) where melody is a list of (note, duration) tuples
                  and harmony is a list of chord names
        """
        # Get scale notes
        scale_notes = get_scale_notes(scale)
        
        # Extend chord progression to cover all bars
        extended_progression = chord_progression * (num_bars // len(chord_progression) + 1)
        harmony = extended_progression[:num_bars]
        
        # Generate melody
        melody = []
        
        for bar, chord in enumerate(harmony):
            # Get chord notes
            chord_notes = get_chord_notes(chord, scale)
            
            # Choose a rhythm pattern based on complexity and tempo
            rhythm = self._get_rhythm_pattern(complexity, tempo)
            
            # Choose a melodic pattern based on mood
            melodic_pattern = self._get_melodic_pattern(mood)
            
            # Determine starting note (prefer chord tones)
            chord_indices = [i for i, note in enumerate(scale_notes) if note in chord_notes]
            if chord_indices:
                start_note_idx = random.choice(chord_indices)
            else:
                start_note_idx = random.randint(0, len(scale_notes) - 1)
            
            # Apply melodic pattern
            notes = self._apply_melodic_pattern(melodic_pattern, scale_notes, start_note_idx)
            
            # Combine notes with rhythm
            # If we have more notes than rhythm values, repeat the rhythm pattern
            extended_rhythm = rhythm * (len(notes) // len(rhythm) + 1)
            extended_rhythm = extended_rhythm[:len(notes)]
            
            for note, duration in zip(notes, extended_rhythm):
                melody.append((note, duration))
        
        return melody, harmony