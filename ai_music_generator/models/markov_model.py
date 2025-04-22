"""
Markov Chain model for melody generation.
"""

import random
import numpy as np
from utils.music_theory import note_to_midi, midi_to_note

class MarkovModel:
    """
    Markov Chain model for melody generation.
    
    This model uses transition matrices to generate melodies based on
    probabilities of note transitions.
    """
    
    def __init__(self):
        # Pre-defined transition matrices for different genres and moods
        self.transition_matrices = self._initialize_transition_matrices()
        
    def _initialize_transition_matrices(self):
        """
        Initialize transition matrices for different genres and moods.
        
        Returns:
            dict: Dictionary of transition matrices
        """
        matrices = {}
        
        # Example: Pop genre, Happy mood
        # This is a simple first-order Markov chain where each row represents
        # the current note (as scale degree) and each column represents the
        # probability of transitioning to the next note.
        # 
        # Scale degrees: 1, 2, 3, 4, 5, 6, 7, 8 (octave)
        # 
        # For example, if we're on scale degree 1 (first row), we have:
        # - 10% chance to stay on 1
        # - 30% chance to go to 3
        # - 25% chance to go to 5
        # - 20% chance to go to 8 (octave)
        # - 15% chance to go to 2
        
        matrices[('Pop', 'Happy')] = np.array([
            [0.10, 0.15, 0.30, 0.00, 0.25, 0.00, 0.00, 0.20],  # From scale degree 1
            [0.20, 0.05, 0.40, 0.10, 0.15, 0.05, 0.05, 0.00],  # From scale degree 2
            [0.15, 0.10, 0.05, 0.25, 0.30, 0.10, 0.05, 0.00],  # From scale degree 3
            [0.05, 0.15, 0.20, 0.05, 0.40, 0.15, 0.00, 0.00],  # From scale degree 4
            [0.30, 0.05, 0.15, 0.10, 0.05, 0.20, 0.05, 0.10],  # From scale degree 5
            [0.10, 0.20, 0.10, 0.15, 0.25, 0.05, 0.15, 0.00],  # From scale degree 6
            [0.40, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.15],  # From scale degree 7
            [0.50, 0.10, 0.10, 0.05, 0.15, 0.05, 0.05, 0.00],  # From scale degree 8 (octave)
        ])
        
        # Pop genre, Sad mood
        matrices[('Pop', 'Sad')] = np.array([
            [0.15, 0.10, 0.05, 0.25, 0.10, 0.20, 0.10, 0.05],  # From scale degree 1
            [0.20, 0.10, 0.15, 0.30, 0.05, 0.15, 0.05, 0.00],  # From scale degree 2
            [0.10, 0.20, 0.10, 0.15, 0.10, 0.25, 0.10, 0.00],  # From scale degree 3
            [0.15, 0.10, 0.20, 0.10, 0.15, 0.20, 0.10, 0.00],  # From scale degree 4
            [0.20, 0.05, 0.10, 0.15, 0.10, 0.25, 0.15, 0.00],  # From scale degree 5
            [0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.20, 0.00],  # From scale degree 6
            [0.30, 0.10, 0.05, 0.15, 0.10, 0.20, 0.05, 0.05],  # From scale degree 7
            [0.40, 0.15, 0.10, 0.10, 0.15, 0.05, 0.05, 0.00],  # From scale degree 8 (octave)
        ])
        
        # Rock genre, Energetic mood
        matrices[('Rock', 'Energetic')] = np.array([
            [0.05, 0.10, 0.15, 0.05, 0.35, 0.05, 0.15, 0.10],  # From scale degree 1
            [0.15, 0.05, 0.20, 0.15, 0.25, 0.10, 0.10, 0.00],  # From scale degree 2
            [0.10, 0.15, 0.05, 0.10, 0.30, 0.15, 0.15, 0.00],  # From scale degree 3
            [0.05, 0.20, 0.25, 0.05, 0.20, 0.15, 0.10, 0.00],  # From scale degree 4
            [0.25, 0.05, 0.15, 0.10, 0.05, 0.15, 0.15, 0.10],  # From scale degree 5
            [0.10, 0.15, 0.20, 0.15, 0.20, 0.05, 0.15, 0.00],  # From scale degree 6
            [0.35, 0.10, 0.10, 0.05, 0.20, 0.10, 0.05, 0.05],  # From scale degree 7
            [0.40, 0.15, 0.10, 0.05, 0.20, 0.05, 0.05, 0.00],  # From scale degree 8 (octave)
        ])
        
        # Classical genre, Calm mood
        matrices[('Classical', 'Calm')] = np.array([
            [0.15, 0.25, 0.15, 0.10, 0.20, 0.05, 0.05, 0.05],  # From scale degree 1
            [0.20, 0.10, 0.25, 0.15, 0.10, 0.15, 0.05, 0.00],  # From scale degree 2
            [0.15, 0.20, 0.10, 0.25, 0.15, 0.10, 0.05, 0.00],  # From scale degree 3
            [0.10, 0.15, 0.20, 0.05, 0.30, 0.15, 0.05, 0.00],  # From scale degree 4
            [0.25, 0.10, 0.15, 0.10, 0.05, 0.25, 0.05, 0.05],  # From scale degree 5
            [0.15, 0.25, 0.15, 0.10, 0.20, 0.05, 0.10, 0.00],  # From scale degree 6
            [0.30, 0.15, 0.05, 0.10, 0.15, 0.15, 0.05, 0.05],  # From scale degree 7
            [0.35, 0.20, 0.15, 0.05, 0.15, 0.05, 0.05, 0.00],  # From scale degree 8 (octave)
        ])
        
        # Default matrix (used when specific genre/mood combination is not available)
        matrices['default'] = np.array([
            [0.10, 0.15, 0.20, 0.10, 0.20, 0.10, 0.05, 0.10],  # From scale degree 1
            [0.20, 0.05, 0.25, 0.15, 0.15, 0.15, 0.05, 0.00],  # From scale degree 2
            [0.15, 0.15, 0.05, 0.20, 0.25, 0.15, 0.05, 0.00],  # From scale degree 3
            [0.10, 0.15, 0.20, 0.05, 0.30, 0.15, 0.05, 0.00],  # From scale degree 4
            [0.25, 0.05, 0.15, 0.15, 0.05, 0.20, 0.10, 0.05],  # From scale degree 5
            [0.15, 0.20, 0.15, 0.15, 0.20, 0.05, 0.10, 0.00],  # From scale degree 6
            [0.35, 0.10, 0.05, 0.10, 0.15, 0.15, 0.05, 0.05],  # From scale degree 7
            [0.40, 0.15, 0.15, 0.05, 0.15, 0.05, 0.05, 0.00],  # From scale degree 8 (octave)
        ])
        
        return matrices
    
    def _get_transition_matrix(self, genre, mood):
        """
        Get the appropriate transition matrix based on genre and mood.
        
        Args:
            genre (str): Music genre
            mood (str): Mood/theme
            
        Returns:
            numpy.ndarray: Transition matrix
        """
        # Try to find an exact match
        key = (genre, mood)
        if key in self.transition_matrices:
            return self.transition_matrices[key]
        
        # Try to find a match with just the genre
        for k in self.transition_matrices:
            if isinstance(k, tuple) and k[0] == genre:
                return self.transition_matrices[k]
        
        # Return default matrix if no match found
        return self.transition_matrices['default']
    
    def _adjust_for_complexity(self, matrix, complexity):
        """
        Adjust transition matrix based on complexity level.
        
        Args:
            matrix (numpy.ndarray): Original transition matrix
            complexity (str): Complexity level - Simple/Intermediate/Complex
            
        Returns:
            numpy.ndarray: Adjusted transition matrix
        """
        if complexity == 'Simple':
            # For simple melodies, increase probability of common transitions
            # (e.g., 1->3, 5->1, etc.) and reduce probability of uncommon ones
            adjusted = matrix.copy()
            
            # Increase probability of moving to scale degrees 1, 3, and 5
            for i in range(8):
                total_increase = 0
                for j in [0, 2, 4]:  # Scale degrees 1, 3, 5 (0-indexed)
                    increase = min(0.1, 1.0 - adjusted[i, j])
                    adjusted[i, j] += increase
                    total_increase += increase
                
                # Decrease other probabilities proportionally
                if total_increase > 0:
                    other_indices = [j for j in range(8) if j not in [0, 2, 4]]
                    total_other = sum(adjusted[i, j] for j in other_indices)
                    
                    if total_other > 0:
                        for j in other_indices:
                            adjusted[i, j] -= (adjusted[i, j] / total_other) * total_increase
            
            # Normalize rows to ensure they sum to 1
            for i in range(8):
                adjusted[i] = adjusted[i] / adjusted[i].sum()
                
            return adjusted
            
        elif complexity == 'Complex':
            # For complex melodies, make transitions more uniform
            # to increase unpredictability and use more varied notes
            adjusted = matrix.copy()
            
            # Make distribution more uniform by moving probabilities toward 1/8
            for i in range(8):
                for j in range(8):
                    adjusted[i, j] = 0.7 * adjusted[i, j] + 0.3 * (1/8)
            
            # Normalize rows to ensure they sum to 1
            for i in range(8):
                adjusted[i] = adjusted[i] / adjusted[i].sum()
                
            return adjusted
            
        else:  # 'Intermediate' or any other value
            # Return the original matrix
            return matrix
    
    def generate_melody(self, scale_notes, num_bars, complexity='Simple', mood='Neutral', genre='Pop', chord_progression=None):
        """
        Generate a melody using the Markov model.
        
        Args:
            scale_notes (list): List of notes in the scale
            num_bars (int): Number of bars to generate
            complexity (str): Complexity level - Simple/Intermediate/Complex
            mood (str): Mood/theme
            genre (str): Music genre
            chord_progression (list, optional): List of chords to influence the melody
            
        Returns:
            list: List of (note, duration) tuples representing the melody
        """
        # Get transition matrix based on genre and mood
        matrix = self._get_transition_matrix(genre, mood)
        
        # Adjust matrix based on complexity
        matrix = self._adjust_for_complexity(matrix, complexity)
        
        # Determine notes per bar based on complexity
        if complexity == 'Simple':
            notes_per_bar = 4  # Quarter notes
        elif complexity == 'Intermediate':
            notes_per_bar = 8  # Eighth notes
        else:  # 'Complex'
            notes_per_bar = 16  # Sixteenth notes
        
        # Initialize melody with scale degree 1 (tonic)
        current_degree = 0  # 0-indexed (scale degree 1)
        melody = []
        
        # Generate notes for each bar
        for bar in range(num_bars):
            # If chord progression is provided, use it to influence note selection
            if chord_progression and bar < len(chord_progression):
                current_chord = chord_progression[bar]
                # Extract chord notes (e.g., for C major, extract C, E, G)
                chord_notes = [note for note in current_chord.split() if note in scale_notes]
                
                # Increase probability of chord tones for this bar
                temp_matrix = matrix.copy()
                for i in range(8):
                    for j, note in enumerate(scale_notes):
                        if j < 8 and note in chord_notes:
                            # Increase probability of chord tones
                            temp_matrix[i, j] *= 1.5
                    
                    # Normalize row
                    temp_matrix[i] = temp_matrix[i] / temp_matrix[i].sum()
            else:
                temp_matrix = matrix
            
            # Generate notes for this bar
            for _ in range(notes_per_bar):
                # Get next note based on transition probabilities
                next_degree = np.random.choice(8, p=temp_matrix[current_degree])
                
                # Determine note duration based on position in bar and complexity
                if complexity == 'Simple':
                    duration = 0.25  # Quarter note
                elif complexity == 'Intermediate':
                    # Mix of eighth and quarter notes
                    duration = random.choice([0.125, 0.25])
                else:  # 'Complex'
                    # Mix of sixteenth, eighth, and quarter notes
                    duration = random.choice([0.0625, 0.125, 0.25])
                
                # Add note to melody
                note = scale_notes[next_degree % len(scale_notes)]
                # Adjust octave for higher scale degrees
                octave = 4 + (next_degree // len(scale_notes))
                melody.append((f"{note}{octave}", duration))
                
                # Update current degree
                current_degree = next_degree % 8
        
        return melody