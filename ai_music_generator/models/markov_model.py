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
        matrices[('Pop', 'Happy')] = np.array([
            [0.10, 0.15, 0.30, 0.00, 0.25, 0.00, 0.00, 0.20],
            [0.20, 0.05, 0.40, 0.10, 0.15, 0.05, 0.05, 0.00],
            [0.15, 0.10, 0.05, 0.25, 0.30, 0.10, 0.05, 0.00],
            [0.05, 0.15, 0.20, 0.05, 0.40, 0.15, 0.00, 0.00],
            [0.30, 0.05, 0.15, 0.10, 0.05, 0.20, 0.05, 0.10],
            [0.10, 0.20, 0.10, 0.15, 0.25, 0.05, 0.15, 0.00],
            [0.40, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.15],
            [0.50, 0.10, 0.10, 0.05, 0.15, 0.05, 0.05, 0.00],
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
        
        # ... (other matrices unchanged) ...
        matrices['default'] = np.array([
            [0.10, 0.15, 0.20, 0.10, 0.20, 0.10, 0.05, 0.10],
            [0.20, 0.05, 0.25, 0.15, 0.15, 0.15, 0.05, 0.00],
            [0.15, 0.15, 0.05, 0.20, 0.25, 0.15, 0.05, 0.00],
            [0.10, 0.15, 0.20, 0.05, 0.30, 0.15, 0.05, 0.00],
            [0.25, 0.05, 0.15, 0.15, 0.05, 0.20, 0.10, 0.05],
            [0.15, 0.20, 0.15, 0.15, 0.20, 0.05, 0.10, 0.00],
            [0.35, 0.10, 0.05, 0.10, 0.15, 0.15, 0.05, 0.05],
            [0.40, 0.15, 0.15, 0.05, 0.15, 0.05, 0.05, 0.00],
        ])
        return matrices

    def _get_transition_matrix(self, genre, mood):
        key = (genre, mood)
        if key in self.transition_matrices:
            return self.transition_matrices[key]
        for k in self.transition_matrices:
            if isinstance(k, tuple) and k[0] == genre:
                return self.transition_matrices[k]
        return self.transition_matrices['default']

    def _adjust_for_complexity(self, matrix, complexity):
        if complexity == 'Simple':
            adjusted = matrix.copy()
            for i in range(8):
                total_increase = 0
                for j in [0, 2, 4]:
                    increase = min(0.1, 1.0 - adjusted[i, j])
                    adjusted[i, j] += increase
                    total_increase += increase
                if total_increase > 0:
                    other = [j for j in range(8) if j not in [0, 2, 4]]
                    total_other = sum(adjusted[i, j] for j in other)
                    if total_other > 0:
                        for j in other:
                            adjusted[i, j] -= (adjusted[i, j] / total_other) * total_increase
            for i in range(8):
                adjusted[i] /= adjusted[i].sum()
            return adjusted
        elif complexity == 'Complex':
            adjusted = matrix.copy()
            for i in range(8):
                for j in range(8):
                    adjusted[i, j] = 0.7 * adjusted[i, j] + 0.3 * (1/8)
            for i in range(8):
                adjusted[i] /= adjusted[i].sum()
            return adjusted
        else:
            return matrix

    def generate_melody(self, scale_notes, num_bars, complexity='Simple', mood='Neutral', genre='Pop', chord_progression=None, tempo='Medium'):
        matrix = self._get_transition_matrix(genre, mood)
        matrix = self._adjust_for_complexity(matrix, complexity)
        
        # Adjust notes per bar based on tempo and complexity
        if tempo == 'Slow':
            if complexity == 'Simple':
                notes_per_bar = 2  # Fewer notes for slow tempo
            elif complexity == 'Intermediate':
                notes_per_bar = 4
            else:
                notes_per_bar = 8
        elif tempo == 'Fast':
            if complexity == 'Simple':
                notes_per_bar = 8  # More notes for fast tempo
            elif complexity == 'Intermediate':
                notes_per_bar = 12
            else:
                notes_per_bar = 16
        else:  # Medium tempo
            if complexity == 'Simple':
                notes_per_bar = 4
            elif complexity == 'Intermediate':
                notes_per_bar = 8
            else:
                notes_per_bar = 12
        
        current_degree = 0
        melody = []
        
        # Adjust the transition matrix based on tempo
        if tempo == 'Fast':
            # For fast tempo, increase probability of larger jumps
            for i in range(8):
                for j in range(8):
                    # Increase probability of jumps of 3 or more steps
                    if abs(i - j) >= 3:
                        matrix[i, j] *= 1.5
                # Renormalize
                matrix[i] /= matrix[i].sum()
        elif tempo == 'Slow':
            # For slow tempo, increase probability of smaller steps
            for i in range(8):
                for j in range(8):
                    # Increase probability of steps of 2 or less
                    if abs(i - j) <= 2:
                        matrix[i, j] *= 1.5
                # Renormalize
                matrix[i] /= matrix[i].sum()
        
        for bar in range(num_bars):
            if chord_progression and bar < len(chord_progression):
                chord_notes = [note for note in chord_progression[bar].split() if note in scale_notes]
                temp_matrix = matrix.copy()
                for i in range(8):
                    for j, note in enumerate(scale_notes):
                        if note in chord_notes:
                            temp_matrix[i, j] *= 1.5
                    temp_matrix[i] /= temp_matrix[i].sum()
            else:
                temp_matrix = matrix
            
            for _ in range(notes_per_bar):
                # Safely sample next note: clip negatives and renormalize
                row = temp_matrix[current_degree].copy()
                row = np.clip(row, 0, None)
                total = row.sum()
                if total <= 0:
                    row = np.ones_like(row) / len(row)
                else:
                    row = row / total
                next_degree = np.random.choice(len(row), p=row)

                # Adjust note durations based on tempo
                if tempo == 'Slow':
                    if complexity == 'Simple':
                        duration = random.choice([0.5, 0.25])  # Longer notes for slow tempo
                    elif complexity == 'Intermediate':
                        duration = random.choice([0.25, 0.5])
                    else:
                        duration = random.choice([0.125, 0.25, 0.5])
                elif tempo == 'Fast':
                    if complexity == 'Simple':
                        duration = random.choice([0.125, 0.25])  # Shorter notes for fast tempo
                    elif complexity == 'Intermediate':
                        duration = random.choice([0.0625, 0.125, 0.25])
                    else:
                        duration = random.choice([0.0625, 0.125])
                else:  # Medium tempo
                    if complexity == 'Simple':
                        duration = 0.25
                    elif complexity == 'Intermediate':
                        duration = random.choice([0.125, 0.25])
                    else:
                        duration = random.choice([0.0625, 0.125, 0.25])

                note = scale_notes[next_degree % len(scale_notes)]
                octave = 4 + (next_degree // len(scale_notes))
                melody.append((f"{note}{octave}", duration))
                current_degree = next_degree % 8
        
        return melody
