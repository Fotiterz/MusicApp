"""
Music theory utilities for the AI Music Generator.
"""

import re

# Define note to MIDI number mapping
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

# Define scale patterns (semitone intervals)
SCALE_PATTERNS = {
    'Major': [0, 2, 4, 5, 7, 9, 11],
    'Minor': [0, 2, 3, 5, 7, 8, 10],
    'Dorian': [0, 2, 3, 5, 7, 9, 10],
    'Phrygian': [0, 1, 3, 5, 7, 8, 10],
    'Lydian': [0, 2, 4, 6, 7, 9, 11],
    'Mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'Locrian': [0, 1, 3, 5, 6, 8, 10],
    'Blues': [0, 3, 5, 6, 7, 10],
    'Pentatonic Major': [0, 2, 4, 7, 9],
    'Pentatonic Minor': [0, 3, 5, 7, 10],
}

# Define chord patterns (semitone intervals from root)
CHORD_PATTERNS = {
    'Major': [0, 4, 7],
    'Minor': [0, 3, 7],
    'Diminished': [0, 3, 6],
    'Augmented': [0, 4, 8],
    'Sus2': [0, 2, 7],
    'Sus4': [0, 5, 7],
    'Major 7': [0, 4, 7, 11],
    'Minor 7': [0, 3, 7, 10],
    'Dominant 7': [0, 4, 7, 10],
    'Diminished 7': [0, 3, 6, 9],
    'Half Diminished 7': [0, 3, 6, 10],
    'Augmented 7': [0, 4, 8, 10],
}

# Define chord symbols to chord types mapping
CHORD_SYMBOLS = {
    'I': 'Major',
    'i': 'Minor',
    'II': 'Major',
    'ii': 'Minor',
    'III': 'Major',
    'iii': 'Minor',
    'IV': 'Major',
    'iv': 'Minor',
    'V': 'Major',
    'v': 'Minor',
    'VI': 'Major',
    'vi': 'Minor',
    'VII': 'Major',
    'vii': 'Diminished',
    'bII': 'Major',
    'bIII': 'Major',
    'bV': 'Major',
    'bVI': 'Major',
    'bVII': 'Major',
}

# Define scale degree to semitone mapping for major and minor scales
SCALE_DEGREE_TO_SEMITONE = {
    'Major': {
        'I': 0, 'II': 2, 'III': 4, 'IV': 5, 'V': 7, 'VI': 9, 'VII': 11,
        'i': 0, 'ii': 2, 'iii': 4, 'iv': 5, 'v': 7, 'vi': 9, 'vii': 11,
        'bII': 1, 'bIII': 3, 'bV': 6, 'bVI': 8, 'bVII': 10,
    },
    'Minor': {
        'I': 0, 'II': 2, 'III': 3, 'IV': 5, 'V': 7, 'VI': 8, 'VII': 10,
        'i': 0, 'ii': 2, 'iii': 3, 'iv': 5, 'v': 7, 'vi': 8, 'vii': 10,
        'bII': 1, 'bIII': 2, 'bV': 6, 'bVI': 7, 'bVII': 9,
    }
}

def note_to_midi(note, octave=4):
    """
    Convert a note name to MIDI number.
    
    Args:
        note (str): Note name (e.g., 'C', 'C#', 'Db')
        octave (int): Octave number
        
    Returns:
        int: MIDI number
    """
    if note in NOTE_TO_MIDI:
        return NOTE_TO_MIDI[note] + (octave + 1) * 12
    else:
        raise ValueError(f"Invalid note: {note}")

def midi_to_note(midi_num):
    """
    Convert a MIDI number to note name and octave.
    
    Args:
        midi_num (int): MIDI number
        
    Returns:
        tuple: (note, octave)
    """
    octave = (midi_num // 12) - 1
    note_num = midi_num % 12
    
    # Use sharps by default
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    return notes[note_num], octave

def parse_scale(scale_str):
    """
    Parse a scale string into root note and scale type.
    
    Args:
        scale_str (str): Scale string (e.g., 'C Major', 'A Minor')
        
    Returns:
        tuple: (root_note, scale_type)
    """
    parts = scale_str.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid scale format: {scale_str}. Expected format: 'C Major'")
    
    root_note = parts[0]
    scale_type = ' '.join(parts[1:])
    
    if root_note not in NOTE_TO_MIDI:
        raise ValueError(f"Invalid root note: {root_note}")
    
    if scale_type not in SCALE_PATTERNS:
        raise ValueError(f"Invalid scale type: {scale_type}")
    
    return root_note, scale_type

def get_scale_notes(scale_str, octave=4):
    """
    Get the notes in a scale.
    
    Args:
        scale_str (str): Scale string (e.g., 'C Major', 'A Minor')
        octave (int): Base octave
        
    Returns:
        list: List of notes in the scale with octave numbers
    """
    root_note, scale_type = parse_scale(scale_str)
    root_midi = note_to_midi(root_note, octave)
    
    scale_notes = []
    for interval in SCALE_PATTERNS[scale_type]:
        midi_num = root_midi + interval
        note, note_octave = midi_to_note(midi_num)
        scale_notes.append(f"{note}{note_octave}")
    
    return scale_notes

def get_chord_progression(genre, scale_str):
    """
    Get a chord progression for a genre in a specific scale.
    
    Args:
        genre (str): Music genre
        scale_str (str): Scale string (e.g., 'C Major', 'A Minor')
        
    Returns:
        list: List of chord names
    """
    root_note, scale_type = parse_scale(scale_str)
    
    # Define common chord progressions for different genres
    progressions = {
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
    
    # Get a progression for the genre
    if genre in progressions:
        progression = progressions[genre][0]  # Use the first progression for simplicity
    else:
        # Default to pop progression
        progression = progressions['Pop'][0]
    
    # Convert chord symbols to actual chord names
    chord_names = []
    for symbol in progression:
        # Get the scale degree (Roman numeral)
        scale_degree = re.sub(r'[^IViv]', '', symbol)
        
        # Get any modifiers (e.g., 7, maj7, etc.)
        modifiers = re.sub(r'[IViv]', '', symbol)
        
        # Determine if it's a major or minor scale
        scale_mode = 'Major' if scale_type == 'Major' else 'Minor'
        
        # Get the semitone offset for this scale degree
        if scale_degree in SCALE_DEGREE_TO_SEMITONE[scale_mode]:
            semitone_offset = SCALE_DEGREE_TO_SEMITONE[scale_mode][scale_degree]
        else:
            # Default to tonic if scale degree not found
            semitone_offset = 0
        
        # Calculate the root note of the chord
        root_midi = note_to_midi(root_note, 4) + semitone_offset
        chord_root, chord_octave = midi_to_note(root_midi)
        
        # Determine chord type based on the symbol
        if scale_degree.islower():
            chord_type = 'Minor'
        elif scale_degree == 'vii' or scale_degree == 'VII' and scale_mode == 'Major':
            chord_type = 'Diminished'
        else:
            chord_type = 'Major'
        
        # Add modifiers
        if '7' in modifiers:
            if chord_type == 'Major':
                chord_type = 'Dominant 7'
            elif chord_type == 'Minor':
                chord_type = 'Minor 7'
            elif chord_type == 'Diminished':
                chord_type = 'Diminished 7'
        
        # Create chord name
        chord_name = f"{chord_root} {chord_type}"
        chord_names.append(chord_name)
    
    return chord_names

def get_chord_notes(chord_name, scale_str=None):
    """
    Get the notes in a chord.
    
    Args:
        chord_name (str): Chord name (e.g., 'C Major', 'A Minor 7')
        scale_str (str, optional): Scale string for context
        
    Returns:
        list: List of notes in the chord
    """
    parts = chord_name.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid chord format: {chord_name}. Expected format: 'C Major'")
    
    root_note = parts[0]
    chord_type = ' '.join(parts[1:])
    
    if root_note not in NOTE_TO_MIDI:
        raise ValueError(f"Invalid root note: {root_note}")
    
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Invalid chord type: {chord_type}")
    
    root_midi = note_to_midi(root_note, 4)
    
    chord_notes = []
    for interval in CHORD_PATTERNS[chord_type]:
        midi_num = root_midi + interval
        note, octave = midi_to_note(midi_num)
        chord_notes.append(f"{note}{octave}")
    
    return chord_notes