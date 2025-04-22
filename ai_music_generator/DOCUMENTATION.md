# AI Music Generator - Technical Documentation

## System Architecture

The AI Music Generator is built with a modular architecture consisting of the following components:

1. **Core Module**: Contains the main music generation logic that combines different AI approaches.
2. **Models Module**: Contains the implementation of different AI models (Markov Chain and Rule-Based).
3. **Utils Module**: Contains utility functions for music theory, MIDI generation, and audio conversion.
4. **UI Module**: Contains the web interface for user interaction.

### Directory Structure

```
ai_music_generator/
├── core/
│   ├── __init__.py
│   └── music_generator.py
├── models/
│   ├── __init__.py
│   ├── markov_model.py
│   └── rule_based_model.py
├── utils/
│   ├── __init__.py
│   ├── music_theory.py
│   └── midi_utils.py
├── ui/
│   ├── __init__.py
│   ├── web_interface.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── data/
├── output/
├── __init__.py
├── main.py
├── example.py
├── requirements.txt
├── README.md
└── DOCUMENTATION.md
```

## AI Models

### Markov Chain Model

The Markov Chain model is implemented in `models/markov_model.py`. It uses transition matrices to generate melodies based on probabilities of note transitions.

#### Key Components:

1. **Transition Matrices**: Pre-defined matrices for different genres and moods that define the probability of transitioning from one note to another.

2. **Complexity Adjustment**: The model adjusts the transition matrices based on the desired complexity level:
   - **Simple**: Increases the probability of common transitions (e.g., to scale degrees 1, 3, and 5).
   - **Complex**: Makes transitions more uniform to increase unpredictability.
   - **Intermediate**: Uses the original transition matrix.

3. **Melody Generation**: The model generates a melody by:
   - Starting with a note (typically the tonic)
   - Using the transition matrix to probabilistically determine the next note
   - Repeating until the desired length is reached

#### Example Transition Matrix:

```python
# Pop genre, Happy mood
matrix = np.array([
    [0.10, 0.15, 0.30, 0.00, 0.25, 0.00, 0.00, 0.20],  # From scale degree 1
    [0.20, 0.05, 0.40, 0.10, 0.15, 0.05, 0.05, 0.00],  # From scale degree 2
    [0.15, 0.10, 0.05, 0.25, 0.30, 0.10, 0.05, 0.00],  # From scale degree 3
    [0.05, 0.15, 0.20, 0.05, 0.40, 0.15, 0.00, 0.00],  # From scale degree 4
    [0.30, 0.05, 0.15, 0.10, 0.05, 0.20, 0.05, 0.10],  # From scale degree 5
    [0.10, 0.20, 0.10, 0.15, 0.25, 0.05, 0.15, 0.00],  # From scale degree 6
    [0.40, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.15],  # From scale degree 7
    [0.50, 0.10, 0.10, 0.05, 0.15, 0.05, 0.05, 0.00],  # From scale degree 8 (octave)
])
```

In this matrix, each row represents the current note (as scale degree) and each column represents the probability of transitioning to the next note. For example, if we're on scale degree 1 (first row), we have:
- 10% chance to stay on 1
- 30% chance to go to 3
- 25% chance to go to 5
- 20% chance to go to 8 (octave)
- 15% chance to go to 2

### Rule-Based Model

The Rule-Based model is implemented in `models/rule_based_model.py`. It applies music theory principles to generate melodies and harmonies.

#### Key Components:

1. **Chord Progressions**: Pre-defined chord progressions for different genres.

2. **Rhythm Patterns**: Pre-defined rhythm patterns for different complexity levels.

3. **Melodic Patterns**: Pre-defined melodic patterns for different moods.

4. **Composition Generation**: The model generates a composition by:
   - Selecting a chord progression based on the genre
   - Selecting rhythm patterns based on the complexity
   - Selecting melodic patterns based on the mood
   - Combining these elements to create a melody and harmony

#### Example Chord Progressions:

```python
chord_progressions = {
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
    # ...
}
```

#### Example Rhythm Patterns:

```python
rhythm_patterns = {
    'Simple': [
        [0.25, 0.25, 0.25, 0.25],  # Four quarter notes
        [0.5, 0.25, 0.25],         # Half note + two quarter notes
        [0.25, 0.25, 0.5],         # Two quarter notes + half note
    ],
    'Intermediate': [
        [0.125, 0.125, 0.25, 0.125, 0.125, 0.25],  # Eighth notes + quarter notes
        # ...
    ],
    # ...
}
```

#### Example Melodic Patterns:

```python
melodic_patterns = {
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
    # ...
}
```

## Hybrid Approach

The hybrid approach is implemented in the `core/music_generator.py` file. It combines the strengths of both the Markov Chain and Rule-Based models:

1. **Rule-Based for Harmony**: Uses the Rule-Based model to generate the harmony (chord progression) and overall structure.

2. **Markov Chain for Melody**: Uses the Markov Chain model to generate the melody, influenced by the harmony.

This approach leverages the strengths of both models:
- The Rule-Based model provides a solid musical foundation with proper chord progressions and structure.
- The Markov Chain model adds variety and unpredictability to the melody.

## Music Theory Utilities

The music theory utilities are implemented in `utils/music_theory.py`. They provide functions for working with notes, scales, chords, and other music theory concepts.

### Key Components:

1. **Note to MIDI Conversion**: Functions to convert between note names and MIDI numbers.

2. **Scale Patterns**: Definitions of different scale patterns (e.g., Major, Minor, Dorian, etc.).

3. **Chord Patterns**: Definitions of different chord patterns (e.g., Major, Minor, Diminished, etc.).

4. **Chord Symbols**: Mapping between chord symbols (e.g., I, IV, V) and chord types.

5. **Scale Degree to Semitone Mapping**: Mapping between scale degrees and semitone offsets for major and minor scales.

## MIDI and Audio Utilities

The MIDI and audio utilities are implemented in `utils/midi_utils.py`. They provide functions for creating MIDI files and converting them to MP3.

### Key Components:

1. **Instrument Mapping**: Mapping between instrument names and General MIDI program numbers.

2. **MIDI File Creation**: Function to create a MIDI file from melody and harmony.

3. **MP3 Conversion**: Function to convert a MIDI file to MP3 using FluidSynth and ffmpeg.

## Web Interface

The web interface is implemented in `ui/web_interface.py`. It provides a user-friendly interface for setting parameters and generating music.

### Key Components:

1. **Flask Server**: A Flask server that handles HTTP requests and serves the web interface.

2. **HTML Template**: A template for the main page with forms for parameter input.

3. **CSS Styles**: Styles for the web interface.

4. **JavaScript**: Client-side logic for form handling and AJAX requests.

## Usage Examples

### Programmatic Usage

```python
from core.music_generator import MusicGenerator

# Create a MusicGenerator instance
generator = MusicGenerator()

# Generate a simple pop song
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
```

### Web Interface Usage

1. Start the web interface:
   ```
   python main.py --web --port 12000
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:12000
   ```

3. Set your desired parameters and click "Generate Music".

## Extending the System

### Adding New Genres

To add a new genre, you need to:

1. Add chord progressions for the genre in the `chord_progressions` dictionary in `models/rule_based_model.py`.
2. Add transition matrices for the genre in the `_initialize_transition_matrices` method in `models/markov_model.py`.

### Adding New Instruments

To add a new instrument, you need to:

1. Add the instrument to the `INSTRUMENT_MAP` dictionary in `utils/midi_utils.py` with its corresponding General MIDI program number.
2. Add the instrument to the options in the HTML template in `ui/web_interface.py`.

### Adding New Scales

To add a new scale, you need to:

1. Add the scale pattern to the `SCALE_PATTERNS` dictionary in `utils/music_theory.py`.
2. Add the scale to the options in the HTML template in `ui/web_interface.py`.

### Adding Vocal Synthesis

To add vocal synthesis, you would need to:

1. Implement a vocal synthesis module that can generate vocals based on lyrics and melody.
2. Integrate this module with the `create_midi_file` function in `utils/midi_utils.py`.
3. Extend the web interface to allow users to input lyrics or select from pre-defined lyrics.

## Conclusion

The AI Music Generator demonstrates how basic AI techniques can be used to create musical compositions. The modular architecture allows for easy extension and customization, making it a good starting point for more advanced music generation systems.

The combination of Markov Chain and Rule-Based approaches provides a balance between creativity and musical coherence, allowing the system to generate music that is both interesting and pleasing to the ear.