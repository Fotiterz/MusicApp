"""
Core music generation module that combines Markov chain and rule-based approaches.
"""

from models.markov_model import MarkovModel
from models.rule_based_model import RuleBasedModel
from utils.music_theory import get_scale_notes, get_chord_progression
from utils.midi_utils import create_midi_file, midi_to_mp3

class MusicGenerator:
    """Main music generation class that combines different generation approaches."""
    
    def __init__(self):
        self.markov_model = MarkovModel()
        self.rule_based_model = RuleBasedModel()
    
    def generate_music(self, params):
        """
        Generate music based on user parameters.
        
        Args:
            params (dict): Dictionary containing user parameters:
                - genre (str): Music genre (required)
                - instruments (list): List of instruments (optional)
                - scale (str): Musical scale/key (optional, default: "C Major")
                - mood (str): Mood/theme (optional)
                - tempo (str): Tempo - Slow/Medium/Fast (optional, default: "Medium")
                - length (str): Song length - Short/Medium/Long (optional, default: "Medium")
                - vocals (dict): Vocals parameters (optional)
                - complexity (str): Complexity level - Simple/Intermediate/Complex (optional, default: "Simple")
                - mode (str): Generation mode - "markov", "rule", or "hybrid" (optional, default: "hybrid")
        
        Returns:
            str: Path to the generated MP3 file
        """
        # Set default values for optional parameters
        genre = params.get('genre')
        if not genre:
            raise ValueError("Genre is required")
            
        instruments = params.get('instruments', ['Piano'])
        scale = params.get('scale', 'C Major')
        mood = params.get('mood', 'Neutral')
        tempo = params.get('tempo', 'Medium')
        length = params.get('length', 'Medium')
        vocals = params.get('vocals', {'enabled': False})
        complexity = params.get('complexity', 'Simple')
        mode = params.get('mode', 'hybrid')
        
        # Convert tempo to BPM
        tempo_map = {'Slow': 70, 'Medium': 100, 'Fast': 130}
        bpm = tempo_map.get(tempo, 100)
        
        # Convert length to bars
        length_map = {'Short': 16, 'Medium': 32, 'Long': 64}
        num_bars = length_map.get(length, 32)
        
        # Get scale notes and chord progression based on genre and scale
        scale_notes = get_scale_notes(scale)
        chord_progression = get_chord_progression(genre, scale)
        
        # Generate music based on the selected mode
        if mode == 'markov':
            melody = self.markov_model.generate_melody(
                scale_notes=scale_notes,
                num_bars=num_bars,
                complexity=complexity,
                mood=mood,
                tempo=tempo
            )
            harmony = chord_progression * (num_bars // len(chord_progression) + 1)
            harmony = harmony[:num_bars]
            
        elif mode == 'rule':
            melody, harmony = self.rule_based_model.generate_composition(
                scale=scale,
                chord_progression=chord_progression,
                num_bars=num_bars,
                complexity=complexity,
                mood=mood,
                tempo=tempo
            )
            
        else:  # hybrid mode
            # Use rule-based for harmony and structure
            _, harmony = self.rule_based_model.generate_composition(
                scale=scale,
                chord_progression=chord_progression,
                num_bars=num_bars,
                complexity=complexity,
                mood=mood,
                tempo=tempo
            )
            
            # Use Markov for melody
            melody = self.markov_model.generate_melody(
                scale_notes=scale_notes,
                num_bars=num_bars,
                complexity=complexity,
                mood=mood,
                chord_progression=harmony,  # Pass harmony to influence melody
                tempo=tempo  # Pass tempo to influence melody generation
            )
        
        # Create MIDI file
        midi_path = create_midi_file(
            melody=melody,
            harmony=harmony,
            instruments=instruments,
            bpm=bpm,
            vocals=vocals
        )
        
        # Convert MIDI to MP3
        mp3_path = midi_to_mp3(midi_path)
        
        return mp3_path