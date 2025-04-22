"""
MIDI utilities for the AI Music Generator.
"""

import os
import subprocess
import tempfile
from midiutil import MIDIFile
from utils.music_theory import note_to_midi, get_chord_notes

# Define instrument mapping (General MIDI program numbers)
INSTRUMENT_MAP = {
    'Piano': 0,
    'Acoustic Grand Piano': 0,
    'Bright Acoustic Piano': 1,
    'Electric Piano': 4,
    'Electric Piano 1': 4,
    'Electric Piano 2': 5,
    'Harpsichord': 6,
    'Clavinet': 7,
    'Celesta': 8,
    'Glockenspiel': 9,
    'Music Box': 10,
    'Vibraphone': 11,
    'Marimba': 12,
    'Xylophone': 13,
    'Tubular Bells': 14,
    'Dulcimer': 15,
    'Drawbar Organ': 16,
    'Percussive Organ': 17,
    'Rock Organ': 18,
    'Church Organ': 19,
    'Reed Organ': 20,
    'Accordion': 21,
    'Harmonica': 22,
    'Tango Accordion': 23,
    'Acoustic Guitar': 24,
    'Acoustic Guitar (nylon)': 24,
    'Acoustic Guitar (steel)': 25,
    'Electric Guitar': 26,
    'Electric Guitar (jazz)': 26,
    'Electric Guitar (clean)': 27,
    'Electric Guitar (muted)': 28,
    'Overdriven Guitar': 29,
    'Distortion Guitar': 30,
    'Guitar Harmonics': 31,
    'Acoustic Bass': 32,
    'Electric Bass': 33,
    'Electric Bass (finger)': 33,
    'Electric Bass (pick)': 34,
    'Fretless Bass': 35,
    'Slap Bass 1': 36,
    'Slap Bass 2': 37,
    'Synth Bass 1': 38,
    'Synth Bass 2': 39,
    'Violin': 40,
    'Viola': 41,
    'Cello': 42,
    'Contrabass': 43,
    'Tremolo Strings': 44,
    'Pizzicato Strings': 45,
    'Orchestral Harp': 46,
    'Timpani': 47,
    'String Ensemble': 48,
    'String Ensemble 1': 48,
    'String Ensemble 2': 49,
    'Synth Strings 1': 50,
    'Synth Strings 2': 51,
    'Choir Aahs': 52,
    'Voice Oohs': 53,
    'Synth Voice': 54,
    'Orchestra Hit': 55,
    'Trumpet': 56,
    'Trombone': 57,
    'Tuba': 58,
    'Muted Trumpet': 59,
    'French Horn': 60,
    'Brass Section': 61,
    'Synth Brass 1': 62,
    'Synth Brass 2': 63,
    'Soprano Sax': 64,
    'Alto Sax': 65,
    'Tenor Sax': 66,
    'Baritone Sax': 67,
    'Oboe': 68,
    'English Horn': 69,
    'Bassoon': 70,
    'Clarinet': 71,
    'Piccolo': 72,
    'Flute': 73,
    'Recorder': 74,
    'Pan Flute': 75,
    'Blown Bottle': 76,
    'Shakuhachi': 77,
    'Whistle': 78,
    'Ocarina': 79,
    'Lead 1 (square)': 80,
    'Lead 2 (sawtooth)': 81,
    'Lead 3 (calliope)': 82,
    'Lead 4 (chiff)': 83,
    'Lead 5 (charang)': 84,
    'Lead 6 (voice)': 85,
    'Lead 7 (fifths)': 86,
    'Lead 8 (bass + lead)': 87,
    'Pad 1 (new age)': 88,
    'Pad 2 (warm)': 89,
    'Pad 3 (polysynth)': 90,
    'Pad 4 (choir)': 91,
    'Pad 5 (bowed)': 92,
    'Pad 6 (metallic)': 93,
    'Pad 7 (halo)': 94,
    'Pad 8 (sweep)': 95,
    'FX 1 (rain)': 96,
    'FX 2 (soundtrack)': 97,
    'FX 3 (crystal)': 98,
    'FX 4 (atmosphere)': 99,
    'FX 5 (brightness)': 100,
    'FX 6 (goblins)': 101,
    'FX 7 (echoes)': 102,
    'FX 8 (sci-fi)': 103,
    'Sitar': 104,
    'Banjo': 105,
    'Shamisen': 106,
    'Koto': 107,
    'Kalimba': 108,
    'Bagpipe': 109,
    'Fiddle': 110,
    'Shanai': 111,
    'Tinkle Bell': 112,
    'Agogo': 113,
    'Steel Drums': 114,
    'Woodblock': 115,
    'Taiko Drum': 116,
    'Melodic Tom': 117,
    'Synth Drum': 118,
    'Reverse Cymbal': 119,
    'Guitar Fret Noise': 120,
    'Breath Noise': 121,
    'Seashore': 122,
    'Bird Tweet': 123,
    'Telephone Ring': 124,
    'Helicopter': 125,
    'Applause': 126,
    'Gunshot': 127,
    'Drums': 0,  # Special case, will be handled separately
    'Synth': 80,  # Default to Lead 1 (square)
    'Guitar': 24,  # Default to Acoustic Guitar (nylon)
    'Bass': 33,  # Default to Electric Bass (finger)
    'Strings': 48,  # Default to String Ensemble 1
}

def parse_note(note_str):
    """
    Parse a note string into note name and octave.
    
    Args:
        note_str (str): Note string (e.g., 'C4', 'D#5')
        
    Returns:
        tuple: (note_name, octave)
    """
    # Check if the note string ends with a digit (octave)
    if note_str[-1].isdigit():
        octave = int(note_str[-1])
        note_name = note_str[:-1]
    else:
        # Default to octave 4 if not specified
        octave = 4
        note_name = note_str
    
    return note_name, octave

def create_midi_file(melody, harmony, instruments=['Piano'], bpm=100, vocals=None):
    """
    Create a MIDI file from melody and harmony.
    
    Args:
        melody (list): List of (note, duration) tuples
        harmony (list): List of chord names
        instruments (list): List of instrument names
        bpm (int): Tempo in beats per minute
        vocals (dict, optional): Vocals parameters
        
    Returns:
        str: Path to the created MIDI file
    """
    # Create a MIDI file with 1 track
    midi = MIDIFile(len(instruments) + 1)  # +1 for harmony
    
    # Set tempo
    midi.addTempo(0, 0, bpm)
    
    # Add melody track(s)
    for i, instrument in enumerate(instruments):
        track = i
        
        # Get the MIDI program number for the instrument
        if instrument in INSTRUMENT_MAP:
            program = INSTRUMENT_MAP[instrument]
        else:
            # Default to piano if instrument not found
            program = 0
        
        # Set instrument
        midi.addProgramChange(track, 0, 0, program)
        
        # Add melody notes
        time = 0
        for note_str, duration in melody:
            note_name, octave = parse_note(note_str)
            
            try:
                midi_note = note_to_midi(note_name, octave)
                
                # Add note with velocity 100 (medium-loud)
                midi.addNote(track, 0, midi_note, time, duration * 4, 100)
            except ValueError:
                # Skip invalid notes
                pass
            
            time += duration * 4
    
    # Add harmony track
    harmony_track = len(instruments)
    
    # Set instrument for harmony (use piano by default)
    midi.addProgramChange(harmony_track, 0, 0, 0)
    
    # Add harmony chords
    time = 0
    for chord_name in harmony:
        # Get chord notes
        try:
            chord_notes = get_chord_notes(chord_name)
            
            # Add each note in the chord
            for note_str in chord_notes:
                note_name, octave = parse_note(note_str)
                
                try:
                    midi_note = note_to_midi(note_name, octave)
                    
                    # Add note with velocity 80 (softer than melody)
                    midi.addNote(harmony_track, 0, midi_note, time, 4, 80)
                except ValueError:
                    # Skip invalid notes
                    pass
        except ValueError:
            # Skip invalid chords
            pass
        
        # Move to next bar
        time += 4
    
    # Add vocals if specified
    if vocals and vocals.get('enabled', False):
        # This is a placeholder for vocal synthesis
        # In a real implementation, this would integrate with a vocal synthesis library
        pass
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Write the MIDI file
    midi_path = os.path.join('output', 'generated_music.mid')
    with open(midi_path, 'wb') as f:
        midi.writeFile(f)
    
    return midi_path

def midi_to_mp3(midi_path):
    """
    Convert a MIDI file to MP3 using FluidSynth and ffmpeg.
    
    Args:
        midi_path (str): Path to the MIDI file
        
    Returns:
        str: Path to the created MP3 file
    """
    # Check if FluidSynth is installed
    try:
        subprocess.run(['fluidsynth', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("FluidSynth not found. Installing...")
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'fluidsynth'], check=True)
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("ffmpeg not found. Installing...")
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
    
    # Check if SoundFont is available
    soundfont_path = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
    if not os.path.exists(soundfont_path):
        print("SoundFont not found. Installing...")
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'fluid-soundfont-gm'], check=True)
    
    # Create temporary WAV file
    wav_path = os.path.join(tempfile.gettempdir(), 'generated_music.wav')
    
    # Convert MIDI to WAV using FluidSynth
    subprocess.run([
        'fluidsynth',
        '-ni',
        soundfont_path,
        midi_path,
        '-F', wav_path,
        '-r', '44100'
    ], check=True)
    
    # Convert WAV to MP3 using ffmpeg
    mp3_path = midi_path.replace('.mid', '.mp3')
    subprocess.run([
        'ffmpeg',
        '-i', wav_path,
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',
        '-y',  # Overwrite output file if it exists
        mp3_path
    ], check=True)
    
    # Remove temporary WAV file
    os.remove(wav_path)
    
    return mp3_path