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
    'Drums': 0,  # special-case for percussion channel override
    'Synth': 80,
    'Guitar': 24,
    'Bass': 33,
    'Strings': 48,
}


def parse_note(note_str):
    """
    Parse a note string into note name and octave.
    """
    if note_str[-1].isdigit():
        octave = int(note_str[-1])
        note_name = note_str[:-1]
    else:
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
    # Total tracks = melody + harmony for each instrument
    num_tracks = len(instruments) * 2
    midi = MIDIFile(num_tracks)
    midi.addTempo(0, 0, bpm)

    # --- Melody Tracks ---
    for i, instrument in enumerate(instruments):
        track = i
        # assign percussion to channel 9, others to unique channels
        channel = 9 if instrument == 'Drums' else i
        program = INSTRUMENT_MAP.get(instrument, 0)
        midi.addProgramChange(track, channel, 0, program)

        time = 0
        for idx, (note_str, duration) in enumerate(melody):
            note_name, octave = parse_note(note_str)
            try:
                midi_note = note_to_midi(note_name, octave)
                # instrument-specific patterns
                if instrument == 'Piano':
                    midi.addNote(track, channel, midi_note, time, duration*4, 100)
                elif instrument == 'Guitar' and idx % 3 == 0:
                    midi.addNote(track, channel, midi_note+12, time, duration*4, 90)
                elif instrument == 'Bass' and idx % 4 == 0:
                    midi.addNote(track, channel, midi_note-12, time, duration*4, 95)
                elif instrument == 'Drums' and idx % 2 == 0:
                    midi.addNote(track, channel, 38, time, duration*4, 100)
                    if idx % 4 == 0:
                        midi.addNote(track, channel, 36, time, duration*4, 110)
                elif instrument == 'Violin' and idx % 2 == 1:
                    midi.addNote(track, channel, midi_note+12, time, duration*4, 85)
                elif instrument == 'Synth' and idx % 3 == 1:
                    midi.addNote(track, channel, midi_note+7, time, duration*4, 80)
                elif instrument == 'Flute' and idx % 3 == 2:
                    midi.addNote(track, channel, midi_note+24, time, duration*4, 75)
                elif instrument == 'Trumpet' and idx % 4 == 2:
                    midi.addNote(track, channel, midi_note+12, time, duration*4, 90)
                else:
                    midi.addNote(track, channel, midi_note, time, duration*4, 90)
            except ValueError:
                pass
            time += duration*4

    # --- Harmony Tracks ---
    for i, instrument in enumerate(instruments):
        harmony_track = len(instruments) + i
        channel = 9 if instrument == 'Drums' else i
        program = INSTRUMENT_MAP.get(instrument, 0)
        midi.addProgramChange(harmony_track, channel, 0, program)

        time = 0
        for chord_name in harmony:
            try:
                chord_notes = get_chord_notes(chord_name)
                # distribute chord notes per instrument
                for note_idx, note_str in enumerate(chord_notes):
                    # each instrument plays its slice
                    if note_idx % len(instruments) == i:
                        note_name, octave = parse_note(note_str)
                        try:
                            midi_note = note_to_midi(note_name, octave)
                            midi.addNote(harmony_track, channel, midi_note, time, 4, 80)
                        except ValueError:
                            pass
            except ValueError:
                pass
            time += 4

    # Vocals (placeholder)
    if vocals and vocals.get('enabled', False):
        pass

    # Write out
    os.makedirs('output', exist_ok=True)
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
    import platform
    
    # Detect operating system
    system = platform.system()
    
    # Create data directory for soundfonts if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Default soundfont path based on OS
    if system == 'Windows':
        soundfont_path = os.path.join(data_dir, 'FluidR3_GM.sf2')
    else:  # Linux, macOS
        soundfont_path = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
    
    # Check if FluidSynth is installed
    fluidsynth_installed = False
    fluidsynth_binary_path = None
    
    # Check for custom path in config file
    custom_path = None
    config_path = os.path.join(os.getcwd(), 'fluidsynth_path.txt')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                custom_path = f.read().strip()
                print(f"Found custom FluidSynth path in config: {custom_path}")
        except Exception as e:
            print(f"Error reading config file: {e}")
    
    # Common FluidSynth installation paths on Windows
    windows_fluidsynth_paths = [
        # First check the custom path from config if available
        custom_path,
        # Prioritize the specific path mentioned by the user
        'C:\\Program Files\\FluidSynth\\bin\\fluidsynth.exe',
        # User directory installation (from local installer)
        os.path.join(os.path.expanduser('~'), 'FluidSynth', 'bin', 'fluidsynth.exe'),
        # Other common paths as fallbacks
        os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'FluidSynth', 'bin', 'fluidsynth.exe'),
        os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'FluidSynth', 'bin', 'fluidsynth.exe'),
        os.path.join('C:\\tools', 'fluidsynth', 'bin', 'fluidsynth.exe'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin', 'fluidsynth.exe')
    ]
    
    # Filter out None values
    windows_fluidsynth_paths = [p for p in windows_fluidsynth_paths if p]
    
    try:
        # Try the standard command first
        subprocess.run(['fluidsynth', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fluidsynth_installed = True
        fluidsynth_binary_path = 'fluidsynth'  # Use command name directly
    except FileNotFoundError:
        # On Windows, check common installation paths
        if system == 'Windows':
            for path in windows_fluidsynth_paths:
                if os.path.exists(path):
                    fluidsynth_installed = True
                    fluidsynth_binary_path = path
                    print(f"Found FluidSynth at: {path}")
                    break
        
        if not fluidsynth_installed:
            print("FluidSynth not found. Checking for alternatives...")
            
            if system == 'Linux':
                # Linux installation
                try:
                    subprocess.run(['apt-get', 'update'], check=True)
                    subprocess.run(['apt-get', 'install', '-y', 'fluidsynth'], check=True)
                    fluidsynth_installed = True
                    fluidsynth_binary_path = 'fluidsynth'
                except Exception as e:
                    print(f"Failed to install FluidSynth: {e}")
            elif system == 'Windows':
                # Windows - try to use pyfluidsynth
                print("Attempting to use Python FluidSynth library...")
                try:
                    import fluidsynth
                    print("Python FluidSynth library found. Will use it for MIDI conversion.")
                    # We'll use the Python library directly, so mark as not installed for command-line usage
                    fluidsynth_installed = False
                except ImportError:
                    # Installation instructions
                    print("\nPlease install FluidSynth for Windows:")
                    print("1. Download from: https://github.com/FluidSynth/fluidsynth/releases")
                    print("2. Extract to C:\\Program Files\\FluidSynth")
                    print("3. Add the bin directory to your PATH environment variable")
                    
                    # Create a helper batch file to install FluidSynth
                    create_fluidsynth_installer_batch()
                    
                    print("\nA helper batch file 'install_fluidsynth.bat' has been created in the current directory.")
                    print("Run it as administrator to automatically download and install FluidSynth.")
                    
                    print("\nAttempting to install pyfluidsynth...")
                    try:
                        subprocess.run(['pip', 'install', 'pyfluidsynth'], check=True)
                        print("pyfluidsynth installed. You still need the FluidSynth binary.")
                        try:
                            import fluidsynth
                            print("Python FluidSynth library successfully imported.")
                        except ImportError:
                            print("Failed to import fluidsynth module after installation.")
                    except Exception as e:
                        print(f"Failed to install pyfluidsynth: {e}")
            elif system == 'Darwin':  # macOS
                print("For macOS, install FluidSynth using Homebrew:")
                print("brew install fluid-synth")
    
    # Check if ffmpeg is installed
    ffmpeg_installed = False
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffmpeg_installed = True
    except FileNotFoundError:
        print("ffmpeg not found. Installing...")
        
        if system == 'Linux':
            # Linux installation
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
            ffmpeg_installed = True
        elif system == 'Windows':
            # Windows installation instructions
            print("Please install ffmpeg for Windows:")
            print("1. Download from: https://ffmpeg.org/download.html")
            print("2. Add the bin directory to your PATH environment variable")
        elif system == 'Darwin':  # macOS
            print("For macOS, install ffmpeg using Homebrew:")
            print("brew install ffmpeg")
    
    # Check if SoundFont is available
    soundfont_found = False
    
    # First check the specified path
    if os.path.exists(soundfont_path) and os.path.getsize(soundfont_path) > 0:
        print(f"SoundFont found at {soundfont_path}")
        soundfont_found = True
    else:
        print(f"SoundFont not found at {soundfont_path}")
        
        # Check alternative locations based on OS
        if system == 'Linux':
            alternative_paths = [
                '/usr/share/sounds/sf2/default.sf2',
                '/usr/share/soundfonts/FluidR3_GM.sf2',
                '/usr/share/soundfonts/default.sf2'
            ]
        elif system == 'Darwin':  # macOS
            alternative_paths = [
                '/usr/local/share/fluidsynth/soundfonts/FluidR3_GM.sf2',
                '/usr/local/share/soundfonts/FluidR3_GM.sf2'
            ]
        elif system == 'Windows':
            alternative_paths = [
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'FluidSynth', 'share', 'soundfonts', 'FluidR3_GM.sf2'),
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'FluidSynth', 'share', 'soundfonts', 'FluidR3_GM.sf2'),
                os.path.join(os.path.expanduser('~'), 'FluidSynth', 'share', 'soundfonts', 'FluidR3_GM.sf2')
            ]
        else:
            alternative_paths = []
        
        # Check all alternative paths
        for path in alternative_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"SoundFont found at alternative location: {path}")
                soundfont_path = path
                soundfont_found = True
                break
        
        # If still not found, try to install or download
        if not soundfont_found:
            print("SoundFont not found in any standard location. Attempting to install/download...")
            
            if system == 'Linux':
                try:
                    print("Trying to install fluid-soundfont-gm package...")
                    subprocess.run(['apt-get', 'update'], check=True)
                    subprocess.run(['apt-get', 'install', '-y', 'fluid-soundfont-gm'], check=True)
                    
                    # Check if installation succeeded
                    if os.path.exists('/usr/share/sounds/sf2/FluidR3_GM.sf2'):
                        soundfont_path = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
                        soundfont_found = True
                        print(f"SoundFont installed at {soundfont_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install fluid-soundfont-gm: {e}")
            
            # If still not found or not on Linux, download it
            if not soundfont_found:
                print(f"Downloading SoundFont to {soundfont_path}...")
                soundfont_found = download_soundfont(soundfont_path)
    
    # Final check - if we still don't have a SoundFont, we can't proceed
    if not soundfont_found:
        print("ERROR: Could not find or download a SoundFont file.")
        print("Please download a SoundFont file manually and place it at:")
        print(soundfont_path)
        return None  # Return None to indicate failure
    
    # Create temporary WAV file
    wav_path = os.path.join(tempfile.gettempdir(), 'generated_music.wav')
    
    # Convert MIDI to WAV using FluidSynth
    if fluidsynth_installed and fluidsynth_binary_path:
        try:
            # Use the detected binary path with correct syntax
            # The correct syntax is: fluidsynth -ni -F output.wav -r 44100 soundfont.sf2 input.mid
            subprocess.run([
                fluidsynth_binary_path,
                '-ni',
                '-F', wav_path,
                '-r', '44100',
                soundfont_path,
                midi_path
            ], check=True)
            print(f"Successfully converted MIDI to WAV using FluidSynth at {fluidsynth_binary_path}")
        except Exception as e:
            print(f"Error using FluidSynth binary: {e}")
            print("Trying alternative methods...")
            if system == 'Windows':
                try:
                    import fluidsynth
                    print("Using Python FluidSynth library instead...")
                    convert_midi_to_wav_with_pyfluidsynth(midi_path, wav_path, soundfont_path)
                except ImportError:
                    print("Python FluidSynth library not available.")
    elif system == 'Windows':
        # Try to use pyfluidsynth if available, even if binary not found
        try:
            import fluidsynth
            print("Using Python FluidSynth library for MIDI conversion...")
            convert_midi_to_wav_with_pyfluidsynth(midi_path, wav_path, soundfont_path)
        except ImportError:
            print("Python FluidSynth library not available. Cannot convert MIDI to audio.")
    
    # Convert WAV to MP3 using ffmpeg
    mp3_path = midi_path.replace('.mid', '.mp3')
    
    if not fluidsynth_installed:
        print("WARNING: FluidSynth not installed. Cannot convert MIDI to audio.")
        # Create an empty MP3 file as a placeholder
        with open(mp3_path, 'w') as f:
            f.write('')
        return mp3_path
    
    if ffmpeg_installed:
        subprocess.run([
            'ffmpeg',
            '-i', wav_path,
            '-codec:a', 'libmp3lame',
            '-qscale:a', '2',
            '-y',  # Overwrite output file if it exists
            mp3_path
        ], check=True)
        
        # Remove temporary WAV file
        if os.path.exists(wav_path):
            os.remove(wav_path)
    else:
        print("WARNING: ffmpeg not installed. Using WAV file instead of MP3.")
        mp3_path = wav_path
    
    return mp3_path

def convert_midi_to_wav_with_pyfluidsynth(midi_path, wav_path, soundfont_path):
    """
    Convert MIDI to WAV using the Python FluidSynth library.
    
    Args:
        midi_path (str): Path to the MIDI file
        wav_path (str): Path where the WAV file should be saved
        soundfont_path (str): Path to the SoundFont file
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        import fluidsynth
        import mido
        import numpy as np
        from scipy.io import wavfile
        
        print(f"Using Python FluidSynth library to convert MIDI to WAV")
        print(f"MIDI file: {midi_path}")
        print(f"SoundFont: {soundfont_path}")
        print(f"Output WAV: {wav_path}")
        
        # Verify files exist
        if not os.path.exists(midi_path):
            print(f"ERROR: MIDI file not found: {midi_path}")
            return False
            
        if not os.path.exists(soundfont_path):
            print(f"ERROR: SoundFont file not found: {soundfont_path}")
            return False
        
        # Load MIDI file
        try:
            midi_file = mido.MidiFile(midi_path)
            print(f"Successfully loaded MIDI file: {len(midi_file.tracks)} tracks")
        except Exception as e:
            print(f"ERROR: Failed to load MIDI file: {e}")
            return False
        
        # Initialize FluidSynth
        fs = fluidsynth.Synth()
        fs.start()
        print("FluidSynth initialized")
        
        # Load SoundFont
        try:
            sfid = fs.sfload(soundfont_path)
            print(f"Successfully loaded SoundFont (ID: {sfid})")
        except Exception as e:
            print(f"ERROR: Failed to load SoundFont: {e}")
            fs.delete()
            return False
            
        try:
            # Select program (instrument)
            fs.program_select(0, sfid, 0, 0)
            
            # Process MIDI events
            sample_rate = 44100
            
            # Calculate total time
            total_time = 0
            for track in midi_file.tracks:
                track_time = sum(msg.time for msg in track)
                total_time = max(total_time, track_time)
            total_time *= midi_file.ticks_per_beat
            
            # Create sample array with extra buffer
            buffer_size = int(total_time * sample_rate / 1000) + sample_rate  # Add 1 second buffer
            samples = np.zeros(buffer_size, dtype=np.int16)
            
            print(f"Processing MIDI file: {total_time/1000:.2f} seconds of audio")
            
            current_time = 0
            for msg in midi_file:
                if msg.type == 'note_on':
                    fs.noteon(msg.channel, msg.note, msg.velocity)
                elif msg.type == 'note_off':
                    fs.noteoff(msg.channel, msg.note)
                
                # Render audio for this time step
                if msg.time > 0:
                    current_time += msg.time
                    start_idx = int((current_time - msg.time) * sample_rate / 1000)
                    end_idx = int(current_time * sample_rate / 1000)
                    
                    # Make sure we don't go out of bounds
                    if end_idx > len(samples):
                        # Extend the samples array if needed
                        samples = np.pad(samples, (0, end_idx - len(samples) + 1000), 'constant')
                    
                    chunk = fs.get_samples(end_idx - start_idx)
                    
                    # Make sure chunk fits in the target range
                    if len(chunk) > (end_idx - start_idx):
                        chunk = chunk[:(end_idx - start_idx)]
                    elif len(chunk) < (end_idx - start_idx):
                        chunk = np.pad(chunk, (0, (end_idx - start_idx) - len(chunk)), 'constant')
                    
                    samples[start_idx:end_idx] = chunk
            
            # Save as WAV
            wavfile.write(wav_path, sample_rate, samples)
            print(f"Successfully wrote WAV file: {wav_path}")
            
            # Clean up
            fs.delete()
            return True
            
        except Exception as e:
            print(f"ERROR during MIDI processing: {e}")
            fs.delete()
            return False
            
    except ImportError as e:
        print(f"ERROR: Missing required libraries for MIDI to WAV conversion: {e}")
        print("Please install the required libraries with: pip install pyfluidsynth mido scipy numpy")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error in MIDI to WAV conversion: {e}")
        return False

def create_fluidsynth_installer_batch():
    """
    Create batch files to help Windows users install FluidSynth automatically.
    Two batch files are created:
    1. install_fluidsynth.bat - For administrator installation to Program Files
    2. install_fluidsynth_local.bat - For local user installation without admin rights
    """
    # Admin version - installs to Program Files
    admin_batch_content = """@echo off
echo Installing FluidSynth for Windows (Administrator version)...
echo.

:: Create temporary directory
mkdir %TEMP%\\fluidsynth_install
cd %TEMP%\\fluidsynth_install

:: Download latest FluidSynth release
echo Downloading FluidSynth...
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.3/fluidsynth-2.3.3-win10-x64.zip' -OutFile 'fluidsynth.zip'}"

:: Extract the zip file
echo Extracting files...
powershell -Command "& {Expand-Archive -Path 'fluidsynth.zip' -DestinationPath '%TEMP%\\fluidsynth_install' -Force}"

:: Create the destination directory
echo Creating installation directory...
mkdir "C:\\Program Files\\FluidSynth"

:: Copy files to Program Files
echo Copying files to C:\\Program Files\\FluidSynth...
xcopy /E /I /Y "%TEMP%\\fluidsynth_install\\fluidsynth-2.3.3-win10-x64\\*" "C:\\Program Files\\FluidSynth\\"

:: Add to PATH
echo Adding FluidSynth to PATH...
setx PATH "%PATH%;C:\\Program Files\\FluidSynth\\bin" /M

:: Clean up
echo Cleaning up temporary files...
cd %USERPROFILE%
rmdir /S /Q %TEMP%\\fluidsynth_install

echo.
echo FluidSynth has been installed to C:\\Program Files\\FluidSynth
echo The bin directory has been added to your system PATH.
echo You may need to restart your computer for the PATH changes to take effect.
echo.
pause
"""

    # Non-admin version - installs to user directory
    local_batch_content = """@echo off
echo Installing FluidSynth for Windows (Local user version)...
echo This version doesn't require administrator privileges.
echo.

:: Create temporary directory
mkdir %TEMP%\\fluidsynth_install
cd %TEMP%\\fluidsynth_install

:: Download latest FluidSynth release
echo Downloading FluidSynth...
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.3/fluidsynth-2.3.3-win10-x64.zip' -OutFile 'fluidsynth.zip'}"

:: Extract the zip file
echo Extracting files...
powershell -Command "& {Expand-Archive -Path 'fluidsynth.zip' -DestinationPath '%TEMP%\\fluidsynth_install' -Force}"

:: Create the destination directory in user's folder
echo Creating installation directory...
mkdir "%USERPROFILE%\\FluidSynth"

:: Copy files to user directory
echo Copying files to %USERPROFILE%\\FluidSynth...
xcopy /E /I /Y "%TEMP%\\fluidsynth_install\\fluidsynth-2.3.3-win10-x64\\*" "%USERPROFILE%\\FluidSynth\\"

:: Add to PATH (user level)
echo Adding FluidSynth to PATH (user level)...
setx PATH "%PATH%;%USERPROFILE%\\FluidSynth\\bin"

:: Create a config file for the application
echo Creating configuration file...
echo %USERPROFILE%\\FluidSynth\\bin\\fluidsynth.exe > "%~dp0fluidsynth_path.txt"

:: Clean up
echo Cleaning up temporary files...
cd %USERPROFILE%
rmdir /S /Q %TEMP%\\fluidsynth_install

echo.
echo FluidSynth has been installed to %USERPROFILE%\\FluidSynth
echo The bin directory has been added to your user PATH.
echo You may need to restart your command prompt for the PATH changes to take effect.
echo.
pause
"""
    
    # Write the admin batch file to the current directory
    admin_batch_path = os.path.join(os.getcwd(), 'install_fluidsynth.bat')
    with open(admin_batch_path, 'w') as f:
        f.write(admin_batch_content)
    
    # Write the local user batch file to the current directory
    local_batch_path = os.path.join(os.getcwd(), 'install_fluidsynth_local.bat')
    with open(local_batch_path, 'w') as f:
        f.write(local_batch_content)
    
    print(f"Created FluidSynth installer batch files:")
    print(f"1. {admin_batch_path} (requires administrator privileges)")
    print(f"2. {local_batch_path} (for local user installation, no admin required)")
    
    # Also check for fluidsynth_path.txt and add it to the paths to check
    config_path = os.path.join(os.getcwd(), 'fluidsynth_path.txt')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                custom_path = f.read().strip()
                if custom_path and os.path.exists(custom_path):
                    print(f"Found custom FluidSynth path in config: {custom_path}")
                    return admin_batch_path, local_batch_path, custom_path
        except Exception as e:
            print(f"Error reading config file: {e}")
    
    return admin_batch_path, local_batch_path

def download_soundfont(soundfont_path):
    """
    Download a SoundFont file from the internet.
    
    Args:
        soundfont_path (str): Path where the SoundFont file should be saved
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    import requests
    import shutil
    
    print("Downloading SoundFont file...")
    print(f"Target path: {soundfont_path}")
    
    # Primary URL
    soundfont_url = "https://archive.org/download/fluidr3-gm-gs/FluidR3_GM.sf2"
    
    # Backup URLs in case the primary one fails
    backup_urls = [
        "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/FluidR3_GM.sf2",
        "https://www.dropbox.com/s/4z8ycbm682g3v0l/FluidR3_GM.sf2?dl=1"
    ]
    
    # Try the primary URL first, then the backups
    all_urls = [soundfont_url] + backup_urls
    
    for url in all_urls:
        try:
            print(f"Trying to download from: {url}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(soundfont_path), exist_ok=True)
            
            # Download with progress reporting
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Get total file size if available
                total_size = int(response.headers.get('content-length', 0))
                if total_size:
                    print(f"File size: {total_size / (1024 * 1024):.2f} MB")
                
                # Download and save the file
                with open(soundfont_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Verify the file was downloaded
            if os.path.exists(soundfont_path) and os.path.getsize(soundfont_path) > 0:
                print(f"SoundFont successfully downloaded to {soundfont_path}")
                print(f"File size: {os.path.getsize(soundfont_path) / (1024 * 1024):.2f} MB")
                return True
            else:
                print("Downloaded file is empty or not found. Trying next URL...")
                
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
            print("Trying next URL...")
    
    # If we get here, all URLs failed
    print("All download attempts failed.")
    print("Please download the SoundFont manually from one of these URLs:")
    for url in all_urls:
        print(f"- {url}")
    print(f"And save it to: {soundfont_path}")
    return False
