from flask import Flask, request, send_file, render_template, jsonify, send_from_directory
import os
import uuid
import atexit
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
import tensorflow as tf
from rule_based import generate_rule_based_music
from utils import save_stream_to_midi, convert_midi_to_mp3
from lstm_model import build_lstm_model_with_duration, generate_hybrid_music
from music21 import stream, note, chord

app = Flask(__name__)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sequence_to_stream_with_duration(sequence, pitchnames, durationnames, length=100, scale_factor=1.0):
    """
    Convert a sequence of (pitch_index, duration_index) tuples to a music21 stream using pitch and duration vocabularies.
    Apply scale_factor to durations to adjust for tempo.
    """
    s = stream.Stream()
    for idx_pair in sequence[:length]:
        pitch_idx, dur_idx = idx_pair
        if pitch_idx == 0:
            # Treat 0 as a rest
            dur = (durationnames[dur_idx] if dur_idx < len(durationnames) else 0.25) * scale_factor
            n = note.Rest(quarterLength=dur)
        else:
            if pitch_idx >= len(pitchnames):
                # Skip invalid indices
                continue
            pitch_name = pitchnames[pitch_idx]
            dur = (durationnames[dur_idx] if dur_idx < len(durationnames) else 0.25) * scale_factor
            if '.' in pitch_name:
                # It's a chord
                chord_notes = pitch_name.split('.')
                try:
                    chord_notes = [int(n) for n in chord_notes]
                    c = chord.Chord(chord_notes, quarterLength=dur)
                    n = c
                except Exception:
                    # fallback to treat as notes separated by dots
                    notes = [note.Note(n, quarterLength=dur) for n in pitch_name.split('.')]
                    n = chord.Chord(notes, quarterLength=dur)
            else:
                # Fix pitch name format if octave is before pitch letter
                # Example error: '5' instead of 'C5'
                # Check if pitch_name is a digit or starts with digit
                if pitch_name and (pitch_name[0].isdigit()):
                    # Attempt to reorder to pitch letter + octave
                    # Find first letter in pitch_name
                    letters = [c for c in pitch_name if c.isalpha()]
                    digits = [c for c in pitch_name if c.isdigit()]
                    if letters and digits:
                        # More robust correction: find first letter and first digit positions
                        first_letter_index = next((i for i, c in enumerate(pitch_name) if c.isalpha()), None)
                        first_digit_index = next((i for i, c in enumerate(pitch_name) if c.isdigit()), None)
                        if first_letter_index is not None and first_digit_index is not None:
                            # Reorder pitch name to letter(s) + digit(s)
                            letters_part = ''.join([c for c in pitch_name if c.isalpha()])
                            digits_part = ''.join([c for c in pitch_name if c.isdigit()])
                            corrected_pitch = letters_part + digits_part
                            try:
                                n = note.Note(corrected_pitch, quarterLength=dur)
                            except Exception:
                                # fallback to original pitch_name if corrected fails
                                n = note.Rest(quarterLength=dur)
                        else:
                            # fallback to original pitch_name
                            n = note.Rest(quarterLength=dur)
                    else:
                        # fallback to original pitch_name
                        n = note.Rest(quarterLength=dur)
                else:
                    # Additional check: if pitch_name is a single digit, treat as rest
                    if pitch_name.isdigit():
                        n = note.Rest(quarterLength=dur)
                    else:
                        n = note.Note(pitch_name, quarterLength=dur)
            s.append(n)
    return s


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/generate', methods=['POST'])
def generate():
    print("Received /generate request with data:", request.json)
    data = request.json
    mode = data.get('mode', 'rule')
    # Ensure tempo is taken as float from user input, fallback to 120 if missing or invalid
    try:
        tempo_value = float(data.get('tempo', 120))
    except Exception:
        tempo_value = 120.0
    tempo = tempo_value
    key_sig = data.get('key', 'C')
    length = int(data.get('length', 8))
    instruments = data.get('instruments', ['Piano'])
    if isinstance(instruments, str):
        instruments = [instr.strip() for instr in instruments.split(',') if instr.strip()]
    elif isinstance(instruments, list):
        instruments = [str(instr).strip() for instr in instruments if str(instr).strip()]
    else:
        instruments = ['Piano']
    
    if mode == 'rule':
        music_stream = generate_rule_based_music(tempo, key_sig, length, instruments)
    elif mode == 'lstm':
        # Load trained LSTM model if available, else build new
        import tensorflow as tf
        import pickle
        from music21 import key as m21key, pitch as m21pitch
        model_path = 'lstm_trained_model.h5'
        vocab_path = 'pitchnames.pkl'
        duration_vocab_path = 'durationnames.pkl'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded model with output shape: {model.output_shape}")
        else:
            input_shape = (50, 2)  # Adjusted input shape for trained model with pitch and duration
            pitch_output_dim = 88  # Placeholder, should be loaded from vocab
            duration_output_dim = 10  # Placeholder, should be loaded from vocab
            model = build_lstm_model_with_duration(input_shape, pitch_output_dim, duration_output_dim)
        # Load pitchnames vocabulary
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                pitchnames = pickle.load(f)
            print(f"Loaded pitchnames length: {len(pitchnames)}")
        else:
            pitchnames = []
        # Load durationnames vocabulary
        if os.path.exists(duration_vocab_path):
            with open(duration_vocab_path, 'rb') as f:
                durationnames = pickle.load(f)
            print(f"Loaded durationnames length: {len(durationnames)}")
            if len(durationnames) == 0:
                return jsonify({'error': 'Duration vocabulary is empty. Cannot generate music.'}), 500
        else:
            durationnames = []
        # Create a seed sequence with shape (50, 2) filled with zeros
        seed_sequence = np.zeros((50, 2))
        print(f"Seed sequence shape: {seed_sequence.shape}")
        generated_sequence = generate_hybrid_music(model, seed_sequence, length=length*50, temperature=1.0, key_signature=key_sig, pitchnames=pitchnames, durationnames=durationnames)  # length scaled

        def apply_music_theory_constraints(sequence, pitchnames, key_signature):
            """
            Adjust the generated sequence to fit the key signature scale.
            """
            k = m21key.Key(key_signature)
            scale_pitches = [p.nameWithOctave for p in k.pitches]
            adjusted_sequence = []
            for idx_pair in sequence:
                pitch_idx, dur_idx = idx_pair
                if pitch_idx == 0:
                    # Rest note, keep as is
                    adjusted_sequence.append((pitch_idx, dur_idx))
                    continue
                if pitch_idx >= len(pitchnames):
                    # Invalid index, skip
                    continue
                pitch_name = pitchnames[pitch_idx]
                # Skip chords (dot-separated pitch names)
                if '.' in pitch_name:
                    adjusted_sequence.append((pitch_idx, dur_idx))
                    continue
                # Check if pitch is in scale, if not, adjust to closest scale pitch
                try:
                    base_pitch = m21pitch.Pitch(pitch_name)
                except Exception:
                    # If pitch creation fails, keep original
                    adjusted_sequence.append((pitch_idx, dur_idx))
                    continue
                if base_pitch.nameWithOctave in scale_pitches:
                    adjusted_sequence.append((pitch_idx, dur_idx))
                else:
                    # Find closest pitch in scale by semitone distance
                    distances = [(abs(base_pitch.midi - m21pitch.Pitch(p).midi), p) for p in scale_pitches]
                    distances.sort(key=lambda x: x[0])
                    closest_pitch_name = distances[0][1]
                    # Map closest pitch name back to index in pitchnames
                    if closest_pitch_name in pitchnames:
                        adjusted_idx = pitchnames.index(closest_pitch_name)
                        adjusted_sequence.append((adjusted_idx, dur_idx))
                    else:
                        # If not found, keep original
                        adjusted_sequence.append((pitch_idx, dur_idx))
            return adjusted_sequence

        adjusted_sequence = apply_music_theory_constraints(generated_sequence, pitchnames, key_sig)
        # Calculate scale factor for tempo scaling
        scale_factor = 120.0 / tempo_value if tempo_value != 0 else 1.0
        music_stream = sequence_to_stream_with_duration(adjusted_sequence, pitchnames, durationnames, length=length*50, scale_factor=scale_factor)
    # Apply tempo, key signature, and instruments to the music stream
    # Set tempo
    from music21 import tempo, key, instrument
    # Set tempo marking
    # Ensure tempo is a number, convert if string
    try:
        tempo_value = float(tempo)
    except Exception:
        tempo_value = 120.0
    # Use dynamic tempo value from user input, default 120
    tempo_marking = tempo.MetronomeMark(number=tempo_value)
    music_stream.insert(0, tempo_marking)

    # Scale note durations according to tempo (default 120 bpm)
    # Use 1.0 as base tempo for scaling to make tempo scaling fluid according to user input
    base_tempo = 1.0
    scale_factor = base_tempo / tempo_value if tempo_value != 0 else 1.0

    # Removed redundant scaling after stream creation to avoid conflicts
    # Scaling is applied in sequence_to_stream_with_duration function now

    # Set key signature
    try:
        key_signature = key.Key(key_sig)
        music_stream.insert(0, key_signature)
    except Exception:
        pass
    # Set instruments for all parts (if applicable)
    # Since the generated stream may be a single part, assign instrument to all notes
    instr_map = {
        'Piano': instrument.Piano(),
        'Violin': instrument.Violin(),
        'Flute': instrument.Flute(),
        'Guitar': instrument.Guitar(),
        'Trumpet': instrument.Trumpet(),
        'Clarinet': instrument.Clarinet(),
        'Saxophone': instrument.Saxophone(),
        'Cello': instrument.Violoncello(),
    # 'Drums': instrument.DrumKit()  # Removed because music21.instrument has no DrumKit attribute
    }
    selected_instruments = []
    for instr_name in instruments:
        if instr_name in instr_map:
            selected_instruments.append(instr_map[instr_name])
    if selected_instruments:
        # Insert instruments at the beginning of the stream
        for instr_obj in selected_instruments:
            music_stream.insert(0, instr_obj)
    else:
        return jsonify({'error': 'Invalid mode'}), 400
    
    midi_filename = f"{uuid.uuid4()}.mid"
    midi_path = os.path.join(OUTPUT_DIR, midi_filename)
    save_stream_to_midi(music_stream, midi_path)
    
    mp3_filename = midi_filename.replace('.mid', '.mp3')
    mp3_path = os.path.join(OUTPUT_DIR, mp3_filename)
    try:
        convert_midi_to_mp3(midi_path, mp3_path)
    except Exception as e:
        import traceback
        print("Error during MIDI to MP3 conversion:", e)
        traceback.print_exc()
        return jsonify({'error': 'MIDI to MP3 conversion failed: ' + str(e)}), 500
    
    response = {
        'mp3_url': f"/download/{mp3_filename}",
        'midi_url': f"/download/{midi_filename}"
    }
    print("Sending response:", response)
    return jsonify(response)

@app.route('/download/<filename>')
def download_file(filename):
    if filename.lower().endswith('.mp3'):
        mimetype = 'audio/mpeg'
    elif filename.lower().endswith('.mid') or filename.lower().endswith('.midi'):
        mimetype = 'audio/midi'
    else:
        mimetype = 'application/octet-stream'
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True, mimetype=mimetype)

def cleanup_output_files():
    print("Cleaning up output files...")
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

atexit.register(cleanup_output_files)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
