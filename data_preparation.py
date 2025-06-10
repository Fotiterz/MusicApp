import numpy as np
from music21 import converter, instrument, note, chord, stream
import os
import glob

def get_notes_and_durations_from_midi(midi_files):
    """Extract notes/chords and their durations from a list of MIDI files."""
    notes_durations = []
    for file in midi_files:
        midi = converter.parse(file)
        print(f"Parsing {file}")
        parts = instrument.partitionByInstrument(midi)
        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            duration = element.quarterLength
            if isinstance(element, note.Note):
                notes_durations.append((str(element.pitch), duration))
            elif isinstance(element, chord.Chord):
                # Convert chord pitches to pitch names instead of normalOrder integers
                chord_pitches = '.'.join(str(pitch.nameWithOctave) for pitch in element.pitches)
                notes_durations.append((chord_pitches, duration))
    return notes_durations

def prepare_sequences_with_durations(notes_durations, sequence_length=50):
    """Prepare sequences including pitch and duration for the Neural Network."""
    # Separate pitches and durations
    pitches = [nd[0] for nd in notes_durations]
    durations = [nd[1] for nd in notes_durations]

    # Create vocabularies for pitches and durations
    pitchnames = sorted(set(pitches))
    durationnames = sorted(set(durations))

    # Create dictionaries to map pitches and durations to integers
    pitch_to_int = dict((pitch, number) for number, pitch in enumerate(pitchnames))
    duration_to_int = dict((dur, number) for number, dur in enumerate(durationnames))

    network_input_pitch = []
    network_input_duration = []
    network_output_pitch = []
    network_output_duration = []

    for i in range(len(notes_durations) - sequence_length):
        seq_in_pitch = pitches[i:i + sequence_length]
        seq_in_duration = durations[i:i + sequence_length]
        seq_out_pitch = pitches[i + sequence_length]
        seq_out_duration = durations[i + sequence_length]

        network_input_pitch.append([pitch_to_int[p] for p in seq_in_pitch])
        network_input_duration.append([duration_to_int[d] for d in seq_in_duration])
        network_output_pitch.append(pitch_to_int[seq_out_pitch])
        network_output_duration.append(duration_to_int[seq_out_duration])

    n_patterns = len(network_input_pitch)

    # Reshape and normalize inputs
    network_input_pitch = np.reshape(network_input_pitch, (n_patterns, sequence_length, 1))
    network_input_duration = np.reshape(network_input_duration, (n_patterns, sequence_length, 1))

    network_input_pitch = network_input_pitch / float(len(pitchnames))
    network_input_duration = network_input_duration / float(len(durationnames))

    # One hot encode outputs
    network_output_pitch = np.eye(len(pitchnames))[network_output_pitch]
    network_output_duration = np.eye(len(durationnames))[network_output_duration]

    # Combine pitch and duration inputs into one array with two features per timestep
    network_input = np.concatenate((network_input_pitch, network_input_duration), axis=2)

    # Combine pitch and duration outputs into a tuple (or handle separately in model)
    network_output = (network_output_pitch, network_output_duration)

    return network_input, network_output, pitchnames, durationnames

def load_midi_files(data_dir='data'):
    """Load all MIDI files from the data directory and subdirectories recursively."""
    midi_files = glob.glob(os.path.join(data_dir, '**', '*.mid'), recursive=True)
    return midi_files

if __name__ == "__main__":
    data_dir = 'data'
    midi_files = load_midi_files(data_dir)
    notes = get_notes_from_midi(midi_files)
    X, y, pitchnames = prepare_sequences(notes)
    print(f"Total sequences: {len(X)}")
