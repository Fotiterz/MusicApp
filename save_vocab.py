import pickle
from data_preparation import get_notes_and_durations_from_midi, load_midi_files

def save_vocabulary(data_dir='data', vocab_path='pitchnames.pkl'):
    midi_files = load_midi_files(data_dir)
    notes_durations = get_notes_and_durations_from_midi(midi_files)
    pitches = [nd[0] for nd in notes_durations]
    pitchnames = sorted(set(pitches))
    with open(vocab_path, 'wb') as f:
        pickle.dump(pitchnames, f)
    print(f"Vocabulary saved to {vocab_path} with {len(pitchnames)} entries.")

def save_duration_vocabulary(data_dir='data', vocab_path='durationnames.pkl'):
    midi_files = load_midi_files(data_dir)
    notes_durations = get_notes_and_durations_from_midi(midi_files)
    durations = [nd[1] for nd in notes_durations]
    durationnames = sorted(set(durations))
    with open(vocab_path, 'wb') as f:
        pickle.dump(durationnames, f)
    print(f"Duration vocabulary saved to {vocab_path} with {len(durationnames)} entries.")

if __name__ == "__main__":
    save_vocabulary()
    save_duration_vocabulary()
