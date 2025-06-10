import os
from lstm_model import build_lstm_model_with_duration, train_lstm_model
from data_preparation import load_midi_files, get_notes_and_durations_from_midi, prepare_sequences_with_durations
from save_vocab import save_vocabulary
import numpy as np

def train_model(data_dir='data', epochs=50, batch_size=64):
    # Regenerate vocabulary dynamically based on current training data
    save_vocabulary(data_dir=data_dir)
    
    midi_files = load_midi_files(data_dir)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {data_dir}. Please add training data.")
    notes_durations = get_notes_and_durations_from_midi(midi_files)
    X_train, y_train, pitchnames, durationnames = prepare_sequences_with_durations(notes_durations)
    input_shape = (X_train.shape[1], X_train.shape[2])
    pitch_output_dim = y_train[0].shape[1]
    duration_output_dim = y_train[1].shape[1]
    model = build_lstm_model_with_duration(input_shape, pitch_output_dim, duration_output_dim)
    model.compile(loss={'pitch_output': 'categorical_crossentropy', 'duration_output': 'categorical_crossentropy'}, optimizer='adam', loss_weights=[1.0, 1.0])
    model = train_lstm_model(model, X_train=X_train, y_train=(y_train[0], y_train[1]), epochs=epochs, batch_size=batch_size)
    model.save('lstm_trained_model.h5')
    print("Model trained and saved as lstm_trained_model.h5")

if __name__ == "__main__":
    train_model()
