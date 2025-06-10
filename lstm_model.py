import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical

def build_lstm_model_with_duration(input_shape, pitch_output_dim, duration_output_dim):
    """
    Build an LSTM model that predicts both pitch and duration for polyphonic music generation.
    """
    inputs = Input(shape=input_shape)  # shape: (sequence_length, 2) for pitch and duration features

    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation='relu')(x)

    pitch_output = Dense(pitch_output_dim, activation='softmax', name='pitch_output')(x)
    duration_output = Dense(duration_output_dim, activation='softmax', name='duration_output')(x)

    model = Model(inputs=inputs, outputs=[pitch_output, duration_output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', loss_weights=[1.0, 1.0])
    return model

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def build_lstm_model(input_shape, output_dim):
    """
    Build a simple LSTM model for polyphonic music generation.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def sample_with_temperature(preds, temperature=1.0):
    """
    Sample an index from a probability array reweighted by temperature.
    """
    preds = np.asarray(preds).astype('float64')
    preds = preds.flatten()  # Flatten to 1D array
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_music_with_duration(model, seed_sequence, length=100, temperature=1.0, key_signature=None, pitchnames=None, durationnames=None):
    """
    Generate a sequence of music notes and durations from the LSTM model with guided generation.
    
    Parameters:
    - model: Trained LSTM model
    - seed_sequence: Initial input sequence (numpy array) with pitch and duration features
    - length: Number of notes to generate
    - temperature: Sampling temperature for diversity
    - key_signature: Key signature string for music theory guidance
    - pitchnames: List of pitch names corresponding to vocab indices
    - durationnames: List of duration values corresponding to vocab indices
    
    Returns:
    - generated sequence as a list of (pitch_index, duration_index) tuples
    """
    from music21 import key as m21key, pitch as m21pitch
    
    generated = []
    sequence = seed_sequence.copy()
    recent_notes = []
    k = m21key.Key(key_signature) if key_signature else None
    scale_pitches = [p.name for p in k.pitches] if k else None
    
    for _ in range(length):
        pitch_pred, duration_pred = model.predict(sequence[np.newaxis, :, :], verbose=0)
        
        # Apply music theory mask to pitch prediction probabilities
        if scale_pitches and pitchnames:
            mask = np.zeros_like(pitch_pred)
            for i, pitch_name in enumerate(pitchnames):
                # Skip chord pitch names containing dots
                if '.' in pitch_name:
                    mask[0, i] = 1
                    continue
                base_pitch = m21pitch.Pitch(pitch_name)
                if base_pitch.name in scale_pitches:
                    mask[0, i] = 1
            pitch_pred = pitch_pred * mask
            if np.sum(pitch_pred) == 0:
                pitch_pred = np.ones_like(pitch_pred)
            pitch_pred = pitch_pred / np.sum(pitch_pred)
        
        # Apply repetition penalty to pitch prediction
        penalty_strength = 0.7
        for note_idx in recent_notes:
            pitch_pred[0, note_idx] *= (1 - penalty_strength)
        pitch_pred = pitch_pred / np.sum(pitch_pred)
        
        # Sample next pitch and duration indices with temperature
        pitch_index = sample_with_temperature(pitch_pred, temperature)
        duration_index = sample_with_temperature(duration_pred, temperature)
        generated.append((pitch_index, duration_index))
        
        # Update recent notes list
        recent_notes.append(pitch_index)
        if len(recent_notes) > 5:
            recent_notes.pop(0)
        
        # Normalize the predicted indices for next input
        normalized_pitch = pitch_index / float(len(pitchnames))
        normalized_duration = duration_index / float(len(durationnames))
        # Append normalized pitch and duration to sequence, remove first element
        next_input = np.array([[normalized_pitch, normalized_duration]])
        sequence = np.append(sequence[1:], next_input, axis=0)
    return generated

# Musical rules for Manos Hatzidakis style
melodic_motifs = {
    'example_folk_motif': [0, 2, 2, -1],
    'archontissa_theme': [0, 2, -1, 2, -2]
}

common_progressions = {
    'I_IV_V': [0, 5, 7],
    'I_vi_ii_V': [0, 9, 2, 7],
    'I_Vii_i_V': [0, 10, 0, 7]
}

rhythmic_templates = {
    'hasapiko': '2/4',
    'kalamatianos': '7/8',
    'syrtos': '4/4',
    'tsamikos': '3/4',
    'rhythmology': ['5/8', '7/8', '9/8', '15/8']
}

modal_scales = {
    'Dorian': [0, 2, 3, 5, 7, 9, 10],
    'Aeolian': [0, 2, 3, 5, 7, 8, 10],
    'Phrygian': [0, 1, 3, 5, 7, 8, 10]
}

def is_pitch_in_modal_scale(pitch_index, tonic_index, modal_scale, pitch_vocab):
    """
    Check if a pitch index corresponds to a pitch in the given modal scale relative to tonic.
    pitch_vocab: list of pitch names or MIDI numbers indexed by pitch_index
    tonic_index: index of tonic pitch in pitch_vocab
    modal_scale: list of semitone intervals defining the scale
    """
    from music21 import pitch as m21pitch
    pitch_name = pitch_vocab[pitch_index]
    tonic_name = pitch_vocab[tonic_index]
    # Filter out invalid pitch names that start with '.' or contain unsupported characters
    if pitch_name.startswith('.') or any(c not in 'ABCDEFGabcdefg#-0123456789' for c in pitch_name):
        return False
    if tonic_name.startswith('.') or any(c not in 'ABCDEFGabcdefg#-0123456789' for c in tonic_name):
        return False
    pitch_midi = m21pitch.Pitch(pitch_name).midi
    tonic_midi = m21pitch.Pitch(tonic_name).midi
    interval = (pitch_midi - tonic_midi) % 12
    return interval in modal_scale

def apply_musical_rules_mask(pitch_pred, pitch_vocab, tonic_index, modal_scale_name='Dorian'):
    """
    Apply a mask to pitch prediction probabilities to favor pitches in the modal scale.
    """
    modal_scale = modal_scales.get(modal_scale_name, modal_scales['Dorian'])
    mask = np.zeros_like(pitch_pred)
    for i in range(len(pitch_pred)):
        if is_pitch_in_modal_scale(i, tonic_index, modal_scale, pitch_vocab):
            mask[i] = 1
    masked_pred = pitch_pred * mask
    if np.sum(masked_pred) == 0:
        masked_pred = np.ones_like(pitch_pred)
    masked_pred = masked_pred / np.sum(masked_pred)
    return masked_pred

def generate_hybrid_music(model, seed_sequence, length=100, temperature=1.0, key_signature=None, pitchnames=None, durationnames=None, tonic_pitch_name=None, modal_scale_name='Dorian'):
    """
    Generate music sequence using the trained LSTM model with musical rules integrated as constraints.
    
    Parameters:
    - model: Trained LSTM model
    - seed_sequence: Initial input sequence (numpy array) with pitch and duration features
    - length: Number of notes to generate
    - temperature: Sampling temperature for diversity
    - key_signature: Key signature string for music theory guidance (not used directly here)
    - pitchnames: List of pitch names or MIDI numbers corresponding to vocab indices
    - durationnames: List of duration values corresponding to vocab indices
    - tonic_pitch_name: Name or MIDI number of the tonic pitch for modal scale reference
    - modal_scale_name: Name of the modal scale to apply (default 'Dorian')
    
    Returns:
    - generated sequence as a list of (pitch_index, duration_index) tuples
    """
    from music21 import pitch as m21pitch
    
    generated = []
    sequence = seed_sequence.copy()
    recent_notes = []
    
    # Determine tonic index in pitch vocab
    if tonic_pitch_name and pitchnames:
        if tonic_pitch_name in pitchnames:
            tonic_index = pitchnames.index(tonic_pitch_name)
        else:
            # fallback to first pitch as tonic
            tonic_index = 0
    else:
        tonic_index = 0
    
    for _ in range(length):
        pitch_pred, duration_pred = model.predict(sequence[np.newaxis, :, :], verbose=0)
        pitch_pred = pitch_pred.flatten()
        duration_pred = duration_pred.flatten()
        
        # Apply modal scale mask to pitch prediction
        pitch_pred = apply_musical_rules_mask(pitch_pred, pitchnames, tonic_index, modal_scale_name)
        
        # Apply repetition penalty to pitch prediction
        penalty_strength = 0.7
        for note_idx in recent_notes:
            pitch_pred[note_idx] *= (1 - penalty_strength)
        pitch_pred = pitch_pred / np.sum(pitch_pred)
        
        # Sample next pitch and duration indices with temperature
        pitch_index = sample_with_temperature(pitch_pred, temperature)
        duration_index = sample_with_temperature(duration_pred, temperature)
        generated.append((pitch_index, duration_index))
        
        # Update recent notes list
        recent_notes.append(pitch_index)
        if len(recent_notes) > 5:
            recent_notes.pop(0)
        
        # Normalize the predicted indices for next input
        normalized_pitch = pitch_index / float(len(pitchnames))
        normalized_duration = duration_index / float(len(durationnames))
        # Append normalized pitch and duration to sequence, remove first element
        next_input = np.array([[normalized_pitch, normalized_duration]])
        sequence = np.append(sequence[1:], next_input, axis=0)
    return generated

def train_lstm_model(model, X_train, y_train, epochs=20, batch_size=64):
    """
    Train the LSTM model on the given training data.
    
    Parameters:
    - model: LSTM model to train
    - X_train: Input sequences (numpy array)
    - y_train: Target outputs (tuple of numpy arrays for pitch and duration)
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    
    Returns:
    - Trained model
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

if __name__ == "__main__":
    # Example usage: build model with dummy input shape and output dims
    model = build_lstm_model_with_duration((50, 2), 88, 10)
    model.summary()
