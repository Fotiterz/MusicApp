# Music Generation App

This project is a music generation application that uses both rule-based and LSTM-based methods to generate polyphonic music. It provides a Flask web interface for users to generate music by selecting parameters such as tempo, key, length, instruments, and generation mode.

## Features

- Rule-based music generation using predefined chord progressions and melodic variations.
- LSTM-based music generation trained on MIDI datasets, predicting pitch and duration.
- Web interface to generate music and download MIDI and MP3 files.
- Support for multiple instruments and key signatures.
- Utilities for MIDI to MP3 conversion using FluidSynth and pydub.

## Installation

1. Clone the repository.

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have FluidSynth installed on your system and the `FluidR3_GM.sf2` soundfont file placed in the project root directory.
Download the `FluidR3_GM.sf2` from: https://member.keymusician.com/Member/FluidR3_GM/index.html

- On Ubuntu/Debian:

```bash
sudo apt-get install fluidsynth
```

- On macOS (using Homebrew):

```bash
brew install fluid-synth
```

## Usage

### Running the Web Application

Start the Flask web server:

```bash
python3 main.py
```

Open your browser and navigate to `http://localhost:5000` to access the web interface.

### Training the LSTM Model

To train the LSTM model on your MIDI dataset:

```bash
python3 train_lstm.py
```

Make sure your MIDI files are placed in the `data/` directory.

There is an already trained model avielable (lstm_trained_model.h5) so this step is not necessary.

### Generating Music

Use the web interface to select generation mode (`Rule-Based` or `LSTM Model`), tempo, key, length, and instruments. Generated music can be downloaded as MIDI and MP3 files.

## Project Structure

- `main.py`: Flask web application with routes for music generation and file download.
- `rule_based.py`: Rule-based music generation logic.
- `lstm_model.py`: LSTM model building, training, and music generation functions.
- `train_lstm.py`: Script to train the LSTM model.
- `data_preparation.py`: Functions to load and preprocess MIDI data for training.
- `save_vocab.py`: Utilities to save pitch and duration vocabularies.
- `utils.py`: Utilities for saving MIDI files and converting MIDI to MP3.
- `data/`: Directory containing MIDI dataset files.
- `output/`: Directory where generated MIDI and MP3 files are saved.
- `static/` and `templates/`: Web interface static assets and HTML templates.
- `FluidR3_GM.sf2`: Soundfont file required for MIDI to MP3 conversion.

## Dependencies

- Python 3.x
- Flask
- TensorFlow
- music21
- numpy
- pydub
- FluidSynth (system dependency)

## Notes

- Ensure the `FluidR3_GM.sf2` soundfont file is present in the project root for audio conversion.
- The project supports polyphonic music generation with pitch and duration modeling.
- The LSTM model expects MIDI data prepared in the `data/` directory.

## License

This project is provided as-is for educational and research purposes.

## Authors

Fotis Terzenidis

Myrto Panagiotopoulou
