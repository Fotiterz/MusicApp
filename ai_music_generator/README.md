# AI Music Generator

A thesis project on "The Impact of Artificial Intelligence on the Music Industry".

## Overview

This application leverages basic AI techniques—Markov Chains and Rule-Based Systems—to create simple musical compositions based on user input. It allows users to generate original songs based on high-level musical preferences, using lightweight AI approaches that require minimal computational resources.

## Features

- Generate music using Markov Chain model, Rule-Based model, or a hybrid approach
- Customize music generation with various parameters:
  - Music Genre (required): e.g., Rock, Pop, Jazz, Classical, etc.
  - Instruments (optional): e.g., Piano, Guitar, Drums, Violin, Synth.
  - Scale/Key (optional): e.g., C Major, A Minor, Dorian, Blues. Default is C Major.
  - Mood/Theme (optional): e.g., Happy, Sad, Energetic, Calm.
  - Tempo (optional): Slow, Medium, Fast.
  - Song Length (optional): Short (1–2 mins), Medium (3–4 mins), Long (5+ mins).
  - Vocals (optional): Yes/No – if Yes: Male / Female / Duet.
  - Complexity Level (optional): Simple / Intermediate / Complex. Default is Simple.
- Export generated music as MP3 files
- Simple web interface for parameter input and music playback

## System Architecture

The application consists of the following components:

1. **Markov Chain Model**: Generates melodies using a transition matrix built from predefined data.
2. **Rule-Based Model**: Applies music theory rules (chord progressions, harmonies, beat structure).
3. **Hybrid Model**: Combines rule-based chord logic with Markov-generated melodic lines.
4. **Web Interface**: Provides a user-friendly interface for parameter input and music playback.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai_music_generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install additional system dependencies:
   ```
   apt-get update
   apt-get install -y fluidsynth fluid-soundfont-gm ffmpeg
   ```

## Usage

1. Start the web interface:
   ```
   python main.py --web --port 12000
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:12000
   ```

3. Set your desired parameters and click "Generate Music".

4. Listen to the generated music and download it as an MP3 file if desired.

## Technical Details

### Markov Chain Model

The Markov Chain model uses transition matrices to generate melodies based on probabilities of note transitions. The model is pre-trained with different transition matrices for various genres and moods.

Example transition matrix for Pop/Happy:
```
[0.10, 0.15, 0.30, 0.00, 0.25, 0.00, 0.00, 0.20]  # From scale degree 1
[0.20, 0.05, 0.40, 0.10, 0.15, 0.05, 0.05, 0.00]  # From scale degree 2
...
```

### Rule-Based Model

The Rule-Based model applies music theory principles to generate melodies and harmonies. It uses predefined chord progressions, rhythm patterns, and melodic patterns based on genre, complexity, and mood.

Example chord progression for Pop:
```
['I', 'V', 'vi', 'IV']  # Most common pop progression
```

### Hybrid Model

The Hybrid model combines the strengths of both approaches:
- Uses the Rule-Based model for harmony and structure
- Uses the Markov Chain model for melody generation

## Future Improvements

- Add more sophisticated AI models (e.g., neural networks)
- Improve vocal synthesis capabilities
- Add more instruments and genres
- Implement user feedback mechanism to improve generation quality
- Add more advanced music theory rules

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- [Your Name] - [Your Email]