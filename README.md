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
   cd MusicApp
   ```

2. Install the required Python dependencies:
   ```
   pip install -r ai_music_generator/requirements.txt
   ```

3. Install additional system dependencies:

   ### Linux (Debian/Ubuntu)
   ```
   sudo apt-get update
   sudo apt-get install -y fluidsynth fluid-soundfont-gm ffmpeg
   ```

   ### macOS
   ```
   brew install fluid-synth ffmpeg
   ```

   ### Windows
   
   #### Automatic Installation (Recommended)
   
   When you first run the application, if FluidSynth is not found, two helper batch files 
   will be created in the current directory:
   
   1. `install_fluidsynth.bat` - Run as administrator to:
      - Download FluidSynth
      - Install it to `C:\Program Files\FluidSynth`
      - Add it to your system PATH environment variable
   
   2. `install_fluidsynth_local.bat` - For users without administrator privileges:
      - Download FluidSynth
      - Install it to your user directory (`%USERPROFILE%\FluidSynth`)
      - Add it to your user PATH environment variable
      - Create a configuration file for the application to find FluidSynth
   
   #### Manual Installation
   
   If the automatic installation doesn't work, follow these steps:
   
   ##### FluidSynth
   1. Download FluidSynth from: https://github.com/FluidSynth/fluidsynth/releases
   2. Extract the ZIP file to `C:\Program Files\FluidSynth`
   3. Add the bin directory to your PATH environment variable:
      - Right-click on "This PC" or "My Computer" and select "Properties"
      - Click on "Advanced system settings"
      - Click on "Environment Variables"
      - Under "System variables", find and select "Path", then click "Edit"
      - Click "New" and add `C:\Program Files\FluidSynth\bin`
      - Click "OK" on all dialogs
   
   ##### FFmpeg
   1. Download FFmpeg from: https://ffmpeg.org/download.html
   2. Extract the ZIP file to a location on your computer (e.g., `C:\Program Files\ffmpeg`)
   3. Add the bin directory to your PATH environment variable (as described above)
   
   #### SoundFont
   The application will automatically download the required SoundFont file on first run.
   If the download fails, you can manually download it from:
   https://archive.org/download/fluidr3-gm-gs/FluidR3_GM.sf2
   
   Place the downloaded file in the `ai_music_generator/data` directory.

## Usage

1. Navigate to the project directory:
   ```
   cd MusicApp
   ```

2. Start the web interface:
   ```
   python ai_music_generator/main.py --web --port 12000
   ```

3. Open a web browser and navigate to:
   ```
   http://localhost:12000
   ```

4. Set your desired parameters and click "Generate Music".

5. Listen to the generated music and download it as an MP3 file if desired.

### Troubleshooting

#### Windows-specific issues:

- If you see "FluidSynth not found" or "The system cannot find the path specified: 'C:\tools\fluidsynth\bin'" errors:
  1. Run the `install_fluidsynth.bat` file that was created in your current directory (run as administrator)
     - If you don't have administrator privileges, use `install_fluidsynth_local.bat` instead
  2. After installation, restart your command prompt or terminal
  3. If you still have issues, manually install FluidSynth to `C:\Program Files\FluidSynth` as described in the installation section
  4. Verify that FluidSynth is in your PATH by running `fluidsynth --version` in a command prompt
  5. If you've installed FluidSynth but it's still not found, restart your computer

- If you encounter issues with the SoundFont file:
  1. Download it manually from the link provided in the installation section
  2. Create a folder named `data` in the `ai_music_generator` directory
  3. Place the downloaded SoundFont file in this folder
  4. The file should be at: `ai_music_generator/data/FluidR3_GM.sf2`

- If you see "fluid_is_soundfont(): fopen() failed: 'File does not exist'" errors:
  1. This means FluidSynth is installed but can't find the SoundFont file
  2. Make sure the SoundFont file exists at `ai_music_generator/data/FluidR3_GM.sf2`
  3. If you've placed the file there and still get the error, try using an absolute path:
     - Create a file named `fluidsynth_path.txt` in the project root directory
     - Add the full path to your SoundFont file, e.g., `C:\Users\username\MusicApp\ai_music_generator\data\FluidR3_GM.sf2`
  4. Restart the application

- If you get permission errors when installing FluidSynth:
  1. Make sure you're running the batch file or commands as administrator
  2. Right-click on Command Prompt or PowerShell and select "Run as administrator"

#### Linux/macOS-specific issues:

- If you encounter permission issues when installing system dependencies:
  1. Make sure you're using `sudo` with the installation commands
  2. If you don't have sudo access, consider using a virtual environment or container

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
