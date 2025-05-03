"""
Web interface for the AI Music Generator.
"""

import os
import json
from flask import Flask, request, render_template, send_from_directory, jsonify
from core.music_generator import MusicGenerator

def start_web_server(port=12000):
    """
    Start the web interface for the AI Music Generator.
    
    Args:
        port (int): Port number to run the server on
    """
    app = Flask(__name__, template_folder='../ui/templates', static_folder='../ui/static')
    music_generator = MusicGenerator()
    
    # Create templates and static directories if they don't exist
    os.makedirs('../ui/templates', exist_ok=True)
    os.makedirs('../ui/static', exist_ok=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('../output', exist_ok=True)
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/generate', methods=['POST'])
    def generate():
        """Generate music based on user parameters."""
        try:
            # Get parameters from request
            params = request.json
            
            # Log the parameters for debugging
            print(f"Generating music with parameters: {params}")
            
            # Ensure instruments is a list
            if 'instruments' in params and not isinstance(params['instruments'], list):
                params['instruments'] = [params['instruments']]
            
            # Generate music
            mp3_path = music_generator.generate_music(params)
            
            # Return the path to the generated MP3 file
            return jsonify({
                'success': True,
                'mp3_path': os.path.basename(mp3_path)
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/output/<path:filename>')
    def download_file(filename):
        """Serve generated music files."""
        return send_from_directory('../output', filename)
    
    # Create HTML template
    create_html_template()
    
    # Create CSS file
    create_css_file()
    
    # Create JavaScript file
    create_js_file()
    
    # Start the server
    app.run(host='0.0.0.0', port=port, debug=True)

def create_html_template():
    """Create the HTML template for the web interface."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Music Generator</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>AI Music Generator</h1>
            <p>Generate original music using AI techniques</p>
        </header>
        
        <main>
            <div class="form-container">
                <h2>Music Parameters</h2>
                
                <div class="form-group">
                    <label for="genre">Genre (required):</label>
                    <select id="genre" required>
                        <option value="Pop">Pop</option>
                        <option value="Rock">Rock</option>
                        <option value="Jazz">Jazz</option>
                        <option value="Classical">Classical</option>
                        <option value="Blues">Blues</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="instruments">Instruments:</label>
                    <select id="instruments" multiple>
                        <option value="Piano" selected>Piano</option>
                        <option value="Guitar">Guitar</option>
                        <option value="Bass">Bass</option>
                        <option value="Drums">Drums</option>
                        <option value="Violin">Violin</option>
                        <option value="Synth">Synth</option>
                        <option value="Flute">Flute</option>
                        <option value="Trumpet">Trumpet</option>
                    </select>
                    <small>Hold Ctrl/Cmd to select multiple</small>
                </div>
                
                <div class="form-group">
                    <label for="scale">Scale/Key:</label>
                    <select id="scale">
                        <option value="C Major" selected>C Major</option>
                        <option value="A Minor">A Minor</option>
                        <option value="G Major">G Major</option>
                        <option value="E Minor">E Minor</option>
                        <option value="F Major">F Major</option>
                        <option value="D Minor">D Minor</option>
                        <option value="D Major">D Major</option>
                        <option value="B Minor">B Minor</option>
                        <option value="A Major">A Major</option>
                        <option value="F# Minor">F# Minor</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="mood">Mood/Theme:</label>
                    <select id="mood">
                        <option value="Happy">Happy</option>
                        <option value="Sad">Sad</option>
                        <option value="Energetic">Energetic</option>
                        <option value="Calm" selected>Calm</option>
                        <option value="Neutral">Neutral</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="tempo">Tempo:</label>
                    <select id="tempo">
                        <option value="Slow">Slow</option>
                        <option value="Medium" selected>Medium</option>
                        <option value="Fast">Fast</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="length">Song Length:</label>
                    <select id="length">
                        <option value="Short">Short (1-2 mins)</option>
                        <option value="Medium" selected>Medium (3-4 mins)</option>
                        <option value="Long">Long (5+ mins)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="vocals-enabled">Vocals:</label>
                    <input type="checkbox" id="vocals-enabled">
                    
                    <div id="vocals-options" class="hidden">
                        <label for="vocals-type">Type:</label>
                        <select id="vocals-type">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Duet">Duet</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="complexity">Complexity Level:</label>
                    <select id="complexity">
                        <option value="Simple" selected>Simple</option>
                        <option value="Intermediate">Intermediate</option>
                        <option value="Complex">Complex</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="mode">Generation Mode:</label>
                    <select id="mode">
                        <option value="markov">Markov Chain</option>
                        <option value="rule">Rule-Based</option>
                        <option value="hybrid" selected>Hybrid</option>
                    </select>
                </div>
                
                <button id="generate-btn">Generate Music</button>
            </div>
            
            <div class="player-container hidden" id="player-section">
                <h2>Generated Music</h2>
                <audio id="audio-player" controls></audio>
                <div class="player-controls">
                    <button id="regenerate-btn">Regenerate</button>
                    <a id="download-link" href="#" download>Download MP3</a>
                </div>
            </div>
        </main>
        
        <div id="loading" class="hidden" style="display: none;">
            <div class="spinner"></div>
            <p>Generating music...</p>
        </div>
        
        <footer>
            <p>AI Music Generator - Thesis Project</p>
            <p>"The Impact of Artificial Intelligence on the Music Industry"</p>
        </footer>
    </div>
    
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
"""
    
    # Create templates directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    
    # Write the HTML template
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w') as f:
        f.write(html)

def create_css_file():
    """Create the CSS file for the web interface."""
    css = """/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header styles */
header {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background-color: #2c3e50;
    color: white;
    border-radius: 5px;
}

header h1 {
    margin-bottom: 10px;
}

/* Main content styles */
main {
    display: flex;
    flex-wrap: wrap;
    gap: 30px;
    margin-bottom: 30px;
}

.form-container {
    flex: 1;
    min-width: 300px;
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.form-container h2 {
    margin-bottom: 20px;
    color: #2c3e50;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

.form-group select,
.form-group input[type="text"],
.form-group input[type="number"] {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
}

.form-group select[multiple] {
    height: 120px;
}

.form-group small {
    display: block;
    margin-top: 5px;
    color: #666;
    font-size: 12px;
}

button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

/* Player styles */
.player-container {
    flex: 1;
    min-width: 300px;
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.player-container h2 {
    margin-bottom: 20px;
    color: #2c3e50;
}

audio {
    width: 100%;
    margin-bottom: 15px;
}

.player-controls {
    display: flex;
    gap: 10px;
}

#download-link {
    display: inline-block;
    background-color: #2ecc71;
    color: white;
    text-decoration: none;
    padding: 10px 15px;
    border-radius: 4px;
    transition: background-color 0.3s;
}

#download-link:hover {
    background-color: #27ae60;
}

/* Loading spinner */
#loading {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    color: white;
}

.spinner {
    border: 5px solid #f3f3f3;
    border-top: 5px solid #3498db;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 2s linear infinite;
    margin-bottom: 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Footer styles */
footer {
    text-align: center;
    padding: 20px;
    background-color: #2c3e50;
    color: white;
    border-radius: 5px;
}

/* Utility classes */
.hidden {
    display: none !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    main {
        flex-direction: column;
    }
}
"""
    
    # Create static directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    # Write the CSS file
    with open(os.path.join(os.path.dirname(__file__), 'static', 'style.css'), 'w') as f:
        f.write(css)

def create_js_file():
    """Create the JavaScript file for the web interface."""
    js = """document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const generateBtn = document.getElementById('generate-btn');
    const regenerateBtn = document.getElementById('regenerate-btn');
    const vocalsEnabled = document.getElementById('vocals-enabled');
    const vocalsOptions = document.getElementById('vocals-options');
    const playerSection = document.getElementById('player-section');
    const audioPlayer = document.getElementById('audio-player');
    const downloadLink = document.getElementById('download-link');
    const loading = document.getElementById('loading');
    
    // Ensure loading spinner is hidden on page load
    loading.classList.add('hidden');
    loading.style.display = 'none';
    
    // Show/hide vocals options based on checkbox
    vocalsEnabled.addEventListener('change', function() {
        if (this.checked) {
            vocalsOptions.classList.remove('hidden');
        } else {
            vocalsOptions.classList.add('hidden');
        }
    });
    
    // Generate music
    generateBtn.addEventListener('click', generateMusic);
    regenerateBtn.addEventListener('click', generateMusic);
    
    function generateMusic() {
        // Show loading spinner
        loading.classList.remove('hidden');
        loading.style.display = 'flex';
        
        // Get selected instruments
        const instrumentsSelect = document.getElementById('instruments');
        const selectedInstruments = Array.from(instrumentsSelect.selectedOptions).map(option => option.value);
        
        // Get vocals parameters
        const vocalsParams = {
            enabled: vocalsEnabled.checked,
            type: document.getElementById('vocals-type').value
        };
        
        // Prepare parameters
        const params = {
            genre: document.getElementById('genre').value,
            instruments: selectedInstruments.length > 0 ? selectedInstruments : ['Piano'],
            scale: document.getElementById('scale').value,
            mood: document.getElementById('mood').value,
            tempo: document.getElementById('tempo').value,
            length: document.getElementById('length').value,
            vocals: vocalsParams,
            complexity: document.getElementById('complexity').value,
            mode: document.getElementById('mode').value
        };
        
        // Send request to server
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loading.classList.add('hidden');
            loading.style.display = 'none';
            
            if (data.success) {
                // Show player section
                playerSection.classList.remove('hidden');
                
                // Set audio source
                const mp3Url = `/output/${data.mp3_path}`;
                audioPlayer.src = mp3Url;
                audioPlayer.play();
                
                // Set download link
                downloadLink.href = mp3Url;
                downloadLink.download = data.mp3_path;
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            // Hide loading spinner
            loading.classList.add('hidden');
            loading.style.display = 'none';
            
            alert(`Error: ${error.message}`);
        });
    }
});
"""
    
    # Create static directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    # Write the JavaScript file
    with open(os.path.join(os.path.dirname(__file__), 'static', 'script.js'), 'w') as f:
        f.write(js)
