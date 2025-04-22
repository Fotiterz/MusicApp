document.addEventListener('DOMContentLoaded', function() {
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
