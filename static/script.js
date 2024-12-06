document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('search-form');
    const searchType = document.getElementById('search-type');
    const weightInput = document.getElementById('weight-input');

    // Function to update form visibility based on search type
    function updateFormVisibility() {
        const type = searchType.value;
        // Only show weight input for hybrid search
        weightInput.style.display = type === 'hybrid' ? 'block' : 'none';
    }

    // Initial visibility update and add listener for changes
    updateFormVisibility();
    searchType.addEventListener('change', updateFormVisibility);

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formData = new FormData(this);
        const resultsDiv = document.getElementById('results');
        
        // Validate inputs
        const weight = parseFloat(formData.get('weight'));
        const kComponents = parseInt(formData.get('k_components'));
        
        if (weight < 0 || weight > 1) {
            alert('Weight must be between 0 and 1');
            return;
        }
        
        if (kComponents < 0) {
            alert('Number of PCA components must be non-negative');
            return;
        }
        
        // Show loading state
        resultsDiv.innerHTML = 'Searching...';
        
        fetch('/search', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            resultsDiv.innerHTML = '';
            
            data.results.forEach(result => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'result-item';
                
                const img = document.createElement('img');
                img.src = result.image_path;
                
                const score = document.createElement('p');
                score.textContent = `Similarity: ${result.similarity.toFixed(4)}`;
                
                resultDiv.appendChild(img);
                resultDiv.appendChild(score);
                resultsDiv.appendChild(resultDiv);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            resultsDiv.innerHTML = 'An error occurred while searching.';
        });
    });
});