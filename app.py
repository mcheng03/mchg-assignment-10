import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import open_clip
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load CLIP model and preprocessor
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load pre-computed image embeddings
df = pd.read_pickle('image_embeddings.pickle').head(50000) 
all_embeddings = torch.tensor(np.stack(df['embedding'].values))

# Initialize PCA for the alternative embedding option
def init_pca(k_components):
    pca = PCA(n_components=k_components)
    embeddings_np = np.stack(df['embedding'].values)
    pca.fit(embeddings_np[:10000])  
    return pca

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    text_query = request.form.get('text_query', '')
    weight = float(request.form.get('weight', 1.0))
    k_components = int(request.form.get('k_components', 0))
    
    # Process uploaded image if present
    image_file = request.files.get('image_query')
    
    # Get embeddings based on queries
    if image_file:
        # Process image query
        image = Image.open(image_file)
        image_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = F.normalize(model.encode_image(image_tensor), dim=1)
    
    if text_query:
        # Process text query
        text = tokenizer([text_query])
        with torch.no_grad():
            text_embedding = F.normalize(model.encode_text(text), dim=1)
    
    # Combine queries if both present
    if image_file and text_query:
        query = F.normalize(weight * text_embedding + (1.0 - weight) * image_embedding, dim=1)
    elif image_file:
        query = image_embedding
    else:
        query = text_embedding
    
    # Use PCA if requested
    if k_components > 0 and image_file:
        pca = init_pca(k_components)
        query_np = query.cpu().numpy()
        embeddings_np = all_embeddings.cpu().numpy()
        
        # Transform query and database embeddings
        query_pca = torch.tensor(pca.transform(query_np))
        embeddings_pca = torch.tensor(pca.transform(embeddings_np))
        
        # Normalize PCA embeddings
        query = F.normalize(query_pca, dim=1)
        embeddings_search = F.normalize(embeddings_pca, dim=1)
    else:
        embeddings_search = all_embeddings
    
    # Compute similarities
    similarities = torch.matmul(embeddings_search, query.T).squeeze()
    
    # Get top 5 results
    top_k = 5
    top_indices = torch.topk(similarities, top_k).indices.tolist()
    top_scores = torch.topk(similarities, top_k).values.tolist()
    
    # Prepare results
    results = []
    for idx, score in zip(top_indices, top_scores):
        results.append({
            'image_path': os.path.join('coco_images_resized', df.iloc[idx]['file_name']),
            'similarity': float(score)
        })
    
    return jsonify({'results': results})

@app.route('/coco_images_resized/<path:filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized', filename)

if __name__ == '__main__':
    app.run(debug=True)