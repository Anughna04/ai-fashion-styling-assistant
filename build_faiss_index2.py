"""
Build FAISS index from fashion dataset
Run this ONCE before starting the app
"""

import torch
import numpy as np
import faiss
import pickle
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

print("Loading fashion dataset from HuggingFace...")
# Using a smaller subset for faster processing
dataset = load_dataset("detection-datasets/fashionpedia", split="train[:1000]")

print("Processing fashion items and generating embeddings...")

fashion_items = []
embeddings_list = []

for idx, item in enumerate(tqdm(dataset)):
    try:
        # Extract image and metadata
        image = item['image']
        
        # Generate text description
        text_desc = f"fashion clothing item"
        
        # Generate CLIP embedding from image
        with torch.no_grad():
            inputs = clip_processor(images=image, return_tensors="pt")
            image_features = clip_model.get_image_features(**inputs)
            embedding = image_features.cpu().numpy()[0]
        
        # Create fashion item entry
        fashion_item = {
            'id': idx,
            'name': f'Fashion Item {idx}',
            'category': 'Clothing',
            'color': 'Various',
            'pattern': 'Various',
            'fabric': 'Various',
            'occasions': ['Casual'],
            'seasons': ['Spring', 'Summer', 'Fall', 'Winter'],
            'price': f'${np.random.randint(20, 200)}',
            'description': text_desc,
            'styling_tip': 'Versatile piece for various occasions',
            'image': image  # Store PIL image
        }
        
        fashion_items.append(fashion_item)
        embeddings_list.append(embedding)
        
        if len(fashion_items) >= 100:  # Limit to 100 items for demo
            break
            
    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        continue

# Convert to numpy array
embeddings = np.array(embeddings_list).astype('float32')

print(f"\nBuilding FAISS index with {len(embeddings)} items...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save everything
print("Saving FAISS index and fashion data...")
faiss.write_index(index, "fashion_index.faiss")

with open("fashion_items.pkl", "wb") as f:
    pickle.dump(fashion_items, f)

with open("fashion_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print(f"\n✅ Successfully built FAISS index!")
print(f"   - Items: {len(fashion_items)}")
print(f"   - Embedding dimension: {dimension}")
print(f"   - Files created: fashion_index.faiss, fashion_items.pkl, fashion_embeddings.pkl")