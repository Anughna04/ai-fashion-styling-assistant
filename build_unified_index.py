import torch
import numpy as np
import faiss
import pickle
import pandas as pd
from fashion_models import FashionCLIP
from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

print("Loading CLIP model from fashion_models...")
clip_model = FashionCLIP()

# Initialize lists
fashion_items = []
embeddings_list = []
idx = 0

print("1. Processing existing HuggingFace dataset (Western/Global)...")
dataset = load_dataset("detection-datasets/fashionpedia", split="train[:100]") # Limit to 100 for speed
for item in tqdm(dataset):
    try:
        image = item['image']
        embedding = clip_model.encode_image(image)[0]

        
        fashion_item = {
            'id': idx,
            'name': f'Western Item {idx}',
            'category': 'Western Clothing',
            'origin': 'Western',
            'occasions': ['Casual', 'Party', 'Formal'],
            'seasons': ['Spring', 'Summer', 'Fall', 'Winter'],
            'price': f'${np.random.randint(20, 200)}',
            'description': 'Global fashion piece',
            'image': image
        }
        
        fashion_items.append(fashion_item)
        embeddings_list.append(embedding)
        idx += 1
    except Exception as e:
        print(f"HF Error: {e}")
        continue

print("2. Processing Indian Dataset...")
# Path to kaggle dataset
csv_path = os.path.join("indian_data", "data.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    # Using only top 100 to avoid memory/time overload
    for _, row in tqdm(df.head(100).iterrows(), total=100):
        try:
            # Assuming images are in indian_data/data/ or similar. Adjust path as needed.
            # Some datasets have images in same folder or subfolder. 
            image_name = row['image']
            img_path = os.path.join("indian_data", "data", str(image_name))
            
            if not os.path.exists(img_path):
                # Try just indian_data directory
                img_path = os.path.join("indian_data", str(image_name))
                
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                embedding = clip_model.encode_image(image)[0]
                
                cat = str(row.get('category', 'Ethnic'))
                desc = str(row.get('description', 'Indian ethnic wear'))
                name = str(row.get('display name', f'Indian Item {idx}'))

                fashion_item = {
                    'id': idx,
                    'name': name,
                    'category': cat,
                    'origin': 'Indian',  # Mark as Indian
                    'occasions': ['Wedding', 'Festive', 'Casual'],
                    'seasons': ['Spring', 'Summer', 'Fall', 'Winter'],
                    'price': f'₹{np.random.randint(500, 5000)}',
                    'description': desc,
                    'image': image
                }
                
                fashion_items.append(fashion_item)
                embeddings_list.append(embedding)
                idx += 1
            else:
                print(f"File not found: {img_path}")
        except Exception as e:
            print(f"Indian Data Error: {e}")
            continue
else:
    print(f"Warning: {csv_path} not found. Skipping Indian dataset integration.")

if len(embeddings_list) > 0:
    # Convert to numpy array
    embeddings = np.array(embeddings_list).astype('float32')

    print(f"\nBuilding FAISS index with {len(embeddings)} items...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save everything
    print("Saving UNIFIED FAISS index and fashion data...")
    faiss.write_index(index, "fashion_index.faiss")

    with open("fashion_items.pkl", "wb") as f:
        pickle.dump(fashion_items, f)

    with open("fashion_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"\n✅ Successfully built UNIFIED FAISS index!")
    print(f"   - Total Items: {len(fashion_items)}")
    print(f"   - Embedding dimension: {dimension}")
else:
    print("No items processed.")
