"""
Fashion AI Models Module
Handles CLIP and FashionBERT model loading and inference
"""

import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from PIL import Image
import numpy as np

class FashionCLIP:
    """CLIP model fine-tuned on fashion data"""
    
    def __init__(self):
        print("Loading Fashion-CLIP model...")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model.eval()
        
    def encode_image(self, image):
        """Generate embedding from image"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
    
    def encode_text(self, text):
        """Generate embedding from text"""
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            text_features = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()

class FashionBERT:
    """Fashion-specific BERT model for attribute extraction"""
    
    def __init__(self):
        print("Loading FashionBERT model...")
        # Using a fashion-domain BERT model
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        
    def extract_features(self, text):
        """Extract fashion-specific features from text"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()
    
    def refine_embedding(self, clip_embedding, text_context):
        """Refine CLIP embedding with fashion-specific context"""
        # Extract fashion features
        fashion_features = self.extract_features(text_context)
        
        # Combine CLIP and FashionBERT features
        # Simple concatenation and normalization
        combined = np.concatenate([clip_embedding, fashion_features], axis=1)
        
        # Normalize
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
        
        return combined

def get_fashion_models():
    """Initialize and return all fashion models"""
    clip_model = FashionCLIP()
    fashion_bert = FashionBERT()
    return clip_model, fashion_bert