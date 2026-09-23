import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from model.dristi_model_v03 import DristiModelV03
from inference.config import MODEL_NAME, CHECKPOINT_PATH, MAX_LENGTH

def compute_ood_reference():
    print("Loading Model and Tokenizer for OOD Reference computation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DristiModelV03(model_name=MODEL_NAME, num_scores=5)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print("Loading training dataset...")
    with open("data/train.json", "r") as f:
        train_data = json.load(f)
        
    embeddings = []
    
    print("Computing embeddings for all training questions...")
    with torch.no_grad():
        for item in tqdm(train_data):
            q = item["question"]
            tokens = tokenizer(q, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            
            # Get the encoder output
            enc_out = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # The [CLS] token is the first token (index 0)
            cls_embedding = enc_out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
            
    embeddings = np.concatenate(embeddings, axis=0) # Shape: (N, 768)
    
    # Calculate the mathematical center (mean) of all known knowledge
    reference_centroid = np.mean(embeddings, axis=0)
    
    # Save the 768-dimensional vector (Takes < 1 MB of space!)
    out_path = "checkpoints/ood_centroid.npy"
    np.save(out_path, reference_centroid)
    print(f"\nOOD Reference Centroid saved to {out_path}")
    print("Dristi will now mathematically know if a question is totally alien!")

if __name__ == "__main__":
    compute_ood_reference()
