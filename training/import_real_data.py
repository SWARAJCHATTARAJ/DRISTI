import json
import os
from datasets import load_dataset

def main():
    print("Downloading 'emotion' dataset from Hugging Face...")
    print("(This contains 16,000 real-world human-labeled messages)")
    
    # The 'emotion' dataset classifies text into 6 categories:
    # 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    dataset = load_dataset("dair-ai/emotion", "split")
    
    options_map = {
        0: "Sadness",
        1: "Joy",
        2: "Love",
        3: "Anger",
        4: "Fear",
        5: "Surprise"
    }
    
    all_options = list(options_map.values())
    
    def process_split(split_name, out_filename):
        print(f"\nProcessing {split_name} split...")
        data = dataset[split_name]
        
        dristi_format = []
        for row in data:
            text = row["text"]
            label_idx = row["label"]
            
            # Map to Dristi's expected schema
            # We'll set binary YES if it's a positive emotion (Joy/Love), NO otherwise
            is_positive = label_idx in [1, 2]
            
            # For the ordinal score (1-5), we'll map intensity. 
            # Let's say Joy/Anger/Fear are intense (5), Sadness/Love are medium (3), Surprise is (4)
            # This is a bit arbitrary for the demo, but fits the schema.
            score_map = {0: 3, 1: 5, 2: 3, 3: 5, 4: 5, 5: 4}
            
            dristi_item = {
                "question": f"What is the emotion of this text: '{text}'?",
                "options": all_options,
                "yes": 1 if is_positive else 0,
                "choice": label_idx,
                "score": score_map[label_idx]  # Dataset class automatically subtracts 1
            }
            dristi_format.append(dristi_item)
            
        out_path = os.path.join("data", out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dristi_format, f, indent=2)
            
        print(f"Saved {len(dristi_format)} real-world examples to {out_path}")

    process_split("train", "train_real.json")
    process_split("validation", "validation_real.json")
    process_split("test", "test_real.json")
    
    print("\nReal-World Data Import Complete!")
    print("You are now ready to point Dristi's training script at 'data/train_real.json'!")

if __name__ == "__main__":
    main()
