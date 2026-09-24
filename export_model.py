import os
import torch
import shutil
from transformers import AutoTokenizer

# Export Directories
EXPORT_BASE = "hf_export"
EXPORT_V1 = os.path.join(EXPORT_BASE, "dristi-v1-fast")
EXPORT_V2 = os.path.join(EXPORT_BASE, "dristi-v2-heavy")

def export_model_for_hf(checkpoint_path, base_model_name, export_dir, custom_code_path):
    print(f"\n--- Exporting model to: {export_dir} ---")
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_weights = checkpoint["model_state_dict"]
    
    # Save raw PyTorch weights
    model_out = os.path.join(export_dir, "pytorch_model.bin")
    print(f"Saving weights to {model_out}")
    torch.save(model_weights, model_out)
    
    print("Downloading and saving tokenizer/config files...")
    # This ensures the config.json and tokenizer files are present in the Hugging Face repo
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(export_dir)
    
    print(f"Copying custom model architecture code: {custom_code_path}")
    shutil.copy(custom_code_path, os.path.join(export_dir, "dristi_model.py"))

    print(f"Successfully exported to {export_dir}")


def export_dual_engine():
    print(f"Creating export directory: {EXPORT_BASE}")
    os.makedirs(EXPORT_BASE, exist_ok=True)
    
    # 1. Export V1 (Speed Demon)
    v1_checkpoint = "checkpoints/dristi_v041_best.pt"
    if os.path.exists(v1_checkpoint):
        export_model_for_hf(
            checkpoint_path=v1_checkpoint,
            base_model_name="distilbert-base-uncased",
            export_dir=EXPORT_V1,
            custom_code_path="model/dristi_model_v03.py" # DistilBERT-based model
        )
    else:
        print(f"Warning: V1 checkpoint {v1_checkpoint} not found.")

    # 2. Export V2 (Heavy Thinker)
    v2_checkpoint = "checkpoints/dristi_v06_deberta_best.pt"
    if os.path.exists(v2_checkpoint):
        export_model_for_hf(
            checkpoint_path=v2_checkpoint,
            base_model_name="microsoft/deberta-v3-small",
            export_dir=EXPORT_V2,
            custom_code_path="model/dristi_model_v04.py" # DeBERTa-based model
        )
    else:
        print(f"\n[SKIP] V2 checkpoint {v2_checkpoint} not found.")
        print("-> DeBERTa Heavy model hasn't been trained yet.")
        print("-> Run `python -m training.train_dristi_v06_deberta` first before exporting V2.")
        
    print("\n==================================================")
    print("EXPORT SCRIPT FINISHED!")
    print("==================================================")

if __name__ == "__main__":
    export_dual_engine()
