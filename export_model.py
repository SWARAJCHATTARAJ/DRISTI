import os
import torch
import shutil

CHECKPOINT_IN = "checkpoints/dristi_v05_real_best.pt"
CALIBRATION_IN = "checkpoints/dristi_v041_calibration.json"

EXPORT_DIR = "hf_export"
MODEL_OUT = os.path.join(EXPORT_DIR, "pytorch_model.bin")
CALIBRATION_OUT = os.path.join(EXPORT_DIR, "calibration.json")

def export_lightweight_model():
    print(f"Creating export directory: {EXPORT_DIR}")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    print(f"Loading heavy checkpoint: {CHECKPOINT_IN}")
    # Load the heavy checkpoint (contains model weights + optimizer states for training)
    checkpoint = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)
    
    # We only need the model_state_dict for inference!
    model_weights = checkpoint["model_state_dict"]
    
    print(f"Saving lightweight inference model to: {MODEL_OUT}")
    torch.save(model_weights, MODEL_OUT)
    
    if os.path.exists(CALIBRATION_IN):
        print("Copying calibration parameters...")
        shutil.copy(CALIBRATION_IN, CALIBRATION_OUT)
        
    print("\n==================================================")
    print("EXPORT COMPLETE!")
    print("==================================================")
    print(f"The '{EXPORT_DIR}' folder now contains your production-ready model.")
    print("You can upload the contents of this folder directly to Hugging Face!")

if __name__ == "__main__":
    export_lightweight_model()
