import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "distilbert/distilbert-base-uncased"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "dristi_v041_best.pt")
CALIBRATION_PATH = os.path.join(BASE_DIR, "checkpoints", "dristi_v041_calibration.json")

MAX_LENGTH = 128
