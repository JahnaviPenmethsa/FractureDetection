import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

# Ensure save directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model & Training Parameters
IMG_SIZE = (128, 128) # (Height, Width)
CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 20
TEST_SPLIT = 0.2
LEARNING_RATE = 0.001

# Labels
LABELS = ['Fracture', 'Nofracture']
NUM_CLASSES = len(LABELS)