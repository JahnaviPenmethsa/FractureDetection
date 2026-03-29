# src/data_loader.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src import config

def get_label_index(name):
    try:
        return config.LABELS.index(name)
    except ValueError:
        return -1

def load_and_preprocess_data(dataset_path=config.DATA_DIR, force_reload=False):
    """
    Loads images from the dataset directory or from saved numpy arrays.
    Returns: X_train, X_test, y_train, y_test
    """
    x_save_path = os.path.join(config.PROCESSED_DIR, 'X.npy')
    y_save_path = os.path.join(config.PROCESSED_DIR, 'Y.npy')

    # Load from cached numpy arrays if they exist and we aren't forcing a reload
    if os.path.exists(x_save_path) and not force_reload:
        print("Loading data from saved numpy arrays...")
        X = np.load(x_save_path)
        Y = np.load(y_save_path)
    else:
        print("Processing raw images. This may take a moment...")
        X, Y = [], []
        
        for root, _, files in os.walk(dataset_path):
            folder_name = os.path.basename(root)
            label = get_label_index(folder_name)
            
            # Skip folders that aren't in our LABELS list
            if label == -1:
                continue
                
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, config.IMG_SIZE)
                        X.append(img)
                        Y.append(label)

        X = np.array(X, dtype='float32') / 255.0  # Normalize immediately
        Y = np.array(Y)
        
        # Save for faster loading next time
        np.save(x_save_path, X)
        np.save(y_save_path, Y)
        print(f"Processed and saved {len(X)} images.")

    # Shuffle and Split
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    Y = to_categorical(Y, num_classes=config.NUM_CLASSES)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=config.TEST_SPLIT, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Quick test to see if it works
    x_tr, x_te, y_tr, y_te = load_and_preprocess_data()
    print(f"Train shapes: X={x_tr.shape}, Y={y_tr.shape}")