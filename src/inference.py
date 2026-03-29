# src/inference.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from src import config

def load_trained_model():
    """Loads the saved JSON architecture and H5 weights."""
    json_path = os.path.join(config.MODEL_SAVE_DIR, "resnet_model.json")
    weights_path = os.path.join(config.MODEL_SAVE_DIR, "resnet_model.weights.h5")
    
    if not os.path.exists(json_path) or not os.path.exists(weights_path):
        raise FileNotFoundError("Model files not found. Please run train.py first.")
        
    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()
        
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model

def predict_single_image(image_path, model):
    """Reads an image path, preprocesses it, and returns the prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Preprocess exactly how we did during training
    img_resized = cv2.resize(img, config.IMG_SIZE)
    img_array = np.array(img_resized, dtype='float32') / 255.0
    img_expanded = np.expand_dims(img_array, axis=0) # Add batch dimension: (1, 64, 64, 3)
    
    # Predict
    prediction_probs = model.predict(img_expanded)[0]
    predicted_class_index = np.argmax(prediction_probs)
    predicted_label = config.LABELS[predicted_class_index]
    confidence = prediction_probs[predicted_class_index] * 100
    
    return predicted_label, confidence, img