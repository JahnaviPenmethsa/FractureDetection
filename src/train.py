# src/train.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src import config
from src.data_loader import load_and_preprocess_data
from src.model import build_resnet_model

def train_model():
    print("Step 1: Loading Data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Step 2: Building Model Architecture...")
    model = build_resnet_model(input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], config.CHANNELS))
    
    print("Step 3: Training the Model (This will take a while)...")
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_test, y_test),
        shuffle=True,
        verbose=1
    )
    
    print("Step 4: Saving Model and Weights...")
    # Save the architecture
    model_json = model.to_json()
    with open(os.path.join(config.MODEL_SAVE_DIR, "resnet_model.json"), "w") as json_file:
        json_file.write(model_json)

    # Save the weights
    model.save_weights(os.path.join(config.MODEL_SAVE_DIR, "resnet_model.weights.h5"))
    
    # Save the training history for the graphs
    with open(os.path.join(config.MODEL_SAVE_DIR, "resnet_history.pckl"), "wb") as f:
        pickle.dump(history.history, f)

    print("Step 5: Evaluating Model on Test Data...")
    evaluate_model(model, X_test, y_test)
    print("\nTraining Complete! Model saved successfully.")

def evaluate_model(model, X_test, y_test):
    """Generates and prints evaluation metrics and plots a confusion matrix."""
    predictions = model.predict(X_test)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true_classes, y_pred_classes) * 100
    precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0) * 100
    recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0) * 100
    
    print("\n--- Model Performance ---")
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {precision:.2f}%")
    print(f"Recall    : {recall:.2f}%")
    print(f"F1-Score  : {f1:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, xticklabels=config.LABELS, yticklabels=config.LABELS, annot=True, fmt="g", cmap="Blues")
    plt.title("Test Set Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    # Save the plot instead of pausing the script to show it
    plt.savefig(os.path.join(config.MODEL_SAVE_DIR, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    train_model()