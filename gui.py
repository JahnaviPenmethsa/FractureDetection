# gui.py
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, END, messagebox

# Import our clean backend logic
from src import config
from src.data_loader import load_and_preprocess_data
from src.train import train_model
from src.inference import load_trained_model, predict_single_image

class TraumaDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Detection System of Fractures")
        self.root.geometry("1000x600")
        self.root.config(bg='LightSteelBlue1')
        
        self.trained_model = None

        # Title
        title_font = ('times', 16, 'bold')
        title = tk.Label(root, text='Transfer Learning for Maxillofacial Trauma Fracture Detection', 
                         bg='DarkGoldenrod1', fg='black', font=title_font, height=2)
        title.pack(fill=tk.X, pady=10)

        # Main Layout Frames
        button_frame = tk.Frame(root, bg='LightSteelBlue1')
        button_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.Y)

        text_frame = tk.Frame(root, bg='LightSteelBlue1')
        text_frame.pack(side=tk.RIGHT, padx=20, pady=20, expand=True, fill=tk.BOTH)

        # Console Output Box
        self.console = tk.Text(text_frame, height=25, width=80, font=('times', 12))
        scroll = tk.Scrollbar(text_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        btn_font = ('times', 12, 'bold')
        tk.Button(button_frame, text="1. Process Dataset", font=btn_font, command=self.process_data, width=25).pack(pady=10)
        tk.Button(button_frame, text="2. Train ResNet50 Model", font=btn_font, command=self.run_training, width=25).pack(pady=10)
        tk.Button(button_frame, text="3. Show Training Graphs", font=btn_font, command=self.show_graphs, width=25).pack(pady=10)
        tk.Button(button_frame, text="4. Predict Single Image", font=btn_font, command=self.predict_image, width=25).pack(pady=10)
        tk.Button(button_frame, text="Exit", font=btn_font, command=root.destroy, width=25, bg='indianred').pack(pady=40)

        self.log_to_console("System Initialized. Ready for inputs.")

    def log_to_console(self, message):
        """Helper to print messages to the GUI text box."""
        self.console.insert(END, message + "\n\n")
        self.console.see(END)
        self.root.update()

    def process_data(self):
        self.log_to_console(f"Processing dataset from: {config.DATA_DIR}...")
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data()
            self.log_to_console(f"Success! Total images: {len(X_train) + len(X_test)}")
            self.log_to_console(f"Training set: {len(X_train)} images")
            self.log_to_console(f"Test set: {len(X_test)} images")
        except Exception as e:
            self.log_to_console(f"Error processing data: {str(e)}")

    def run_training(self):
        self.log_to_console("Starting Training process. Please look at your command line terminal for progress bar...")
        try:
            train_model()
            self.log_to_console("Training Complete! Model weights and history saved in the 'models/saved_models' directory.")
        except Exception as e:
            self.log_to_console(f"Training Error: {str(e)}")

    def show_graphs(self):
        history_path = os.path.join(config.MODEL_SAVE_DIR, 'resnet_history.pckl')
        if not os.path.exists(history_path):
            messagebox.showerror("Error", "No training history found. Train the model first.")
            return
            
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
            
        plt.figure(figsize=(10, 5))
        plt.plot(history.get('accuracy', history.get('acc', [])), 'o-', color='green', label='Train Accuracy')
        plt.plot(history.get('val_accuracy', history.get('val_acc', [])), 'o-', color='blue', label='Val Accuracy')
        plt.title('ResNet50 Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_image(self):
        if self.trained_model is None:
            try:
                self.trained_model = load_trained_model()
            except Exception as e:
                messagebox.showerror("Error", "Could not load model. Have you trained it yet?")
                return

        file_path = filedialog.askopenfilename(title="Select X-Ray Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not file_path:
            return
            
        try:
            label, confidence, original_img = predict_single_image(file_path, self.trained_model)
            self.log_to_console(f"File: {os.path.basename(file_path)}")
            self.log_to_console(f"Prediction: {label} ({confidence:.2f}% confidence)")
            
            # Show image via OpenCV
            display_img = cv2.resize(original_img, (500, 500))
            color = (0, 0, 255) if label == 'Fracture' else (0, 255, 0)
            cv2.putText(display_img, f"{label} ({confidence:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Prediction Result", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.log_to_console(f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TraumaDetectionApp(root)
    root.mainloop()