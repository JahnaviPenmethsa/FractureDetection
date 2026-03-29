# Maxillofacial Trauma Fracture Detection System

An automated, deep learning-based diagnostic tool designed to detect fractures in patients with maxillofacial trauma using X-ray imagery. 

## Overview
This project leverages **Transfer Learning** using the **ResNet50** architecture to classify medical images into two categories: `Fracture` and `Nofracture`. The system includes a robust data preprocessing pipeline and a user-friendly Tkinter GUI for easy interaction and individual image inference.

## Features
* **Data Pipeline:** Automated resizing (128x128), normalization, and train/test splitting.
* **Transfer Learning:** Utilizes ResNet50 (pre-trained on ImageNet) with a custom dense classification head and dropout layers to prevent overfitting.
* **High Performance:** Achieves ~98% Accuracy, Precision, and Recall on the validation set.
* **Interactive GUI:** Allows users to easily train the model, view performance graphs, and predict outcomes on new, single X-ray images.

## Project Structure
```text
TraumaDetection/
│
├── data/
│   ├── raw/                # contains 'fracture' and 'nofracture' folders 
│   └── processed/          # Caches .npy arrays for faster loading
│
├── models/
│   └── saved_models/       # Stores trained weights (.weights.h5) and architecture (.json)
│
├── src/
│   ├── config.py           # Centralized hyperparameters (Epochs, Batch Size, etc.)
│   ├── data_loader.py      # Logic for image preprocessing
│   ├── model.py            # ResNet50 architecture definition
│   ├── train.py            # Training loop and metric evaluation
│   └── inference.py        # Logic for single-image prediction
│
├── gui.py                  # The main Tkinter application
├── requirements.txt        # Project dependencies
└── README.md
```

## Setup & Installation

1. Clone the repository
```bash
git clone [https://github.com/JahnaviPenmethsa/FractureDetection.git](https://github.com/JahnaviPenmethsa/FractureDetection.git)
cd TraumaDetection
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```


3. Add Data
Create the data/raw/ directory and place your two folders (fracture and nofracture) containing the dataset images inside.

4. Run the Application
```bash
python gui.py
```

## How to Use

* Launch the application using python gui.py.

* Click Process Dataset to prepare the images.

* Click Train ResNet50 Model to begin training (progress will display in your terminal).

* Once complete, click Show Training Graphs to view model performance.

* Use Predict Single Image to test the model on a new X-ray.

## Built With

* TensorFlow/Keras - Deep Learning Framework

* OpenCV - Image Processing

* Scikit-Learn - Evaluation Metrics

* Tkinter - Graphical User Interface