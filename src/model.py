# src/model.py
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from src import config

def build_resnet_model(input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], config.CHANNELS)):
    """
    Builds and compiles the ResNet50 Transfer Learning model.
    """
    # 1. Load pre-trained ResNet50 (without the final classification layer)
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # 2. Freeze the base model so we don't destroy pre-trained features
    for layer in base_model.layers:
        layer.trainable = False
        
    # 3. Build our custom classification head
    model = Sequential()
    model.add(base_model)
    
    # GlobalAveragePooling is generally better than Flatten for ResNet
    model.add(GlobalAveragePooling2D()) 
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5)) # Prevents the model from memorizing the data
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    # 4. Compile the model
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Quick test to see the architecture
    test_model = build_resnet_model()
    test_model.summary()