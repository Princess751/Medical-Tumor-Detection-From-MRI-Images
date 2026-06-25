#!/usr/bin/env python
"""
Script to generate a sample pre-trained model for demonstration
Run this script to create a dummy tumor_model.h5 before building the Docker image
"""

import tensorflow as tf
import os

def create_and_save_model(model_path='models/tumor_model.h5'):
    """Create and save a simple CNN model"""
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")
    print(f"Model Summary:")
    model.summary()

if __name__ == '__main__':
    create_and_save_model()
