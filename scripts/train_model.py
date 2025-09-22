"""
Author: Nikabou Gaou Nadjombe
Project: ML Iris Classification
Description:
This script trains a simple neural network using TensorFlow/Keras
to classify Iris flower species based on sepal and petal measurements.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the target
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the neural network
model = Sequential([
    Dense(16, input_shape=(4,), activation='relu', name='hidden_layer_1'),
    Dense(16, activation='relu', name='hidden_layer_2'),
    Dense(3, activation='softmax', name='output_layer')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.2)

# Save the trained model
model.save('../models/iris_model.h5')

print("Model training complete. Model saved in 'models/iris_model.h5'.")
