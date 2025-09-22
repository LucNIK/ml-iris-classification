# Models Folder – ML Iris Classification

## Description
This folder contains the trained neural network models generated
during the training process of the Iris classification project.

## Contents
- `iris_model.h5` – Keras/TensorFlow saved model for Iris classification

## Notes
- The model architecture consists of:
  - Input layer: 4 features
  - Two hidden layers with 16 neurons each and ReLU activation
  - Output layer: 3 neurons with softmax activation
- Loss function: categorical_crossentropy
- Optimizer: Adam
- Training epochs: 50
- Batch size: 5

## Usage
- Load the model in Python:
```python
from tensorflow.keras.models import load_model
model = load_model('models/iris_model.h5')
