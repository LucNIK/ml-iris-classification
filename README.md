# ML Iris Classification

---

## Description
This project demonstrates a complete Machine Learning workflow for classifying **Iris flower species** using a neural network implemented with **TensorFlow/Keras**.  
The model is trained on the well-known Iris dataset, which includes measurements of **sepal length, sepal width, petal length, and petal width**.  

The trained model achieves high accuracy on unseen test data and can be easily deployed for inference.  

---

## Features
- Data loading and preprocessing using **scikit-learn**.
- One-hot encoding of target labels for multi-class classification.
- Train-test split with validation set.
- Fully connected neural network with multiple hidden layers.
- Model training with configurable epochs and batch size.
- Model saved via **TensorFlow/Keras**, compatible with **HDF5** and future Keras native formats.
- Versioned model management using **Git LFS** for large files.

---

## Author
**Nikabou Gaou Nadjombe**  
- Email: [your_email@example.com]  
- GitHub: [https://github.com/LucNIK](https://github.com/LucNIK)  
- Project License: **MIT License** (or specify another license)

---

## Project Structure
<img width="293" height="114" alt="image" src="https://github.com/user-attachments/assets/49b552c5-1493-4fd3-9ff0-f9b530669925" />

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/LucNIK/ml-iris-classification.git
cd ml-iris-classification
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Train the model
```bash
python scripts/train_model.py
```

## .The trained model will be saved in models/iris_model.h5.

## .Training metrics and accuracy will be displayed in the console.

## Load the model for inference

from tensorflow.keras.models import load_model
```bash
model = load_model("models/iris_model.h5")
predictions = model.predict(new_data)
```

## Screenshots

<img width="667" height="293" alt="image" src="https://github.com/user-attachments/assets/97719b3c-70ed-4f87-9f68-efa4f68b93a8" />


## License

© 2025 Nikabou Gaou Nadjombe
All rights reserved.

This project is licensed under the MIT License — see the LICENSE file for details.

## Future Work

.Implement hyperparameter tuning for optimal model performance.

.Add interactive web app for real-time inference.

.Include visualizations for data exploration and model evaluation.

.Extend to other datasets or classification tasks.

