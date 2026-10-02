import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(seed=42):
    # Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Normalize pixels to [0, 1]
    X = X / 16.0
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Create one-hot labels
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1.0
    
    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1.0
    
    return X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test
