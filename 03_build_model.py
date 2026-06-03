from sklearn.neural_network import MLPRegressor
import numpy as np
import pickle
import os

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def build_model(input_dim, hidden_layer_sizes=(256, 128, 64), 
                learning_rate=0.001, alpha=0.0001, random_state=42):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate,
        alpha=alpha,
        batch_size=64,
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
        warm_start=False
    )
    
    return model


if __name__ == '__main__':
    X_train_proc = np.load('data/X_train_proc.npy')
    
    input_dim = X_train_proc.shape[1]
    model = build_model(input_dim)
    
    print("="*60)
    print("MLP Regression Model Architecture")
    print("="*60)
    print(f"Input Layer:  {input_dim} neurons")
    print(f"Hidden Layer 1: 256 neurons (ReLU activation)")
    print(f"Hidden Layer 2: 128 neurons (ReLU activation)")
    print(f"Hidden Layer 3: 64 neurons (ReLU activation)")
    print(f"Output Layer: 1 neuron (linear activation)")
    print(f"\nOptimizer: Adam (lr=0.001)")
    print(f"Loss Function: MSE (Mean Squared Error)")
    print(f"Regularization: L2 (alpha=0.0001)")
    print(f"Early Stopping: Enabled (patience=20 iterations)")
    print(f"Batch Size: 64")
    print("="*60)
    
    with open('logs/model_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("MLP Regression Model Architecture\n")
        f.write("="*60 + "\n")
        f.write(f"Input Layer:  {input_dim} neurons\n")
        f.write(f"Hidden Layer 1: 256 neurons (ReLU activation)\n")
        f.write(f"Hidden Layer 2: 128 neurons (ReLU activation)\n")
        f.write(f"Hidden Layer 3: 64 neurons (ReLU activation)\n")
        f.write(f"Output Layer: 1 neuron (linear activation)\n\n")
        f.write(f"Optimizer: Adam (lr=0.001)\n")
        f.write(f"Loss Function: MSE (Mean Squared Error)\n")
        f.write(f"Regularization: L2 (alpha=0.0001)\n")
        f.write(f"Early Stopping: Enabled (patience=20 iterations)\n")
        f.write(f"Batch Size: 64\n")
        f.write("="*60 + "\n")
    
    with open('models/model_object.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel architecture saved to 'logs/model_summary.txt'")
    print("Model object saved to 'models/model_object.pkl'")
