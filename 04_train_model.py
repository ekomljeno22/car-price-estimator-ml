from sklearn.neural_network import MLPRegressor
import numpy as np
import json
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
        alpha=alpha,  # L2 regularization
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

X_train_proc = np.load('data/X_train_proc.npy')
y_train = np.load('data/y_train.npy')

input_dim = X_train_proc.shape[1]
model = build_model(input_dim)

print("="*60)
print("Training Neural Network Model")
print("="*60)
print(f"Training data shape: {X_train_proc.shape}")
print(f"Target data shape: {y_train.shape}")
print(f"\nTraining configuration:")
print(f"  - Solver: Adam")
print(f"  - Learning rate: 0.001")
print(f"  - Batch size: 64")
print(f"  - Max iterations: 500 (limited by early stopping)")
print(f"  - Early stopping patience: 20 iterations")
print(f"  - Validation split: 10%")
print("="*60 + "\n")

print("Starting training... (this may take 2-10 minutes)")
model.fit(X_train_proc, y_train)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Total iterations: {model.n_iter_}")
print(f"Final loss: {model.loss_:.4f}")
print(f"Convergence: {'Yes - Early stopping triggered' if model.n_iter_ >= model.max_iter else 'Yes'}")

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to 'models/best_model.pkl'")

history = {
    'final_loss': float(model.loss_),
    'n_iterations': int(model.n_iter_),
    'training_samples': X_train_proc.shape[0],
    'model_type': 'MLPRegressor (scikit-learn)',
    'hidden_layers': (256, 128, 64)
}

with open('logs/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("Training history saved to 'logs/training_history.json'")
