import matplotlib.pyplot as plt
import json
import os

os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

with open('logs/training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.text(0.5, 0.9, 'Training Summary', ha='center', fontsize=14, fontweight='bold')
plt.text(0.1, 0.75, f"Final Loss: {history.get('final_loss', 'N/A'):.4f}", fontsize=11)
plt.text(0.1, 0.65, f"Iterations: {history.get('n_iterations', 'N/A')}", fontsize=11)
plt.text(0.1, 0.55, f"Training Samples: {history.get('training_samples', 'N/A')}", fontsize=11)
plt.text(0.1, 0.45, f"Model: {history.get('model_type', 'N/A')}", fontsize=11)
plt.text(0.1, 0.35, f"Hidden Layers: {history.get('hidden_layers', 'N/A')}", fontsize=11)
plt.axis('off')

plt.subplot(1, 2, 2)
arch_text = """
Neural Network Architecture:

Input Layer
    ↓
  [256] neurons (ReLU)
    ↓
  [128] neurons (ReLU)
    ↓
   [64] neurons (ReLU)
    ↓
  [1] neuron (Linear)
    ↓
Output (Price Prediction)

Optimizer: Adam
Regularization: L2 (α=0.0001)
Batch Size: 64
"""
plt.text(0.1, 0.5, arch_text, fontsize=9, family='monospace', verticalalignment='center')
plt.axis('off')

plt.tight_layout()
plt.savefig('results/learning_curves_m6.png', dpi=200)
print("Training summary visualization saved as 'results/learning_curves_m6.png'")
plt.show()

print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"Final Loss: {history.get('final_loss', 'N/A'):.4f}")
print(f"Iterations: {history.get('n_iterations', 'N/A')}")
print(f"Training Samples: {history.get('training_samples', 'N/A')}")
print(f"Model Type: {history.get('model_type', 'N/A')}")
print(f"Hidden Layers: {history.get('hidden_layers', 'N/A')}")
print("="*60)
