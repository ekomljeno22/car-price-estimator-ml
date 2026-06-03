from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

os.makedirs('results', exist_ok=True)

X_test_proc = np.load('data/X_test_proc.npy')
y_test = np.load('data/y_test.npy')

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Evaluating model on test set...")
print(f"Test data shape: {X_test_proc.shape}\n")

y_pred_log = model.predict(X_test_proc)

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print("=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"MAE  (Mean Absolute Error):    ${mae:,.2f}")
print(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"R²   (Coefficient of Determination): {r2:.4f}")
print("=" * 60)

plt.figure(figsize=(9, 7))
plt.scatter(y_true, y_pred, alpha=0.4, s=25, color='teal', edgecolor='none')

ideal_range = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
plt.plot(ideal_range, ideal_range, 'r--', linewidth=2, label='Ideal prediction (y = x)')

plt.xlabel('Actual car price (USD)', fontsize=11)
plt.ylabel('Predicted car price (USD)', fontsize=11)
plt.title('Scatter Plot: Predicted vs. Actual Values', fontsize=13, pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig('results/prediction_vs_actual.png', dpi=200)
print("\nPlot saved as 'results/prediction_vs_actual.png'")
plt.show()

residuals = y_true - y_pred
print(f"\nResidual Analysis:")
print(f"Mean residual: ${np.mean(residuals):,.2f}")
print(f"Std residual: ${np.std(residuals):,.2f}")
print(f"Min residual: ${np.min(residuals):,.2f}")
print(f"Max residual: ${np.max(residuals):,.2f}")
