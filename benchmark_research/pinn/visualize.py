import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR

def plot_learning_curves():
    curve_path = os.path.join(RESULTS_DIR, 'learning_curves.npz')
    if not os.path.exists(curve_path): return
    
    data = np.load(curve_path)
    train_loss = data['train']
    val_loss = data['val']
    phys_loss = data['physics']
    
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train MSE (Data)')
    plt.plot(epochs, val_loss, label='Validation Loss (Total)')
    plt.plot(epochs, phys_loss, label='Physics Loss (Residual)')
    plt.title('PINN Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'learning_curves.png'))
    plt.close()

def plot_predictions():
    pred_path = os.path.join(RESULTS_DIR, 'predictions.npz')
    if not os.path.exists(pred_path): return
    
    data = np.load(pred_path)
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    targets = ['Density', 'Queue Length', 'Throughput', 'Average Speed', 'Fuel Consumption']
    
    # 1. Prediction vs Actual Scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, target in enumerate(targets):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.1, color='blue')
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f'Actual vs Predicted {target}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'))
    plt.close()
    
    # 2. Residual Error Distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, target in enumerate(targets):
        ax = axes[i]
        residuals = y_true[:, i] - y_pred[:, i]
        sns.histplot(residuals, kde=True, ax=ax, color='purple', bins=50)
        ax.set_title(f'{target} Residuals')
        ax.set_xlabel('Error (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residual_distributions.png'))
    plt.close()
    
    # 3. Correlation Matrix
    df = pd.DataFrame(y_pred, columns=[f'Pred {t}' for t in targets])
    for i, t in enumerate(targets):
        df[f'True {t}'] = y_true[:, i]
        
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix of Predictions and True Values')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'))
    plt.close()

def main():
    print("Generating PINN Visualization Plots...")
    plot_learning_curves()
    plot_predictions()
    print("Plots saved in results/pinn/")

if __name__ == '__main__':
    main()
