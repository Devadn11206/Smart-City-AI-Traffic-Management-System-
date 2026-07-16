import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from config import RESULTS_DIR

def plot_diagnostics():
    diag_dir = os.path.join(RESULTS_DIR, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    # Load Predictions
    preds_path = os.path.join(RESULTS_DIR, 'predictions.npz')
    if not os.path.exists(preds_path):
        print("Predictions not found. Run evaluate_pinn.py first.")
        return
        
    data = np.load(preds_path)
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    targets = ['Density', 'Queue Length', 'Throughput', 'Average Speed', 'Fuel Consumption']
    
    # 1. Actual vs Predicted Scatter & Residuals
    for i, target in enumerate(targets):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        residuals = y_t - y_p
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter
        axes[0].scatter(y_t, y_p, alpha=0.3, s=5, c='blue')
        min_val, max_val = np.min(y_t), np.max(y_t)
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(f'{target}: Actual vs Predicted')
        
        # Residuals
        axes[1].hist(residuals, bins=50, color='orange', alpha=0.7)
        axes[1].set_xlabel('Error (Actual - Predicted)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{target}: Error Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f'{target.replace(" ", "_").lower()}_diagnostics.png'))
        plt.close()
        
        # Time Series plot (first 200 steps)
        plt.figure(figsize=(10, 4))
        plt.plot(y_t[:200], label='Actual', alpha=0.8)
        plt.plot(y_p[:200], label='Predicted', alpha=0.8, linestyle='--')
        plt.legend()
        plt.title(f'{target}: Time Series (First 200 Test Steps)')
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f'{target.replace(" ", "_").lower()}_timeseries.png'))
        plt.close()
        
    # 2. Plot Physics Loss Audit
    curves_path = os.path.join(RESULTS_DIR, 'learning_curves.npz')
    if os.path.exists(curves_path):
        curves = np.load(curves_path)
        train_losses = curves['train']
        val_losses = curves['val']
        physics_losses = curves['physics']
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Data Loss (MSE)')
        plt.plot(epochs, physics_losses, label='Physics Loss', linestyle='--')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.title('PINN Loss Decomposition')
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, 'loss_decomposition.png'))
        plt.close()
        
    print(f"Diagnostics generated in {diag_dir}")

def generate_report():
    diag_dir = os.path.join(RESULTS_DIR, 'diagnostics')
    metrics_path = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')
    
    if not os.path.exists(metrics_path):
        return
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    report = []
    report.append("# PINN Evaluation Report")
    report.append("This report audits the Physics-Informed Neural Network after mathematical fixes.")
    report.append("")
    
    report.append("## Evaluation Metrics")
    report.append("| Target | MAE | RMSE | WMAPE | R2 |")
    report.append("|---|---|---|---|---|")
    for k, v in metrics.items():
        report.append(f"| {k} | {v['MAE']:.2f} | {v['RMSE']:.2f} | {v['WMAPE']:.2f}% | {v['R2']:.4f} |")
    report.append("")
    
    report.append("## Diagnostic Plots")
    targets = ['Density', 'Queue Length', 'Throughput', 'Average Speed', 'Fuel Consumption']
    for t in targets:
        slug = t.replace(" ", "_").lower()
        report.append(f"### {t}")
        report.append(f"![{t} Diagnostics](diagnostics/{slug}_diagnostics.png)")
        report.append(f"![{t} Time Series](diagnostics/{slug}_timeseries.png)")
        report.append("")
        
    report.append("## Physics Loss Audit")
    report.append("![Loss Decomposition](diagnostics/loss_decomposition.png)")
    report.append("If the Physics Loss declines alongside Data Loss, it means the network is successfully internalizing the traffic flow continuity equation constraint, anchoring the predictions in reality.")
    
    report_path = os.path.join(RESULTS_DIR, 'PINN_Evaluation_Report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
        
    print(f"Report generated at {report_path}")

if __name__ == '__main__':
    plot_diagnostics()
    generate_report()
