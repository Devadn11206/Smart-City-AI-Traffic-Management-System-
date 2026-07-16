import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, RESULTS_DIR

def validate_dataset():
    diag_dir = os.path.join(RESULTS_DIR, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    features = ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption', 'current_phase', 'phase_duration']
    
    report_lines = []
    report_lines.append("# Dataset Validation Report")
    report_lines.append(f"**Total Records:** {len(df)}")
    report_lines.append("")
    
    # Missing values
    missing = df[features].isnull().sum()
    report_lines.append("## Missing Values")
    if missing.sum() == 0:
        report_lines.append("No missing values found.")
    else:
        report_lines.append(missing.to_markdown())
    report_lines.append("")
    
    # Descriptive Stats
    stats = df[features].describe().T
    stats['unique_values'] = df[features].nunique()
    report_lines.append("## Descriptive Statistics")
    report_lines.append(stats.to_markdown())
    report_lines.append("")
    
    # Constant columns check
    constants = [col for col in features if df[col].nunique() <= 1]
    report_lines.append("## Variance Check")
    if constants:
        report_lines.append(f"**WARNING:** The following columns have constant values and zero variance: {constants}")
    else:
        report_lines.append("All features possess sufficient variance (no constant columns).")
    report_lines.append("")
    
    # Outlier Analysis (Z-score > 3)
    report_lines.append("## Outlier Report (Z-score > 3)")
    outlier_summary = {}
    for col in features:
        if df[col].std() > 0:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > 3).sum()
            outlier_summary[col] = outliers
    
    outlier_df = pd.DataFrame(list(outlier_summary.items()), columns=['Feature', 'Outlier Count'])
    report_lines.append(outlier_df.to_markdown())
    report_lines.append("")
    
    # Correlation Matrix
    corr = df[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    corr_path = os.path.join(diag_dir, 'correlation_matrix.png')
    plt.savefig(corr_path)
    plt.close()
    
    report_lines.append("## Feature Correlation Matrix")
    report_lines.append(f"![Correlation Matrix]({corr_path})")
    report_lines.append("")
    
    # Histograms
    report_lines.append("## Feature Distributions")
    for col in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        hist_path = os.path.join(diag_dir, f'hist_{col}.png')
        plt.savefig(hist_path)
        plt.close()
        report_lines.append(f"![{col}]({hist_path})")
    
    # Write report
    report_path = os.path.join(RESULTS_DIR, 'dataset_validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Validation complete. Report saved to {report_path}")

if __name__ == '__main__':
    validate_dataset()
