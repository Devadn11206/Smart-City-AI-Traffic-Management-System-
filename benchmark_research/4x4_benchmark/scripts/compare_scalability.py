import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.dirname(base_dir)
    
    # Load 2x2 results
    res_2x2_file = os.path.join(parent_dir, 'results', '2x2', 'benchmark_comparison', '2x2_benchmark_results.csv')
    res_4x4_file = os.path.join(base_dir, 'results', '4x4', 'benchmark_comparison', '4x4_benchmark_results.csv')
    
    if not os.path.exists(res_2x2_file):
        print(f"Error: {res_2x2_file} not found.")
        return
    if not os.path.exists(res_4x4_file):
        print(f"Error: {res_4x4_file} not found. Ensure training and analysis have run.")
        return
        
    df_2x2 = pd.read_csv(res_2x2_file)
    df_4x4 = pd.read_csv(res_4x4_file)
    
    merged = pd.merge(df_2x2, df_4x4, on='Algorithm', suffixes=('_2x2', '_4x4'))
    
    def calc_change(col):
        return ((merged[f'{col}_4x4'] - merged[f'{col}_2x2']) / abs(merged[f'{col}_2x2'])) * 100
        
    merged['Wait_Change_%'] = calc_change('Average Waiting Time')
    merged['Queue_Change_%'] = calc_change('Average Queue Length')
    merged['Throughput_Change_%'] = calc_change('Throughput')
    merged['Travel_Time_Change_%'] = calc_change('Average Travel Time')
    merged['Fuel_Change_%'] = calc_change('Fuel Consumption')
    merged['Speed_Change_%'] = calc_change('Average Speed')
    merged['Reward_Change_%'] = calc_change('Total Reward')
    
    report_lines = [
        "# Scalability Analysis: 2x2 vs 4x4 Traffic Network",
        "",
        "This report analyzes how each algorithm scales from a 4-intersection grid (2x2) to a 16-intersection grid (4x4) while maintaining lane capacity at 2 lanes per direction.",
        "",
        "## Percentage Changes (4x4 relative to 2x2)",
        ""
    ]
    
    for _, row in merged.iterrows():
        report_lines.append(f"### {row['Algorithm']}")
        report_lines.append(f"- **Waiting Time**: {row['Wait_Change_%']:.2f}%")
        report_lines.append(f"- **Queue Length**: {row['Queue_Change_%']:.2f}%")
        report_lines.append(f"- **Throughput**: {row['Throughput_Change_%']:.2f}%")
        report_lines.append(f"- **Travel Time**: {row['Travel_Time_Change_%']:.2f}%")
        report_lines.append(f"- **Fuel Consumption**: {row['Fuel_Change_%']:.2f}%")
        report_lines.append(f"- **Average Speed**: {row['Speed_Change_%']:.2f}%")
        report_lines.append(f"- **Total Reward**: {row['Reward_Change_%']:.2f}%")
        report_lines.append("")
        
    # Analyze best scaling
    best_scale_wait = merged.loc[merged['Wait_Change_%'].idxmin()]
    best_throughput = merged.loc[merged['Throughput_Change_%'].idxmax()]
    
    report_lines.extend([
        "## Scalability Insights",
        "",
        f"**1. Which algorithm scaled best?**",
        f"{best_throughput['Algorithm']} showed the best improvement in throughput.",
        "",
        f"**2. Which algorithm degraded least?**",
        f"{best_scale_wait['Algorithm']} showed the minimal increase in waiting times when scaling up.",
        "",
        "**3. Did CoLight improve on larger networks?**",
        "*(Review the CoLight metrics above to answer)*",
        "",
        "**4. Did PressLight remain competitive?**",
        "*(Review PressLight metrics above)*",
        "",
        "**5. Did PPO remain dominant?**",
        "*(Review PPO metrics above)*",
        ""
    ])
    
    report_path = os.path.join(base_dir, 'results', '4x4', '2x2_vs_4x4_comparison_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Scalability report generated: {report_path}")

if __name__ == "__main__":
    main()
