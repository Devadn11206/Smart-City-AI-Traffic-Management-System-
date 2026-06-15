#!/usr/bin/env python3
"""
Recompute PINN ablation study comparison with extended metrics.
Uses already-generated experiment data to create comprehensive comparison table.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def load_experiment_data(exp_name: str) -> Dict[str, Any]:
    """Load the latest summary from an experiment folder."""
    results_dir = Path("results") / f"ablation_experiment_{exp_name}"
    if not results_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {results_dir}")
    
    # Look for _summary.json file
    summary_files = list(results_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {results_dir}")
    
    with open(summary_files[0], 'r') as f:
        return json.load(f)

def compute_metrics(exp_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from experiment data."""
    return {
        "avg_speed": exp_data.get("avg_speed", 0.0),
        "emergency_eta": exp_data.get("eta_seconds", 0.0),
        "waiting_time": exp_data.get("wait", 0.0),
        "queue_length": exp_data.get("queue", 0.0),
        "throughput": exp_data.get("throughput", 0.0),
        "corridor_efficiency": exp_data.get("corridor_efficiency", 0.0),
    }

def compute_percentage_change(baseline: float, with_pinn: float, direction: str = "higher_is_better") -> tuple:
    """Compute percentage change and determine if improved."""
    if baseline == 0:
        pct_change = 0.0
    else:
        pct_change = ((with_pinn - baseline) / baseline) * 100
    
    if direction == "higher_is_better":
        improved = with_pinn > baseline
    else:  # lower_is_better
        improved = with_pinn < baseline
    
    return pct_change, improved

def generate_comparison_report(baseline_data: Dict, pinn_data: Dict):
    """Generate comprehensive comparison report."""
    
    baseline_metrics = compute_metrics(baseline_data)
    pinn_metrics = compute_metrics(pinn_data)
    
    # Define metrics and their optimality directions
    metrics_config = {
        "avg_speed": ("higher_is_better", "Average Speed (m/s)"),
        "emergency_eta": ("lower_is_better", "Emergency ETA (seconds)"),
        "waiting_time": ("lower_is_better", "Waiting Time (seconds)"),
        "queue_length": ("lower_is_better", "Queue Length (vehicles)"),
        "throughput": ("higher_is_better", "Throughput (vehicles/sec)"),
        "corridor_efficiency": ("higher_is_better", "Corridor Efficiency"),
    }
    
    comparison = {}
    metrics_improved = 0
    total_metrics = len(metrics_config)
    
    for metric_key, (direction, metric_name) in metrics_config.items():
        baseline_val = baseline_metrics[metric_key]
        pinn_val = pinn_metrics[metric_key]
        pct_change, improved = compute_percentage_change(baseline_val, pinn_val, direction)
        
        comparison[metric_key] = {
            "metric_name": metric_name,
            "baseline": baseline_val,
            "with_pinn": pinn_val,
            "absolute_difference": pinn_val - baseline_val,
            "percentage_change": pct_change,
            "improved": improved,
            "direction": direction
        }
        
        if improved:
            metrics_improved += 1
    
    return comparison, metrics_improved, total_metrics

def save_json_report(comparison: Dict, metrics_improved: int, total_metrics: int, output_path: str):
    """Save comprehensive JSON report."""
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "comparison_type": "PINN Ablation Study (Extended Metrics)",
            "metrics_improved": metrics_improved,
            "total_metrics": total_metrics,
            "improvement_rate": (metrics_improved / total_metrics * 100) if total_metrics > 0 else 0
        },
        "detailed_metrics": comparison,
        "summary": {
            "baseline_superior": sum(1 for m in comparison.values() if not m["improved"]),
            "pinn_superior": sum(1 for m in comparison.values() if m["improved"]),
            "overall_assessment": "PINN provides benefits" if metrics_improved > (total_metrics // 2) else "PINN provides limited benefits"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ JSON report saved: {output_path}")

def save_csv_table(comparison: Dict, metrics_improved: int, total_metrics: int, output_path: str):
    """Save comprehensive CSV comparison table."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Metric",
            "Baseline (No PINN)",
            "With PINN",
            "Absolute Difference",
            "Percentage Change (%)",
            "Improved?",
            "Direction"
        ])
        
        # Data rows
        for metric_key, data in comparison.items():
            writer.writerow([
                data["metric_name"],
                f"{data['baseline']:.4f}",
                f"{data['with_pinn']:.4f}",
                f"{data['absolute_difference']:.4f}",
                f"{data['percentage_change']:.2f}%",
                "YES" if data["improved"] else "NO",
                data["direction"]
            ])
        
        # Summary row
        writer.writerow([])
        writer.writerow([
            "SUMMARY",
            f"",
            f"",
            f"",
            f"Improvement Rate: {(metrics_improved / total_metrics * 100):.1f}%",
            f"{metrics_improved}/{total_metrics} improved",
            ""
        ])
    
    print(f"✓ CSV table saved: {output_path}")

def main():
    """Main execution."""
    print("=" * 80)
    print("PINN ABLATION STUDY - EXTENDED METRICS COMPARISON")
    print("=" * 80)
    print()
    
    # Load experiment data
    print("Loading experiment data...")
    try:
        baseline_data = load_experiment_data("Baseline (No PINN)")
        pinn_data = load_experiment_data("With PINN")
        print("✓ Baseline data loaded")
        print("✓ PINN data loaded")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    print()
    
    # Generate comparison
    print("Computing metrics...")
    comparison, metrics_improved, total_metrics = generate_comparison_report(baseline_data, pinn_data)
    print("✓ Metrics computed")
    print()
    
    # Display results
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    print(f"{'Metric':<30} {'Baseline':<15} {'With PINN':<15} {'Change (%)':<12} {'Improved':<10}")
    print("-" * 82)
    
    for metric_key, data in comparison.items():
        status = "✓ Yes" if data["improved"] else "✗ No"
        print(f"{data['metric_name']:<30} {data['baseline']:<15.4f} {data['with_pinn']:<15.4f} {data['percentage_change']:<12.2f} {status:<10}")
    
    print()
    print(f"Overall: {metrics_improved}/{total_metrics} metrics improved ({(metrics_improved/total_metrics*100):.1f}%)")
    print()
    
    # Save reports
    print("=" * 80)
    print("SAVING REPORTS")
    print("=" * 80)
    print()
    
    output_dir = Path("results/pinn_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "pinn_ablation_extended_report.json"
    csv_path = output_dir / "pinn_comparison_table_extended.csv"
    
    save_json_report(comparison, metrics_improved, total_metrics, str(json_path))
    save_csv_table(comparison, metrics_improved, total_metrics, str(csv_path))
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Metrics evaluated: {total_metrics}")
    print(f"Metrics improved with PINN: {metrics_improved}")
    print(f"Metrics degraded with PINN: {total_metrics - metrics_improved}")
    print(f"Improvement rate: {(metrics_improved / total_metrics * 100):.1f}%")
    print()
    
    # Key findings
    print("KEY FINDINGS:")
    print()
    
    improved_metrics = [data for data in comparison.values() if data["improved"]]
    degraded_metrics = [data for data in comparison.values() if not data["improved"]]
    
    if improved_metrics:
        print("✓ Metrics where PINN performs BETTER:")
        for metric in improved_metrics:
            print(f"  • {metric['metric_name']}: {metric['percentage_change']:+.2f}%")
        print()
    
    if degraded_metrics:
        print("✗ Metrics where PINN performs WORSE:")
        for metric in degraded_metrics:
            print(f"  • {metric['metric_name']}: {metric['percentage_change']:+.2f}%")
        print()
    
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
