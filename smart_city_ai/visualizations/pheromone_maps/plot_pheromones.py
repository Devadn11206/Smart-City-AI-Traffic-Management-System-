import csv
import os
import matplotlib.pyplot as plt

def plot_pheromones(csv_path, out_dir='visualizations/pheromone_maps'):
    os.makedirs(out_dir, exist_ok=True)
    edges = []
    vals = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(f"{row['from']}->{row['to']}")
            vals.append(float(row['pheromone']))

    if not vals:
        print('No pheromone data found')
        return

    # simple bar chart for pheromone distribution
    plt.figure(figsize=(12, 6))
    idx = range(len(vals))
    plt.bar(idx, vals, color='C1')
    plt.xlabel('Edge (from->to)')
    plt.ylabel('Pheromone strength')
    plt.title('Pheromone distribution (final)')
    plt.xticks(idx, edges, rotation=90, fontsize=6)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'pheromones_final.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved pheromone plot to', out_path)

if __name__ == '__main__':
    import sys
    csvp = sys.argv[1] if len(sys.argv) > 1 else 'results/aco_results/pheromones_final.csv'
    plot_pheromones(csvp)
