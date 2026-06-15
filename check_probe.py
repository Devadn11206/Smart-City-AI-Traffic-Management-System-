import json

with open('results/aco_integration_probe_report.json') as f:
    data = json.load(f)

print(f'Total entries: {len(data)}')
for i, e in enumerate(data[:5]):
    print(f'Entry {i}: pheromone_count={e.get("pheromone_count")}, adjacency_size={e.get("adjacency_size")}, neighbors={len(e.get("source_neighbors", []))}')

# Check classification
print('\nClassifications:')
for e in data:
    print(f'  {e.get("classification")}')
