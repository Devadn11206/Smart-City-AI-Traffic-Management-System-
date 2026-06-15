import json

# Check integration report
with open('results/aco_integration_probe_report.json') as f:
    data = json.load(f)

print(f'Total entries in report: {len(data)}')

# Get statistics
neighbors_populated = sum(1 for e in data if len(e.get('source_neighbors', [])) > 0)
paths_found = sum(1 for e in data if len(e.get('path', [])) > 0)
successful_aco = sum(1 for e in data if e.get('aco_debug', {}).get('successful_paths', 0) > 0)
dijkstra_fallback = sum(1 for e in data if e.get('dijkstra_path_exists', False) and len(e.get('path', [])) == 0)

print(f'\nStatistics:')
print(f'  Entries with source_neighbors populated: {neighbors_populated}/{len(data)}')
print(f'  Paths found (any method): {paths_found}/{len(data)}')
print(f'  ACO successful paths: {successful_aco}/{len(data)}')
print(f'  Dijkstra fallback used: {dijkstra_fallback}/{len(data)}')

# Show first few entries
print(f'\nFirst 5 entries:')
for i in range(min(5, len(data))):
    e = data[i]
    aco_success = e.get('aco_debug', {}).get('successful_paths', 0)
    print(f'{i}: vehicle={e.get("vehicle_id")}, source={e.get("source_edge")}, neighbors={len(e.get("source_neighbors", []))}, aco_paths={aco_success}, path_len={len(e.get("path", []))}')

# Check for reroutes
print(f'\nReroute entries:')
with open('results/aco_test_reroute.csv') as f:
    lines = f.readlines()
    print(f'  Total reroute lines: {len(lines)}')
    if len(lines) > 1:
        print(f'  Sample reroutes (first 3):')
        for line in lines[1:4]:
            fields = line.strip().split(',')
            if len(fields) > 3:
                print(f'    Vehicle: {fields[0]}, Old route: {fields[1][:30]}..., New route: {fields[2][:30]}...')
