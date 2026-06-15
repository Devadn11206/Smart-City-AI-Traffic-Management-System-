import os
import ast
import json
from collections import defaultdict

base_dir = r'c:\Users\ddnan\intenrship_IIIT\smart_city_ai'

# Load the previously generated imports map
with open(os.path.join(base_dir, 'import_analysis.json'), 'r') as f:
    imports_map = json.load(f)

# Build a mapping from module name to file path
module_to_file = {}
for py_file in imports_map.keys():
    # e.g., aco_routing/aco_router.py -> aco_routing.aco_router
    module_name = py_file.replace('/', '.').replace('\\', '.')[:-3]
    if module_name.endswith('.__init__'):
        module_name = module_name[:-9]
    module_to_file[module_name] = py_file

# Resolve imports to file paths
file_deps = defaultdict(list)
for py_file, imports in imports_map.items():
    for imp in imports:
        # Check if imp matches any module exactly
        if imp in module_to_file:
            file_deps[py_file].append(module_to_file[imp])
        else:
            # Check for submodules
            matched = False
            for mod in module_to_file.keys():
                if mod.startswith(imp + '.') or imp.startswith(mod + '.'):
                    file_deps[py_file].append(module_to_file[mod])
                    matched = True
            if not matched:
                # Try relative imports or just fuzzy match
                for mod in module_to_file.keys():
                    if imp.split('.')[-1] == mod.split('.')[-1]:
                        file_deps[py_file].append(module_to_file[mod])

# Deduplicate
for k in file_deps:
    file_deps[k] = list(set(file_deps[k]))

# Known root files that are entry points
roots = [
    'training/train_dqn.py',
    'training/train_ddqn.py',
    'training/train_ppo.py',
    'pinn_models/train_pinn.py',
    'integration/full_hybrid_system.py',
    'integration/run_emergency_system.py',
    'integration/train_hybrid.py',
    'scripts/demo_predictive_corridor.py',
    'scripts/evaluate_milestones.py',
    'scripts/run_simulation.py',
    'dashboards/app.py'
]

# Add roots that might exist based on user prompt
prompt_roots = [
    'main.py', 'evaluate_milestones.py', 'run_demo.py'
]
for p in prompt_roots:
    for py_file in imports_map.keys():
        if py_file.endswith(p) and py_file not in roots:
            roots.append(py_file)

# Find all reachable files from roots
reachable = set()
queue = list(roots)
while queue:
    curr = queue.pop(0)
    if curr not in reachable and curr in imports_map:
        reachable.add(curr)
        for dep in file_deps.get(curr, []):
            if dep not in reachable:
                queue.append(dep)

# Calculate in-degrees for ALL files (who is imported by whom)
in_degree = defaultdict(list)
for f, deps in file_deps.items():
    for d in deps:
        if d != f:
            in_degree[d].append(f)

# Find orphaned/unused files
results = {
    'reachable_from_core': list(reachable),
    'unreachable': [],
    'in_degree': in_degree
}

for py_file in imports_map.keys():
    if py_file not in reachable:
        results['unreachable'].append({
            'file': py_file,
            'imported_by': in_degree.get(py_file, [])
        })

with open(os.path.join(base_dir, 'reachability_analysis.json'), 'w') as f:
    json.dump(results, f, indent=2)

