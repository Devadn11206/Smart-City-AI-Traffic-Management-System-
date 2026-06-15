import os
import ast
import json

def get_imports(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except Exception:
            return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

base_dir = r'c:\Users\ddnan\intenrship_IIIT\smart_city_ai'
all_py_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.py'):
            all_py_files.append(os.path.join(root, file))

imports_map = {}
for py_file in all_py_files:
    rel_path = os.path.relpath(py_file, base_dir).replace('\\', '/')
    imports_map[rel_path] = list(get_imports(py_file))

with open(os.path.join(base_dir, 'import_analysis.json'), 'w') as f:
    json.dump(imports_map, f, indent=2)
