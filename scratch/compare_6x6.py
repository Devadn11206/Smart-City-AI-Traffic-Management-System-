import os
import filecmp
import pandas as pd
import json

dir1 = r"c:\Users\ddnan\intenrship_IIIT\benchmark_research\6x6_benchmark"
dir2 = r"c:\Users\ddnan\intenrship_IIIT\benchmark_research\6x6 bench mark"

if not os.path.exists(dir1) or not os.path.exists(dir2):
    print("One or both directories do not exist.")
    exit(1)

files1 = set([os.path.relpath(os.path.join(dp, f), dir1) for dp, dn, fn in os.walk(dir1) for f in fn])
files2 = set([os.path.relpath(os.path.join(dp, f), dir2) for dp, dn, fn in os.walk(dir2) for f in fn])

print("--- FILE PRESENCE COMPARISON ---")
only_in_1 = files1 - files2
only_in_2 = files2 - files1
print(f"Files only in '{dir1}': {len(only_in_1)}")
for f in only_in_1: print("  ", f)
print(f"Files only in '{dir2}': {len(only_in_2)}")
for f in only_in_2: print("  ", f)

common_files = files1.intersection(files2)
print(f"\nCommon files: {len(common_files)}")

identical_files = []
different_files = []

for f in common_files:
    path1 = os.path.join(dir1, f)
    path2 = os.path.join(dir2, f)
    if filecmp.cmp(path1, path2, shallow=False):
        identical_files.append(f)
    else:
        different_files.append(f)

print(f"\nIdentical files: {len(identical_files)}")
print(f"Different files: {len(different_files)}")

for f in different_files:
    print(f"\nDifferences in {f}:")
    path1 = os.path.join(dir1, f)
    path2 = os.path.join(dir2, f)
    
    if f.endswith('.csv'):
        try:
            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)
            if not df1.equals(df2):
                print(f"  Shape 1: {df1.shape}, Shape 2: {df2.shape}")
                print(f"  Columns 1: {list(df1.columns)}")
                print(f"  Columns 2: {list(df2.columns)}")
                print(f"  CSV content differs.")
        except Exception as e:
            print(f"  Error reading CSV: {e}")
    elif f.endswith('.json'):
        try:
            with open(path1, 'r') as file1, open(path2, 'r') as file2:
                j1 = json.load(file1)
                j2 = json.load(file2)
                if j1 != j2:
                    print("  JSON content differs.")
        except Exception as e:
            print(f"  Error reading JSON: {e}")
    else:
        print("  Size 1:", os.path.getsize(path1))
        print("  Size 2:", os.path.getsize(path2))

