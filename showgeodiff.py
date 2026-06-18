import numpy as np

file_orig = 'lc_GRB260207A_cand41148'
file_geo  = 'lc_GRB260207A_cand41148_geo'

try:
    # usecols=0 only loads the first column to save memory/time
    data_orig = np.loadtxt(file_orig, usecols=0)
    data_geo  = np.loadtxt(file_geo, usecols=0)

    # Check if files match in length
    if len(data_orig) != len(data_geo):
        print(f"Warning: File lengths differ! ({len(data_orig)} vs {len(data_geo)})")
        # Compare only up to the shortest file length
        n = min(len(data_orig), len(data_geo))
        data_orig = data_orig[:n]
        data_geo = data_geo[:n]

    # Calculate row-by-row differences
    diffs = data_orig - data_geo

    # Calculate Stats
    max_diff = np.max(diffs)*24*60*60
    min_diff = np.min(diffs)*24*60*60
    avg_diff = np.mean(diffs)*24*60*60

    # Print results (using .10f to avoid 'e' notation and show high precision)
    print(f"Stats for column 1 differences ({file_orig} minus {file_geo}) converted to seconds:")
    print("-" * 60)
    print(f"Maximum Difference: {max_diff}")
    print(f"Minimum Difference: {min_diff}")
    print(f"Average Difference: {avg_diff}")
    print("-" * 60)

except Exception as e:
    print(f"An error occurred: {e}")
