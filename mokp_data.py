"""
mokp_data.py
Module sinh, lưu, đọc instance MOKP dưới dạng CSV.
"""
import numpy as np
import csv
import os


def generate_mokp_instance(n_items: int, n_objectives: int,
                             seed: int = None, low: int = 1, high: int = 1000):
    rng = np.random.default_rng(seed)
    profits = rng.integers(low, high + 1, size=(n_objectives, n_items))
    weights = rng.integers(low, high + 1, size=(n_objectives, n_items))
    capacities = 0.5 * weights.sum(axis=1)
    return profits, weights, capacities


def save_mokp_csv(path: str, profits, weights, capacities, seed=None):
    n_obj, n_items = profits.shape

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"# seed={seed if seed is not None else -1}"])
        writer.writerow([f"# capacities=" + ";".join(str(c) for c in capacities)])

        header = ["item_id"] + [f"profit_obj{j+1}" for j in range(n_obj)] \
                             + [f"weight_obj{j+1}" for j in range(n_obj)]
        writer.writerow(header)

        for i in range(n_items):
            row = [i + 1]
            row += [int(profits[j, i]) for j in range(n_obj)]
            row += [int(weights[j, i]) for j in range(n_obj)]
            writer.writerow(row)

    print(f"Đã lưu instance dạng CSV vào: {path}")


def load_mokp_csv(path: str):
    with open(path, "r", newline="") as f:
        reader = list(csv.reader(f))

    seed = int(reader[0][0].split("=")[1])
    capacities = np.array([float(x) for x in reader[1][0].split("=")[1].split(";")])
    n_obj = capacities.shape[0]

    data_rows = reader[3:]
    n_items = len(data_rows)

    profits = np.zeros((n_obj, n_items), dtype=int)
    weights = np.zeros((n_obj, n_items), dtype=int)

    for i, row in enumerate(data_rows):
        values = [int(x) for x in row[1:]]
        profits[:, i] = values[:n_obj]
        weights[:, i] = values[n_obj:]

    return profits, weights, capacities, seed