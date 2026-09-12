import os
from mokp_data import generate_mokp_instance, save_mokp_csv

DATA_DIR = "data/Knapsack 0_1.2"
os.makedirs(DATA_DIR, exist_ok=True)

configs = [(2,250),(2,500),(2,750),(3,250),(3,500),(3,750),(4,250),(4,500),(4,750)]

for n_obj, n_items in configs:
    seed = 1000 * n_obj + n_items
    profits, weights, capacities = generate_mokp_instance(n_items, n_obj, seed=seed)
    path = f"{DATA_DIR}/mokp_{n_obj}obj_{n_items}items.csv"
    save_mokp_csv(path, profits, weights, capacities, seed=seed)