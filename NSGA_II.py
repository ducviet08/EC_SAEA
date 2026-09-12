import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.append(PARENT_DIR)
from mokp_data import load_mokp_csv

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.core.repair import Repair
import time
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions


start = time.perf_counter()
csv_path = os.path.join(PARENT_DIR,"mokp_2obj_250items.csv")
profits, weights, capacities, seed, = load_mokp_csv(csv_path)
import random
seed = random.randint(0,10**8)
n_obj , n_items = profits.shape

class KnapsackProblem(ElementwiseProblem):
    def __init__(self,profits,weights,capacities):
        self.weights = np.array(weights)
        self.profits = np.array(profits)
        self.capacities = capacities

        super().__init__(
            n_var=n_items,
            n_obj=n_obj,
            n_constr = 0,
            xl=0,xu=1,
            vtype=bool
        )

    def _evaluate(self, x, out, *args, **kwargs):
        obj_values = (self.profits*x).sum(axis=1)
        out["F"]= -obj_values

class KnapsackRepair(Repair):
    def _do(self,problem,X,**kwargs):
        weights = problem.weights
        capacities = problem.capacities
        profits = problem.profits

        ratios = np.max(profits / np.maximum(weights, 1e-9), axis=0)  # (n_items,)

        for k in range(X.shape[0]):
            x = X[k].astype(int)

            def total_weight(x):
                return (weights * x).sum(axis=1)

            def is_feasible(x):
                return np.all(total_weight(x) <= capacities)

            selected = np.where(x == 1)[0]
            while not is_feasible(x) and len(selected) > 0:
                ratios_selected = ratios[selected]
                worst = selected[np.argmin(ratios_selected)]
                x[worst] = 0
                selected = np.where(x == 1)[0]

            X[k] = x

        return X


problem = KnapsackProblem(profits,weights,capacities)

algorithm = NSGA2(
    pop_size=150,
    sampling=BinaryRandomSampling(),
    crossover=UniformCrossover(prob=0.9),
    mutation=BitflipMutation(prob=0.01),
    repair=KnapsackRepair(),
    eliminate_duplicates=True
)
# algorithm = MOEAD(
#     ref_dirs=get_reference_directions("das-dennis",n_dim=2,n_partitions=149),
#     sampling=BinaryRandomSampling(),
#     crossover=UniformCrossover(prob = 0.9),
#     mutation=BitflipMutation(prob = 0.01),
#     repair = KnapsackRepair()
# )

res = minimize(
    problem,
    algorithm,
    ("n_gen",200),
    seed= seed,
    verbose = True
)
end = time.perf_counter()
print(-res.F)

print(end-start)
# import matplotlib.pyplot as plt

# profit_values = -res.F

# plt.figure(figsize=(8, 6))
# plt.scatter(profit_values[:, 0], profit_values[:, 1],
#             c="red", s=30, edgecolors="black", label="NSGA-II Pareto front")

# plt.xlabel("Objective 1 (Profit)")
# plt.ylabel("Objective 2 (Profit)")
# plt.title(f"Knapsack_{n_obj}_{n_items} (NSGA-II - pymoo) : {end - start:.2f}s")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

from pymoo.indicators.hv import HV
ref_point = ([0.0,0.0])
obj_min = res.F
hv = HV(ref_point=ref_point)(obj_min)
print(hv/10e5)