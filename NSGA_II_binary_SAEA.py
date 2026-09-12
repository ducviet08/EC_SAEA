import sys

sys.path.append(r"D:\MOEA\Data\Knapsack_0-1.2")
from mokp_data import load_mokp_csv

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.core.repair import Repair
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import time

start = time.perf_counter()
profits, weights, capacities, seed = load_mokp_csv(r"D:\MOEA\Data\Knapsack_0-1.2\mokp_2obj_250items.csv")
n_obj, n_items = profits.shape
import random
seed = random.randint(1,10**8)


#Real problem

class KnapsackProblem(ElementwiseProblem):
    def __init__(self, profits, weights, capacities):
        self.weights = np.array(weights)
        self.profits = np.array(profits)
        self.capacities = capacities

        super().__init__(
            n_var=n_items,
            n_obj=n_obj,
            n_constr=0,
            xl=0, xu=1,
            vtype=bool
        )

    def _evaluate(self, x, out, *args, **kwargs):
        obj_values = (self.profits * x).sum(axis=1)
        out["F"] = -obj_values


class KnapsackRepair(Repair):
    def _do(self, problem, X, **kwargs):
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

#surrogate problem
class SurrogateKnapsackProblem(Problem):
    def __init__(self, n_items, n_obj, surrogate_models, weights, capacities, profits):
        super().__init__(
            n_var=n_items,
            n_obj=n_obj,
            n_constr=0,
            xl=0, xu=1,
            vtype=bool
        )
        self.surrogate_models = surrogate_models
        # cần để KnapsackRepair tái sử dụng được trên problem ảo
        self.weights = weights
        self.capacities = capacities
        self.profits = profits

    def _evaluate(self, X, out, *args, **kwargs):
        X_bin = X.astype(int)
        preds = [model.predict(X_bin) for model in self.surrogate_models]
        out["F"] = np.column_stack(preds)


def real_evaluate(X, profits):
    X = np.asarray(X, dtype=int)
    obj_values = X @ profits.T  # (N, n_obj)
    return -obj_values


def make_operators(repair_obj):
    return dict(
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=0.9),
        mutation=BitflipMutation(prob=0.01),
        repair=repair_obj,
        eliminate_duplicates=True
    )

from pymoo.util.ref_dirs import get_reference_directions
ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_partitions=149)

#Vòng lặp chính
def run_saea_knapsack(
    profits, weights, capacities, n_items, n_obj, seed,
    n_init=100,          # kích thước archive khởi tạo 
    n_saea_gens=50,      # số vòng lặp SAEA (fit -> virtual search -> infill)
    n_infill=30,         # số cá thể đưa đi đánh giá thật mỗi vòng
    virtual_pop=150,    # số cá thể trong mỗi quần thể chạy ảo
    virtual_gens=50,    # số thế hệ NSGA-II chạy trên surrogate mỗi vòng
):
    repair = KnapsackRepair()

    #  Khởi tạo archive (DoE) bằng chính NSGA-II thật, chạy vài thế hệ
    # để có các cá thể khả thi/chất lượng ban đầu, không phải random thuần.
    real_problem = KnapsackProblem(profits, weights, capacities)
    init_algorithm = NSGA2(pop_size=n_init, **make_operators(repair))
    #init_algorithm = MOEAD(ref_dirs,n_neighbors=20,prob_neighbor_mating=0.9,**make_operators(repair))
    init_res = minimize(
        real_problem, init_algorithm, ("n_gen", 1),
        seed=seed, verbose=False, save_history=False
    )
    X_archive = np.asarray(init_res.pop.get("X"), dtype=int)
    F_archive = real_evaluate(X_archive, profits)
    n_real_evals = X_archive.shape[0]

    print(f"--- Khởi tạo archive: {n_real_evals} cá thể đánh giá thật ---")

    for gen in range(n_saea_gens):
        #  Fit surrogate cho từng objective trên toàn bộ archive
        surrogate_models = []
        for i in range(n_obj):
            #model = RandomForestRegressor(n_estimators=100, random_state=seed)
            model = Ridge(alpha=1.0)
            model.fit(X_archive, F_archive[:, i])
            surrogate_models.append(model)

        #  Tìm kiếm ảo bằng NSGA-II trên surrogate (rẻ, không tốn eval thật)
        virtual_problem = SurrogateKnapsackProblem(
            n_items, n_obj, surrogate_models, weights, capacities, profits
        )
        virtual_algorithm = NSGA2(pop_size=virtual_pop, **make_operators(repair))
        #virtual_algorithm = MOEAD(ref_dirs,n_neighbors=20,prob_neighbor_mating=0.9,**make_operators(repair))
        virtual_res = minimize(
            virtual_problem, virtual_algorithm, ("n_gen", virtual_gens),
            seed=seed, verbose=False
        )

        X_candidates = np.asarray(virtual_res.pop.get("X"), dtype=int)

        # Infill criterion: loại trùng với archive, chọn n_infill cá thể

        archive_set = set(map(tuple, X_archive))
        is_new_mask = [tuple(row) not in archive_set for row in X_candidates]
        X_new = X_candidates[is_new_mask]

        if len(X_new) == 0:
            print(f"  [Gen {gen+1}] Không có cá thể ảo mới, dừng sớm.")
            break

        X_infill = X_new[:min(n_infill, len(X_new))]

        # Đánh giá thật cho các cá thể infill, cập nhật archive
        F_infill = real_evaluate(X_infill, profits)
        X_archive = np.vstack([X_archive, X_infill])
        F_archive = np.vstack([F_archive, F_infill])
        n_real_evals += len(X_infill)

        print(f"  [Gen {gen+1}] +{len(X_infill)} eval thật "
              f"(tổng: {n_real_evals}) | archive size: {X_archive.shape[0]}")

    # Pareto front cuối cùng từ archive thật
    nds = NonDominatedSorting()
    fronts = nds.do(F_archive, only_non_dominated_front=True)
    pareto_F = -F_archive[fronts]
    pareto_X = X_archive[fronts]

    return pareto_X, pareto_F, n_real_evals


X_pareto, F_pareto, total_real_evals = run_saea_knapsack(
    profits, weights, capacities, n_items, n_obj, seed
)

end = time.perf_counter()

print("\n==================================================")
print(f"Tổng số lần đánh giá THẬT đã dùng: {total_real_evals}")
print(f"(baseline NSGA-II thuần trong bản gốc: {150 * 200} evals cho pop=150, 200 gens)")
print(f"Số nghiệm Pareto tìm được: {len(F_pareto)}")
print(F_pareto)
print(f"Thời gian chạy: {end - start:.2f}s")

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8, 6))
# plt.scatter(F_pareto[:, 0], F_pareto[:, 1],
#             c="red", s=30, edgecolors="black", label="SAEA Pareto front")
# plt.xlabel("Objective 1 (Profit)")
# plt.ylabel("Objective 2 (Profit)")
# plt.title(f"Knapsack_{n_obj}_{n_items} (SAEA - pymoo) : {end - start:.2f}s "
#           f"| real evals: {total_real_evals}")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
#
from pymoo.indicators.hv import HV
ref_point = ([0.0, 0.0])
hv = HV(ref_point=ref_point)(-F_pareto)
print(hv / 10e5)