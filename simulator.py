import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize


# ==================================================
# Topology
# ==================================================
class Topology:
    def __init__(self, kind, size, link_type, alpha):
        self.kind = kind
        self.size = size
        self.link_type = link_type
        self.alpha = alpha  # [s]

    def get_bw(self):
        bw = {
            "infiniband": 0.8,
            "PCIe": 0.256,
            "nvlink": 1.8,
            "l2": 6,
            "HBM": 8,
            "HBI": 10,
            "l1": 20
        }  # [TB/s]
        return bw[self.link_type] * (2 ** 30)  # [KB/s]

    # returns the time it takes to propagate inside the given topology
    def get_scaling_factor(self):
        if self.kind == "tree":
            return np.ceil(np.log2(self.size))
        elif self.kind == "linear":
            return self.size - 1
        else:
            return 1

    def get_t_comm(self, v, c):
        scaling_factor = self.get_scaling_factor()
        t1 = self.alpha * scaling_factor
        t2 = (v * 2 * (self.size - 1)) / (c * self.get_bw())
        return t1 + t2


# ==================================================
# GPU
# ==================================================
class GPU:
    def __init__(self, efficiency_factor, t_launch):
        self.efficiency_factor = efficiency_factor
        self.t_launch = t_launch  # [s]
        self.tensor_memory = 40960  # [KB]

    def get_achieved_tflops(self, element_type):
        peak = {
            "fp32": 495,
            "fp16": 990,
            "bf16": 990,
            "fp8": 1980,
            "nvfp4": 15000
        }  # [TFLOPS]
        return peak[element_type] * self.efficiency_factor


# ==================================================
# Simulator
# ==================================================
class Simulator:
    def __init__(self, inner_topology, outer_topology, gpu, dim, r, element_type):
        self.inner_topology = inner_topology
        self.outer_topology = outer_topology
        self.gpu = gpu
        self.dim = dim
        self.r = r
        self.element_type = element_type

    def get_elem_size(self):
        return {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "nvfp4": 0.5
        }[self.element_type] / (2 ** 10)  # [KB]

    def calculate_t_comp(self, b, c):
        total_gpus = self.inner_topology.size * self.outer_topology.size
        flops = 2 * (b / c) * self.dim * (self.r * self.dim / total_gpus)  # [FLOPs]
        achieved = self.gpu.get_achieved_tflops(self.element_type) * 1e12  # [FLOP / sec]
        return self.gpu.t_launch + (flops / achieved)

    def calculate_t_comm(self, b, c):
        elem_size = self.get_elem_size()
        num_gpus = self.inner_topology.size * self.outer_topology.size

        v_gpu = (1 / num_gpus) * b * self.r * self.dim * elem_size
        return (
                self.inner_topology.get_t_comm(v_gpu, c)
                + self.outer_topology.get_t_comm(v_gpu, c)
        )

    # Objective 1: total time (minimize)
    def opt_func_1(self, p):
        b, c = p
        t_comp = self.calculate_t_comp(b, c)
        t_comm = self.calculate_t_comm(b, c)
        return t_comp + c * max(t_comp, t_comm)

    # Objective 2: throughput (negated so we minimize)
    def opt_func_2(self, p):
        b, c = p
        t_total = self.opt_func_1([b, c])
        b_bytes = b * self.dim * self.r * self.get_elem_size()
        return -(b_bytes / t_total)

    # ------------------------------------------------------------------
    # Method 1: Brute Force Grid Search
    # ------------------------------------------------------------------
    def pareto_front(self, points):
        is_pareto = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            if not is_pareto[i]:
                continue
            dominates = (
                    (points[:, 0] >= p[0]) &
                    (points[:, 1] <= p[1]) &
                    ((points[:, 0] > p[0]) | (points[:, 1] < p[1]))
            )
            is_pareto[dominates] = False
        return is_pareto

    def solve_pareto_method_1(self):
        max_mem = self.gpu.tensor_memory
        combinations = []
        for i in range(20):
            b = 2 ** i
            for c in range(1, b + 1):
                if (b / c) * self.dim * self.get_elem_size() <= max_mem:
                    combinations.append((b, c))

        print(f"[Method 1] Evaluating {len(combinations)} (B, C) pairs...")

        def evaluate(pair):
            b, c = pair
            f1 = self.opt_func_1([b, c])
            f2 = self.opt_func_2([b, c])
            return b, c, f1, -f2

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(evaluate, combinations))

        results = np.array(results, dtype=float)
        f_vals = results[:, 2:4]
        pareto_mask = self.pareto_front(f_vals)
        pareto_points = results[pareto_mask]

        return results, pareto_points

    # ------------------------------------------------------------------
    # Method 2: Pymoo Genetic Algorithm (NSGA-II)
    # ------------------------------------------------------------------
    def solve_pareto_pymoo(self):
        print("[Method 2] Running NSGA-II optimization...")

        # We need a reference to 'self' inside the Problem class
        simulator_instance = self

        class GPUOptimizationProblem(ElementwiseProblem):
            def __init__(self):
                # Decision Variables:
                # x0: exponent for Batch Size B (where B = 2^x0). Range [0, 20]
                # x1: Chunk Size C. Range [1, 2048]
                super().__init__(n_var=2,
                                 n_obj=2,
                                 n_ieq_constr=2,  # 2 constraints
                                 xl=np.array([0, 1]),
                                 xu=np.array([20, 2048]))

            def _evaluate(self, x, out, *args, **kwargs):
                # 1. Decode Variables
                b_exp = int(x[0])
                c = int(x[1])
                b = 2 ** b_exp

                # 2. Objectives
                f1 = simulator_instance.opt_func_1([b, c])  # Total Time
                f2 = simulator_instance.opt_func_2([b, c])  # Negative Throughput (minimized)

                # 3. Constraints (must be <= 0 to be satisfied)

                # Constraint A: Memory Usage <= Max Memory
                # Usage - Max <= 0
                mem_usage = (b / c) * simulator_instance.dim * simulator_instance.get_elem_size()
                g1 = mem_usage - simulator_instance.gpu.tensor_memory

                # Constraint B: Chunk size C cannot be larger than Batch size B
                # C - B <= 0
                g2 = c - b

                out["F"] = [f1, f2]
                out["G"] = [g1, g2]

        # Initialize Problem
        problem = GPUOptimizationProblem()

        # Configure Algorithm (NSGA-II)
        # Using specific integer operators (crossover/mutation)
        algorithm = NSGA2(
            pop_size=50,
            n_offsprings=20,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        # Run Optimization
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 200),  # Number of generations
                       seed=42,
                       verbose=False)

        # Process Results
        # Pymoo returns 'res.F' (objectives) and 'res.X' (variables)
        # We need to format them back into [b, c, f1, throughput] for plotting

        pareto_results = []
        if res.X is not None:
            for i in range(len(res.X)):
                b_exp = int(res.X[i, 0])
                c = int(res.X[i, 1])
                b = 2 ** b_exp

                f1_val = res.F[i, 0]
                f2_val_neg = res.F[i, 1]

                # Store: [b, c, total_time, throughput (positive)]
                pareto_results.append([b, c, f1_val, -f2_val_neg])

        return np.array(pareto_results)

    # ==================================================
    # Plot
    # ==================================================
    def plot_pareto(self, pareto_pts, title_suffix=""):
        plt.figure(figsize=(9, 6))

        plt.plot(
            pareto_pts[:, 2], pareto_pts[:, 3],
            "ro", label="Pareto front"
        )
        # Sort for line plotting
        sorted_indices = np.argsort(pareto_pts[:, 2])
        plt.plot(pareto_pts[sorted_indices, 2], pareto_pts[sorted_indices, 3], "r-", alpha=0.5)

        for point in pareto_pts:
            b_val = int(point[0])
            c_val = int(point[1])
            x_coord = point[2]
            y_coord = point[3]

            label = f"({b_val}, {c_val})"
            plt.annotate(
                label,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(5, 5),
                ha='left',
                fontsize=8
            )

        plt.xlabel("Objective 1: Total Time [sec] (Minimize)")
        plt.ylabel("Objective 2: Throughput [KB / sec] (Maximize)")
        plt.title(f"Pareto Front {title_suffix}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ==================================================
# Run
# ==================================================
if __name__ == "__main__":
    nvlink = Topology("tree", 72, "nvlink", 0.003)
    infiniband = Topology("linear", 2, "infiniband", 0.01)
    gpu = GPU(0.4, 0.01)

    sim = Simulator(
        inner_topology=nvlink,
        outer_topology=infiniband,
        gpu=gpu,
        dim=4096,
        r=4,
        element_type="fp32"
    )

    # --- Run Method 1 (Brute Force) ---
    all_points_1, pareto_points_1 = sim.solve_pareto_method_1()
    sim.plot_pareto(pareto_points_1, title_suffix="(Method 1: Brute Force)")

    # --- Run Method 2 (Pymoo NSGA-II) ---
    # Note: Genetic algorithms don't typically return "all feasible points", only the optimal set
    pareto_points_2 = sim.solve_pareto_pymoo()

    # Check if we found solutions
    if len(pareto_points_2) > 0:
        sim.plot_pareto(pareto_points_2, title_suffix="(Method 2: Pymoo / NSGA-II)")
    else:
        print("Pymoo solver found no feasible solutions.")