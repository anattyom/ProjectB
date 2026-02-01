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
# GPUs
# ==================================================
# ==================================================
# GPUs
# ==================================================
class GPU:
    def __init__(self,
                 efficiency_factor,
                 t_launch,
                 sm_l1_size,
                 number_sm,
                 number_dies,
                 peak_flops_dict,
                 hbi_bw=None,
                 hbi_launch=0.0):
        """
        Base Generic GPU Class.
        """
        self.efficiency_factor = efficiency_factor
        self.t_launch = t_launch  # [s]
        self.sm_l1_size = sm_l1_size  # [KB]
        self.number_sm = number_sm
        self.number_dies = number_dies

        # Auto-calculate total memory
        self.memory = self.sm_l1_size * self.number_sm  # [KB]

        self.peak_flops_dict = peak_flops_dict  # Dictionary of peaks
        self.hbi_bw = hbi_bw  # [KB/s] (Die-to-Die Bandwidth)
        self.hbi_launch = hbi_launch  # [s] (Die-to-Die Latency)

    def get_achieved_tflops(self, element_type):
        if element_type not in self.peak_flops_dict:
            raise ValueError(f"Precision '{element_type}' not supported by this GPU.")

        peak_val = self.peak_flops_dict[element_type]

        if peak_val == "N/A":
            raise ValueError(f"Precision '{element_type}' is N/A for this GPU.")

        return peak_val * self.efficiency_factor

    def get_inner_gpu_overhead(self, v):
        # If single die, no internal communication overhead
        if self.number_dies <= 1 or self.hbi_bw is None:
            return 0

        # Logic: Latency + (Volume / Bandwidth)
        return self.hbi_launch + (v / self.hbi_bw)


class Blackwell(GPU):
    def __init__(self, efficiency_factor, t_launch):
        super().__init__(
            efficiency_factor=efficiency_factor,
            t_launch=t_launch,
            sm_l1_size=256,  # [KB]
            number_sm=160,
            number_dies=2,
            peak_flops_dict={
                "fp32": 495,
                "fp16": 990,
                "bf16": 990,
                "fp8": 1980,
                "nvfp4": 15000
            },
            hbi_bw=10 * (2 ** 30),  # 10 TB/s -> KB/s
            hbi_launch=1e-7  # 100 ns
        )


class RTX3090(GPU):
    def __init__(self, efficiency_factor, t_launch):
        super().__init__(
            efficiency_factor=efficiency_factor,
            t_launch=t_launch,
            sm_l1_size=128,  # [KB]
            number_sm=82,
            number_dies=1,
            peak_flops_dict={
                "fp32": 35.6,
                "fp16": 142,
                "bf16": 71.2,
                "fp8": "N/A",
                "nvfp4": "N/A"
            }
        )


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
        # calculating compute time
        total_gpus = self.inner_topology.size * self.outer_topology.size
        flops = 2 * (b / c) * self.dim * (self.r * self.dim / total_gpus)  # [FLOPs]
        achieved = self.gpu.get_achieved_tflops(self.element_type) * 1e12  # [FLOP / sec]
        # calculating inner chip communication time
        elem_size = self.get_elem_size()
        num_gpus = self.inner_topology.size * self.outer_topology.size
        v_gpu = (1 / num_gpus) * b * self.r * self.dim * elem_size
        inner_gpu_overhead = self.gpu.get_inner_gpu_overhead(v_gpu / self.gpu.number_dies)
        return self.gpu.t_launch + (flops / achieved) + inner_gpu_overhead

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

    def solve_pareto_brute_force(self, fast_mode=False):
        max_mem = self.gpu.memory
        combinations = []
        b_range = [2**i for i in range(20)]
        for b in b_range:
            if fast_mode:
                # SPARSE GRID: Only check powers of 2 up to 32
                c_candidates = [1, 2, 4, 8, 16, 32]
            else:
                # FULL SEARCH: Check every integer (Slow!)
                c_candidates = range(1, b + 1)

            for c in c_candidates:
                if c > b:
                    continue
                elif (b / c) * self.dim * self.get_elem_size() <= max_mem:
                    combinations.append((b, c))
        if not fast_mode:
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
    # Method 2: Pymoo Genetic Algorithm (NSGA-II) - ROBUST VERSION
    # ------------------------------------------------------------------
    def solve_pareto_pymoo(self, fast_mode=False):
        # Configure settings based on mode
        if fast_mode:
            # FAST MODE: Approx. 400 evaluations (20x faster)
            # Good enough for inverse design loops
            pop_size = 20
            n_offsprings = 10
            n_gen = 30
            verbose = False
        else:
            # STANDARD MODE: Approx. 8,000 evaluations
            # High fidelity for final plotting
            pop_size = 100
            n_offsprings = 40
            n_gen = 200
            verbose = False
        if not fast_mode:
            print("\n" + "=" * 40)
            print("[Method 2] Running NSGA-II...")

        simulator_instance = self

        # 1. Define Hard Limits
        max_b_limit = 2 ** 20  # 1,048,576
        max_c_limit = 2048

        class GPUOptimizationProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=2,
                                 n_obj=2,
                                 n_ieq_constr=2,
                                 xl=np.array([1, 1]),
                                 xu=np.array([max_b_limit, max_c_limit]))

            def _evaluate(self, x, out, *args, **kwargs):
                # --- DEFENSIVE CODING START ---
                # Force-Clamp variables to valid range.
                # This prevents the "48 billion" bug if the optimizer drifts.
                b = int(np.clip(x[0], 1, max_b_limit))
                c = int(np.clip(x[1], 1, max_c_limit))
                # --- DEFENSIVE CODING END ---

                # Objectives
                f1 = simulator_instance.opt_func_1([b, c])
                f2 = simulator_instance.opt_func_2([b, c])

                # Constraint A: Memory Check
                mem_usage = (b / c) * simulator_instance.dim * simulator_instance.get_elem_size()
                g1 = mem_usage - simulator_instance.gpu.memory

                # Constraint B: Logical Check (c <= b)
                g2 = c - b

                out["F"] = [f1, f2]
                out["G"] = [g1, g2]

        problem = GPUOptimizationProblem()

        # Configure Algorithm dynamically based on mode
        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 200),
                       seed=42,
                       verbose=verbose)

        # Process Results
        pareto_results = []
        if res.X is not None:
            for i in range(len(res.X)):
                # Apply the same clamping to the results we save
                b = int(np.clip(res.X[i, 0], 1, max_b_limit))
                c = int(np.clip(res.X[i, 1], 1, max_c_limit))

                f1_val = res.F[i, 0]
                f2_val_neg = res.F[i, 1]

                pareto_results.append([b, c, f1_val, -f2_val_neg])

        print(f"[Method 2] Done. Found {len(pareto_results)} solutions.")
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
        plt.ylabel("Objective 2: Throughput [Batch / sec] (Maximize)")
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
    blackwell_gpu = Blackwell(0.5, 0.01)
    rtx3090_gpu = RTX3090(0.5, 0.01)

    sim = Simulator(
        inner_topology=nvlink,
        outer_topology=infiniband,
        gpu=blackwell_gpu,
        dim=4096,
        r=4,
        element_type="fp32"
    )

    sim2 = Simulator(
        inner_topology=nvlink,
        outer_topology=infiniband,
        gpu=rtx3090_gpu,
        dim=4096,
        r=4,
        element_type="fp32"
    )

    # --- Run Method 1 (Brute Force - Blackwell) ---
    all_points_1, pareto_points_1 = sim.solve_pareto_brute_force()
    sim.plot_pareto(pareto_points_1, title_suffix="(Method 1: Brute Force - Blackwell)")

    # --- Run Method 1 (Brute Force - RTX3090) ---
    all_points_r, pareto_points_r = sim2.solve_pareto_brute_force()
    sim.plot_pareto(pareto_points_r, title_suffix="(Method 1: Brute Force - RTX3090)")

    # --- Run Method 2 (Pymoo NSGA-II) ---
    pareto_points_2 = sim.solve_pareto_pymoo()

    # Check if we found solutions
    if len(pareto_points_2) > 0:
        sim.plot_pareto(pareto_points_2, title_suffix="(Method 2: Pymoo / NSGA-II)")
    else:
        print("Pymoo solver found no feasible solutions.")