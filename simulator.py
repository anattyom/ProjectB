# import numpy as np
# import matplotlib.pyplot as plt
# from concurrent.futures import ThreadPoolExecutor
# from scipy.optimize import minimize
#
#
# class Topology:
#     def __init__(self, kind, size, link_type, alpha):
#         self.kind = kind
#         self.size = size
#         self.link_type = link_type
#         self.alpha = alpha #[s]
#
#     def get_bw(self) -> float:
#         bw = {
#             "infiniband" : 0.8,
#             "PCIe" : 0.256,
#             "nvlink" : 1.8,
#             "l2" : 6,
#             "HBM" : 8,
#             "HBI" : 10,
#             "l1" : 20
#         } # [TB/s]
#         return bw[self.link_type] * (2**30) #[KB/s]
#
#     def get_t_comm(self, v, c):
#         log_size = np.ceil(np.log2(self.size))
#         t1 = self.alpha * log_size
#         t2 = (v * log_size * 2) / (c * self.get_bw())
#         return t1 + t2
#
#
# class GPU:
#     def __init__(self, efficiency_factor, t_launch):
#         self.efficiency_factor = efficiency_factor
#         self.t_launch = t_launch # [ms]
#         self.tensor_memory = 192 * (2 ** 20) #[KB]
#
#     def get_achieved_tflops(self, element_type) -> float:
#         peak_tflops = {
#             "fp32" : 495,
#             "fp16" : 990,
#             "bf16" : 990,
#             "fp8" : 1980,
#             "nvfp4" : 15000
#         } # [Tera Floating Point Ops per Second]
#         achieved_tflops = peak_tflops[element_type] * self.efficiency_factor
#         return achieved_tflops
#
#
# class Simulator:
#     def __init__(self, inner_topology, outer_topology, gpu, dim, r, element_type):
#         self.inner_topology = inner_topology
#         self.outer_topology = outer_topology
#         self.gpu = gpu
#         self.dim = dim
#         self.r = r
#         self.element_type = element_type
#
#     def get_elem_size(self) -> float:
#         element_size = {
#             "fp32" : 4,
#             "fp16" : 2,
#             "bf16" : 2,
#             "fp8" : 1,
#             "nvfp4" : 0.5
#         } # Bytes
#         return element_size[self.element_type] / (2 ** 10) #[KB]
#
#     def calculate_t_comp(self, b, c) -> float:
#         total_gpus = self.inner_topology.size * self.outer_topology.size
#         flops = 2 * (b / c) * self.dim * (self.r * self.dim / total_gpus) # make sure this is Tera-FLOPS
#         achieved_flops = self.gpu.get_achieved_tflops(self.element_type)
#         return self.gpu.t_launch + (flops / achieved_flops)
#
#     def calculate_t_comm(self, b, c) -> float:
#         element_size = self.get_elem_size()
#         num_gpus = self.inner_topology.size * self.outer_topology.size
#         v_gpu = (1 / num_gpus) * b * self.r * self.dim * element_size
#         v_outer_top = (1 / self.outer_topology.size) * b * self.r * self.dim * element_size
#         t_inner_top = self.inner_topology.get_t_comm(v_gpu, c)
#         t_outer_top = self.outer_topology.get_t_comm(v_outer_top, c)
#         return t_outer_top + t_inner_top
#
#     def opt_func_1(self, param) -> float:
#         b = param[0]
#         c = param[1]
#         t_comp_a = t_comp_b = self.calculate_t_comp(b, c)
#         t_comm = self.calculate_t_comm(b, c)
#         t_total = t_comp_a + np.maximum(t_comp_b, t_comm)
#         return t_total
#
#     def opt_func_2(self, param) -> float:
#         b = param[0]
#         c = param[1]
#         t_comp_a = t_comp_b = self.calculate_t_comp(b, c)
#         t_comm = self.calculate_t_comm(b, c)
#         t_total = t_comp_a + np.maximum(t_comp_b, t_comm)
#         return - (b / t_total)
#
#     # def optimize(self):
#     #     weights = np.linspace(0, 1, 10)
#     #     x0 = np.array([2, 2])
#     #     b_opt = []
#     #     c_opt = []
#     #     for w in weights:
#     #         objective = lambda p: w * self.opt_func_1(p) + (1 - w) * self.opt_func_2(p)
#     #         bounds = [(1, self.gpu.tensor_memory), (1, 9)]
#     #         res = minimize(objective, x0, bounds=bounds, method="Nelder-Mead")
#     #         if res.success:
#     #             b_opt.append(res.x[0])
#     #             c_opt.append(res.x[1])
#     #         else:
#     #             print("Optimization Failed")
#     #     plt.figure()
#     #     plt.scatter(b_opt, c_opt)
#     #     plt.xlabel('B')
#     #     plt.ylabel('C')
#     #     plt.show()
#
#     def evaluate_point(self, args):
#         """Helper to run in threads. args is a tuple (b, c)."""
#         b, c = args
#         p = np.array([b, c])
#         # Calculate both objectives separately
#         val1 = self.opt_func_1(p)
#         val2 = self.opt_func_2(p)
#         return (b, c, val1, val2)
#
#     def solve_pareto_brute_force(self):
#         # 1. Define the discrete search space
#         max_mem = self.gpu.tensor_memory
#         # Generate powers of 2: [1, 2, 4, 8, ... 128]
#         b_values = [2 ** i for i in range(20) if 2 ** i <= max_mem and 2 ** i > 0]
#
#         combinations = []
#         for b in b_values:
#             # C is between 1 and B (assuming integer steps for chunks)
#             for c in range(1, int(b) + 1):
#                 combinations.append((b, c))
#
#         print(f"Brute forcing {len(combinations)} combinations using threads...")
#
#         # 2. Run evaluations in parallel
#         # Note: Using ThreadPoolExecutor. If your functions are CPU-heavy
#         # and don't release GIL, ProcessPoolExecutor might be faster.
#         raw_results = []
#         with ThreadPoolExecutor() as executor:
#             raw_results = list(executor.map(self.evaluate_point, combinations))
#
#         # 3. Filter for Pareto Frontier (Minimization)
#         # To find the frontier, we sort by Objective 1 and keep points
#         # that improve Objective 2 compared to all previous points.
#
#         # Sort by Objective 1 (val1)
#         sorted_results = sorted(raw_results, key=lambda x: x[2])
#
#         pareto_points = []
#         best_obj2_so_far = float('inf')
#
#         for point in sorted_results:
#             b, c, val1, val2 = point
#
#             # If this point has a better (lower) val2 than any point
#             # with a better (lower) val1 found so far, it is efficient.
#             if val2 < best_obj2_so_far:
#                 pareto_points.append(point)
#                 best_obj2_so_far = val2
#
#         # 4. Extract data for plotting
#         p_b = [p[0] for p in pareto_points]
#         p_c = [p[1] for p in pareto_points]
#         p_v1 = [p[2] for p in pareto_points]
#         p_v2 = [p[3] for p in pareto_points]
#
#         all_v1 = [r[2] for r in raw_results]
#         all_v2 = [r[3] for r in raw_results]
#
#         # 5. Plot
#         plt.figure(figsize=(10, 6))
#
#         # Plot all tested points in gray
#         plt.scatter(all_v1, all_v2, color='lightgray', label='Feasible Points', alpha=0.5)
#
#         # Plot Pareto Frontier in red connected by a line
#         plt.plot(p_v1, p_v2, color='red', marker='o', label='Pareto Frontier')
#
#         # Annotate the B, C values on the frontier points
#         for i, txt in enumerate(zip(p_b, p_c)):
#             plt.annotate(f"({txt[0]}, {txt[1]})", (p_v1[i], p_v2[i]),
#                          fontsize=8, xytext=(5, 5), textcoords='offset points')
#
#         plt.title('Pareto Frontier: Opt Func 1 vs Opt Func 2')
#         plt.xlabel('Objective 1 Value')
#         plt.ylabel('Objective 2 Value')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.show()
#
#         return pareto_points
#
#
# nvlink_topology = Topology("something", 72, "nvlink", 1)
# infiniband_topology = Topology("something", 2, "infiniband", 3)
# gpu = GPU(0.4, 1)
# sim = Simulator(nvlink_topology, infiniband_topology, gpu, 4096, 4, "fp32")
# sim.solve_pareto_brute_force()
#
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


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
            # Default or error handling
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
        #v_outer = (1 / self.outer_topology.size) * b * self.r * self.dim * elem_size

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

    def pareto_front(self, points):
        """
        points[:,0] = f1 (minimize)
        points[:,1] = f2 (minimize)
        """
        is_pareto = np.ones(len(points), dtype=bool)

        for i, p in enumerate(points):
            if not is_pareto[i]:
                continue

                # Filter out points that are WORSE than 'p':
                # 1. Worse X: points[:, 0] >= p[0] (We want smaller X, so larger is bad)
                # 2. Worse Y: points[:, 1] <= p[1] (We want bigger Y, so smaller is bad)
            dominates = (
                    (points[:, 0] >= p[0]) &
                    (points[:, 1] <= p[1]) &
                    # Must be strictly worse in at least one dimension to avoid removing itself
                    ((points[:, 0] > p[0]) | (points[:, 1] < p[1]))
            )

            is_pareto[dominates] = False

        return is_pareto

    def solve_pareto_method_1(self):
        max_mem = self.gpu.tensor_memory

        combinations = []
        for i in range(20):
            b = 2 ** i
            for c in range (1, b + 1):
                if (b / c) * self.dim * self.get_elem_size() <= max_mem:
                    combinations.append((b, c))

        print(f"Evaluating {len(combinations)} (B, C) pairs...")

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

    # ==================================================
    # Plot
    # ==================================================
    def plot_pareto(self, all_pts, pareto_pts):
        plt.figure(figsize=(9, 6))

        # plt.scatter(
        #     all_pts[:, 2], all_pts[:, 3],
        #     color="lightgray", alpha=0.4, label="All feasible points"
        # )

        plt.plot(
            pareto_pts[:, 2], pareto_pts[:, 3],
            "ro-", label="Pareto front"
        )

        for point in pareto_pts:
            b_val = int(point[0])
            c_val = int(point[1])
            x_coord = point[2]
            y_coord = point[3]

            # Annotate text: "(B, C)"
            label = f"({b_val}, {c_val})"

            plt.annotate(
                label,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(5, 5),  # Offset label slightly up and right
                ha='left',
                fontsize=9
            )

        plt.xlabel("Objective 1: Total Time [sec]")
        plt.ylabel("Objective 2: Throughput [KB / sec]")
        plt.title("Pareto Front (Method 1)")
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
    all_points, pareto_points = sim.solve_pareto_method_1()
    sim.plot_pareto(all_points, pareto_points)
