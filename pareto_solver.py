import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
    def __init__(self, kind, size, link_type, alpha, bandwidth=None):
        self.kind = kind
        self.size = size
        self.link_type = link_type
        self.alpha = alpha  # [s]
        self.bandwidth = bandwidth  # [TB/s]

    def get_bw(self):
        if self.bandwidth is not None:
            return self.bandwidth * (2 ** 30)
        bw = {
            "infiniband": 0.8,
            "PCIe": 0.256,
            "nvlink": 1.8,
            "l2": 6,
            "HBM": 8,
            "HBI": 10,
            "l1": 20
        }
        return bw.get(self.link_type, 1.0) * (2 ** 30)

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
class GPU:
    def __init__(self, t_launch, sm_l1_size, number_sm, number_dies, peak_flops_dict, efficiency_factor=0.5,
                 hbi_bw=None, hbi_launch=1e-7):
        self.efficiency_factor = efficiency_factor
        self.t_launch = t_launch
        self.sm_l1_size = sm_l1_size
        self.number_sm = number_sm
        self.number_dies = number_dies
        self.memory = self.sm_l1_size * self.number_sm
        self.peak_flops_dict = peak_flops_dict
        self.hbi_bw = hbi_bw
        self.hbi_launch = hbi_launch

    def get_achieved_tflops(self, element_type):
        return self.peak_flops_dict.get(element_type, 0) * self.efficiency_factor

    def get_inner_gpu_overhead(self, v):
        if self.number_dies <= 1 or self.hbi_bw is None:
            return 0
        return self.hbi_launch + (v / self.hbi_bw)


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
        return {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "nvfp4": 0.5}[self.element_type] / (2 ** 10)

    def calculate_t_comp(self, b, c):
        total_gpus = self.inner_topology.size * self.outer_topology.size
        flops = 2 * (b / c) * self.dim * (self.r * self.dim / total_gpus)
        achieved = self.gpu.get_achieved_tflops(self.element_type) * 1e12
        v_gpu = (1 / total_gpus) * b * self.r * self.dim * self.get_elem_size()
        inner_gpu_overhead = self.gpu.get_inner_gpu_overhead(v_gpu / self.gpu.number_dies)
        return self.gpu.t_launch + (flops / achieved) + inner_gpu_overhead

    def calculate_t_comm(self, b, c):
        total_gpus = self.inner_topology.size * self.outer_topology.size
        v_gpu = (1 / total_gpus) * b * self.r * self.dim * self.get_elem_size()
        return self.inner_topology.get_t_comm(v_gpu, c) + self.outer_topology.get_t_comm(v_gpu, c)

    def opt_func_1(self, p):
        b, c = p
        t_comp = self.calculate_t_comp(b, c)
        t_comm = self.calculate_t_comm(b, c)
        return t_comp + c * max(t_comp, t_comm)

    def opt_func_2(self, p):
        b, c = p
        t_total = self.opt_func_1([b, c])
        b_bytes = b * self.dim * self.r * self.get_elem_size()
        return -(b_bytes / t_total)

    def pareto_front(self, points):
        is_pareto = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            if not is_pareto[i]: continue
            dominates = ((points[:, 0] >= p[0]) & (points[:, 1] <= p[1]) & (
                        (points[:, 0] > p[0]) | (points[:, 1] < p[1])))
            is_pareto[dominates] = False
        return is_pareto

    def solve_pareto_brute_force(self, fast_mode=False, latency_cap=8.0):
        max_mem = self.gpu.memory
        combinations = []
        b_range = [2 ** i for i in range(21)]
        for b in b_range:
            c_candidates = [1, 2, 4, 8, 16, 32] if fast_mode else range(1, b + 1)
            for c in c_candidates:
                if c > b: continue
                if (b / c) * self.dim * self.get_elem_size() <= max_mem:
                    combinations.append((b, c))

        def evaluate(pair):
            b, c = pair
            return b, c, self.opt_func_1([b, c]), -self.opt_func_2([b, c])

        with ThreadPoolExecutor() as executor:
            results = np.array(list(executor.map(evaluate, combinations)))

        # Filter by latency cap before finding pareto front
        results = results[results[:, 2] <= latency_cap]
        if len(results) == 0: return np.zeros((0, 4)), np.zeros((0, 4))

        mask = self.pareto_front(results[:, 2:4])
        pareto = results[mask]
        return results, pareto[pareto[:, 2].argsort()]

    def estimate_pareto_analytical(self, latency_cap=8.0):
        results = []
        b_range = [2 ** i for i in range(21)]
        c_range = [1, 2, 4, 8, 16]

        for b in b_range:
            for c in c_range:
                if c > b: continue
                if (b / c) * self.dim * self.get_elem_size() <= self.gpu.memory:
                    lat = self.opt_func_1([b, c])
                    if lat <= latency_cap:
                        results.append([b, c, lat, -self.opt_func_2([b, c])])

        res_arr = np.array(results)
        if len(res_arr) == 0: return np.zeros((0, 4))

        res_arr = res_arr[res_arr[:, 2].argsort()]
        pareto = []
        curr_max_thr = -1
        for row in res_arr:
            if row[3] > curr_max_thr:
                pareto.append(row)
                curr_max_thr = row[3]
        return np.array(pareto)

    def solve_pareto_pymoo(self, fast_mode=True, latency_cap=8.0):
        pop_size = 40 if fast_mode else 100
        n_gen = 60 if fast_mode else 200
        simulator_instance = self
        max_b_limit = 2 ** 20
        max_c_limit = 2048

        class GPUOptimizationProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=2, n_obj=2, n_ieq_constr=3, xl=np.array([1, 1]),
                                 xu=np.array([max_b_limit, max_c_limit]))

            def _evaluate(self, x, out, *args, **kwargs):
                b = int(np.clip(x[0], 1, max_b_limit))
                c = int(np.clip(x[1], 1, max_c_limit))

                f1 = simulator_instance.opt_func_1([b, c])
                f2 = simulator_instance.opt_func_2([b, c])

                g1 = (
                                 b / c) * simulator_instance.dim * simulator_instance.get_elem_size() - simulator_instance.gpu.memory
                g2 = c - b
                g3 = f1 - latency_cap  # Ensures latency stays under cap

                out["F"] = [f1, f2]
                out["G"] = [g1, g2, g3]

        problem = GPUOptimizationProblem()
        algorithm = NSGA2(pop_size=pop_size, sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=15),
                          mutation=PM(eta=20), eliminate_duplicates=True)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=42, verbose=False)

        pareto_results = []
        if res.X is not None:
            for i in range(len(res.X)):
                b = int(np.clip(res.X[i, 0], 1, max_b_limit))
                c = int(np.clip(res.X[i, 1], 1, max_c_limit))
                pareto_results.append([b, c, res.F[i, 0], -res.F[i, 1]])

        res_arr = np.array(pareto_results)
        if len(res_arr) == 0: return res_arr
        return res_arr[res_arr[:, 2].argsort()]