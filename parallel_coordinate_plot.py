import numpy as np
import multiprocessing
import time
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import warnings

# ==============================================================================
# IMPORTANT: Worker Process Setup
# ==============================================================================
# Silence warnings in worker processes to keep console clean
warnings.filterwarnings("ignore")

# Pymoo Imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.callback import Callback

# Import Simulator
try:
    from pareto_solver import Simulator, GPU, Topology, Blackwell
except ImportError:
    pass


# ==============================================================================
# 0. Custom Callback for Progress & Estimations
# ==============================================================================
class ProgressCallback(Callback):
    def __init__(self, callback_fn, n_gen_total, time_limit=None):
        super().__init__()
        self.callback_fn = callback_fn
        self.n_gen_total = n_gen_total
        self.time_limit = time_limit
        self.start_time = time.time()

    def notify(self, algorithm):
        # 1. Calculate Progress & Time
        gen = algorithm.n_gen
        elapsed = time.time() - self.start_time

        # 2. Check Time Limit (Force Stop if exceeded)
        if self.time_limit and self.time_limit > 0:
            if elapsed >= self.time_limit:
                algorithm.termination.force_termination = True

        # Determine progress
        if self.time_limit and self.time_limit > 0:
            prog_gen = gen / self.n_gen_total if self.n_gen_total > 0 else 0
            prog_time = elapsed / self.time_limit
            progress = min(1.0, max(prog_gen, prog_time))
            eta_seconds = max(0, self.time_limit - elapsed)
        else:
            progress = gen / self.n_gen_total if self.n_gen_total > 0 else 0
            if gen > 0:
                time_per_gen = elapsed / gen
                remaining_gens = self.n_gen_total - gen
                eta_seconds = time_per_gen * remaining_gens
            else:
                eta_seconds = 0

        # 3. Extract Population Statistics
        pop = algorithm.pop
        if pop is not None and len(pop) > 0:
            min_error = pop.get("F").min()
            avg_error = pop.get("F").mean()

            candidates = []
            F = pop.get("F")
            if len(F.shape) > 1:
                F = F.flatten()
            sorted_indices = np.argsort(F)

            for idx in sorted_indices[:50]:
                ind = pop[idx]
                x = ind.X
                candidates.append({
                    "peak_tflops": x[0],
                    "sm_num": int(x[1]),
                    "l1_size": int(x[2]),
                    "dies": int(round(x[3])),
                    "inner_size": int(x[4]),
                    "outer_size": int(x[5]),
                    "hbi_bw": x[6],
                    "nvlink_bw": x[7],
                    "ib_bw": x[8],
                    "error": float(ind.F[0])
                })
        else:
            min_error = 0.0
            avg_error = 0.0
            candidates = []

        # 4. Send Data to UI
        if self.callback_fn:
            progress_data = {
                "progress": progress,
                "gen": gen,
                "total_gen": self.n_gen_total,
                "min_error": float(min_error),
                "avg_error": float(avg_error),
                "eta": eta_seconds,
                "top_candidates": candidates
            }
            self.callback_fn(progress_data)


# ==============================================================================
# 1. Helper: Combined Distance
# ==============================================================================
def calculate_combined_distance(target_front, candidate_front):
    if len(candidate_front) == 0:
        return 1e9

    min_val = np.min(target_front, axis=0)
    max_val = np.max(target_front, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0

    target_norm = (target_front - min_val) / range_val
    cand_norm = (candidate_front - min_val) / range_val

    dists = cdist(cand_norm, target_norm, metric='euclidean')
    gd = np.mean(np.min(dists, axis=1))
    igd = np.mean(np.min(dists, axis=0))

    return 0.5 * (gd + igd)


# ==============================================================================
# 2. Parallel Evaluation Function
# ==============================================================================
def evaluate_hardware_config(args):
    x, target_front, sim_config, fast_mode = args

    flops_val = x[0]
    sm_val = int(x[1])
    l1_val = int(x[2])
    dies_val = int(np.round(x[3]))
    inner_size = int(x[4])
    outer_size = int(x[5])
    hbi_bw = x[6]
    nvlink_bw = x[7]
    ib_bw = x[8]

    candidate_gpu = GPU(
        t_launch=0.01,
        sm_l1_size=l1_val,
        number_sm=sm_val,
        number_dies=dies_val,
        peak_flops_dict={"fp32": flops_val},
        hbi_bw=hbi_bw * (2 ** 30),
        hbi_launch=1e-7
    )

    inner_topo = Topology(
        kind=sim_config["inner_kind"],
        size=inner_size,
        link_type=sim_config["inner_type"],
        alpha=sim_config["inner_alpha"],
        bandwidth=nvlink_bw
    )

    outer_topo = Topology(
        kind=sim_config["outer_kind"],
        size=outer_size,
        link_type=sim_config["outer_type"],
        alpha=sim_config["outer_alpha"],
        bandwidth=ib_bw
    )

    temp_sim = Simulator(
        inner_topology=inner_topo,
        outer_topology=outer_topo,
        gpu=candidate_gpu,
        dim=sim_config["dim"],
        r=sim_config["r"],
        element_type=sim_config["type"]
    )

    cand_pareto = temp_sim.solve_pareto_pymoo(fast_mode=fast_mode)

    if len(cand_pareto) > 0:
        cand_front = cand_pareto[:, 2:4]
        error = calculate_combined_distance(target_front, cand_front)
    else:
        error = 1e9

    return error


# ==============================================================================
# 3. Pymoo Problem Definition
# ==============================================================================
class HardwareInverseProblem(ElementwiseProblem):
    def __init__(self, target_pareto, sim_config, bounds, fast_mode=True):
        self.target_front = target_pareto[:, 2:4]
        self.sim_config = sim_config
        self.fast_mode = fast_mode

        xl = np.array([
            bounds['flops'][0], bounds['sm'][0], bounds['l1'][0], bounds['dies'][0],
            bounds['inner_size'][0], bounds['outer_size'][0],
            bounds['hbi_bw'][0], bounds['nvlink_bw'][0], bounds['ib_bw'][0]
        ])

        xu = np.array([
            bounds['flops'][1], bounds['sm'][1], bounds['l1'][1], bounds['dies'][1],
            bounds['inner_size'][1], bounds['outer_size'][1],
            bounds['hbi_bw'][1], bounds['nvlink_bw'][1], bounds['ib_bw'][1]
        ])

        super().__init__(n_var=9, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        error = evaluate_hardware_config((x, self.target_front, self.sim_config, self.fast_mode))
        out["F"] = error


# ==============================================================================
# 4. Main Optimization Controller
# ==============================================================================
def run_advanced_optimization(target_pareto, simulator_template, bounds, hbi_bw_val=10, time_limit=0,
                              progress_callback=None):
    start_time = time.time()

    if progress_callback:
        progress_callback({"status": "init", "msg": "Initializing Optimization..."})

    N_THREADS = max(1, multiprocessing.cpu_count() - 2)

    sim_config = {
        "inner_kind": simulator_template.inner_topology.kind,
        "inner_type": simulator_template.inner_topology.link_type,
        "inner_alpha": simulator_template.inner_topology.alpha,
        "outer_kind": simulator_template.outer_topology.kind,
        "outer_type": simulator_template.outer_topology.link_type,
        "outer_alpha": simulator_template.outer_topology.alpha,
        "dim": simulator_template.dim,
        "r": simulator_template.r,
        "type": simulator_template.element_type,
        "hbi_bw": hbi_bw_val
    }

    N_GEN = 15
    POP_SIZE = 40

    with Pool(N_THREADS) as pool:
        problem = HardwareInverseProblem(target_pareto, sim_config, bounds, fast_mode=True)

        algorithm = GA(
            pop_size=POP_SIZE,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        termination = DefaultSingleObjectiveTermination(ftol=1e-3, period=4, n_max_gen=N_GEN)

        res = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=1,
            verbose=False,
            runner=pool.starmap,
            callback=ProgressCallback(progress_callback, N_GEN, time_limit)
        )

        # Extract Candidates
        pop = res.pop
        candidates = []
        seen_specs = set()

        if pop is not None:
            for ind in pop:
                x = ind.X
                key = tuple(np.round(x, 2))

                if key not in seen_specs and ind.F[0] < 1000:
                    seen_specs.add(key)
                    candidates.append({
                        "params": x,
                        "peak_tflops": x[0],
                        "sm_num": int(x[1]),
                        "l1_size": int(x[2]),
                        "dies": int(round(x[3])),
                        "inner_size": int(x[4]),
                        "outer_size": int(x[5]),
                        "hbi_bw": x[6],
                        "nvlink_bw": x[7],
                        "ib_bw": x[8],
                        "error": ind.F[0]
                    })

        candidates.sort(key=lambda x: x["error"])
        top_candidates = candidates[:50]

        # --- Phase 2: Logic Check ---
        # If time limit exceeded, skip expensive verification and return current results immediately
        elapsed_now = time.time() - start_time
        if time_limit and time_limit > 0 and elapsed_now >= time_limit:
            if progress_callback:
                progress_callback({"status": "complete", "msg": "Time Limit Reached. Showing current best results."})
            # Just fill in dummy "real_error" to match the expected dict structure, or reuse "error"
            for cand in top_candidates:
                cand["real_error"] = cand["error"]
            return top_candidates

        # Otherwise proceed to verification
        if progress_callback:
            progress_callback({"status": "phase2", "msg": "Verifying top candidates..."})

        tasks = [(cand["params"], target_pareto[:, 2:4], sim_config, False) for cand in top_candidates]
        high_res_errors = pool.map(evaluate_hardware_config, tasks)

        for i, err in enumerate(high_res_errors):
            top_candidates[i]["real_error"] = err
            top_candidates[i]["error"] = err

    return top_candidates