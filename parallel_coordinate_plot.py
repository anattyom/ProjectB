import numpy as np
import multiprocessing
import time
import sys
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings("ignore")

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.callback import Callback

try:
    from pareto_solver import Simulator, GPU, Topology
except ImportError:
    pass


class ProgressCallback(Callback):
    def __init__(self, callback_fn, n_gen_total, time_limit=None):
        super().__init__()
        self.callback_fn = callback_fn
        self.n_gen_total = n_gen_total
        self.time_limit = time_limit
        self.start_time = time.time()

    def notify(self, algorithm):
        gen = algorithm.n_gen
        elapsed = time.time() - self.start_time

        if self.time_limit and self.time_limit > 0 and elapsed >= self.time_limit:
            algorithm.termination.force_termination = True

        progress = min(1.0, elapsed / self.time_limit) if (self.time_limit and self.time_limit > 0) else (
            gen / self.n_gen_total if self.n_gen_total > 0 else 0)
        eta_seconds = max(0, self.time_limit - elapsed) if (self.time_limit and self.time_limit > 0) else (
            (elapsed / gen) * (self.n_gen_total - gen) if gen > 0 else 0)

        pop = algorithm.pop
        if pop is not None and len(pop) > 0:
            min_error = pop.get("F").min()
            avg_error = pop.get("F").mean()
            candidates = []
            F = pop.get("F").flatten() if len(pop.get("F").shape) > 1 else pop.get("F")

            for idx in np.argsort(F)[:50]:
                ind = pop[idx]
                x = ind.X
                candidates.append({
                    "peak_tflops": x[0], "sm_num": int(x[1]), "l1_size": int(x[2]), "dies": int(round(x[3])),
                    "inner_size": int(x[4]), "outer_size": int(x[5]), "hbi_bw": x[6], "nvlink_bw": x[7], "ib_bw": x[8],
                    "error": float(ind.F[0])
                })
        else:
            min_error, avg_error, candidates = 0.0, 0.0, []

        if self.callback_fn:
            self.callback_fn({
                "progress": progress, "gen": gen, "total_gen": self.n_gen_total,
                "min_error": float(min_error), "avg_error": float(avg_error),
                "eta": eta_seconds, "top_candidates": candidates
            })


def calculate_combined_distance(target_front, candidate_front):
    # If no curve was generated, return max error of 100
    if len(candidate_front) == 0: return 100.0

    min_val, max_val = np.min(target_front, axis=0), np.max(target_front, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0

    target_norm = (target_front - min_val) / range_val
    cand_norm = (candidate_front - min_val) / range_val

    dists = cdist(cand_norm, target_norm, metric='euclidean')
    raw_error = 0.5 * (np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0)))

    # The max distance in a normalized 1x1 space is sqrt(1^2 + 1^2) = ~1.414
    # Scale the raw error to a 0-100 range
    max_theoretical_dist = np.sqrt(2)
    scaled_error = (raw_error / max_theoretical_dist) * 100.0

    return min(scaled_error, 100.0)  # Cap at 100 just in case


def evaluate_hardware_config(args):
    x, target_front, sim_config, fast_mode = args
    flops_val, sm_val, l1_val, dies_val = x[0], int(x[1]), int(x[2]), int(np.round(x[3]))
    inner_size, outer_size = int(x[4]), int(x[5])
    hbi_bw, nvlink_bw, ib_bw = x[6], x[7], x[8]

    candidate_gpu = GPU(0.01, sm_l1_size=l1_val, number_sm=sm_val, number_dies=dies_val,
                        peak_flops_dict={"fp32": flops_val}, hbi_bw=hbi_bw * (2 ** 30))
    inner_topo = Topology(sim_config["inner_kind"], inner_size, sim_config["inner_type"], sim_config["inner_alpha"],
                          nvlink_bw)
    outer_topo = Topology(sim_config["outer_kind"], outer_size, sim_config["outer_type"], sim_config["outer_alpha"],
                          ib_bw)
    temp_sim = Simulator(inner_topo, outer_topo, candidate_gpu, sim_config["dim"], sim_config["r"], sim_config["type"])

    cand_pareto = temp_sim.solve_pareto_pymoo(fast_mode=fast_mode, latency_cap=8.0)
    return calculate_combined_distance(target_front, cand_pareto[:, 2:4]) if len(cand_pareto) > 0 else 1e9


class HardwareInverseProblem(ElementwiseProblem):
    def __init__(self, target_pareto, sim_config, bounds, fast_mode=True):
        self.target_front = target_pareto[:, 2:4]
        self.sim_config = sim_config
        self.fast_mode = fast_mode
        xl = np.array([bounds[k][0] for k in
                       ['flops', 'sm', 'l1', 'dies', 'inner_size', 'outer_size', 'hbi_bw', 'nvlink_bw', 'ib_bw']])
        xu = np.array([bounds[k][1] for k in
                       ['flops', 'sm', 'l1', 'dies', 'inner_size', 'outer_size', 'hbi_bw', 'nvlink_bw', 'ib_bw']])
        super().__init__(n_var=9, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = evaluate_hardware_config((x, self.target_front, self.sim_config, self.fast_mode))


def run_advanced_optimization(target_pareto, simulator_template, bounds, hbi_bw_val=10, time_limit=0,
                              progress_callback=None):
    start_time = time.time()
    if progress_callback: progress_callback({"status": "init", "msg": "Initializing..."})

    sim_config = {
        "inner_kind": simulator_template.inner_topology.kind, "inner_type": simulator_template.inner_topology.link_type,
        "inner_alpha": simulator_template.inner_topology.alpha,
        "outer_kind": simulator_template.outer_topology.kind, "outer_type": simulator_template.outer_topology.link_type,
        "outer_alpha": simulator_template.outer_topology.alpha,
        "dim": simulator_template.dim, "r": simulator_template.r, "type": simulator_template.element_type,
        "hbi_bw": hbi_bw_val
    }

    N_GEN, POP_SIZE = 15, 40
    pool = Pool(max(1, multiprocessing.cpu_count() - 2))

    try:
        res = minimize(
            HardwareInverseProblem(target_pareto, sim_config, bounds, fast_mode=True),
            GA(pop_size=POP_SIZE, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
               eliminate_duplicates=True),
            termination=DefaultSingleObjectiveTermination(ftol=1e-3, period=4, n_max_gen=N_GEN),
            seed=1, verbose=False, runner=pool.starmap,
            callback=ProgressCallback(progress_callback, N_GEN, time_limit)
        )

        candidates, seen_specs = [], set()
        if res.pop is not None:
            for ind in res.pop:
                x = ind.X
                key = tuple(np.round(x, 2))
                if key not in seen_specs and ind.F[0] < 1000:
                    seen_specs.add(key)
                    candidates.append({
                        "params": x, "peak_tflops": x[0], "sm_num": int(x[1]), "l1_size": int(x[2]),
                        "dies": int(round(x[3])),
                        "inner_size": int(x[4]), "outer_size": int(x[5]), "hbi_bw": x[6], "nvlink_bw": x[7],
                        "ib_bw": x[8], "error": ind.F[0]
                    })

        top_candidates = sorted(candidates, key=lambda c: c["error"])[:50]

        if time_limit and time_limit > 0 and (time.time() - start_time) >= time_limit:
            if progress_callback: progress_callback({"status": "complete", "msg": "Time Limit Reached."})
            for cand in top_candidates: cand["real_error"] = cand["error"]
            return top_candidates

        if progress_callback: progress_callback({"status": "phase2", "msg": "Verifying..."})
        high_res_errors = pool.map(evaluate_hardware_config,
                                   [(cand["params"], target_pareto[:, 2:4], sim_config, False) for cand in
                                    top_candidates])

        for i, err in enumerate(high_res_errors):
            top_candidates[i]["real_error"] = err
            top_candidates[i]["error"] = err

        return sorted(top_candidates, key=lambda c: c["error"])

    except KeyboardInterrupt:
        print("\nOptimization Interrupted! Killing workers...")
        pool.terminate()
        pool.join()
        sys.exit(0)
    finally:
        pool.close()
        pool.join()