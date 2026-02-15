import numpy as np
import multiprocessing
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

# Import Simulator
try:
    from pareto_solver import Simulator, GPU, Topology, Blackwell
except ImportError:
    pass


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
    """
    Evaluates a single hardware configuration.
    Unpacks 9 variables:
    0:FLOPS, 1:SM, 2:L1, 3:Dies, 4:InnerSize, 5:OuterSize, 6:HBI_BW, 7:NVLink_BW, 8:IB_BW
    """
    # Unpack arguments
    x, target_front, sim_config, fast_mode = args

    # GPU Parameters
    flops_val = x[0]
    sm_val = int(x[1])
    l1_val = int(x[2])
    dies_val = int(np.round(x[3]))

    # Topology Parameters
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
        hbi_bw=hbi_bw * (2 ** 30),  # Convert TB/s to KB/s
        hbi_launch=1e-7
    )

    # Reconstruct Topology Objects dynamically
    # Use template 'kind', 'type', 'alpha' from config, but inject 'size' and 'bandwidth'
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

    # Reconstruct Simulator
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

        # We now have 9 optimization variables
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
def run_advanced_optimization(target_pareto, simulator_template, bounds, progress_callback=None):
    """
    Runs the genetic algorithm using multiprocessing.
    Includes status reporting via progress_callback.
    """

    def report(p, msg):
        print(f"[{p}%] {msg}")
        if progress_callback:
            progress_callback(p, msg)

    report(0, "Initializing Optimization...")

    N_THREADS = max(1, multiprocessing.cpu_count() - 2)

    # Extract template info (static properties of topology)
    # We don't pass 'size' or 'bandwidth' here as they are now variables
    sim_config = {
        "inner_kind": simulator_template.inner_topology.kind,
        "inner_type": simulator_template.inner_topology.link_type,
        "inner_alpha": simulator_template.inner_topology.alpha,

        "outer_kind": simulator_template.outer_topology.kind,
        "outer_type": simulator_template.outer_topology.link_type,
        "outer_alpha": simulator_template.outer_topology.alpha,

        "dim": simulator_template.dim,
        "r": simulator_template.r,
        "type": simulator_template.element_type
    }

    # --- Phase 1: Genetic Algorithm ---
    report(10, "Phase 1: Global Genetic Algorithm (Fast Mode)...")

    with Pool(N_THREADS) as pool:
        problem = HardwareInverseProblem(target_pareto, sim_config, bounds, fast_mode=True)

        algorithm = GA(
            pop_size=100,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        termination = DefaultSingleObjectiveTermination(ftol=1e-3, period=4, n_max_gen=50)

        res = minimize(problem, algorithm, termination=termination, seed=1, verbose=False, runner=pool.starmap)

        report(50, "Phase 1 Complete. Processing Candidates...")

        # Extract Candidates
        pop = res.pop
        candidates = []
        seen_specs = set()

        for ind in pop:
            x = ind.X
            # Key based on GPU params (first 4) + Topology params (next 5)
            # Use tuple to ensure uniqueness check works
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

        # --- Phase 2: High Res Verification ---
        report(60, f"Phase 2: High-Fidelity Verification of {len(top_candidates)} Candidates...")

        tasks = [(cand["params"], target_pareto[:, 2:4], sim_config, False) for cand in top_candidates]
        high_res_errors = pool.map(evaluate_hardware_config, tasks)

        for i, err in enumerate(high_res_errors):
            top_candidates[i]["real_error"] = err
            top_candidates[i]["error"] = err

        report(90, "Phase 2 Complete. Finalizing Results...")

    return top_candidates