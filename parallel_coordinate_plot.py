import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
import os
import multiprocessing
from multiprocessing import Pool

# Pymoo Imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultSingleObjectiveTermination

# Import your Simulator
try:
    from pareto_solver import Simulator, GPU, Topology, Blackwell
except ImportError:
    print("Error: Could not import 'pareto_solver.py'.")
    exit()


# ==============================================================================
# 1. Helper: Combined Distance (GD + IGD)
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

    return gd + igd


# ==============================================================================
# 2. Parallelizable Evaluation Function (Must be outside class for Pickling)
# ==============================================================================
def evaluate_hardware_config(args):
    """
    Unpacks arguments and runs the simulation.
    args: (x, target_front, simulator_template, fast_mode)
    """
    x, target_front, sim_template, fast_mode = args

    flops_val = x[0]
    sm_val = int(x[1])
    l1_val = int(x[2])
    dies_val = int(np.round(x[3]))

    # Build Candidate GPU
    candidate_gpu = GPU(
        t_launch=0.01,
        sm_l1_size=l1_val,
        number_sm=sm_val,
        number_dies=dies_val,
        peak_flops_dict={"fp32": flops_val},
        hbi_bw=10 * (2 ** 30),
        hbi_launch=1e-7
    )

    sim_template.gpu = candidate_gpu

    cand_pareto = sim_template.solve_pareto_pymoo(fast_mode=fast_mode)

    if len(cand_pareto) > 0:
        cand_front = cand_pareto[:, 2:4]
        error = calculate_combined_distance(target_front, cand_front)
    else:
        error = 1e9

    return error


# ==============================================================================
# 3. Pymoo Problem Definition (Parallelized)
# ==============================================================================
class HardwareInverseProblem(ElementwiseProblem):
    def __init__(self, target_pareto, simulator_template, fast_mode=True, n_threads=8):
        self.target_front = target_pareto[:, 2:4]
        self.sim_template = simulator_template
        self.fast_mode = fast_mode
        self.n_threads = n_threads

        super().__init__(n_var=4,
                         n_obj=1,
                         xl=np.array([100.0, 50.0, 64.0, 1.0]),
                         xu=np.array([4000.0, 250.0, 512.0, 2.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        # NOTE: Pymoo calls this function.
        # If we use a parallel runner in 'minimize', 'x' will be a batch.
        # But 'ElementwiseProblem' usually processes one by one.
        # To get true parallelization with Pymoo Elementwise, we often use a ThreadPool inside here
        # OR we rely on Pymoo's internal parallelization if configured.

        # Here, we will just call the helper directly.
        # *Optimization*: We will use the parallel pool in the 'Runner' passed to minimize,
        # so this function assumes it is running on a worker.

        # Pass arguments to the static helper function
        error = evaluate_hardware_config((x, self.target_front, self.sim_template, self.fast_mode))
        out["F"] = error


# ==============================================================================
# 4. Main Workflow Controller
# ==============================================================================
def run_advanced_optimization(target_pareto, simulator_template):
    N_THREADS = multiprocessing.cpu_count()
    print(f"--- Starting Optimization on {N_THREADS} Cores ---")

    # ------------------------------------------------------------------
    # PHASE 1: Global Search (Fast Mode)
    # ------------------------------------------------------------------
    print("\n[Phase 1] Global Evolutionary Search (Fast Mode)...")

    # We use a Pool to parallelize the Pymoo population evaluation
    pool = Pool(N_THREADS)

    # Initialize Problem
    problem = HardwareInverseProblem(target_pareto, simulator_template, fast_mode=True)

    algorithm = GA(
        pop_size=100,  # 100 Candidates per generation
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Define "Smart" Termination
    # 1. ftol=1e-3: Stop if improvement is less than 0.001 (0.1% approx)
    # 2. period=5:  Calculate this trend over the last 5 generations
    # 3. n_max_gen=50: Safety cap (don't run forever if it keeps oscillating)
    termination = DefaultSingleObjectiveTermination(
        ftol=1e-3,
        period=4,
        n_max_gen=50
    )
    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   seed=1,
                   verbose=True,
                   runner=pool.starmap)

    pool.close()  # Close pool to free resources for Phase 2
    pool.join()

    print(f"[Phase 1] Done. Best Fast Error: {res.F[0]:.5f}")

    # Extract Top 100 Unique Candidates
    pop = res.pop
    candidates = []
    seen_specs = set()

    for ind in pop:
        x = ind.X
        # create unique key to avoid duplicates
        key = (int(x[0]), int(x[1]), int(x[2]), int(round(x[3])))
        if key not in seen_specs and ind.F[0] < 1000:
            seen_specs.add(key)
            candidates.append({
                "params": x,
                "peak_tflops": x[0],
                "sm_num": int(x[1]),
                "l1_size": int(x[2]),
                "dies": int(round(x[3])),
                "fast_error": ind.F[0]
            })

    # Sort and slice top 100
    candidates.sort(key=lambda x: x["fast_error"])
    top_candidates = candidates[:100]
    print(f"Extracted {len(top_candidates)} unique candidates for verification.")

    # ------------------------------------------------------------------
    # PHASE 2: Verification (High-Res Mode)
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Verifying Top {len(top_candidates)} with High-Resolution Simulation...")

    pool = Pool(N_THREADS)

    # Prepare High-Res Batches
    tasks = []
    for cand in top_candidates:
        # Tuple: (x, target, simulator, fast_mode=False)
        tasks.append((cand["params"], target_pareto[:, 2:4], simulator_template, False))

    # Run Parallel Verification
    high_res_errors = pool.map(evaluate_hardware_config, tasks)

    pool.close()
    pool.join()

    # Update candidates with real errors
    for i, err in enumerate(high_res_errors):
        top_candidates[i]["real_error"] = err
        top_candidates[i]["error"] = err  # For plotting compatibility

    # Sort by REAL error
    top_candidates.sort(key=lambda x: x["real_error"])

    best_real = top_candidates[0]
    print(f"[Phase 2] Verification Done.")
    print(f"Best Verified Candidate: Err={best_real['real_error']:.5f} | "
          f"Peak={best_real['peak_tflops']:.0f}, SM={best_real['sm_num']}, Dies={best_real['dies']}")

    # ------------------------------------------------------------------
    # PHASE 3: Refinement (Local Perturbation of Winners)
    # ------------------------------------------------------------------
    # The user asked: "create new points based on actual winners... check these fast"
    print("\n[Phase 3] Refining Top 10 Winners (Local Search)...")

    top_10 = top_candidates[:10]
    refined_batch = []

    # For each winner, spawn 10 mutated versions
    for winner in top_10:
        base_x = winner["params"]
        for _ in range(10):
            # Apply slight gaussian noise (Mutation)
            mutated_x = base_x.copy()
            mutated_x[0] += np.random.normal(0, 50)  # +/- 50 TFLOPS
            mutated_x[1] += np.random.normal(0, 5)  # +/- 5 SMs
            mutated_x[2] += np.random.normal(0, 16)  # +/- 16KB L1
            # Keep dies same, it's discrete

            # Clip bounds
            mutated_x[0] = np.clip(mutated_x[0], 100, 4000)
            mutated_x[1] = np.clip(mutated_x[1], 50, 250)
            mutated_x[2] = np.clip(mutated_x[2], 64, 512)

            refined_batch.append(mutated_x)

    # Run Fast Check on Refined Batch
    pool = Pool(N_THREADS)
    fast_tasks = [(x, target_pareto[:, 2:4], simulator_template, True) for x in refined_batch]
    refined_errors = pool.map(evaluate_hardware_config, fast_tasks)
    pool.close()
    pool.join()

    # Store Refined Results
    final_population = top_candidates.copy()  # Keep original winners

    for i, x in enumerate(refined_batch):
        err = refined_errors[i]
        if err < 1000:
            final_population.append({
                "params": x,
                "peak_tflops": x[0],
                "sm_num": int(x[1]),
                "l1_size": int(x[2]),
                "dies": int(round(x[3])),
                "error": err  # Note: This is fast error, but acceptable for final plot
            })

    print(f"[Phase 3] Done. Final Population: {len(final_population)} configs.")
    return final_population


# ==============================================================================
# 5. Plotting
# ==============================================================================
def plot_results(data_dicts, max_lines_to_plot=150):
    if not data_dicts:
        print("No matches found.")
        return

    # Filter top N
    data_dicts.sort(key=lambda x: x["error"])
    data_dicts = data_dicts[:max_lines_to_plot]
    # Reverse for plotting (Best on top)
    data_dicts.sort(key=lambda x: x["error"], reverse=True)

    keys = ["peak_tflops", "sm_num", "l1_size", "dies"]
    axis_labels = ["Peak TFLOPS", "Number of SMs", "L1 Size (KB)", "Dies"]

    raw_matrix = np.array([[d[k] for k in keys] for d in data_dicts])
    mins = raw_matrix.min(axis=0)
    maxs = raw_matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    plot_matrix = np.array([[d[k] for k in keys] for d in data_dicts])
    errors = np.array([d["error"] for d in data_dicts])
    norm_data = (plot_matrix - mins) / ranges

    # Linear Normalization for best contrast
    norm_err = mcolors.Normalize(vmin=errors.min(), vmax=errors.max())
    cmap = plt.cm.coolwarm
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_err)

    fig, ax = plt.subplots(figsize=(14, 7))
    n_axes = len(keys)

    # Grid
    num_ticks = 6
    for j in range(n_axes):
        tick_vals = np.linspace(mins[j], maxs[j], num_ticks)
        tick_pos = np.linspace(0, 1, num_ticks)
        for i, pos in enumerate(tick_pos):
            ax.axhline(pos, color='gray', alpha=0.2, linestyle=':', zorder=0)
            val_str = f"{tick_vals[i]:.0f}"
            ax.text(j - 0.03, pos, val_str, ha='right', va='center', fontsize=9, color='gray')

    # Lines
    for i in range(len(norm_data)):
        val = norm_err(errors[i])
        color = cmap(val)
        dynamic_alpha = 0.9 if val < 0.2 else 0.2
        ax.plot(range(n_axes), norm_data[i], color=color, alpha=dynamic_alpha, linewidth=1.5, zorder=1)

    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(axis_labels, fontsize=12, fontweight='bold', y=1.02)
    ax.set_yticks([])
    for i in range(n_axes):
        ax.axvline(i, color='black', lw=2, zorder=2)

    cbar = plt.colorbar(sm, ax=ax, pad=0.05, aspect=30)
    cbar.set_label('Error (Blue = Best)', rotation=270, labelpad=20, fontsize=11)

    plt.title(f"Inverse Design Solution Space\n(Top {len(norm_data)} from Refinement Phase)", fontsize=15, pad=25)
    plt.tight_layout()
    plt.show()


def sanity_check_plot(target_pareto, candidates, simulator_template):
    print("\n--- Running Final Sanity Check (Brute Force Plotting) ---")

    # 1. Setup Plot
    plt.figure(figsize=(10, 7))

    # Plot Ground Truth (Black, Thick)
    # Assumes column 2 = Step Time, column 3 = Throughput
    plt.plot(target_pareto[:, 2], target_pareto[:, 3],
             color='black', linewidth=2.5, linestyle='-', label='Ground Truth')

    # 2. Pick Top 3 Candidates
    # Sort by error (smallest error first)
    candidates.sort(key=lambda x: x["error"])
    top_3 = candidates[:3]

    # Define shades of Red: [Darkest (Best), Medium, Lightest]
    colors = ['blue', 'green', 'red']
    alphas = [1.0, 0.8, 0.6]

    for i, cand_data in enumerate(top_3):
        print(f"Simulating Rank #{i + 1} (Error: {cand_data['error']:.4f})...")

        # A. Re-create GPU
        cand_gpu = GPU(
            t_launch=0.01,
            sm_l1_size=cand_data['l1_size'],
            number_sm=cand_data['sm_num'],
            number_dies=cand_data['dies'],
            peak_flops_dict={"fp32": cand_data['peak_tflops']},
            hbi_bw=10 * (2 ** 30),
            hbi_launch=1e-7
        )

        # B. Run Brute Force (Absolute Truth Mode)
        simulator_template.gpu = cand_gpu
        _, brute_force_curve = simulator_template.solve_pareto_brute_force(fast_mode=False)

        # C. Plot
        # Rank 1 gets the darkest color, Rank 3 gets the lightest
        c = colors[i]
        a = alphas[i]
        label_str = (f"Rank #{i + 1} (Err: {cand_data['error']:.4f})\n"
                     f"Peak: {cand_data['peak_tflops']:.0f} | SM: {cand_data['sm_num']}")

        plt.plot(brute_force_curve[:, 2], brute_force_curve[:, 3],
                 color=c, alpha=a, linewidth=2, linestyle='--', marker='o', markersize=3,
                 label=label_str)

    # 3. Formatting
    plt.xlabel("Step Time (s)", fontweight='bold')
    plt.ylabel("Throughput (samples/s)", fontweight='bold')
    plt.title("Sanity Check: Top 3 Candidates vs. Ground Truth", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 6. Run
# ==============================================================================
if __name__ == "__main__":
    # Ensure safe multiprocessing on Windows/MacOS
    multiprocessing.freeze_support()

    GROUND_TRUTH_FILE = "ground_truth_pymoo.npy"
    nvlink = Topology("tree", 72, "nvlink", 0.003)
    ib = Topology("linear", 2, "infiniband", 0.01)

    # 1. Setup Ground Truth
    if os.path.exists(GROUND_TRUTH_FILE):
        print(f"1. Loading Target Pareto Curve from '{GROUND_TRUTH_FILE}'...")
        target_pareto = np.load(GROUND_TRUTH_FILE)
    else:
        print("1. Generating Target Pareto Curve (Ground Truth)...")
        secret_gpu = Blackwell()
        sim = Simulator(nvlink, ib, secret_gpu, dim=4096, r=4, element_type="fp32")

        # FIX IS HERE: Capture the single return value
        target_pareto = sim.solve_pareto_pymoo(fast_mode=False)

        np.save(GROUND_TRUTH_FILE, target_pareto)
        print(f"   Saved to '{GROUND_TRUTH_FILE}'")
    target_pareto = target_pareto[target_pareto[:, 2].argsort()]

    # 2. Run Optimization
    dummy_gpu = GPU(0.01, 0, 0, 2, {"fp32": 0})
    solver_sim = Simulator(nvlink, ib, dummy_gpu, dim=4096, r=4, element_type="fp32")

    final_population = run_advanced_optimization(target_pareto, solver_sim)

    # 3. Plot
    plot_results(final_population, max_lines_to_plot=150)
    sanity_check_plot(target_pareto, final_population, solver_sim)