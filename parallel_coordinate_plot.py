import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
from tqdm import tqdm

try:
    from pareto_solver import Simulator, GPU, Topology
except ImportError:
    print("Error: Could not import 'pareto_solver.py'.")
    exit()


# ==============================================================================
# 1. Helper: Generational Distance (Curve Comparison)
# ==============================================================================
def calculate_combined_distance(target_front, candidate_front):
    """
    Measures how close the Candidate Front is to the Target Front.
    Lower is better. Returns 'Average Distance'.
    """
    if len(candidate_front) == 0:
        return float('inf')

    # Normalize based on Target's range
    min_val = np.min(target_front, axis=0)
    max_val = np.max(target_front, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0  # Avoid div by zero

    target_norm = (target_front - min_val) / range_val
    cand_norm = (candidate_front - min_val) / range_val

    dists = cdist(cand_norm, target_norm, metric='euclidean')
    # GD: Distance from Candidate -> Nearest Target (Rows)
    gd = np.mean(np.min(dists, axis=1))

    # IGD: Distance from Target -> Nearest Candidate (Columns)
    igd = np.mean(np.min(dists, axis=0))  # <--- THIS WAS MISSING

    # 4. Return the Sum
    return gd + igd


# ==============================================================================
# 2. The Main Solver
# ==============================================================================
def solve_inverse_problem_curve_matching(target_pareto_points, simulator_template, num_samples=3000, tolerance=0.03):
    print(f"Inverse Solving: Matching CURVES via Simulator Fast Mode ({num_samples} samples)...")

    # Extract [Time, Throughput] from the target curve
    # Format: [B, C, Time, Throughput] -> Columns 2 and 3
    target_front = target_pareto_points[:, 2:4]

    valid_configs = []

    for _ in tqdm(range(num_samples)):
        # --- A. Sample Hardware Specs ---
        l1_sample = int(np.random.uniform(64, 512))
        sm_sample = int(np.random.uniform(50, 200))
        flops_sample = np.random.uniform(100, 800)

        # --- B. Create NEW Candidate GPU ---
        # We do not modify an existing object. We instantiate a fresh one.
        candidate_gpu = GPU(
            efficiency_factor=0.5,  # Fixed at 0.5
            t_launch=0.01,  # Assuming constant launch latency
            sm_l1_size=l1_sample,
            number_sm=sm_sample,
            number_dies=2,  # Assuming standard 2-die layout
            peak_flops_dict={"fp32": flops_sample},  # Inject sampled TFLOPS directly
            hbi_bw=10 * (2 ** 30),  # 10 TB/s
            hbi_launch=1e-7
        )

        # --- C. Update Simulator ---
        # We use the template simulator but swap out the GPU component
        simulator_template.gpu = candidate_gpu

        # --- D. Get Candidate Pareto Front (via FAST MODE) ---
        # This uses the new GPU's memory limits and compute speed
        _, cand_pareto_points = simulator_template.solve_pareto_brute_force(fast_mode=True)

        # --- E. Compare Curves ---
        if len(cand_pareto_points) > 0:
            cand_front = cand_pareto_points[:, 2:4]
            dist = calculate_combined_distance(target_front, cand_front)

            if dist < tolerance:
                valid_configs.append({
                    "peak_tflops": flops_sample,
                    "sm_num": sm_sample,
                    "l1_size": l1_sample,
                    "error": dist
                })

    return valid_configs


# ==============================================================================
# 3. PLOTTING
# ==============================================================================
def plot_results(data_dicts, max_lines_to_plot=150):
    if not data_dicts:
        print("No matches found. Try increasing tolerance.")
        return

    print(f"Found {len(data_dicts)} matches. Filtering top {max_lines_to_plot} for readability...")

    # 1. Sort data by error (best first)
    data_dicts.sort(key=lambda x: x["error"])

    # 2. Filter top N results to reduce line density
    filtered_data = data_dicts[:max_lines_to_plot]
    filtered_data.sort(key=lambda x: x["error"], reverse=True)

    keys = ["peak_tflops", "sm_num", "l1_size"]
    axis_labels = ["Peak TFLOPS", "Number of SMs", "L1 Size (KB)"]

    # Extract raw data for normalization boundaries
    raw_matrix = np.array([[d[k] for k in keys] for d in data_dicts])
    mins = raw_matrix.min(axis=0)
    maxs = raw_matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    # Prepare filtered data for plotting
    plot_matrix = np.array([[d[k] for k in keys] for d in filtered_data])
    errors = np.array([d["error"] for d in filtered_data])

    # Normalize filtered data to [0, 1]
    norm_data = (plot_matrix - mins) / ranges

    # --- Setup Color Mapping (Log Scale) ---
    vmin = max(errors.min(), 1e-5)
    vmax = errors.max()

    # Ensure vmax > vmin (Crashes if range is flat or inverted)
    if vmax <= vmin:
        vmax = vmin + 1.0  # Create an artificial range if all errors are identical

    norm_err = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues_r
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_err)

    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Draw Gridlines and Ticks FIRST (behind lines) ---
    num_ticks = 6
    for j in range(3):  # For each vertical axis
        # Calculate tick values in data domain
        tick_vals = np.linspace(mins[j], maxs[j], num_ticks)
        # Calculate tick positions in normalized domain [0,1]
        tick_pos = np.linspace(0, 1, num_ticks)

        for i, pos in enumerate(tick_pos):
            # Draw faint gray horizontal line
            ax.axhline(pos, color='gray', alpha=0.2, linestyle=':', zorder=0)
            # Add text label next to the axis line
            val_str = f"{tick_vals[i]:.0f}"
            # Offset text slightly to the left of the axis line
            ax.text(j - 0.03, pos, val_str, ha='right', va='center', fontsize=9, color='gray')

    # --- Plot the Data Lines ---
    for i in range(len(norm_data)):
        # Get color based on log-error
        color = cmap(norm_err(errors[i]))
        # Plot line with slight transparency
        ax.plot(range(3), norm_data[i], color=color, alpha=0.5, linewidth=1.5, zorder=1)

    # --- Setup Axes Structure ---
    ax.set_xticks(range(3))
    # Move axis labels higher up
    ax.set_xticklabels(axis_labels, fontsize=12, fontweight='bold', y=1.02)
    ax.set_yticks([])  # Hide standard y-axis

    # Draw main vertical axis lines
    for i in range(3):
        ax.axvline(i, color='black', lw=2, zorder=2)

    # --- Add Colorbar Legend ---
    cbar = plt.colorbar(sm, ax=ax, pad=0.05, aspect=30)
    cbar.set_label('Curve Mismatch Error (Generational Distance) - Log Scale',
                   rotation=270, labelpad=20, fontsize=11)

    plt.title(f"Inverse Design Solution Space\n(Top {len(norm_data)} configurations shown)", fontsize=15, pad=25)
    plt.tight_layout()
    plt.show()


def verify_best_match(target_pareto, best_config, simulator_template):
    print(f"\nVerifying Best Match: Error = {best_config['error']:.4f}")
    print(f"Specs: {best_config['peak_tflops']:.1f} TFLOPS, "
          f"{best_config['sm_num']} SMs, {best_config['l1_size']} KB L1")

    # 1. Re-create the Best GPU
    best_gpu = GPU(
        efficiency_factor=0.5,
        t_launch=0.01,
        sm_l1_size=best_config['l1_size'],
        number_sm=best_config['sm_num'],
        number_dies=2,
        peak_flops_dict={"fp32": best_config['peak_tflops']},
        hbi_bw=10 * (2 ** 30),
        hbi_launch=1e-7
    )

    # 2. Run Simulator
    simulator_template.gpu = best_gpu
    _, best_points = simulator_template.solve_pareto_brute_force()

    # Extract X (Time) and Y (Throughput)
    target_x = target_pareto[:, 2]
    target_y = target_pareto[:, 3]

    best_x = best_points[:, 2]
    best_y = best_points[:, 3]

    # 3. Plot Comparison
    plt.figure(figsize=(10, 6))

    # Plot Ground Truth
    plt.plot(target_x, target_y, 'ko-', label='Ground Truth (Target)',
             linewidth=2, markersize=5, alpha=0.8)

    # Plot Best Candidate
    plt.plot(best_x, best_y, 'r*--', label=f'Best AI Found (Err: {best_config["error"]:.3f})',
             linewidth=2, markersize=8)

    plt.xlabel("Step Time (s)", fontweight='bold')
    plt.ylabel("Throughput (samples/s)", fontweight='bold')
    plt.title("Sanity Check: Target vs. Best Inverse Design Result")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
# ==============================================================================
# 4. Run
# ==============================================================================
if __name__ == "__main__":
    # Configuration
    GROUND_TRUTH_FILE = "ground_truth_pareto.npy"

    # 1. Setup Ground Truth
    nvlink = Topology("tree", 72, "nvlink", 0.003)
    ib = Topology("linear", 2, "infiniband", 0.01)

    # Check if we already have the data saved
    if os.path.exists(GROUND_TRUTH_FILE):
        print(f"1. Loading Target Pareto Curve from '{GROUND_TRUTH_FILE}'...")
        target_pareto = np.load(GROUND_TRUTH_FILE)
    else:
        print("1. Generating Target Pareto Curve (Ground Truth)...")

        # The "Secret" GPU (Ground Truth)
        secret_gpu = GPU(
            efficiency_factor=0.5,
            t_launch=0.01,
            sm_l1_size=192,
            number_sm=132,
            number_dies=2,
            peak_flops_dict={"fp32": 900},  # 900 TFLOPS
            hbi_bw=10 * (2 ** 30),
            hbi_launch=1e-7
        )

        sim = Simulator(nvlink, ib, secret_gpu, dim=4096, r=4, element_type="fp32")

        # Use SLOW mode for the Ground Truth to ensure it is high quality
        _, target_pareto = sim.solve_pareto_brute_force(fast_mode=False)

        # Save it for next time
        np.save(GROUND_TRUTH_FILE, target_pareto)
        print(f"Saved to '{GROUND_TRUTH_FILE}'")

    # 2. Solve Inverse Problem
    print("2. Starting Inverse Solver...")

    # Create dummy template GPU
    dummy_gpu = GPU(1.0, 0.01, 0, 0, 2, {"fp32": 0})
    solver_sim = Simulator(nvlink, ib, dummy_gpu, dim=4096, r=4, element_type="fp32")

    matches = solve_inverse_problem_curve_matching(target_pareto, solver_sim, num_samples=3000, tolerance=0.20)

    plot_results(matches, max_lines_to_plot=200)

    matches.sort(key=lambda x: x["error"])
    best_candidate = matches[0]
    verify_best_match(target_pareto, best_candidate, solver_sim)