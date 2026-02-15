import logging
import os
import sys

# ==========================================
# 1. Logging & Execution Guard
# ==========================================
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime').setLevel(logging.CRITICAL)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

try:
    from pareto_solver import Simulator, GPU, Topology, Blackwell
    from parallel_coordinate_plot import run_advanced_optimization
except ImportError:
    pass


# ==========================================
# 2. Main Application Function
# ==========================================
def main():
    st.set_page_config(page_title="Pareto Inverse Solver", layout="wide")
    st.title("🧩 GPU Hardware Inverse Design")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    # Helper for Floats
    def range_input(label, default_min, default_max, key_prefix, step=1.0):
        col1, col2 = st.sidebar.columns(2)
        min_val = col1.number_input(f"Min {label}", value=float(default_min), step=step, key=f"{key_prefix}_min")
        max_val = col2.number_input(f"Max {label}", value=float(default_max), step=step, key=f"{key_prefix}_max")
        return [min_val, max_val]

    # Helper for Integers
    def int_range_input(label, default_min, default_max, key_prefix):
        col1, col2 = st.sidebar.columns(2)
        min_val = col1.number_input(f"Min {label}", value=int(default_min), step=1, format="%d",
                                    key=f"{key_prefix}_min")
        max_val = col2.number_input(f"Max {label}", value=int(default_max), step=1, format="%d",
                                    key=f"{key_prefix}_max")
        return [min_val, max_val]

    st.sidebar.header("1. GPU Search Space")
    bounds = {}
    bounds['flops'] = range_input("TFLOPS", 100.0, 1000.0, "flops")
    bounds['sm'] = int_range_input("SM Count", 50, 200, "sm")
    bounds['l1'] = int_range_input("L1 Size (KB)", 64, 512, "l1")
    bounds['dies'] = int_range_input("Dies", 1, 2, "dies")

    st.sidebar.divider()
    st.sidebar.header("2. Topology Search Space")
    st.sidebar.caption("Ranges for Topology optimization")

    bounds['inner_size'] = int_range_input("GPUs per Tree (Inner)", 32, 72, "inner_sz")
    bounds['outer_size'] = int_range_input("Num Trees (Outer)", 1, 8, "outer_sz")

    st.sidebar.markdown("---")
    st.sidebar.caption("Bandwidths (TB/s)")
    bounds['hbi_bw'] = range_input("HBI BW", 5.0, 15.0, "hbi", step=0.5)
    bounds['nvlink_bw'] = range_input("NVLink BW", 0.9, 3.6, "nvlink", step=0.1)
    bounds['ib_bw'] = range_input("Infiniband BW", 0.4, 1.6, "ib", step=0.1)

    if st.sidebar.button("❌ Stop App"):
        st.stop()

    # ------------------------------------------------------------------
    # Section 1: Target Curve (Ground Truth)
    # ------------------------------------------------------------------
    st.subheader("2. Define Target Ground Truth")

    if "target_df" not in st.session_state:
        if os.path.exists("ground_truth_pymoo.npy"):
            data = np.load("ground_truth_pymoo.npy")
            df = pd.DataFrame(data[:, 2:4], columns=["Latency (s)", "Throughput"])
            st.session_state.target_df = df.sort_values(by="Latency (s)").reset_index(drop=True)
        else:
            st.error("ground_truth_pymoo.npy not found.")
            st.stop()

    col_controls, col_plot = st.columns([1, 3])

    with col_controls:
        st.markdown("**Curve Modifiers**")
        scale_lat = st.number_input("Scale Latency (x)", 0.1, 5.0, 1.0, 0.1)
        scale_thr = st.number_input("Scale Throughput (y)", 0.1, 5.0, 1.0, 0.1)

        if st.button("Apply Scaling"):
            df_copy = st.session_state.target_df.copy()
            df_copy["Latency (s)"] = df_copy["Latency (s)"] * scale_lat
            df_copy["Throughput"] = df_copy["Throughput"] * scale_thr
            st.session_state.target_df = df_copy.sort_values(by="Latency (s)").reset_index(drop=True)
            st.rerun()

        if st.button("Reset to Default"):
            if os.path.exists("ground_truth_pymoo.npy"):
                data = np.load("ground_truth_pymoo.npy")
                df = pd.DataFrame(data[:, 2:4], columns=["Latency (s)", "Throughput"])
                st.session_state.target_df = df.sort_values(by="Latency (s)").reset_index(drop=True)
                st.rerun()

    with col_plot:
        fig = px.line(st.session_state.target_df, x="Latency (s)", y="Throughput", markers=True)
        fig.update_traces(line_color="#00CC96", marker_size=10)
        fig.update_layout(clickmode='event+select', dragmode='zoom')

        gt_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="gt_plot")

        if "gt_selected_idx" not in st.session_state:
            st.session_state.gt_selected_idx = None

        if gt_event and gt_event.selection and "points" in gt_event.selection:
            points = gt_event.selection["points"]
            if points:
                st.session_state.gt_selected_idx = points[0]["point_index"]

        editor_display_df = st.session_state.target_df.copy()

        if st.session_state.gt_selected_idx is not None:
            sel_idx = st.session_state.gt_selected_idx
            if sel_idx < len(editor_display_df):
                st.info(f"📍 **Editing Point #{sel_idx}:** Moving to top of table.")
                selected_row = editor_display_df.iloc[[sel_idx]]
                rest_rows = editor_display_df.drop(editor_display_df.index[sel_idx])
                editor_display_df = pd.concat([selected_row, rest_rows])
            else:
                st.session_state.gt_selected_idx = None

        edited_df = st.data_editor(
            editor_display_df,
            height=200,
            use_container_width=True,
            num_rows="dynamic",
            key="gt_editor"
        )

        sorted_edited = edited_df.sort_values(by="Latency (s)").reset_index(drop=True)
        if not sorted_edited.equals(st.session_state.target_df):
            st.session_state.target_df = sorted_edited
            st.rerun()

    # ------------------------------------------------------------------
    # Section 2: Run Solver
    # ------------------------------------------------------------------
    st.subheader("3. Run Inverse Solver")

    if st.button("🚀 Solve Inverse Problem", type="primary"):
        target_points = st.session_state.target_df.values
        full_target = np.zeros((len(target_points), 4))
        full_target[:, 2:4] = target_points

        # NOTE: Bounds now contain topology info.
        # We pass dummy topology objects to simulator just to set 'kind', 'type', 'alpha'.
        # The ACTUAL sizes and bandwidths will be determined by the optimizer.
        nvlink = Topology("tree", 1, "nvlink", 0.003)
        ib = Topology("linear", 1, "infiniband", 0.01)
        dummy_gpu = GPU(0.01, 0, 0, 2, {"fp32": 0})

        solver_sim = Simulator(nvlink, ib, dummy_gpu, dim=4096, r=4, element_type="fp32")

        prog_bar = st.progress(0, text="Initializing...")

        def update_ui_progress(p, msg):
            prog_bar.progress(p / 100.0, text=msg)

        try:
            results = run_advanced_optimization(
                full_target,
                solver_sim,
                bounds,
                progress_callback=update_ui_progress
            )
            prog_bar.progress(1.0, text="Optimization Complete!")
            st.session_state.results = results
            st.success(f"Found {len(results)} candidates.")
        except Exception as e:
            st.error(f"Optimization failed: {e}")

    # ------------------------------------------------------------------
    # Section 3: Results Analysis
    # ------------------------------------------------------------------
    if "results" in st.session_state:
        st.divider()
        st.subheader("4. Results Analysis")

        raw_df = pd.DataFrame(st.session_state.results)

        # --- A. Result Filtering ---
        # Note: We now have MANY parameters. Simplified filtering for key ones.
        f_col1, f_col2, f_col3 = st.columns(3)
        min_err = float(raw_df['error'].min())
        max_err = float(raw_df['error'].max())

        err_range = f_col1.slider("Max Error", min_err, max_err, max_err)
        l1_range = f_col2.slider("L1 Size (KB)", int(raw_df['l1_size'].min()), int(raw_df['l1_size'].max()),
                                 (int(raw_df['l1_size'].min()), int(raw_df['l1_size'].max())))
        nvlink_range = f_col3.slider("NVLink BW (TB/s)", float(raw_df['nvlink_bw'].min()),
                                     float(raw_df['nvlink_bw'].max()),
                                     (float(raw_df['nvlink_bw'].min()), float(raw_df['nvlink_bw'].max())))

        filtered_df = raw_df[
            (raw_df['error'] <= err_range) &
            (raw_df['l1_size'] >= l1_range[0]) & (raw_df['l1_size'] <= l1_range[1]) &
            (raw_df['nvlink_bw'] >= nvlink_range[0]) & (raw_df['nvlink_bw'] <= nvlink_range[1])
            ].reset_index(drop=True)

        st.caption(f"Showing {len(filtered_df)} of {len(raw_df)} solutions")

        if filtered_df.empty:
            st.warning("No results match your filters.")
        else:
            # --- B. Detailed Table ---
            st.markdown("### Solution Table")

            selected_idx = None

            # Format ALL columns properly
            formatted_df = filtered_df[
                ["peak_tflops", "sm_num", "l1_size", "dies", "inner_size", "outer_size", "hbi_bw", "nvlink_bw", "ib_bw",
                 "error"]].style.format({
                "peak_tflops": "{:.0f}",
                "sm_num": "{:.0f}",
                "l1_size": "{:.0f}",
                "dies": "{:.0f}",
                "inner_size": "{:.0f}",
                "outer_size": "{:.0f}",
                "hbi_bw": "{:.2f}",
                "nvlink_bw": "{:.2f}",
                "ib_bw": "{:.2f}",
                "error": "{:.8f}"
            })

            table_event = st.dataframe(
                formatted_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                height=350,
                key="table_sel"
            )

            if table_event and table_event.selection and "rows" in table_event.selection:
                rows = table_event.selection["rows"]
                if rows:
                    selected_idx = rows[0]

            # --- C. Highlight Logic ---
            line_color = filtered_df['error']
            line_colorscale = 'Spectral_r'

            if selected_idx is not None:
                if selected_idx < len(filtered_df):
                    row = filtered_df.iloc[selected_idx]
                    st.toast(f"Selected: {row['inner_size']:.0f} GPUs/Tree, {row['nvlink_bw']:.2f} TB/s NVLink")
                    color_mask = np.zeros(len(filtered_df))
                    color_mask[selected_idx] = 1
                    line_color = color_mask
                    line_colorscale = [[0.0, 'rgba(200,200,200,0.1)'], [1.0, '#FF0055']]

            # --- D. Parallel Coordinates Plot (Expanded) ---
            st.markdown("### Parallel Coordinate Paths")
            parcoords = go.Figure(data=go.Parcoords(
                line=dict(
                    color=line_color,
                    colorscale=line_colorscale,
                    showscale=(selected_idx is None),
                    cmin=0,
                    cmax=1 if selected_idx is not None else filtered_df['error'].max(),
                    colorbar={'title': 'Error'} if selected_idx is None else None
                ),
                dimensions=[
                    dict(label='TFLOPS', values=filtered_df['peak_tflops']),
                    dict(label='SMs', values=filtered_df['sm_num']),
                    dict(label='L1 (KB)', values=filtered_df['l1_size']),
                    dict(label='GPUs/Tree', values=filtered_df['inner_size']),
                    dict(label='Num Trees', values=filtered_df['outer_size']),
                    dict(label='NVLink BW', values=filtered_df['nvlink_bw']),
                    dict(label='IB BW', values=filtered_df['ib_bw']),
                    dict(label='Error', values=filtered_df['error'])
                ]
            ))

            parcoords.update_layout(height=550, margin=dict(l=40, r=40, t=50, b=50))
            st.plotly_chart(parcoords, use_container_width=True)


if __name__ == "__main__":
    main()