import logging
import os
import sys
import time

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
    def range_input(label, default_min, default_max, key_prefix, step=1.0):
        col1, col2 = st.sidebar.columns(2)
        min_val = col1.number_input(f"Min {label}", value=float(default_min), step=step, key=f"{key_prefix}_min")
        max_val = col2.number_input(f"Max {label}", value=float(default_max), step=step, key=f"{key_prefix}_max")
        return [min_val, max_val]

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
    bounds['inner_size'] = int_range_input("GPUs per Tree (Inner)", 32, 72, "inner_sz")
    bounds['outer_size'] = int_range_input("Num Trees (Outer)", 1, 8, "outer_sz")

    st.sidebar.markdown("---")
    st.sidebar.caption("Bandwidths (TB/s)")
    # UPDATED DEFAULT RANGES
    bounds['hbi_bw'] = range_input("HBI BW", 5.0, 10.0, "hbi", step=0.5)
    bounds['nvlink_bw'] = range_input("NVLink BW", 0.9, 1.2, "nvlink", step=0.1)
    bounds['ib_bw'] = range_input("Infiniband BW", 0.3, 0.5, "ib", step=0.1)

    if st.sidebar.button("❌ Stop App"):
        st.stop()

    # ------------------------------------------------------------------
    # Section 1: Ground Truth
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

    # Time limit input
    time_limit = st.number_input("Max Time Limit (seconds) [0 = No Limit]", min_value=0, value=0, step=10,
                                 help="Stop the solver after this many seconds and show current best results.")

    if st.button("🚀 Solve Inverse Problem", type="primary"):
        target_points = st.session_state.target_df.values
        full_target = np.zeros((len(target_points), 4))
        full_target[:, 2:4] = target_points

        nvlink = Topology("tree", 1, "nvlink", 0.003)
        ib = Topology("linear", 1, "infiniband", 0.01)
        dummy_gpu = GPU(0.01, 0, 0, 2, {"fp32": 0})
        solver_sim = Simulator(nvlink, ib, dummy_gpu, dim=4096, r=4, element_type="fp32")

        # --- LIVE STATUS CONTAINERS ---
        status_cols = st.columns(4)
        prog_bar = st.progress(0, text="Initializing...")

        metric_time = status_cols[0].empty()
        metric_min_err = status_cols[1].empty()
        metric_avg_err = status_cols[2].empty()

        st.markdown("**Live Top 50 Results**")
        live_table = st.empty()

        def update_ui_progress(data):
            if "status" in data:
                prog_bar.progress(1.0 if data["status"] == "complete" else 0.5, text=data["msg"])
                return

            p = data["progress"]
            gen = data["gen"]
            total = data["total_gen"]

            prog_bar.progress(p, text=f"Optimization Generation {gen}/{total}")

            metric_time.metric("Est. Time Left", f"{data['eta']:.1f}s")
            metric_min_err.metric("Min Error", f"{data['min_error']:.4f}")
            metric_avg_err.metric("Avg Error", f"{data['avg_error']:.4f}")

            if "top_candidates" in data:
                live_df = pd.DataFrame(data["top_candidates"])
                live_table.dataframe(live_df[["peak_tflops", "sm_num", "l1_size", "dies", "inner_size", "outer_size",
                                              "hbi_bw", "nvlink_bw", "ib_bw", "error"]], height=200)

        try:
            results = run_advanced_optimization(
                full_target,
                solver_sim,
                bounds,
                hbi_bw_val=10.0,  # Default or from param if needed
                time_limit=time_limit,
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

        # Filters
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

        col_main, col_detail = st.columns([2, 1])

        with col_main:
            # --- Detailed Table ---
            st.markdown("### Solution Table")

            if "selected_idx_state" not in st.session_state:
                st.session_state.selected_idx_state = None

            display_cols = ["peak_tflops", "sm_num", "l1_size", "dies", "inner_size", "outer_size", "hbi_bw",
                            "nvlink_bw", "ib_bw", "error"]

            formatted_df = filtered_df[display_cols].style.format({
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
                height=300,
                key="table_sel"
            )

            if table_event and table_event.selection and "rows" in table_event.selection:
                rows = table_event.selection["rows"]
                if rows:
                    st.session_state.selected_idx_state = rows[0]

            selected_idx = st.session_state.selected_idx_state

            # --- Parallel Coordinates Plot ---
            st.markdown("### Parallel Coordinate Paths")

            line_color = filtered_df['error']
            line_colorscale = 'Spectral_r'
            if selected_idx is not None and selected_idx < len(filtered_df):
                color_mask = np.zeros(len(filtered_df))
                color_mask[selected_idx] = 1
                line_color = color_mask
                line_colorscale = [[0.0, 'rgba(200,200,200,0.1)'], [1.0, '#007BFF']]

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
                    dict(label='Dies', values=filtered_df['dies']),
                    dict(label='GPUs/Tree', values=filtered_df['inner_size']),
                    dict(label='Num Trees', values=filtered_df['outer_size']),
                    dict(label='HBI BW', values=filtered_df['hbi_bw']),
                    dict(label='NVLink BW', values=filtered_df['nvlink_bw']),
                    dict(label='IB BW', values=filtered_df['ib_bw']),
                    dict(label='Error', values=filtered_df['error'])
                ]
            ))
            # Increased top margin to 100 to show column names
            parcoords.update_layout(height=500, margin=dict(l=50, r=50, t=100, b=50))
            st.plotly_chart(parcoords, use_container_width=True)

        with col_detail:
            st.markdown("### 🔍 Path Analysis")
            if selected_idx is not None and selected_idx < len(filtered_df):
                row = filtered_df.iloc[selected_idx]

                with st.spinner("Generating curve..."):
                    sel_gpu = GPU(
                        t_launch=0.01,
                        sm_l1_size=row['l1_size'],
                        number_sm=row['sm_num'],
                        number_dies=row['dies'],
                        peak_flops_dict={"fp32": row['peak_tflops']},
                        hbi_bw=row['hbi_bw'] * (2 ** 30),
                        hbi_launch=1e-7
                    )
                    sel_nvlink = Topology("tree", row['inner_size'], "nvlink", 0.003, bandwidth=row['nvlink_bw'])
                    sel_ib = Topology("linear", row['outer_size'], "infiniband", 0.01, bandwidth=row['ib_bw'])

                    sel_sim = Simulator(sel_nvlink, sel_ib, sel_gpu, dim=4096, r=4, element_type="fp32")
                    sel_curve = sel_sim.solve_pareto_pymoo(fast_mode=True)

                if len(sel_curve) > 0:
                    fig_comp = go.Figure()

                    fig_comp.add_trace(go.Scatter(
                        x=st.session_state.target_df["Latency (s)"],
                        y=st.session_state.target_df["Throughput"],
                        mode='lines+markers',
                        name='Ground Truth',
                        line=dict(color='lightgrey', width=2)
                    ))

                    sel_curve = sel_curve[sel_curve[:, 2].argsort()]
                    fig_comp.add_trace(go.Scatter(
                        x=sel_curve[:, 2],
                        y=sel_curve[:, 3],
                        mode='lines+markers',
                        name='Selected Solution',
                        line=dict(color='#007BFF', width=3)
                    ))

                    fig_comp.update_layout(
                        title="Performance Comparison",
                        xaxis_title="Latency (s)",
                        yaxis_title="Throughput",
                        legend=dict(
                            yanchor="bottom",
                            y=0.01,
                            xanchor="right",
                            x=0.99
                        ),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=400
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

                    st.success(f"Config Error: {row['error']:.6f}")
                else:
                    st.warning("Could not generate curve for this config.")
            else:
                st.info("Select a row in the table to see its specific Pareto Curve compared to the Ground Truth.")


if __name__ == "__main__":
    main()