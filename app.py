import logging
import os
import sys
import time
import signal
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

os.environ['STREAMLIT_LOG_LEVEL'] = 'error'
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime').setLevel(logging.CRITICAL)

from pareto_solver import Simulator, GPU, Topology
from parallel_coordinate_plot import run_advanced_optimization


def main():
    st.set_page_config(page_title="Pareto Inverse Solver", layout="wide")
    st.title("🧩 GPU Hardware Inverse Design")

    # ------------------------------------------------------------------
    # Sidebar Bounds & Stop Button
    # ------------------------------------------------------------------
    def range_input(label, default_min, default_max, key_prefix, step=1.0):
        col1, col2 = st.sidebar.columns(2)
        min_v = col1.number_input(f"Min {label}", value=float(default_min), step=step, key=f"{key_prefix}_min")
        max_v = col2.number_input(f"Max {label}", value=float(default_max), step=step, key=f"{key_prefix}_max")
        return [min_v, max_v]

    st.sidebar.header("1. Search Space Boundaries")

    bounds = {
        'flops': range_input("Peak Perf. (TFLOPS)", 100.0, 1000.0, "flops", step=10.0),
        'sm': range_input("SM Count", 50, 200, "sm", step=1.0),
        'l1': range_input("L1 Cache (KB)", 64, 512, "l1", step=8.0),
        'dies': range_input("Dies per GPU", 1, 2, "dies", step=1.0),
        'inner_size': range_input("GPUs per Tree", 32, 72, "inner_sz", step=1.0),
        'outer_size': range_input("Number of Trees", 1, 8, "outer_sz", step=1.0),
        'hbi_bw': range_input("HBI BW (TB/s)", 5.0, 10.0, "hbi", step=0.5),
        'nvlink_bw': range_input("NVLink BW (TB/s)", 0.9, 1.2, "nvlink", step=0.1),
        'ib_bw': range_input("Infiniband BW (TB/s)", 0.3, 0.5, "ib", step=0.1)
    }

    st.sidebar.divider()
    if st.sidebar.button("❌ Stop App", use_container_width=True, type="primary"):
        st.warning("Shutting down server...")
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    # ------------------------------------------------------------------
    # Section 2: Ground Truth
    # ------------------------------------------------------------------
    st.subheader("2. Define Target Ground Truth (Cap: 8s)")

    if "target_df" not in st.session_state:
        if os.path.exists("ground_truth_pymoo.npy"):
            data = np.load("ground_truth_pymoo.npy")
            df = pd.DataFrame(data[:, 2:4], columns=["Latency (s)", "Throughput"])
            st.session_state.target_df = df[df["Latency (s)"] <= 8.0].sort_values(by="Latency (s)").reset_index(
                drop=True)
        else:
            st.error("ground_truth_pymoo.npy not found.")
            st.stop()

    col_ctrl, col_plt = st.columns([1, 3])

    with col_plt:
        fig_gt = px.line(st.session_state.target_df, x="Latency (s)", y="Throughput", markers=True)
        fig_gt.update_traces(line_color="#00CC96", marker_size=10, selector=dict(type='scatter'))
        fig_gt.update_layout(clickmode='event+select', height=400)
        gt_event = st.plotly_chart(fig_gt, use_container_width=True, on_select="rerun", key="gt_graph_main")

    with col_ctrl:
        selected_point_idx = None
        if gt_event and gt_event.selection and "points" in gt_event.selection:
            points = gt_event.selection["points"]
            if points:
                selected_point_idx = points[0]["point_index"]

        if selected_point_idx is not None:
            st.markdown(f"### 📍 Edit Point #{selected_point_idx}")
            row = st.session_state.target_df.iloc[selected_point_idx]
            new_lat = st.number_input("Latency (s)", value=float(row["Latency (s)"]), format="%.4f", key="edit_lat")
            new_thr = st.number_input("Throughput", value=float(row["Throughput"]), format="%.2f", key="edit_thr")

            if st.button("Update Point", type="primary", use_container_width=True):
                st.session_state.target_df.at[selected_point_idx, "Latency (s)"] = new_lat
                st.session_state.target_df.at[selected_point_idx, "Throughput"] = new_thr
                st.session_state.target_df = st.session_state.target_df[
                    st.session_state.target_df["Latency (s)"] <= 8.0].sort_values(by="Latency (s)").reset_index(
                    drop=True)
                st.rerun()
        else:
            st.info("Click a point on the graph to edit coordinates.")

        st.divider()
        st.markdown("### ✂️ Region Scaling")
        min_l, max_l = float(st.session_state.target_df["Latency (s)"].min()), float(
            st.session_state.target_df["Latency (s)"].max())
        scale_range = st.slider("Select Region", min_l, max_l, (min_l, max_l), key="range_slider")
        sc_lat = st.number_input("Lat Scale (Multiplier)", 0.1, 5.0, 1.0, 0.1, key="sc_lat_input")
        sc_thr = st.number_input("Thr Scale (Multiplier)", 0.1, 5.0, 1.0, 0.1, key="sc_thr_input")

        if st.button("Apply Region Scaling"):
            df = st.session_state.target_df.copy()
            mask = (df["Latency (s)"] >= scale_range[0]) & (df["Latency (s)"] <= scale_range[1])
            df.loc[mask, "Latency (s)"] *= sc_lat
            df.loc[mask, "Throughput"] *= sc_thr
            st.session_state.target_df = df[df["Latency (s)"] <= 8.0].sort_values(by="Latency (s)").reset_index(
                drop=True)
            st.rerun()

        if st.button("Reset Curve (8s Cap)", use_container_width=True):
            data = np.load("ground_truth_pymoo.npy")
            df = pd.DataFrame(data[:, 2:4], columns=["Latency (s)", "Throughput"])
            st.session_state.target_df = df[df["Latency (s)"] <= 8.0].sort_values(by="Latency (s)").reset_index(
                drop=True)
            st.rerun()

    # ------------------------------------------------------------------
    # Section 3: Run Solver
    # ------------------------------------------------------------------
    st.subheader("3. Run Inverse Solver")
    t_limit = st.number_input("Time Limit (s)", 0, 1000, 0, key="solver_time_limit")

    if st.button("🚀 Solve Inverse Problem", type="primary"):
        target_pts = st.session_state.target_df.values
        full_target = np.zeros((len(target_pts), 4))
        full_target[:, 2:4] = target_pts

        prog_bar = st.progress(0)

        metrics_placeholder = st.empty()
        live_graph_placeholder = st.empty()

        def update_ui_progress(data):
            if "status" in data: return
            prog_bar.progress(data["progress"], text=f"Gen {data['gen']} - Err: {data['min_error']:.2f}/100")

            with metrics_placeholder.container():
                m_cols = st.columns(3)
                m_cols[0].metric("ETA", f"{data['eta']:.1f}s")
                m_cols[1].metric("Best Error", f"{data['min_error']:.2f}")

            if "top_candidates" in data and len(data['top_candidates']) > 0:
                best = data['top_candidates'][0]
                tgpu = GPU(0.01, best['l1_size'], best['sm_num'], best['dies'], {"fp32": best['peak_tflops']},
                           hbi_bw=best['hbi_bw'] * (2 ** 30))
                tsim = Simulator(Topology("tree", best['inner_size'], "nvlink", 0.003, best['nvlink_bw']),
                                 Topology("linear", best['outer_size'], "infiniband", 0.01, best['ib_bw']),
                                 tgpu, 4096, 4, "fp32")

                curve = tsim.estimate_pareto_analytical(latency_cap=8.0)

                fig_l = go.Figure()
                fig_l.add_trace(
                    go.Scatter(x=st.session_state.target_df["Latency (s)"], y=st.session_state.target_df["Throughput"],
                               name="Target", line=dict(color="grey", dash="dot")))
                if len(curve) > 0:
                    fig_l.add_trace(
                        go.Scatter(x=curve[:, 2], y=curve[:, 3], name="Best Live", line=dict(color="#007BFF", width=3)))

                fig_l.update_layout(title=f"Evolution (Gen {data['gen']})", height=450)
                live_graph_placeholder.plotly_chart(fig_l, use_container_width=True,
                                                    key=f"live_chart_gen_{data['gen']}")

        results = run_advanced_optimization(full_target, Simulator(Topology("tree", 1, "nvlink", 0.003),
                                                                   Topology("linear", 1, "infiniband", 0.01),
                                                                   GPU(0.01, 0, 0, 2, {"fp32": 0}), 4096, 4, "fp32"),
                                            bounds, time_limit=t_limit, progress_callback=update_ui_progress)
        st.session_state.results = results
        st.rerun()

    # ------------------------------------------------------------------
    # Section 4: Results Analysis
    # ------------------------------------------------------------------
    if "results" in st.session_state:
        st.divider()
        st.write("### 4. Final Analysis")
        raw_df = pd.DataFrame(st.session_state.results)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Solution Table")
            display_cols = ["peak_tflops", "sm_num", "l1_size", "dies", "inner_size", "outer_size", "hbi_bw",
                            "nvlink_bw", "ib_bw", "error"]

            # 1. Render Table First
            table_event = st.dataframe(raw_df[display_cols], use_container_width=True, on_select="rerun",
                                       selection_mode="single-row", hide_index=True, key="t_sel")

            # 2. Extract Selection Logic
            selected_idx = table_event.selection["rows"][0] if (
                        table_event and table_event.selection and table_event.selection["rows"]) else None

            # 3. Handle Color Masking for Parallel Coordinates
            if selected_idx is not None:
                line_color = np.zeros(len(raw_df))
                line_color[selected_idx] = 1  # Mark selected as 1
                line_colorscale = [[0.0, 'rgba(150, 150, 150, 0.2)'], [1.0, '#007BFF']]  # 0=Gray/Fade, 1=Blue Highlight
                showscale = False
            else:
                line_color = raw_df['error']
                line_colorscale = 'Spectral_r'
                showscale = True

            st.markdown("### Parallel Coordinates")
            parcoords = go.Figure(data=go.Parcoords(
                line=dict(color=line_color, colorscale=line_colorscale, showscale=showscale),
                dimensions=[
                    dict(label='TFLOPS', values=raw_df['peak_tflops']),
                    dict(label='SMs', values=raw_df['sm_num']),
                    dict(label='L1 (KB)', values=raw_df['l1_size']),
                    dict(label='Dies', values=raw_df['dies']),
                    dict(label='GPUs/Tree', values=raw_df['inner_size']),
                    dict(label='Trees', values=raw_df['outer_size']),
                    dict(label='HBI BW', values=raw_df['hbi_bw']),
                    dict(label='NVLink', values=raw_df['nvlink_bw']),
                    dict(label='IB BW', values=raw_df['ib_bw']),
                    dict(label='Error', values=raw_df['error'])
                ]
            ))
            # Fixed the cut-off text by pushing the top margin down
            parcoords.update_layout(height=450, margin=dict(l=40, r=40, t=80, b=40))
            st.plotly_chart(parcoords, use_container_width=True, key="final_pcp_chart")

        with c2:
            st.markdown("### 🔍 Path Analysis")
            if selected_idx is not None:
                row = raw_df.iloc[selected_idx]
                with st.spinner("Generating curve..."):
                    sel_gpu = GPU(0.01, row['l1_size'], row['sm_num'], row['dies'], {"fp32": row['peak_tflops']},
                                  hbi_bw=row['hbi_bw'] * (2 ** 30))
                    sel_sim = Simulator(Topology("tree", row['inner_size'], "nvlink", 0.003, row['nvlink_bw']),
                                        Topology("linear", row['outer_size'], "infiniband", 0.01, row['ib_bw']),
                                        sel_gpu, 4096, 4, "fp32")
                    sel_curve = sel_sim.solve_pareto_pymoo(fast_mode=False, latency_cap=8.0)

                if len(sel_curve) > 0:
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(x=st.session_state.target_df["Latency (s)"],
                                                  y=st.session_state.target_df["Throughput"], mode='lines+markers',
                                                  name='Ground Truth', line=dict(color='lightgrey', width=2)))
                    fig_comp.add_trace(
                        go.Scatter(x=sel_curve[:, 2], y=sel_curve[:, 3], mode='lines+markers', name='Selected Solution',
                                   line=dict(color='#007BFF', width=3)))
                    fig_comp.update_layout(title="Performance Comparison", xaxis_title="Latency (s)",
                                           yaxis_title="Throughput",
                                           legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                                           margin=dict(l=20, r=20, t=40, b=20), height=400)
                    st.plotly_chart(fig_comp, use_container_width=True, key="path_analysis_chart")
                    st.success(f"Config Error: {row['error']:.2f}")
                else:
                    st.warning("Could not generate curve for this config within 8s latency.")
            else:
                st.info("Select a row in the table to see its Pareto Curve compared to the Ground Truth.")


if __name__ == "__main__":
    main()