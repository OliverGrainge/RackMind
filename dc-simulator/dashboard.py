"""
Data Centre Simulator Dashboard

Professional monitoring dashboard for the DC simulator.
Visualizes thermal, power, carbon, workload, and failure state.
Requires the API server to be running: uvicorn dc_sim.main:app
"""

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"

# --- Colour constants ---
COLOUR_GREEN = "#22c55e"
COLOUR_AMBER = "#f59e0b"
COLOUR_RED = "#ef4444"
COLOUR_BLUE = "#3b82f6"
COLOUR_PURPLE = "#8b5cf6"
COLOUR_TEAL = "#14b8a6"
COLOUR_SLATE = "#64748b"
COLOUR_BG_DARK = "#0f172a"
COLOUR_BG_CARD = "#1e293b"


def fetch_json(path: str, base_url: str) -> dict | None:
    """Fetch JSON from API. Returns None on failure."""
    try:
        r = httpx.get(f"{base_url}{path}", timeout=2.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_json(path: str, base_url: str, json: dict | None = None) -> dict | None:
    """POST to API. Returns response JSON or None."""
    try:
        r = httpx.post(f"{base_url}{path}", json=json or {}, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def make_gauge(value: float, title: str, max_val: float, unit: str = "", thresholds: list | None = None) -> go.Figure:
    """Create a compact gauge chart."""
    if thresholds is None:
        thresholds = [0.6, 0.85]  # fractions of max

    colour = COLOUR_GREEN
    frac = value / max_val if max_val > 0 else 0
    if frac > thresholds[1]:
        colour = COLOUR_RED
    elif frac > thresholds[0]:
        colour = COLOUR_AMBER

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 20, "color": "white"}},
        title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#475569", "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": colour, "thickness": 0.7},
            "bgcolor": "#334155",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_val * thresholds[0]], "color": "rgba(34,197,94,0.1)"},
                {"range": [max_val * thresholds[0], max_val * thresholds[1]], "color": "rgba(245,158,11,0.1)"},
                {"range": [max_val * thresholds[1], max_val], "color": "rgba(239,68,68,0.1)"},
            ],
        },
    ))
    fig.update_layout(
        height=160, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
    )
    return fig


def make_thermal_heatmap(thermal_racks: list) -> go.Figure:
    """Create a heatmap of rack inlet temperatures."""
    rack_ids = [f"Rack {r['rack_id']}" for r in thermal_racks]
    temps = [r["inlet_temp_c"] for r in thermal_racks]
    humidity = [r.get("humidity_pct", 45) for r in thermal_racks]

    fig = go.Figure()

    # Temperature bars
    fig.add_trace(go.Bar(
        x=rack_ids, y=temps, name="Inlet Temp",
        marker=dict(
            color=temps,
            colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
            cmin=18, cmax=45,
            colorbar=dict(title="Temp", ticksuffix=" C", len=0.5, y=0.75),
        ),
        text=[f"{t:.1f}C" for t in temps],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))

    # Throttle line
    fig.add_hline(y=40, line_dash="dash", line_color=COLOUR_RED, opacity=0.7,
                  annotation_text="Throttle (40C)", annotation_font_color=COLOUR_RED)
    fig.add_hline(y=35, line_dash="dot", line_color=COLOUR_AMBER, opacity=0.5,
                  annotation_text="Warning (35C)", annotation_font_color=COLOUR_AMBER)

    fig.update_layout(
        height=300, margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, yaxis_title="Inlet Temp (C)",
        yaxis=dict(gridcolor="rgba(100,116,139,0.2)", range=[15, 50]),
        xaxis=dict(gridcolor="rgba(100,116,139,0.2)"),
        showlegend=False,
    )
    return fig


def make_power_breakdown(power_racks: list) -> go.Figure:
    """Stacked bar chart of power per rack."""
    rack_ids = [f"Rack {r['rack_id']}" for r in power_racks]
    power_kw = [r["total_power_kw"] for r in power_racks]

    fig = go.Figure(go.Bar(
        x=rack_ids, y=power_kw,
        marker_color=COLOUR_BLUE,
        text=[f"{p:.1f}" for p in power_kw],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, yaxis_title="Power (kW)",
        yaxis=dict(gridcolor="rgba(100,116,139,0.2)"),
        xaxis=dict(gridcolor="rgba(100,116,139,0.2)"),
    )
    return fig


def make_timeseries(df: pd.DataFrame, columns: list, title: str, ylabel: str, colours: list | None = None) -> go.Figure:
    """Create a multi-line time-series chart."""
    fig = go.Figure()
    default_colours = [COLOUR_BLUE, COLOUR_GREEN, COLOUR_AMBER, COLOUR_RED, COLOUR_PURPLE, COLOUR_TEAL, "#f472b6", "#a78bfa"]
    colours = colours or default_colours

    for i, col in enumerate(columns):
        fig.add_trace(go.Scatter(
            x=df["tick"], y=df[col], mode="lines",
            name=col.replace("_", " ").title(),
            line=dict(color=colours[i % len(colours)], width=2),
        ))

    fig.update_layout(
        height=300, margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, yaxis_title=ylabel,
        title=dict(text=title, font=dict(size=14, color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(100,116,139,0.2)"),
        xaxis=dict(gridcolor="rgba(100,116,139,0.2)", title="Tick"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


def main():
    st.set_page_config(
        page_title="DC Simulator",
        page_icon="DC",
        layout="wide",
    )

    # Custom CSS for dark theme and card styling
    st.markdown("""
    <style>
        .stApp { background-color: #0f172a; }
        .metric-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 16px 20px;
            border: 1px solid #334155;
        }
        .metric-value { font-size: 28px; font-weight: 700; color: white; }
        .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-delta { font-size: 13px; }
        .section-header {
            font-size: 18px; font-weight: 600; color: #e2e8f0;
            margin-bottom: 8px; padding-bottom: 4px;
            border-bottom: 2px solid #334155;
        }
        .status-ok { color: #22c55e; }
        .status-warn { color: #f59e0b; }
        .status-crit { color: #ef4444; }
        div[data-testid="stMetric"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 12px 16px;
        }
        div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 12px !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("### Settings")
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

    status = fetch_json("/status", api_url)
    if status is None:
        st.error(
            "Cannot connect to simulator API. Start with:\n\n"
            "```bash\nuvicorn dc_sim.main:app\n```"
        )
        return

    sim_status = fetch_json("/sim/status", api_url) or {}
    is_running = sim_status.get("running", False)

    # Header
    elapsed = status.get("current_time", 0)
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    tick_count = status.get("tick_count", 0)

    header_cols = st.columns([3, 1, 1, 1, 1, 1])
    with header_cols[0]:
        st.markdown(f"## Data Centre Simulator")
        st.caption(f"Sim time: **{h:02d}:{m:02d}:{s:02d}** | Tick: **{tick_count}** | {'Running' if is_running else 'Paused'}")

    # Controls in header row
    with header_cols[1]:
        if is_running:
            if st.button("Pause", use_container_width=True):
                post_json("/sim/pause", api_url)
                st.rerun()
        else:
            if st.button("Play", use_container_width=True):
                post_json("/sim/run?tick_interval_s=0.5", api_url)
                st.rerun()
    with header_cols[2]:
        if st.button("+1 Tick", use_container_width=True):
            post_json("/sim/tick?n=1", api_url)
            st.rerun()
    with header_cols[3]:
        if st.button("+10 Ticks", use_container_width=True):
            post_json("/sim/tick?n=10", api_url)
            st.rerun()
    with header_cols[4]:
        if st.button("+60 Ticks", use_container_width=True):
            post_json("/sim/tick?n=60", api_url)
            st.rerun()
    with header_cols[5]:
        if st.button("Reset", use_container_width=True):
            post_json("/sim/pause", api_url)
            post_json("/sim/reset", api_url)
            st.rerun()

    st.divider()

    # =====================================================
    # ROW 1: Key metrics — Power | Carbon | Cost | Workload
    # =====================================================
    power = status.get("power", {})
    carbon = status.get("carbon", {})
    thermal = status.get("thermal", {})

    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("IT Power", f"{power.get('it_power_kw', 0):.1f} kW")
    m2.metric("Total Power", f"{power.get('total_power_kw', 0):.1f} kW")
    m3.metric("PUE", f"{power.get('pue', 0):.2f}")
    headroom = power.get("headroom_kw", 0)
    m4.metric("Headroom", f"{headroom:.1f} kW")

    ci = carbon.get("carbon_intensity_gco2_kwh", 0)
    m5.metric("Carbon Intensity", f"{ci:.0f} gCO2/kWh")
    m6.metric("Total Carbon", f"{carbon.get('cumulative_carbon_kg', 0):.1f} kg")

    price = carbon.get("electricity_price_gbp_kwh", 0)
    m7.metric("Elec. Price", f"{price:.3f} GBP/kWh")
    m8.metric("Total Cost", f"{carbon.get('cumulative_cost_gbp', 0):.2f} GBP")

    st.divider()

    # =====================================================
    # ROW 2: Thermal heatmap | Power breakdown
    # =====================================================
    col_thermal, col_power = st.columns(2)

    thermal_racks = thermal.get("racks", [])
    with col_thermal:
        st.markdown('<div class="section-header">Thermal — Rack Inlet Temperatures</div>', unsafe_allow_html=True)
        ambient = thermal.get("ambient_temp_c", 22)
        humidity = thermal.get("avg_humidity_pct", 45)
        st.caption(f"Ambient: {ambient:.1f}C | Avg Humidity: {humidity:.0f}% RH")
        if thermal_racks:
            fig = make_thermal_heatmap(thermal_racks)
            st.plotly_chart(fig, use_container_width=True, key="thermal_heatmap")

    power_racks = power.get("racks", [])
    with col_power:
        st.markdown('<div class="section-header">Power — Per-Rack Breakdown</div>', unsafe_allow_html=True)
        exceeded = power.get("power_cap_exceeded", False)
        if exceeded:
            st.error("Power cap EXCEEDED")
        else:
            st.caption(f"Facility headroom: {headroom:.1f} kW")
        if power_racks:
            fig = make_power_breakdown(power_racks)
            st.plotly_chart(fig, use_container_width=True, key="power_breakdown")

    st.divider()

    # =====================================================
    # ROW 3: Carbon & Cost gauges | Workload summary
    # =====================================================
    col_carbon, col_workload = st.columns(2)

    with col_carbon:
        st.markdown('<div class="section-header">Carbon & Cost</div>', unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            fig = make_gauge(ci, "Grid Carbon", 400, "g/kWh", [0.5, 0.75])
            st.plotly_chart(fig, use_container_width=True, key="carbon_gauge")
        with cc2:
            cost_rate = carbon.get("cost_rate_gbp_h", 0)
            fig = make_gauge(cost_rate, "Cost Rate", 30, "GBP/h", [0.5, 0.8])
            st.plotly_chart(fig, use_container_width=True, key="cost_gauge")
        with cc3:
            fig = make_gauge(power.get("pue", 1.4), "PUE", 2.5, "", [0.56, 0.72])
            st.plotly_chart(fig, use_container_width=True, key="pue_gauge")

    with col_workload:
        st.markdown('<div class="section-header">Workload</div>', unsafe_allow_html=True)
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Pending", status.get("workload_pending", 0))
        wc2.metric("Running", status.get("workload_running", 0))
        wc3.metric("Completed", status.get("workload_completed", 0))
        wc4.metric("SLA Violations", status.get("sla_violations", 0))

        running = fetch_json("/workload/running", api_url)
        if running and running.get("running"):
            jobs = running["running"]
            df_jobs = pd.DataFrame(jobs)
            if not df_jobs.empty:
                display_cols = ["name", "job_type", "gpu_requirement", "assigned_servers"]
                available = [c for c in display_cols if c in df_jobs.columns]
                st.dataframe(
                    df_jobs[available],
                    hide_index=True,
                    use_container_width=True,
                    height=min(200, 35 * len(df_jobs) + 38),
                )
        else:
            st.info("No running jobs")

    st.divider()

    # =====================================================
    # ROW 4: Failures | Audit log
    # =====================================================
    col_fail, col_audit = st.columns(2)

    with col_fail:
        st.markdown('<div class="section-header">Active Failures</div>', unsafe_allow_html=True)
        failures = fetch_json("/failures/active", api_url)
        if failures and failures.get("active"):
            df_fail = pd.DataFrame(failures["active"])
            st.dataframe(df_fail, hide_index=True, use_container_width=True)
        else:
            st.success("No active failures")

    with col_audit:
        st.markdown('<div class="section-header">Audit Log (Recent Actions)</div>', unsafe_allow_html=True)
        audit = fetch_json("/audit?last_n=10", api_url)
        if audit and audit.get("entries"):
            df_audit = pd.DataFrame(audit["entries"])
            st.dataframe(df_audit, hide_index=True, use_container_width=True)
        else:
            st.info("No actions recorded yet")

    st.divider()

    # =====================================================
    # ROW 5: Time-series history
    # =====================================================
    st.markdown('<div class="section-header">Time-Series History</div>', unsafe_allow_html=True)

    history = fetch_json("/telemetry/history?last_n=120", api_url)
    if history and history.get("history"):
        rows = history["history"]
        records = []
        for i, r in enumerate(rows):
            state_data = r["state"]
            rec = {
                "tick": i,
                "it_power_kw": state_data["power"]["it_power_kw"],
                "total_power_kw": state_data["power"]["total_power_kw"],
                "pue": state_data["power"]["pue"],
            }
            # Carbon data
            carbon_data = state_data.get("carbon", {})
            rec["carbon_intensity"] = carbon_data.get("carbon_intensity_gco2_kwh", 0)
            rec["elec_price"] = carbon_data.get("electricity_price_gbp_kwh", 0)
            rec["cumulative_carbon_kg"] = carbon_data.get("cumulative_carbon_kg", 0)
            rec["cumulative_cost_gbp"] = carbon_data.get("cumulative_cost_gbp", 0)

            # Thermal data
            thermal_data = state_data.get("thermal", {})
            rec["ambient_temp"] = thermal_data.get("ambient_temp_c", 22)
            racks = thermal_data.get("racks", [])
            for rack in racks:
                rec[f"rack_{rack['rack_id']}_inlet"] = rack["inlet_temp_c"]
            records.append(rec)
        df_hist = pd.DataFrame(records)

        tab_power, tab_thermal, tab_carbon, tab_cost = st.tabs(
            ["Power & PUE", "Rack Temperatures", "Carbon", "Cost"]
        )

        with tab_power:
            c1, c2 = st.columns(2)
            with c1:
                fig = make_timeseries(df_hist, ["it_power_kw", "total_power_kw"], "Power Draw", "kW", [COLOUR_BLUE, COLOUR_PURPLE])
                st.plotly_chart(fig, use_container_width=True, key="ts_power")
            with c2:
                fig = make_timeseries(df_hist, ["pue"], "Power Usage Effectiveness", "PUE", [COLOUR_TEAL])
                st.plotly_chart(fig, use_container_width=True, key="ts_pue")

        with tab_thermal:
            inlet_cols = [c for c in df_hist.columns if c.startswith("rack_") and "_inlet" in c]
            if inlet_cols:
                c1, c2 = st.columns(2)
                with c1:
                    fig = make_timeseries(df_hist, inlet_cols, "Rack Inlet Temperatures", "Temp (C)")
                    st.plotly_chart(fig, use_container_width=True, key="ts_temps")
                with c2:
                    fig = make_timeseries(df_hist, ["ambient_temp"], "Ambient Temperature", "Temp (C)", [COLOUR_SLATE])
                    st.plotly_chart(fig, use_container_width=True, key="ts_ambient")

        with tab_carbon:
            c1, c2 = st.columns(2)
            with c1:
                fig = make_timeseries(df_hist, ["carbon_intensity"], "Grid Carbon Intensity", "gCO2/kWh", [COLOUR_GREEN])
                st.plotly_chart(fig, use_container_width=True, key="ts_carbon")
            with c2:
                fig = make_timeseries(df_hist, ["cumulative_carbon_kg"], "Cumulative Carbon Emissions", "kg CO2", [COLOUR_RED])
                st.plotly_chart(fig, use_container_width=True, key="ts_cum_carbon")

        with tab_cost:
            c1, c2 = st.columns(2)
            with c1:
                fig = make_timeseries(df_hist, ["elec_price"], "Electricity Spot Price", "GBP/kWh", [COLOUR_AMBER])
                st.plotly_chart(fig, use_container_width=True, key="ts_price")
            with c2:
                fig = make_timeseries(df_hist, ["cumulative_cost_gbp"], "Cumulative Cost", "GBP", [COLOUR_PURPLE])
                st.plotly_chart(fig, use_container_width=True, key="ts_cum_cost")
    else:
        st.info("Advance the simulation to see history.")

    # Sidebar: auto-refresh
    st.sidebar.divider()
    refresh_interval = st.sidebar.slider("Refresh interval (s)", 1, 10, 2)
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh", value=is_running,
        help="Reload data periodically",
    )
    if st.sidebar.button("Refresh now"):
        st.rerun()

    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
