"""
Data Centre Simulator Dashboard — Terminal/Ops Console Style

Visualizes thermal, power, carbon, workload, and failure state
in a dark terminal aesthetic inspired by infrastructure monitoring consoles.
Requires the API server to be running: uvicorn dc_sim.main:app
"""

import httpx
import pandas as pd
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"

# ── Colour palette ──────────────────────────────────────────
C_BG = "#0a0e17"
C_CARD = "#0d1321"
C_BORDER = "#1a2332"
C_GREEN = "#00ff88"
C_AMBER = "#ffaa00"
C_RED = "#ff3355"
C_CYAN = "#00d4ff"
C_PURPLE = "#aa66ff"
C_BLUE = "#3388ff"
C_MUTED = "#4a5568"
C_LABEL = "#667788"
C_TEXT = "#c0ccdd"
C_WHITE = "#e8eef5"


def hex_to_rgba(hex_colour: str, alpha: float = 0.1) -> str:
    """Convert '#RRGGBB' → 'rgba(R,G,B,alpha)'."""
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def fetch_json(path: str, base_url: str) -> dict | None:
    try:
        r = httpx.get(f"{base_url}{path}", timeout=2.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_json(path: str, base_url: str, json: dict | None = None) -> dict | None:
    try:
        r = httpx.post(f"{base_url}{path}", json=json or {}, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── HTML helper components ──────────────────────────────────

def _progress_bar(value: float, max_val: float, colour: str = C_GREEN,
                  label_left: str = "", label_right: str = "") -> str:
    pct = min(100, max(0, (value / max_val) * 100)) if max_val > 0 else 0
    return (
        f'<div style="width:100%;margin:4px 0;">'
        f'<div style="height:6px;background:{C_BORDER};border-radius:3px;overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{colour};border-radius:3px;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:{C_MUTED};'
        f"font-family:'Courier New',monospace;margin-top:2px;\">"
        f'<span>{label_left}</span><span>{label_right}</span>'
        f'</div></div>'
    )


def _stat_card(title: str, value: str, colour: str = C_GREEN,
               icon: str = "", subtitle: str = "", bar_html: str = "") -> str:
    icon_part = (f'<span style="float:right;font-size:22px;opacity:0.3;">{icon}</span>'
                 if icon else "")
    sub_part = (f'<div style="color:{C_MUTED};font-size:11px;margin-top:4px;">{subtitle}</div>'
                if subtitle else "")
    return (
        f'<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:6px;'
        f'padding:14px 18px;height:100%;">'
        f'{icon_part}'
        f'<div style="color:{colour};font-size:11px;font-family:\'Courier New\',monospace;'
        f'text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px;">'
        f'{title}</div>'
        f'<div style="color:{C_WHITE};font-size:26px;font-family:\'Courier New\',monospace;'
        f'font-weight:700;line-height:1.1;">{value}</div>'
        f'{sub_part}'
        f'{bar_html}'
        f'</div>'
    )


def _section_title(text: str, colour: str = C_CYAN) -> str:
    return (
        f'<div style="color:{colour};font-size:12px;font-family:\'Courier New\',monospace;'
        f'text-transform:uppercase;letter-spacing:2px;font-weight:700;'
        f'border-bottom:1px solid {C_BORDER};padding-bottom:6px;margin:16px 0 10px 0;">'
        f'{text}</div>'
    )


def _server_cell(server_id: str, temp: float, power_kw: float,
                 util: float, status: str) -> str:
    if status == "throttled":
        bg, border_c, dot = "#1a1a00", C_AMBER, C_AMBER
    elif status == "offline":
        bg, border_c, dot = "#1a0011", C_RED, C_RED
    else:
        bg, border_c, dot = "#001a0d", "#0a3320", C_GREEN

    bar_h = min(95, max(5, util * 100))
    label = server_id.split("-")[-1].upper()

    return (
        f'<div style="background:{bg};border:1px solid {border_c};border-radius:4px;'
        f'padding:0;position:relative;overflow:hidden;aspect-ratio:1.3;min-height:60px;"'
        f' title="{server_id} | Temp: {temp:.1f}C | Power: {power_kw:.0f}W | Util: {util*100:.0f}%">'
        f'<div style="position:absolute;bottom:0;right:4px;width:4px;height:{bar_h}%;'
        f'background:{border_c};border-radius:2px 2px 0 0;opacity:0.7;"></div>'
        f'<div style="position:absolute;top:3px;right:5px;width:5px;height:5px;'
        f'background:{dot};border-radius:50%;"></div>'
        f'<div style="position:absolute;bottom:3px;left:5px;font-size:9px;color:{C_MUTED};'
        f"font-family:'Courier New',monospace;\">{label}</div>"
        f'</div>'
    )


def _workload_row(name: str, status: str, gpus: int, job_type: str) -> str:
    status_colour = {"running": C_GREEN, "queued": C_AMBER, "completed": C_MUTED,
                     "failed": C_RED, "preempted": C_RED}.get(status, C_MUTED)
    type_colour = {"training": C_PURPLE, "inference": C_CYAN,
                   "batch": C_BLUE}.get(job_type, C_MUTED)
    return (
        f'<tr style="border-bottom:1px solid {C_BORDER};">'
        f'<td style="padding:6px 8px;font-size:11px;color:{C_TEXT};'
        f"font-family:'Courier New',monospace;\">{name}</td>"
        f'<td style="padding:6px 8px;font-size:11px;color:{status_colour};'
        f"font-family:'Courier New',monospace;text-transform:uppercase;font-weight:600;\">"
        f'{status}</td>'
        f'<td style="padding:6px 8px;font-size:11px;color:{type_colour};'
        f"font-family:'Courier New',monospace;text-transform:uppercase;\">{job_type}</td>"
        f'<td style="padding:6px 8px;font-size:11px;color:{C_TEXT};'
        f"font-family:'Courier New',monospace;text-align:right;\">{gpus}</td>"
        f'</tr>'
    )


def _alert_row(failure_type: str, target: str, effect: str) -> str:
    return (
        f'<div style="background:#1a0011;border:1px solid {C_RED};border-radius:4px;'
        f'padding:8px 12px;margin-bottom:4px;">'
        f'<span style="color:{C_RED};font-size:11px;font-family:\'Courier New\',monospace;'
        f'font-weight:700;text-transform:uppercase;">{failure_type}</span>'
        f'<span style="color:{C_MUTED};font-size:11px;font-family:\'Courier New\',monospace;'
        f'margin-left:8px;">{target} &mdash; {effect}</span>'
        f'</div>'
    )


def _node_telemetry_table(thermal_racks: list, power_racks: list) -> str:
    header = (
        f'<table style="width:100%;border-collapse:collapse;font-family:\'Courier New\',monospace;">'
        f'<thead><tr style="border-bottom:1px solid {C_BORDER};">'
        f'<th style="text-align:left;padding:8px;font-size:11px;color:{C_LABEL};font-weight:600;">RACK_ID</th>'
        f'<th style="text-align:left;padding:8px;font-size:11px;color:{C_LABEL};font-weight:600;">STATUS</th>'
        f'<th style="text-align:left;padding:8px;font-size:11px;color:{C_LABEL};font-weight:600;">TEMP (C)</th>'
        f'<th style="text-align:left;padding:8px;font-size:11px;color:{C_LABEL};font-weight:600;">POWER (kW)</th>'
        f'<th style="text-align:left;padding:8px;font-size:11px;color:{C_LABEL};font-weight:600;">HUMIDITY</th>'
        f'</tr></thead><tbody>'
    )

    rows = ""
    for rack in thermal_racks:
        rid = rack["rack_id"]
        temp = rack["inlet_temp_c"]
        throttled = rack.get("throttled", False)
        humidity = rack.get("humidity_pct", 45)

        rack_power = 0
        for pr in power_racks:
            if pr["rack_id"] == rid:
                rack_power = pr["total_power_kw"]
                break

        if throttled:
            status_text, status_colour = "THROTTLED", C_AMBER
        elif temp >= 35:
            status_text, status_colour = "WARNING", C_AMBER
        else:
            status_text, status_colour = "OPTIMAL", C_GREEN

        temp_colour = C_RED if temp >= 40 else (C_AMBER if temp >= 35 else C_GREEN)
        temp_pct = min(100, max(0, (temp - 15) / 35 * 100))
        temp_bar = (
            f'<span style="color:{temp_colour};">{temp:.1f}</span>'
            f'<span style="display:inline-block;width:50px;height:4px;background:{C_BORDER};'
            f'border-radius:2px;vertical-align:middle;margin-left:6px;overflow:hidden;">'
            f'<span style="display:block;width:{temp_pct:.0f}%;height:100%;background:{temp_colour};'
            f'border-radius:2px;"></span></span>'
        )

        humidity_pct = min(100, max(0, (humidity - 20) / 60 * 100))
        humidity_bar = (
            f'{humidity:.0f}%'
            f'<span style="display:inline-block;width:50px;height:4px;background:{C_BORDER};'
            f'border-radius:2px;vertical-align:middle;margin-left:6px;overflow:hidden;">'
            f'<span style="display:block;width:{humidity_pct:.0f}%;height:100%;background:{C_BLUE};'
            f'border-radius:2px;"></span></span>'
        )

        rows += (
            f'<tr style="border-bottom:1px solid {C_BORDER};">'
            f'<td style="padding:8px;font-size:12px;color:{C_TEXT};">R{rid:03d}</td>'
            f'<td style="padding:8px;font-size:11px;color:{status_colour};font-weight:600;">{status_text}</td>'
            f'<td style="padding:8px;font-size:12px;">{temp_bar}</td>'
            f'<td style="padding:8px;font-size:12px;color:{C_TEXT};">{rack_power:.1f}</td>'
            f'<td style="padding:8px;font-size:12px;">{humidity_bar}</td>'
            f'</tr>'
        )

    return header + rows + "</tbody></table>"


# ── Main dashboard ──────────────────────────────────────────

def main():
    st.set_page_config(page_title="DC Simulator", page_icon="DC", layout="wide")

    # ── Global styles ────────────────────────────────────
    st.html(
        f'<style>'
        f"@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');"
        f'.stApp {{ background-color: {C_BG}; }}'
        f"* {{ font-family: 'JetBrains Mono', 'Courier New', monospace !important; }}"
        f'.block-container {{ padding-top: 1rem !important; max-width: 100% !important; }}'
        f'header[data-testid="stHeader"] {{ display: none !important; }}'
        f'section[data-testid="stSidebar"] {{ background: {C_CARD} !important; border-right: 1px solid {C_BORDER}; }}'
        f'section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}'
        f'.stButton > button {{'
        f'  background: {C_CARD} !important; color: {C_TEXT} !important;'
        f'  border: 1px solid {C_BORDER} !important; border-radius: 4px !important;'
        f'  font-size: 11px !important; padding: 4px 12px !important;'
        f'  text-transform: uppercase !important; letter-spacing: 1px !important;'
        f'}}'
        f'.stButton > button:hover {{'
        f'  border-color: {C_CYAN} !important; color: {C_CYAN} !important;'
        f'}}'
        f'.stTabs [data-baseweb="tab-list"] {{ background: transparent; gap: 0; border-bottom: 1px solid {C_BORDER}; }}'
        f'.stTabs [data-baseweb="tab"] {{'
        f'  background: transparent !important; color: {C_MUTED} !important;'
        f'  border: none !important; border-bottom: 2px solid transparent !important;'
        f'  font-size: 11px !important; text-transform: uppercase !important;'
        f'  letter-spacing: 1.5px !important; padding: 8px 16px !important;'
        f'}}'
        f'.stTabs [aria-selected="true"] {{'
        f'  color: {C_CYAN} !important; border-bottom-color: {C_CYAN} !important;'
        f'  background: rgba(0,212,255,0.05) !important;'
        f'}}'
        f'div[data-testid="stMetric"] {{ display: none; }}'
        f'.stDataFrame {{ border: 1px solid {C_BORDER} !important; border-radius: 4px; }}'
        f'hr {{ border-color: {C_BORDER} !important; margin: 8px 0 !important; }}'
        f'</style>'
    )

    # ── Sidebar ─────────────────────────────────────────
    st.sidebar.markdown(
        f'<div style="color:{C_CYAN};font-size:11px;letter-spacing:2px;'
        f'text-transform:uppercase;margin-bottom:12px;">SETTINGS</div>',
        unsafe_allow_html=True,
    )
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL,
                                     label_visibility="collapsed")

    status = fetch_json("/status", api_url)
    if status is None:
        st.html(
            f'<div style="text-align:center;padding:80px 20px;">'
            f'<div style="color:{C_RED};font-size:14px;font-family:\'Courier New\',monospace;'
            f'letter-spacing:2px;text-transform:uppercase;">CONNECTION FAILED</div>'
            f'<div style="color:{C_MUTED};font-size:12px;margin-top:12px;">'
            f'Start the simulator: <code>uvicorn dc_sim.main:app</code></div>'
            f'</div>'
        )
        return

    sim_status = fetch_json("/sim/status", api_url) or {}
    is_running = sim_status.get("running", False)

    # ── Top navbar ──────────────────────────────────────
    elapsed = status.get("current_time", 0)
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    tick_count = status.get("tick_count", 0)

    power = status.get("power", {})
    carbon = status.get("carbon", {})
    thermal = status.get("thermal", {})
    ci = carbon.get("carbon_intensity_gco2_kwh", 0)
    cost_total = carbon.get("cumulative_cost_gbp", 0)
    cost_rate = carbon.get("cost_rate_gbp_h", 0)

    if is_running:
        running_dot = (
            f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
            f'background:{C_GREEN};margin-right:4px;animation:pulse 1.5s infinite;"></span>'
        )
    else:
        running_dot = (
            f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
            f'background:{C_MUTED};margin-right:4px;"></span>'
        )

    st.html(
        f'<style>@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}</style>'
        f'<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:6px;padding:8px 20px;'
        f'display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">'
        f'<div style="display:flex;align-items:center;gap:16px;">'
        f'{running_dot}'
        f'<span style="color:{C_GREEN};font-size:14px;font-weight:700;letter-spacing:1px;">DC_SIM</span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:20px;font-size:11px;">'
        f'<span style="color:{C_MUTED};">TICK: <span style="color:{C_WHITE};">{tick_count}</span></span>'
        f'<span style="color:{C_MUTED};">TIME: <span style="color:{C_WHITE};">{h:02d}:{m:02d}:{s:02d}</span></span>'
        f'<span style="color:{C_MUTED};">GRID: <span style="color:{C_CYAN};">{ci:.0f}g</span></span>'
        f'<span style="color:{C_MUTED};">COST/H: <span style="color:{C_GREEN};">&pound;{cost_rate:.1f}</span></span>'
        f'<span style="color:{C_MUTED};">TOTAL: <span style="color:{C_GREEN};">&pound;{cost_total:.2f}</span></span>'
        f'</div></div>'
    )

    # ── Controls row ────────────────────────────────────
    ctrl_cols = st.columns([1, 1, 1, 1, 1, 1, 6])
    with ctrl_cols[0]:
        if is_running:
            if st.button("PAUSE", use_container_width=True):
                post_json("/sim/pause", api_url)
                st.rerun()
        else:
            if st.button("PLAY", use_container_width=True):
                post_json("/sim/run?tick_interval_s=0.5", api_url)
                st.rerun()
    with ctrl_cols[1]:
        if st.button("+1", use_container_width=True):
            post_json("/sim/tick?n=1", api_url)
            st.rerun()
    with ctrl_cols[2]:
        if st.button("+10", use_container_width=True):
            post_json("/sim/tick?n=10", api_url)
            st.rerun()
    with ctrl_cols[3]:
        if st.button("+60", use_container_width=True):
            post_json("/sim/tick?n=60", api_url)
            st.rerun()
    with ctrl_cols[4]:
        if st.button("RESET", use_container_width=True):
            post_json("/sim/pause", api_url)
            post_json("/sim/reset", api_url)
            st.rerun()

    # ── Tabs ────────────────────────────────────────────
    tab_overview, tab_infra, tab_fleet, tab_gpu_tab, tab_net_tab, tab_storage_tab, tab_cool_tab, tab_carbon_tab, tab_eval, tab_leaderboard = st.tabs(
        ["OVERVIEW", "INFRASTRUCTURE", "FLEET", "GPU", "NETWORK", "STORAGE", "COOLING", "CARBON", "EVAL", "LEADERBOARD"]
    )

    thermal_racks = thermal.get("racks", [])
    power_racks = power.get("racks", [])
    gpu_data = status.get("gpu", {})
    net_data = status.get("network", {})
    storage_data = status.get("storage", {})
    cooling_data = status.get("cooling", {})

    # ════════════════════════════════════════════════════
    # TAB: OVERVIEW
    # ════════════════════════════════════════════════════
    with tab_overview:
        total_servers = 8 * 4
        throttled_count = sum(1 for r in thermal_racks if r.get("throttled", False))
        optimal_count = total_servers - throttled_count
        avg_temp = (sum(r["inlet_temp_c"] for r in thermal_racks) / max(1, len(thermal_racks))
                    if thermal_racks else 22)
        total_pwr = power.get("total_power_kw", 0)
        pue = power.get("pue", 1.4)
        sla = status.get("sla_violations", 0)
        running_count = status.get("workload_running", 0)
        pending_count = status.get("workload_pending", 0)
        completed_count = status.get("workload_completed", 0)
        sla_colour = C_RED if sla > 0 else C_GREEN

        # ── Top stat cards (single HTML block for all 4) ──
        fleet_bar = _progress_bar(optimal_count, total_servers, C_GREEN,
                                  f"{optimal_count} OPTIMAL", f"{throttled_count} WARN")
        card1 = _stat_card("FLEET HEALTH", f"{optimal_count/total_servers*100:.0f}%",
                           C_GREEN, "", f"{total_servers} TOTAL NODES", fleet_bar)
        card2 = _stat_card("AVG TEMP", f"{avg_temp:.1f}C",
                           C_AMBER if avg_temp >= 35 else C_GREEN, "", "TARGET: 22.0C")
        card3 = _stat_card("TOTAL POWER", f"{total_pwr:.1f} kW",
                           C_CYAN, "", f"PUE: {pue:.2f}")
        card4 = _stat_card("WORKLOAD", f"{running_count} RUNNING",
                           C_GREEN, "",
                           f'{pending_count} queued / {completed_count} done / '
                           f'<span style="color:{sla_colour}">{sla} SLA</span>')

        st.html(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:8px;">'
            f'{card1}{card2}{card3}{card4}'
            f'</div>'
        )

        # ── Second row: GPU, Network, Storage, Cooling summaries ──
        gpu_healthy = gpu_data.get("healthy_gpus", 0)
        gpu_total = gpu_data.get("total_gpus", 128)
        gpu_mem_used = gpu_data.get("total_gpu_mem_used_mib", 0)
        gpu_mem_total = gpu_data.get("total_gpu_mem_total_mib", 1)
        gpu_bar = _progress_bar(gpu_healthy, gpu_total, C_GREEN,
                                f"{gpu_healthy} HEALTHY", f"{gpu_data.get('throttled_gpus', 0)} THROT")
        card_gpu = _stat_card("GPU FLEET", f"{gpu_data.get('avg_sm_util_pct', 0):.0f}% SM",
                              C_CYAN, "", f"MEM: {gpu_mem_used/1024:.0f}/{gpu_mem_total/1024:.0f} GiB", gpu_bar)

        net_ew = net_data.get("total_east_west_gbps", 0)
        net_ns = net_data.get("total_north_south_gbps", 0)
        net_lat = net_data.get("avg_fabric_latency_us", 5)
        card_net = _stat_card("NETWORK", f"{net_ew + net_ns:.1f} Gbps",
                              C_BLUE, "", f"LAT: {net_lat:.0f}us / RDMA: {net_data.get('total_rdma_gbps', 0):.1f}G")

        sto_iops = storage_data.get("total_read_iops", 0) + storage_data.get("total_write_iops", 0)
        sto_used = storage_data.get("total_used_tb", 0)
        sto_cap = storage_data.get("total_capacity_tb", 1)
        sto_bar = _progress_bar(sto_used, sto_cap, C_AMBER, f"{sto_used:.0f}TB", f"{sto_cap:.0f}TB")
        card_sto = _stat_card("STORAGE", f"{sto_iops/1000:.0f}K IOPS",
                              C_AMBER, "", f"R: {storage_data.get('avg_read_latency_us', 0):.0f}us", sto_bar)

        cool_cop = cooling_data.get("cop", 4.0)
        cool_load = cooling_data.get("cooling_load_pct", 0)
        cool_bar = _progress_bar(cool_load, 100, C_GREEN if cool_load < 70 else (C_AMBER if cool_load < 90 else C_RED),
                                 f"{cool_load:.0f}%", "100%")
        card_cool = _stat_card("COOLING", f"COP {cool_cop:.1f}",
                               C_GREEN if cool_cop > 3.5 else C_AMBER, "",
                               f"CHW: {cooling_data.get('chw_plant_supply_temp_c', 7):.1f}C", cool_bar)

        st.html(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:8px;">'
            f'{card_gpu}{card_net}{card_sto}{card_cool}'
            f'</div>'
        )

        # ── Left: stat panels | Right: system alerts ────
        left_col, right_col = st.columns([3, 1])

        with left_col:
            # Unit economics + Carbon cards
            price = carbon.get("electricity_price_gbp_kwh", 0)
            bar_econ = _progress_bar(
                price, 0.30,
                C_GREEN if price < 0.15 else (C_AMBER if price < 0.22 else C_RED),
                f"SPOT: {price:.3f}", "CAP: 0.30")
            card_econ = _stat_card("UNIT ECONOMICS", f"&pound;{cost_rate:.2f}",
                                   C_GREEN, "$", "GBP / HOUR", bar_econ)

            carbon_rate = carbon.get("carbon_rate_gco2_s", 0)
            bar_carb = _progress_bar(
                ci, 400,
                C_GREEN if ci < 180 else (C_AMBER if ci < 260 else C_RED),
                f"GRID: {ci:.0f}g", "SOLAR: 0kW")
            card_carb = _stat_card("CARBON INTENSITY", f"{carbon_rate*3.6:.1f}",
                                   C_GREEN, "", "kgCO2/h", bar_carb)

            st.html(
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px;">'
                f'{card_econ}{card_carb}'
                f'</div>'
            )

            # Power headroom bar
            headroom = power.get("headroom_kw", 0)
            bar_pw = _progress_bar(
                total_pwr, max(total_pwr, 120),
                C_GREEN if headroom > 20 else (C_AMBER if headroom > 0 else C_RED),
                f"USED: {total_pwr:.1f}kW", "CAP: 120kW")
            card_pw = _stat_card(
                "POWER BUDGET", f"{headroom:.1f} kW",
                C_GREEN if headroom > 20 else (C_AMBER if headroom > 0 else C_RED),
                "", "HEADROOM", bar_pw)
            st.html(card_pw)

        with right_col:
            # System alerts panel
            st.html(_section_title("SYSTEM ALERTS", C_RED))
            failures = fetch_json("/failures/active", api_url)
            if failures and failures.get("active"):
                alerts_html = ""
                for f in failures["active"]:
                    alerts_html += _alert_row(f["type"], f["target"], f["effect"])
                st.html(alerts_html)
            else:
                st.html(
                    f'<div style="text-align:center;padding:20px;color:{C_MUTED};'
                    f'font-size:11px;">NO ACTIVE ALERTS</div>'
                )

            # Audit log
            st.html(_section_title("MESSAGE_BUS", C_MUTED))
            audit = fetch_json("/audit?last_n=5", api_url)
            if audit and audit.get("entries"):
                bus_html = ""
                for e in audit["entries"][-5:]:
                    t = e.get("timestamp", 0)
                    hh, mm, ss = int(t // 3600), int((t % 3600) // 60), int(t % 60)
                    bus_html += (
                        f'<div style="font-size:10px;color:{C_MUTED};font-family:\'Courier New\',monospace;'
                        f'padding:2px 0;border-bottom:1px solid {C_BORDER};">'
                        f'<span style="color:{C_CYAN};">{hh:02d}:{mm:02d}:{ss:02d}</span>'
                        f'<span style="color:{C_GREEN};margin-left:6px;">{e.get("action","")}</span>'
                        f'<span style="color:{C_MUTED};margin-left:6px;">{e.get("result","")}</span>'
                        f'</div>'
                    )
                st.html(bus_html)
            else:
                st.html(
                    f'<div style="font-size:10px;color:{C_GREEN};padding:4px 0;">'
                    f'&bull; STREAM ACTIVE</div>'
                )

        # ── Workload queue ──────────────────────────────
        wl_left, wl_right = st.columns(2)
        with wl_left:
            st.html(_section_title("WORKLOAD QUEUE"))
            st.html(
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">'
                f'<div style="text-align:center;background:{C_CARD};border:1px solid {C_BORDER};'
                f'border-radius:4px;padding:12px;">'
                f'<div style="color:{C_GREEN};font-size:24px;font-weight:700;">{running_count}</div>'
                f'<div style="color:{C_MUTED};font-size:10px;text-transform:uppercase;">RUNNING</div></div>'
                f'<div style="text-align:center;background:{C_CARD};border:1px solid {C_BORDER};'
                f'border-radius:4px;padding:12px;">'
                f'<div style="color:{C_AMBER};font-size:24px;font-weight:700;">{pending_count}</div>'
                f'<div style="color:{C_MUTED};font-size:10px;text-transform:uppercase;">QUEUED</div></div>'
                f'<div style="text-align:center;background:{C_CARD};border:1px solid {C_BORDER};'
                f'border-radius:4px;padding:12px;">'
                f'<div style="color:{C_CYAN};font-size:24px;font-weight:700;">{completed_count}</div>'
                f'<div style="color:{C_MUTED};font-size:10px;text-transform:uppercase;">DONE</div></div>'
                f'</div>'
            )

        with wl_right:
            st.html(_section_title("RUNNING JOBS"))
            running_data = fetch_json("/workload/running", api_url)
            if running_data and running_data.get("running"):
                table_html = (
                    f'<table style="width:100%;border-collapse:collapse;">'
                    f'<thead><tr style="border-bottom:1px solid {C_BORDER};">'
                    f'<th style="text-align:left;padding:6px 8px;font-size:10px;color:{C_LABEL};">NAME</th>'
                    f'<th style="text-align:left;padding:6px 8px;font-size:10px;color:{C_LABEL};">STATUS</th>'
                    f'<th style="text-align:left;padding:6px 8px;font-size:10px;color:{C_LABEL};">TYPE</th>'
                    f'<th style="text-align:right;padding:6px 8px;font-size:10px;color:{C_LABEL};">GPUS</th>'
                    f'</tr></thead><tbody>'
                )
                for j in running_data["running"][:8]:
                    table_html += _workload_row(
                        j["name"], "running",
                        j["gpu_requirement"], j.get("job_type", "batch"))
                table_html += "</tbody></table>"
                st.html(table_html)
            else:
                st.html(
                    f'<div style="color:{C_MUTED};font-size:11px;padding:10px;">NO RUNNING JOBS</div>'
                )

    # ════════════════════════════════════════════════════
    # TAB: INFRASTRUCTURE
    # ════════════════════════════════════════════════════
    with tab_infra:
        st.html(_section_title("INFRASTRUCTURE_MAP", C_CYAN))

        # Legend
        st.html(
            f'<div style="text-align:right;margin-bottom:8px;font-size:10px;">'
            f'<span style="display:inline-block;width:8px;height:8px;background:{C_GREEN};'
            f'border-radius:50%;margin-right:3px;"></span>'
            f'<span style="color:{C_MUTED};margin-right:12px;">OPTIMAL</span>'
            f'<span style="display:inline-block;width:8px;height:8px;background:{C_AMBER};'
            f'border-radius:50%;margin-right:3px;"></span>'
            f'<span style="color:{C_MUTED};margin-right:12px;">THROTTLED</span>'
            f'<span style="display:inline-block;width:8px;height:8px;background:{C_RED};'
            f'border-radius:50%;margin-right:3px;"></span>'
            f'<span style="color:{C_MUTED};">OFFLINE</span>'
            f'</div>'
        )

        # Build server grid — organized by rack
        if thermal_racks and power_racks:
            num_racks = len(thermal_racks)
            servers_per_rack = 4

            running_data = fetch_json("/workload/running", api_url) or {"running": []}
            server_utils: dict[str, float] = {}
            for j in running_data.get("running", []):
                for srv in j.get("assigned_servers", []):
                    server_utils[srv] = server_utils.get(srv, 0) + 0.2

            # Build all racks as a single HTML grid
            rack_html_parts = []
            for rack_idx in range(num_racks):
                rack = thermal_racks[rack_idx] if rack_idx < len(thermal_racks) else {}
                pr = power_racks[rack_idx] if rack_idx < len(power_racks) else {}
                throttled = rack.get("throttled", False)
                inlet = rack.get("inlet_temp_c", 22)
                rack_power_kw = pr.get("total_power_kw", 0)
                temp_colour = C_AMBER if throttled else C_GREEN

                cells = ""
                for srv_idx in range(servers_per_rack):
                    sid = f"rack-{rack_idx}-srv-{srv_idx}"
                    util = min(1.0, server_utils.get(sid, 0.05))
                    srv_power = rack_power_kw / servers_per_rack * 1000
                    srv_status = "throttled" if throttled else "optimal"
                    cells += _server_cell(sid, inlet, srv_power, util, srv_status)

                rack_html_parts.append(
                    f'<div style="display:flex;flex-direction:column;gap:3px;">'
                    f'<div style="display:grid;grid-template-columns:1fr;gap:3px;">{cells}</div>'
                    f'<div style="text-align:center;padding:6px 0;">'
                    f'<div style="font-size:10px;color:{C_MUTED};">R{rack_idx:02d}</div>'
                    f'<div style="font-size:10px;color:{temp_colour};">{inlet:.1f}C</div>'
                    f'<div style="font-size:9px;color:{C_MUTED};">{rack_power_kw:.1f}kW</div>'
                    f'</div></div>'
                )

            grid_cols = f"repeat({num_racks}, 1fr)"
            st.html(
                f'<div style="display:grid;grid-template-columns:{grid_cols};gap:8px;">'
                + "".join(rack_html_parts)
                + '</div>'
            )

        st.html(f'<div style="height:16px"></div>')

        # Node telemetry table
        st.html(_section_title("NODE TELEMETRY"))
        if thermal_racks:
            st.html(_node_telemetry_table(thermal_racks, power_racks))

    # ════════════════════════════════════════════════════
    # TAB: FLEET
    # ════════════════════════════════════════════════════
    with tab_fleet:
        st.html(_section_title("FLEET TELEMETRY"))

        history = fetch_json("/telemetry/history?last_n=120", api_url)
        if history and history.get("history"):
            rows = history["history"]
            records = []
            for i, r in enumerate(rows):
                sd = r["state"]
                rec = {
                    "tick": i,
                    "it_power_kw": sd["power"]["it_power_kw"],
                    "total_power_kw": sd["power"]["total_power_kw"],
                    "pue": sd["power"]["pue"],
                }
                cd = sd.get("carbon", {})
                rec["carbon_intensity"] = cd.get("carbon_intensity_gco2_kwh", 0)
                rec["elec_price"] = cd.get("electricity_price_gbp_kwh", 0)
                rec["cumulative_carbon_kg"] = cd.get("cumulative_carbon_kg", 0)
                rec["cumulative_cost_gbp"] = cd.get("cumulative_cost_gbp", 0)
                td = sd.get("thermal", {})
                rec["ambient_temp"] = td.get("ambient_temp_c", 22)
                for rack in td.get("racks", []):
                    rec[f"rack_{rack['rack_id']}_inlet"] = rack["inlet_temp_c"]
                records.append(rec)
            df = pd.DataFrame(records)

            import plotly.graph_objects as go

            def _term_chart(df_in, cols, title, ylabel, colours=None):
                fig = go.Figure()
                default_c = [C_GREEN, C_CYAN, C_AMBER, C_RED, C_PURPLE, C_BLUE,
                             "#ff66aa", "#aa88ff"]
                colours = colours or default_c
                for i_c, col in enumerate(cols):
                    fig.add_trace(go.Scatter(
                        x=df_in["tick"], y=df_in[col], mode="lines",
                        name=col.replace("_", " ").upper(),
                        line=dict(color=colours[i_c % len(colours)], width=1.5),
                    ))
                fig.update_layout(
                    height=260,
                    margin=dict(l=40, r=10, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C_MUTED, size=10, family="Courier New"),
                    title=dict(text=title, font=dict(size=11, color=C_LABEL)),
                    yaxis=dict(gridcolor="rgba(30,50,70,0.5)", title=ylabel),
                    xaxis=dict(gridcolor="rgba(30,50,70,0.5)", title="TICK"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                )
                return fig

            c1, c2 = st.columns(2)
            with c1:
                fig = _term_chart(df, ["it_power_kw", "total_power_kw"],
                                  "POWER_DRAW", "kW", [C_CYAN, C_PURPLE])
                st.plotly_chart(fig, use_container_width=True, key="fleet_power")
            with c2:
                fig = _term_chart(df, ["pue"], "PUE_TREND", "RATIO", [C_GREEN])
                st.plotly_chart(fig, use_container_width=True, key="fleet_pue")

            inlet_cols = [c for c in df.columns if c.startswith("rack_") and "_inlet" in c]
            if inlet_cols:
                c1, c2 = st.columns(2)
                with c1:
                    fig = _term_chart(df, inlet_cols, "RACK_INLET_TEMPS", "TEMP (C)")
                    st.plotly_chart(fig, use_container_width=True, key="fleet_temps")
                with c2:
                    fig = _term_chart(df, ["ambient_temp"], "AMBIENT_TEMP",
                                      "TEMP (C)", [C_MUTED])
                    st.plotly_chart(fig, use_container_width=True, key="fleet_ambient")
        else:
            st.html(
                f'<div style="color:{C_MUTED};font-size:11px;padding:20px;">'
                f'ADVANCE SIMULATION TO SEE DATA</div>'
            )

    # ════════════════════════════════════════════════════
    # TAB: GPU
    # ════════════════════════════════════════════════════
    with tab_gpu_tab:
        st.html(_section_title("GPU TELEMETRY"))

        # Summary cards
        gpu_detail = fetch_json("/gpu", api_url) or {}
        g_total = gpu_detail.get("total_gpus", 0)
        g_healthy = gpu_detail.get("healthy_gpus", 0)
        g_throttled = gpu_detail.get("throttled_gpus", 0)
        g_ecc = gpu_detail.get("ecc_error_gpus", 0)
        g_temp = gpu_detail.get("avg_gpu_temp_c", 35)
        g_util = gpu_detail.get("avg_sm_util_pct", 0)
        g_mem_u = gpu_detail.get("total_gpu_mem_used_mib", 0)
        g_mem_t = gpu_detail.get("total_gpu_mem_total_mib", 1)

        bar_h = _progress_bar(g_healthy, g_total, C_GREEN, f"{g_healthy}", f"{g_total}")
        gc1 = _stat_card("HEALTHY GPUS", f"{g_healthy}/{g_total}",
                         C_GREEN if g_healthy == g_total else C_AMBER, "", "", bar_h)
        bar_u = _progress_bar(g_util, 100, C_CYAN, f"{g_util:.0f}%", "100%")
        gc2 = _stat_card("AVG SM UTIL", f"{g_util:.1f}%", C_CYAN, "", "", bar_u)
        gc3 = _stat_card("AVG GPU TEMP", f"{g_temp:.1f}C",
                         C_RED if g_temp > 80 else (C_AMBER if g_temp > 65 else C_GREEN), "",
                         f"THROTTLED: {g_throttled}")
        bar_m = _progress_bar(g_mem_u, g_mem_t, C_PURPLE, f"{g_mem_u/1024:.0f}GiB", f"{g_mem_t/1024:.0f}GiB")
        gc4 = _stat_card("GPU MEMORY", f"{g_mem_u/g_mem_t*100:.0f}%", C_PURPLE, "",
                         f"ECC ERRORS: {g_ecc}", bar_m)

        st.html(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
            f'{gc1}{gc2}{gc3}{gc4}'
            f'</div>'
        )

        # Per-server GPU table
        st.html(_section_title("PER-SERVER GPU DETAIL"))
        gpu_full = fetch_json("/gpu", api_url) or {}
        # Show table for first few servers
        for rack_id in range(min(4, 8)):  # Show first 4 racks
            srv_id = f"rack-{rack_id}-srv-0"
            srv_data = fetch_json(f"/gpu/{srv_id}", api_url)
            if srv_data and srv_data.get("gpus"):
                gpus = srv_data["gpus"]
                rows_html = ""
                for g in gpus:
                    temp_c = C_RED if g["gpu_temp_c"] > 80 else (C_AMBER if g["gpu_temp_c"] > 65 else C_GREEN)
                    mem_pct = g["mem_used_mib"] / max(1, g["mem_total_mib"]) * 100
                    thr_badge = (f'<span style="color:{C_RED};font-size:9px;"> THR</span>'
                                 if g["thermal_throttle"] else "")
                    rows_html += (
                        f'<tr style="border-bottom:1px solid {C_BORDER};">'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_TEXT};">{g["gpu_id"].split("-")[-1].upper()}</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_CYAN};">{g["sm_utilisation_pct"]:.0f}%</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{temp_c};">{g["gpu_temp_c"]:.0f}C{thr_badge}</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_TEXT};">{g["power_draw_w"]:.0f}W</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_PURPLE};">{mem_pct:.0f}%</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_TEXT};">{g["sm_clock_mhz"]}</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_MUTED};">{g["fan_speed_pct"]:.0f}%</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_BLUE};">{g["pcie_tx_gbps"]:.1f}/{g["pcie_rx_gbps"]:.1f}</td>'
                        f'<td style="padding:4px 6px;font-size:10px;color:{C_MUTED};">{g["ecc_sbe_count"]}/{g["ecc_dbe_count"]}</td>'
                        f'</tr>'
                    )
                st.html(
                    f'<div style="margin-bottom:8px;">'
                    f'<div style="color:{C_CYAN};font-size:10px;margin-bottom:4px;">{srv_id.upper()}</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-family:\'Courier New\',monospace;">'
                    f'<thead><tr style="border-bottom:1px solid {C_BORDER};">'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">GPU</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">SM%</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">TEMP</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">PWR</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">MEM%</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">CLK</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">FAN</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">PCIe TX/RX</th>'
                    f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">ECC S/D</th>'
                    f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
                )

    # ════════════════════════════════════════════════════
    # TAB: NETWORK
    # ════════════════════════════════════════════════════
    with tab_net_tab:
        st.html(_section_title("NETWORK FABRIC"))

        net_detail = fetch_json("/network", api_url) or {}
        n_ew = net_detail.get("total_east_west_gbps", 0)
        n_ns = net_detail.get("total_north_south_gbps", 0)
        n_rdma = net_detail.get("total_rdma_gbps", 0)
        n_lat = net_detail.get("avg_fabric_latency_us", 5)
        n_loss = net_detail.get("total_packet_loss_pct", 0)
        n_crc = net_detail.get("total_crc_errors", 0)

        nc1 = _stat_card("EAST-WEST", f"{n_ew:.1f} Gbps", C_GREEN, "", "INTRA-DC TRAFFIC")
        nc2 = _stat_card("NORTH-SOUTH", f"{n_ns:.1f} Gbps", C_CYAN, "", "EXTERNAL TRAFFIC")
        nc3 = _stat_card("RDMA FABRIC", f"{n_rdma:.1f} Gbps", C_PURPLE, "", "GPU-TO-GPU")
        loss_c = C_RED if n_loss > 0.1 else (C_AMBER if n_loss > 0.01 else C_GREEN)
        nc4 = _stat_card("FABRIC LATENCY", f"{n_lat:.0f} us", loss_c, "",
                         f"LOSS: {n_loss:.3f}% / CRC: {n_crc}")

        st.html(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
            f'{nc1}{nc2}{nc3}{nc4}'
            f'</div>'
        )

        # Per-rack ToR switch table
        st.html(_section_title("TOR SWITCH TELEMETRY"))
        net_racks = net_detail.get("racks", [])
        if net_racks:
            nr_rows = ""
            for r in net_racks:
                tor_c = C_RED if r["tor_utilisation_pct"] > 80 else (C_AMBER if r["tor_utilisation_pct"] > 50 else C_GREEN)
                nr_rows += (
                    f'<tr style="border-bottom:1px solid {C_BORDER};">'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">R{r["rack_id"]:03d}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_GREEN};">{r["ingress_gbps"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_CYAN};">{r["egress_gbps"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_MUTED};">{r["intra_rack_gbps"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{tor_c};">{r["tor_utilisation_pct"]:.0f}%</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">{r["avg_latency_us"]:.0f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_AMBER};">{r["p99_latency_us"]:.0f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_PURPLE};">{r["rdma_tx_gbps"]:.1f}/{r["rdma_rx_gbps"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_MUTED};">{r["active_ports"]}/{r["total_ports"]}</td>'
                    f'</tr>'
                )
            st.html(
                f'<table style="width:100%;border-collapse:collapse;font-family:\'Courier New\',monospace;">'
                f'<thead><tr style="border-bottom:1px solid {C_BORDER};">'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">RACK</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">IN Gbps</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">OUT Gbps</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">INTRA</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">ToR%</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">AVG us</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">P99 us</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">RDMA TX/RX</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">PORTS</th>'
                f'</tr></thead><tbody>{nr_rows}</tbody></table>'
            )

        # Spine links
        spine = net_detail.get("spine_links", [])
        if spine:
            st.html(_section_title("SPINE FABRIC LINKS"))
            sp_rows = ""
            for s in spine:
                sp_rows += (
                    f'<tr style="border-bottom:1px solid {C_BORDER};">'
                    f'<td style="padding:4px 6px;font-size:10px;color:{C_TEXT};">R{s["src_rack_id"]:02d} ↔ R{s["dst_rack_id"]:02d}</td>'
                    f'<td style="padding:4px 6px;font-size:10px;color:{C_CYAN};">{s["bandwidth_gbps"]:.1f}</td>'
                    f'<td style="padding:4px 6px;font-size:10px;color:{C_AMBER};">{s["utilisation_pct"]:.1f}%</td>'
                    f'<td style="padding:4px 6px;font-size:10px;color:{C_TEXT};">{s["latency_us"]:.1f}</td>'
                    f'</tr>'
                )
            st.html(
                f'<table style="width:50%;border-collapse:collapse;font-family:\'Courier New\',monospace;">'
                f'<thead><tr style="border-bottom:1px solid {C_BORDER};">'
                f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">LINK</th>'
                f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">BW Gbps</th>'
                f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">UTIL%</th>'
                f'<th style="text-align:left;padding:4px 6px;font-size:9px;color:{C_LABEL};">LAT us</th>'
                f'</tr></thead><tbody>{sp_rows}</tbody></table>'
            )

    # ════════════════════════════════════════════════════
    # TAB: STORAGE
    # ════════════════════════════════════════════════════
    with tab_storage_tab:
        st.html(_section_title("STORAGE I/O"))

        sto_detail = fetch_json("/storage", api_url) or {}
        s_r_iops = sto_detail.get("total_read_iops", 0)
        s_w_iops = sto_detail.get("total_write_iops", 0)
        s_r_tp = sto_detail.get("total_read_throughput_gbps", 0)
        s_w_tp = sto_detail.get("total_write_throughput_gbps", 0)
        s_used = sto_detail.get("total_used_tb", 0)
        s_cap = sto_detail.get("total_capacity_tb", 1)
        s_r_lat = sto_detail.get("avg_read_latency_us", 80)
        s_w_lat = sto_detail.get("avg_write_latency_us", 20)

        sc1 = _stat_card("READ IOPS", f"{s_r_iops/1000:.0f}K", C_GREEN, "", f"LATENCY: {s_r_lat:.0f}us")
        sc2 = _stat_card("WRITE IOPS", f"{s_w_iops/1000:.0f}K", C_AMBER, "", f"LATENCY: {s_w_lat:.0f}us")
        bar_tp = _progress_bar(s_r_tp + s_w_tp, 200, C_CYAN,
                               f"R:{s_r_tp:.1f}G", f"W:{s_w_tp:.1f}G")
        sc3 = _stat_card("THROUGHPUT", f"{s_r_tp + s_w_tp:.1f} Gbps", C_CYAN, "", "", bar_tp)
        bar_cap = _progress_bar(s_used, s_cap, C_AMBER if s_used/s_cap > 0.7 else C_GREEN,
                                f"{s_used:.0f}TB", f"{s_cap:.0f}TB")
        sc4 = _stat_card("CAPACITY", f"{s_used/s_cap*100:.0f}%", C_AMBER, "", "USED", bar_cap)

        st.html(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
            f'{sc1}{sc2}{sc3}{sc4}'
            f'</div>'
        )

        # Per-rack storage table
        st.html(_section_title("PER-RACK NVMe SHELVES"))
        sto_racks = sto_detail.get("racks", [])
        if sto_racks:
            sr_rows = ""
            for r in sto_racks:
                health_c = C_GREEN if r["drive_health_pct"] > 90 else (C_AMBER if r["drive_health_pct"] > 70 else C_RED)
                sr_rows += (
                    f'<tr style="border-bottom:1px solid {C_BORDER};">'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">R{r["rack_id"]:03d}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_GREEN};">{r["read_iops"]/1000:.0f}K</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_AMBER};">{r["write_iops"]/1000:.0f}K</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_CYAN};">{r["read_throughput_gbps"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_CYAN};">{r["write_throughput_gbps"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">{r["avg_read_latency_us"]:.0f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_AMBER};">{r["p99_read_latency_us"]:.0f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">{r["used_tb"]:.1f}/{r["total_tb"]:.0f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{health_c};">{r["drive_health_pct"]:.1f}%</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_MUTED};">{r["queue_depth"]}</td>'
                    f'</tr>'
                )
            st.html(
                f'<table style="width:100%;border-collapse:collapse;font-family:\'Courier New\',monospace;">'
                f'<thead><tr style="border-bottom:1px solid {C_BORDER};">'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">RACK</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">R IOPS</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">W IOPS</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">R Gbps</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">W Gbps</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">R LAT</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">P99 LAT</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">USED/CAP TB</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">HEALTH</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">QD</th>'
                f'</tr></thead><tbody>{sr_rows}</tbody></table>'
            )

    # ════════════════════════════════════════════════════
    # TAB: COOLING
    # ════════════════════════════════════════════════════
    with tab_cool_tab:
        st.html(_section_title("COOLING PLANT"))

        cool_detail = fetch_json("/cooling", api_url) or {}
        c_cop = cool_detail.get("cop", 4.0)
        c_load = cool_detail.get("cooling_load_pct", 0)
        c_output = cool_detail.get("total_cooling_output_kw", 0)
        c_capacity = cool_detail.get("total_cooling_capacity_kw", 1)
        c_power = cool_detail.get("cooling_power_kw", 0)
        c_chw_s = cool_detail.get("chw_plant_supply_temp_c", 7)
        c_chw_r = cool_detail.get("chw_plant_return_temp_c", 12)
        c_chw_dt = cool_detail.get("chw_plant_delta_t_c", 5)
        c_pump_pwr = cool_detail.get("pump_power_kw", 2)
        c_pump_flow = cool_detail.get("pump_flow_rate_lps", 20)

        cop_c = C_GREEN if c_cop > 3.5 else (C_AMBER if c_cop > 2.5 else C_RED)
        bar_load = _progress_bar(c_load, 100,
                                 C_GREEN if c_load < 70 else (C_AMBER if c_load < 90 else C_RED),
                                 f"{c_load:.0f}%", "100%")
        cl1 = _stat_card("COP", f"{c_cop:.2f}", cop_c, "", "EFFICIENCY RATIO", bar_load)
        cl2 = _stat_card("COOLING OUTPUT", f"{c_output:.0f} kW", C_CYAN, "",
                         f"CAPACITY: {c_capacity:.0f} kW")
        cl3 = _stat_card("CHW TEMPS", f"{c_chw_s:.1f}C / {c_chw_r:.1f}C", C_BLUE, "",
                         f"DELTA-T: {c_chw_dt:.1f}C")
        cl4 = _stat_card("COOLING POWER", f"{c_power:.1f} kW", C_AMBER, "",
                         f"PUMP: {c_pump_pwr:.1f}kW / {c_pump_flow:.0f} L/s")

        st.html(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
            f'{cl1}{cl2}{cl3}{cl4}'
            f'</div>'
        )

        # Cooling tower
        tower = cool_detail.get("cooling_tower", {})
        if tower:
            st.html(_section_title("COOLING TOWER"))
            st.html(
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
                + _stat_card("WET BULB", f"{tower.get('wet_bulb_temp_c', 18):.1f}C", C_BLUE, "", "AMBIENT")
                + _stat_card("CONDENSER", f"{tower.get('condenser_supply_temp_c', 28):.1f}C / {tower.get('condenser_return_temp_c', 33):.1f}C",
                             C_CYAN, "", f"APPROACH: {tower.get('approach_temp_c', 5):.1f}C")
                + _stat_card("TOWER FAN", f"{tower.get('fan_speed_pct', 40):.0f}%", C_MUTED, "",
                             f"REJECTION: {tower.get('heat_rejection_kw', 0):.0f} kW")
                + f'</div>'
            )

        # CRAC units table
        crac_units = cool_detail.get("crac_units", [])
        if crac_units:
            st.html(_section_title("CRAC UNITS"))
            cr_rows = ""
            for u in crac_units:
                op_c = C_GREEN if u["operational"] else C_RED
                op_text = "ONLINE" if u["operational"] else "FAULT"
                cr_rows += (
                    f'<tr style="border-bottom:1px solid {C_BORDER};">'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">CRAC-{u["unit_id"]:02d}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{op_c};font-weight:600;">{op_text}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_CYAN};">{u["supply_air_temp_c"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_AMBER};">{u["return_air_temp_c"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">{u["fan_speed_pct"]:.0f}%</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">{u["airflow_cfm"]:.0f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_BLUE};">{u["chw_supply_temp_c"]:.1f}/{u["chw_return_temp_c"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_TEXT};">{u["chw_flow_rate_lps"]:.1f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_GREEN};">{u["cooling_output_kw"]:.0f}/{u["cooling_capacity_kw"]:.0f}</td>'
                    f'<td style="padding:6px;font-size:11px;color:{C_MUTED};">{u["load_pct"]:.0f}%</td>'
                    f'</tr>'
                )
            st.html(
                f'<table style="width:100%;border-collapse:collapse;font-family:\'Courier New\',monospace;">'
                f'<thead><tr style="border-bottom:1px solid {C_BORDER};">'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">UNIT</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">STATUS</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">SUPPLY C</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">RETURN C</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">FAN%</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">CFM</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">CHW S/R</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">FLOW L/s</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">OUT/CAP kW</th>'
                f'<th style="text-align:left;padding:6px;font-size:10px;color:{C_LABEL};">LOAD%</th>'
                f'</tr></thead><tbody>{cr_rows}</tbody></table>'
            )

    # ════════════════════════════════════════════════════
    # TAB: CARBON
    # ════════════════════════════════════════════════════
    with tab_carbon_tab:
        st.html(_section_title("CARBON & COST TRACKING"))

        bar_ci = _progress_bar(
            ci, 400,
            C_GREEN if ci < 180 else (C_AMBER if ci < 260 else C_RED),
            "LOW", "HIGH")
        cc1 = _stat_card("GRID CARBON", f"{ci:.0f} g/kWh",
                         C_GREEN if ci < 180 else C_AMBER, "", "", bar_ci)
        cc2 = _stat_card("CUMULATIVE CO2",
                         f"{carbon.get('cumulative_carbon_kg', 0):.1f} kg",
                         C_RED, "", "SINCE START")
        cc3 = _stat_card("ELEC PRICE",
                         f"&pound;{carbon.get('electricity_price_gbp_kwh', 0):.3f}",
                         C_AMBER, "", "GBP/kWh")
        cc4 = _stat_card("TOTAL COST",
                         f"&pound;{carbon.get('cumulative_cost_gbp', 0):.2f}",
                         C_PURPLE, "", "SINCE START")

        st.html(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
            f'{cc1}{cc2}{cc3}{cc4}'
            f'</div>'
        )

        if history and history.get("history"):
            import plotly.graph_objects as go

            def _term_chart2(df_in, cols, title, ylabel, colours=None):
                fig = go.Figure()
                default_c = [C_GREEN, C_CYAN, C_AMBER, C_RED, C_PURPLE, C_BLUE]
                colours = colours or default_c
                for i_c, col in enumerate(cols):
                    fig.add_trace(go.Scatter(
                        x=df_in["tick"], y=df_in[col], mode="lines",
                        name=col.replace("_", " ").upper(),
                        line=dict(color=colours[i_c % len(colours)], width=1.5),
                    ))
                fig.update_layout(
                    height=260,
                    margin=dict(l=40, r=10, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C_MUTED, size=10, family="Courier New"),
                    title=dict(text=title, font=dict(size=11, color=C_LABEL)),
                    yaxis=dict(gridcolor="rgba(30,50,70,0.5)", title=ylabel),
                    xaxis=dict(gridcolor="rgba(30,50,70,0.5)", title="TICK"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                )
                return fig

            c1, c2 = st.columns(2)
            with c1:
                fig = _term_chart2(df, ["carbon_intensity"],
                                   "GRID_CARBON_INTENSITY", "gCO2/kWh", [C_GREEN])
                st.plotly_chart(fig, use_container_width=True, key="carb_intensity")
            with c2:
                fig = _term_chart2(df, ["cumulative_carbon_kg"],
                                   "CUMULATIVE_EMISSIONS", "kg CO2", [C_RED])
                st.plotly_chart(fig, use_container_width=True, key="carb_cum")

            c1, c2 = st.columns(2)
            with c1:
                fig = _term_chart2(df, ["elec_price"],
                                   "SPOT_PRICE", "GBP/kWh", [C_AMBER])
                st.plotly_chart(fig, use_container_width=True, key="carb_price")
            with c2:
                fig = _term_chart2(df, ["cumulative_cost_gbp"],
                                   "CUMULATIVE_COST", "GBP", [C_PURPLE])
                st.plotly_chart(fig, use_container_width=True, key="carb_cost")

    # ════════════════════════════════════════════════════
    # TAB: EVAL
    # ════════════════════════════════════════════════════
    with tab_eval:
        st.html(_section_title("EVALUATION & SCORING"))

        # ── Agent + Scenario selector ─────────────────────
        agents_data = fetch_json("/eval/agents", api_url)
        agent_list = agents_data.get("agents", []) if agents_data else []
        agent_names = [a["name"] for a in agent_list] if agent_list else []

        scenarios = fetch_json("/eval/scenarios", api_url)
        scenario_list = scenarios.get("scenarios", []) if scenarios else []
        scenario_names = {s["scenario_id"]: f'{s["name"]}  ({s["duration_hours"]:.0f}h / {s["failure_count"]} failures)' for s in scenario_list}

        ev_a1, ev_a2 = st.columns([2, 3])
        with ev_a1:
            selected_agent = st.selectbox(
                "AGENT",
                options=agent_names if agent_names else ["(none)"],
                label_visibility="collapsed",
            )
        with ev_a2:
            selected_scenario = st.selectbox(
                "SCENARIO",
                options=[s["scenario_id"] for s in scenario_list] if scenario_list else ["steady_state"],
                format_func=lambda x: scenario_names.get(x, x),
                label_visibility="collapsed",
            )

        # Show scenario description
        sel_info = next((s for s in scenario_list if s["scenario_id"] == selected_scenario), None)
        if sel_info:
            st.html(
                f'<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:4px;'
                f'padding:10px 14px;margin:8px 0;font-size:11px;color:{C_MUTED};'
                f"font-family:'Courier New',monospace;\">"
                f'{sel_info.get("description", sel_info.get("name", ""))}'
                f'</div>'
            )

        # ── Custom scenario parameters (expandable) ───────
        _FAILURE_TYPES = ["crac_degraded", "crac_failure", "gpu_degraded", "pdu_spike", "network_partition"]

        # Defaults from selected predefined scenario
        default_duration = sel_info["duration_ticks"] if sel_info else 240
        default_seed = sel_info.get("rng_seed", 42) if sel_info else 42
        default_arrival = sel_info.get("mean_job_arrival_interval_s", 300.0) if sel_info else 300.0
        default_failures = sel_info.get("failure_injections", []) if sel_info else []

        with st.expander("CUSTOM SCENARIO PARAMETERS", expanded=False):
            cust_c1, cust_c2, cust_c3 = st.columns(3)
            with cust_c1:
                custom_duration = st.number_input(
                    "DURATION (ticks)",
                    min_value=10, max_value=5000,
                    value=default_duration,
                    key="custom_duration",
                )
            with cust_c2:
                custom_arrival = st.number_input(
                    "JOB ARRIVAL INTERVAL (s)",
                    min_value=10.0, max_value=3600.0,
                    value=float(default_arrival), step=10.0,
                    key="custom_arrival",
                )
            with cust_c3:
                custom_seed = st.number_input(
                    "RNG SEED",
                    min_value=0, max_value=99999,
                    value=default_seed,
                    key="custom_seed",
                )

            st.html(
                f'<div style="color:{C_RED};font-size:10px;letter-spacing:1.5px;'
                f'text-transform:uppercase;font-weight:600;margin:12px 0 6px 0;'
                f'border-bottom:1px solid {C_BORDER};padding-bottom:4px;">FAILURE INJECTIONS</div>'
            )

            # Initialise failure rows from scenario defaults (once)
            if "custom_failures_init" not in st.session_state:
                st.session_state["custom_failures_init"] = True
                st.session_state["num_custom_failures"] = len(default_failures)
                for i, fi in enumerate(default_failures):
                    st.session_state[f"fi_tick_{i}"] = fi.get("at_tick", 30)
                    st.session_state[f"fi_type_{i}"] = fi.get("failure_type", "crac_failure")
                    st.session_state[f"fi_target_{i}"] = fi.get("target", "crac-0")
                    st.session_state[f"fi_dur_{i}"] = fi.get("duration_s", 1800) or 0

            num_failures = st.session_state.get("num_custom_failures", 0)
            custom_failures: list[dict] = []
            for i in range(num_failures):
                fc1, fc2, fc3, fc4 = st.columns([1, 2, 2, 1])
                with fc1:
                    fi_tick = st.number_input("TICK", key=f"fi_tick_{i}", min_value=0, value=st.session_state.get(f"fi_tick_{i}", 30))
                with fc2:
                    fi_default_idx = _FAILURE_TYPES.index(st.session_state.get(f"fi_type_{i}", "crac_failure")) if st.session_state.get(f"fi_type_{i}", "crac_failure") in _FAILURE_TYPES else 0
                    fi_type = st.selectbox("TYPE", _FAILURE_TYPES, key=f"fi_type_{i}", index=fi_default_idx)
                with fc3:
                    fi_target = st.text_input("TARGET", key=f"fi_target_{i}", value=st.session_state.get(f"fi_target_{i}", "crac-0"))
                with fc4:
                    fi_dur = st.number_input("DUR (s)", key=f"fi_dur_{i}", min_value=0, value=st.session_state.get(f"fi_dur_{i}", 1800))
                custom_failures.append({"at_tick": fi_tick, "failure_type": fi_type, "target": fi_target, "duration_s": fi_dur if fi_dur > 0 else None})

            fb_c1, fb_c2, _ = st.columns([1, 1, 3])
            with fb_c1:
                if st.button("+ ADD FAILURE", use_container_width=True):
                    st.session_state["num_custom_failures"] = num_failures + 1
                    st.rerun()
            with fb_c2:
                if st.button("CLEAR FAILURES", use_container_width=True):
                    # Remove all fi_* keys
                    for k in list(st.session_state.keys()):
                        if k.startswith("fi_"):
                            del st.session_state[k]
                    st.session_state["num_custom_failures"] = 0
                    st.rerun()

        # ── Build override body ───────────────────────────
        def _build_overrides() -> dict:
            """Return custom override fields only if they differ from scenario defaults."""
            overrides: dict = {}
            if custom_duration != default_duration:
                overrides["duration_ticks"] = custom_duration
            if custom_seed != default_seed:
                overrides["rng_seed"] = custom_seed
            if abs(custom_arrival - default_arrival) > 0.01:
                overrides["mean_job_arrival_interval_s"] = custom_arrival
            # Compare failure injections
            if num_failures != len(default_failures):
                overrides["failure_injections"] = custom_failures
            else:
                # Check if any individual failure changed
                changed = False
                for i, fi in enumerate(custom_failures):
                    df = default_failures[i] if i < len(default_failures) else {}
                    if (fi.get("at_tick") != df.get("at_tick") or
                        fi.get("failure_type") != df.get("failure_type") or
                        fi.get("target") != df.get("target") or
                        fi.get("duration_s") != df.get("duration_s")):
                        changed = True
                        break
                if changed:
                    overrides["failure_injections"] = custom_failures
            return overrides

        # ── Control buttons ───────────────────────────────
        btn_c1, btn_c2 = st.columns([3, 1])
        with btn_c1:
            run_agent_btn = st.button("▸ RUN AGENT", use_container_width=True, type="primary")
        with btn_c2:
            run_baseline_btn = st.button("RUN BASELINE", use_container_width=True)

        # ── Execute runs ──────────────────────────────────
        eval_result = None
        baseline_result = None

        if run_agent_btn and selected_agent and selected_agent != "(none)":
            req_body = {"agent_name": selected_agent, "scenario_id": selected_scenario}
            req_body.update(_build_overrides())
            with st.spinner(f"RUNNING AGENT [{selected_agent.upper()}] ON {selected_scenario.upper()}..."):
                eval_result = post_json("/eval/run-agent", api_url, json=req_body)
                if eval_result:
                    st.session_state["last_eval"] = eval_result

        if run_baseline_btn:
            req_body = {"scenario_id": selected_scenario}
            req_body.update(_build_overrides())
            with st.spinner("COMPUTING BASELINE (no agent)..."):
                baseline_result = post_json("/eval/run-baseline", api_url, json=req_body)
                if baseline_result:
                    st.session_state["last_baseline"] = baseline_result

        # Use last results from session state
        if "last_eval" in st.session_state:
            eval_result = st.session_state["last_eval"]
        if "last_baseline" in st.session_state:
            baseline_result = st.session_state["last_baseline"]

        if eval_result or baseline_result:
            primary = eval_result or baseline_result

            # ── Composite score hero card ─────────────────
            comp_score = primary.get("composite_score", 0)
            if comp_score >= 70:
                score_colour = C_GREEN
                grade = "EXCELLENT"
            elif comp_score >= 50:
                score_colour = C_CYAN
                grade = "GOOD"
            elif comp_score >= 30:
                score_colour = C_AMBER
                grade = "FAIR"
            else:
                score_colour = C_RED
                grade = "POOR"

            run_label = primary.get("run_type", "agent").upper()
            dur_ticks = primary.get("duration_ticks", 0)

            # Baseline comparison text
            compare_html = ""
            if eval_result and baseline_result:
                delta = eval_result.get("composite_score", 0) - baseline_result.get("composite_score", 0)
                delta_c = C_GREEN if delta >= 0 else C_RED
                delta_sign = "+" if delta >= 0 else ""
                compare_html = (
                    f'<div style="color:{delta_c};font-size:14px;margin-top:6px;">'
                    f'{delta_sign}{delta:.1f} vs BASELINE ({baseline_result.get("composite_score", 0):.1f})'
                    f'</div>'
                )

            st.html(
                f'<div style="background:{C_CARD};border:2px solid {score_colour};border-radius:8px;'
                f'padding:24px;text-align:center;margin:12px 0;">'
                f'<div style="color:{C_MUTED};font-size:11px;letter-spacing:2px;text-transform:uppercase;">'
                f'COMPOSITE SCORE — {run_label} — {selected_scenario.upper()}</div>'
                f'<div style="color:{score_colour};font-size:64px;font-weight:700;'
                f"font-family:'JetBrains Mono','Courier New',monospace;line-height:1.1;margin:8px 0;\">"
                f'{comp_score:.1f}</div>'
                f'<div style="color:{score_colour};font-size:14px;letter-spacing:3px;font-weight:600;">'
                f'{grade}</div>'
                f'{compare_html}'
                f'<div style="color:{C_MUTED};font-size:10px;margin-top:8px;">'
                f'{dur_ticks} ticks / {dur_ticks * 60 / 3600:.1f}h simulated</div>'
                f'</div>'
            )

            # ── Radar chart ──────────────────────────────
            dims = primary.get("dimensions", [])
            if dims:
                import plotly.graph_objects as go

                dim_names = [d["name"].replace("_", " ").upper() for d in dims]
                dim_scores = [d["score"] for d in dims]

                # Close the polygon
                radar_names = dim_names + [dim_names[0]]
                radar_scores = dim_scores + [dim_scores[0]]

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=radar_scores,
                    theta=radar_names,
                    fill="toself",
                    name=run_label,
                    line=dict(color=C_CYAN, width=2),
                    fillcolor="rgba(0,212,255,0.15)",
                ))

                # Overlay baseline if available
                if baseline_result and baseline_result.get("dimensions"):
                    b_dims = baseline_result["dimensions"]
                    b_scores = [d["score"] for d in b_dims] + [b_dims[0]["score"]]
                    fig.add_trace(go.Scatterpolar(
                        r=b_scores,
                        theta=radar_names,
                        fill="toself",
                        name="BASELINE",
                        line=dict(color=C_MUTED, width=1, dash="dot"),
                        fillcolor="rgba(74,85,104,0.1)",
                    ))

                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            gridcolor="rgba(30,50,70,0.5)",
                            linecolor=C_BORDER,
                            tickfont=dict(color=C_MUTED, size=9),
                        ),
                        angularaxis=dict(
                            gridcolor="rgba(30,50,70,0.5)",
                            linecolor=C_BORDER,
                            tickfont=dict(color=C_LABEL, size=10),
                        ),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C_TEXT, family="Courier New"),
                    showlegend=True,
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=C_TEXT)),
                    height=400,
                    margin=dict(l=60, r=60, t=30, b=30),
                )

                st.plotly_chart(fig, use_container_width=True, key="eval_radar")

            # ── Dimension gauge cards ─────────────────────
            if dims:
                st.html(_section_title("DIMENSION SCORES"))

                dim_icons = {
                    "sla_quality": "SLA",
                    "energy_efficiency": "PWR",
                    "carbon": "CO2",
                    "thermal_safety": "TMP",
                    "cost": "GBP",
                    "infra_health": "HW",
                    "failure_response": "FIX",
                }

                dim_colours = {
                    "sla_quality": C_GREEN,
                    "energy_efficiency": C_CYAN,
                    "carbon": C_GREEN,
                    "thermal_safety": C_AMBER,
                    "cost": C_PURPLE,
                    "infra_health": C_BLUE,
                    "failure_response": C_RED,
                }

                cards_html = ""
                for d in dims:
                    dname = d["name"]
                    dscore = d["score"]
                    dweight = d["weight"]

                    if dscore >= 70:
                        s_colour = C_GREEN
                    elif dscore >= 50:
                        s_colour = C_CYAN
                    elif dscore >= 30:
                        s_colour = C_AMBER
                    else:
                        s_colour = C_RED

                    accent = dim_colours.get(dname, C_CYAN)
                    icon = dim_icons.get(dname, "")

                    # Top 2 metrics as subtitle
                    metrics = d.get("metrics", {})
                    metric_parts = []
                    for mk, mv in list(metrics.items())[:3]:
                        short_key = mk.replace("_pct", "%").replace("_gbp", "£").replace("_kg", "kg")
                        short_key = short_key.replace("_", " ").upper()[:16]
                        if isinstance(mv, float):
                            metric_parts.append(f"{short_key}: {mv:.1f}")
                        else:
                            metric_parts.append(f"{short_key}: {mv}")
                    metric_text = " / ".join(metric_parts)

                    bar_html = _progress_bar(dscore, 100, s_colour, f"{dscore:.0f}", "100")

                    cards_html += (
                        f'<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:6px;'
                        f'padding:14px 16px;position:relative;">'
                        f'<div style="position:absolute;top:10px;right:12px;color:{accent};font-size:10px;'
                        f'opacity:0.5;font-weight:700;letter-spacing:1px;">{icon}</div>'
                        f'<div style="color:{accent};font-size:10px;letter-spacing:1.5px;'
                        f'text-transform:uppercase;font-weight:600;margin-bottom:4px;">'
                        f'{dname.replace("_", " ")}</div>'
                        f'<div style="color:{s_colour};font-size:28px;font-weight:700;'
                        f"font-family:'JetBrains Mono','Courier New',monospace;\">{dscore:.1f}</div>"
                        f'<div style="color:{C_MUTED};font-size:9px;margin:2px 0;">WEIGHT: {dweight:.0%}</div>'
                        f'{bar_html}'
                        f'<div style="color:{C_MUTED};font-size:9px;margin-top:6px;'
                        f"font-family:'Courier New',monospace;word-break:break-all;\">"
                        f'{metric_text}</div>'
                        f'</div>'
                    )

                # Responsive grid: 4 cols on first row, 3 on second
                st.html(
                    f'<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:10px;">'
                    f'{cards_html}'
                    f'</div>'
                )

            # ── Expandable raw metrics ────────────────────
            if dims:
                with st.expander("RAW METRICS"):
                    for d in dims:
                        st.html(
                            f'<div style="color:{C_CYAN};font-size:11px;letter-spacing:1.5px;'
                            f'text-transform:uppercase;margin:12px 0 6px 0;font-weight:600;'
                            f'border-bottom:1px solid {C_BORDER};padding-bottom:4px;">'
                            f'{d["name"].replace("_", " ")}</div>'
                        )
                        metrics = d.get("metrics", {})
                        rows_html = ""
                        for mk, mv in metrics.items():
                            formatted = f"{mv:.4f}" if isinstance(mv, float) else str(mv)
                            rows_html += (
                                f'<tr style="border-bottom:1px solid {C_BORDER};">'
                                f'<td style="padding:4px 8px;font-size:10px;color:{C_LABEL};'
                                f"font-family:'Courier New',monospace;\">{mk}</td>"
                                f'<td style="padding:4px 8px;font-size:10px;color:{C_WHITE};'
                                f"font-family:'Courier New',monospace;text-align:right;\">{formatted}</td>"
                                f'</tr>'
                            )
                        if rows_html:
                            st.html(
                                f'<table style="width:100%;border-collapse:collapse;">'
                                f'<tbody>{rows_html}</tbody></table>'
                            )

        else:
            st.html(
                f'<div style="text-align:center;padding:60px 20px;">'
                f'<div style="color:{C_MUTED};font-size:13px;letter-spacing:2px;'
                f'text-transform:uppercase;">NO EVALUATION RESULTS</div>'
                f'<div style="color:{C_MUTED};font-size:11px;margin-top:12px;">'
                f'Select an agent and scenario above, then click RUN AGENT to begin.</div>'
                f'</div>'
            )

    # ════════════════════════════════════════════════════
    # TAB: LEADERBOARD
    # ════════════════════════════════════════════════════
    with tab_leaderboard:
        st.html(_section_title("AGENT LEADERBOARD"))

        leaderboard_data = fetch_json("/eval/leaderboard", api_url)
        lb_entries = leaderboard_data.get("entries", []) if leaderboard_data else []

        if lb_entries:
            lb_df = pd.DataFrame(lb_entries)

            # ── Summary cards ────────────────────────────
            total_runs = len(lb_df)
            unique_agents = lb_df["agent_name"].nunique()
            best_score = lb_df["composite_score"].max()
            best_agent = lb_df.loc[lb_df["composite_score"].idxmax(), "agent_name"] if not lb_df.empty else "—"

            lb_s1 = _stat_card("TOTAL RUNS", str(total_runs), C_CYAN, "", "ALL TIME")
            lb_s2 = _stat_card("UNIQUE AGENTS", str(unique_agents), C_PURPLE, "", "REGISTERED")
            lb_s3 = _stat_card("BEST SCORE", f"{best_score:.1f}", C_GREEN, "",
                               f"AGENT: {best_agent.upper()}")
            avg_score = lb_df["composite_score"].mean()
            lb_s4 = _stat_card("AVG SCORE", f"{avg_score:.1f}",
                               C_AMBER if avg_score < 50 else C_GREEN, "", "ALL RUNS")

            st.html(
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:12px;">'
                f'{lb_s1}{lb_s2}{lb_s3}{lb_s4}'
                f'</div>'
            )

            # ── Scenario filter ──────────────────────────
            all_scenarios = ["ALL"] + sorted(lb_df["scenario_id"].unique().tolist())
            lb_filter = st.selectbox("FILTER BY SCENARIO", options=all_scenarios,
                                     label_visibility="collapsed", key="lb_filter")
            filtered_df = lb_df if lb_filter == "ALL" else lb_df[lb_df["scenario_id"] == lb_filter]

            # ── Leaderboard table ────────────────────────
            dim_cols = ["sla_quality", "energy_efficiency", "carbon", "thermal_safety",
                        "cost", "infra_health", "failure_response"]

            sorted_df = filtered_df.sort_values("composite_score", ascending=False)

            header_cells = (
                f'<th style="text-align:left;padding:8px;font-size:10px;color:{C_LABEL};font-weight:600;">RANK</th>'
                f'<th style="text-align:left;padding:8px;font-size:10px;color:{C_LABEL};font-weight:600;">AGENT</th>'
                f'<th style="text-align:left;padding:8px;font-size:10px;color:{C_LABEL};font-weight:600;">SCENARIO</th>'
                f'<th style="text-align:right;padding:8px;font-size:10px;color:{C_LABEL};font-weight:600;">COMPOSITE</th>'
            )
            for dc in dim_cols:
                short = dc.replace("_", " ").upper()[:10]
                header_cells += (
                    f'<th style="text-align:right;padding:8px;font-size:9px;color:{C_LABEL};font-weight:600;">{short}</th>'
                )
            header_cells += (
                f'<th style="text-align:right;padding:8px;font-size:10px;color:{C_LABEL};font-weight:600;">TIME</th>'
            )

            rows_html = ""
            for rank, (_, row) in enumerate(sorted_df.head(20).iterrows(), 1):
                comp = row.get("composite_score", 0)
                if rank == 1:
                    rank_colour = C_GREEN
                elif rank <= 3:
                    rank_colour = C_CYAN
                else:
                    rank_colour = C_TEXT

                comp_colour = C_GREEN if comp >= 70 else (C_CYAN if comp >= 50 else (C_AMBER if comp >= 30 else C_RED))

                cells = (
                    f'<td style="padding:8px;font-size:11px;color:{rank_colour};font-weight:700;">#{rank}</td>'
                    f'<td style="padding:8px;font-size:11px;color:{C_WHITE};font-weight:600;">{row["agent_name"].upper()}</td>'
                    f'<td style="padding:8px;font-size:11px;color:{C_MUTED};">{row["scenario_id"]}</td>'
                    f'<td style="padding:8px;font-size:12px;color:{comp_colour};font-weight:700;text-align:right;">{comp:.1f}</td>'
                )
                for dc in dim_cols:
                    val = row.get(dc, 0)
                    dc_c = C_GREEN if val >= 70 else (C_CYAN if val >= 50 else (C_AMBER if val >= 30 else C_RED))
                    cells += f'<td style="padding:8px;font-size:10px;color:{dc_c};text-align:right;">{val:.0f}</td>'

                ts = str(row.get("timestamp", ""))[:16]
                cells += f'<td style="padding:8px;font-size:9px;color:{C_MUTED};text-align:right;">{ts}</td>'

                rows_html += f'<tr style="border-bottom:1px solid {C_BORDER};">{cells}</tr>'

            st.html(
                f'<table style="width:100%;border-collapse:collapse;font-family:\'Courier New\',monospace;">'
                f'<thead><tr style="border-bottom:1px solid {C_BORDER};">{header_cells}</tr></thead>'
                f'<tbody>{rows_html}</tbody></table>'
            )

            # ── Comparison radar chart ───────────────────
            st.html(_section_title("AGENT COMPARISON"))

            unique_agent_list = sorted(lb_df["agent_name"].unique().tolist())
            if len(unique_agent_list) >= 1:
                compare_agents = st.multiselect(
                    "SELECT AGENTS TO COMPARE",
                    options=unique_agent_list,
                    default=unique_agent_list[:min(3, len(unique_agent_list))],
                    label_visibility="collapsed",
                    key="lb_compare",
                )

                if compare_agents:
                    import plotly.graph_objects as go

                    radar_colours = [C_CYAN, C_GREEN, C_PURPLE, C_AMBER, C_RED, C_BLUE]
                    fig = go.Figure()

                    for i, agent_name in enumerate(compare_agents[:5]):
                        agent_df = filtered_df[filtered_df["agent_name"] == agent_name]
                        if agent_df.empty:
                            continue
                        # Use best run for this agent
                        best_row = agent_df.loc[agent_df["composite_score"].idxmax()]
                        scores = [best_row.get(dc, 0) for dc in dim_cols]
                        labels = [dc.replace("_", " ").upper() for dc in dim_cols]

                        # Close polygon
                        scores_closed = scores + [scores[0]]
                        labels_closed = labels + [labels[0]]

                        colour = radar_colours[i % len(radar_colours)]
                        fig.add_trace(go.Scatterpolar(
                            r=scores_closed,
                            theta=labels_closed,
                            fill="toself",
                            name=agent_name.upper(),
                            line=dict(color=colour, width=2),
                            fillcolor=hex_to_rgba(colour, 0.1) if colour.startswith("#") else colour,
                        ))

                    fig.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(
                                visible=True, range=[0, 100],
                                gridcolor="rgba(30,50,70,0.5)",
                                linecolor=C_BORDER,
                                tickfont=dict(color=C_MUTED, size=9),
                            ),
                            angularaxis=dict(
                                gridcolor="rgba(30,50,70,0.5)",
                                linecolor=C_BORDER,
                                tickfont=dict(color=C_LABEL, size=10),
                            ),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=C_TEXT, family="Courier New"),
                        showlegend=True,
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=C_TEXT)),
                        height=400,
                        margin=dict(l=60, r=60, t=30, b=30),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="lb_radar")

            # ── Score history chart ──────────────────────
            if "timestamp" in lb_df.columns and len(lb_df) > 1:
                st.html(_section_title("SCORE HISTORY"))

                import plotly.graph_objects as go

                history_colours = [C_CYAN, C_GREEN, C_PURPLE, C_AMBER, C_RED, C_BLUE]
                fig_hist = go.Figure()

                for i, agent_name in enumerate(unique_agent_list[:6]):
                    agent_df = filtered_df[filtered_df["agent_name"] == agent_name].sort_values("timestamp")
                    if len(agent_df) < 1:
                        continue
                    colour = history_colours[i % len(history_colours)]
                    fig_hist.add_trace(go.Scatter(
                        x=list(range(len(agent_df))),
                        y=agent_df["composite_score"].tolist(),
                        mode="lines+markers",
                        name=agent_name.upper(),
                        line=dict(color=colour, width=2),
                        marker=dict(size=6),
                    ))

                fig_hist.update_layout(
                    height=260,
                    margin=dict(l=40, r=10, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C_MUTED, size=10, family="Courier New"),
                    title=dict(text="COMPOSITE_SCORE_OVER_TIME", font=dict(size=11, color=C_LABEL)),
                    yaxis=dict(gridcolor="rgba(30,50,70,0.5)", title="SCORE", range=[0, 100]),
                    xaxis=dict(gridcolor="rgba(30,50,70,0.5)", title="RUN #"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                )
                st.plotly_chart(fig_hist, use_container_width=True, key="lb_history")

        else:
            st.html(
                f'<div style="text-align:center;padding:60px 20px;">'
                f'<div style="color:{C_MUTED};font-size:13px;letter-spacing:2px;'
                f'text-transform:uppercase;">NO LEADERBOARD DATA</div>'
                f'<div style="color:{C_MUTED};font-size:11px;margin-top:12px;">'
                f'Run an agent from the EVAL tab to populate the leaderboard.</div>'
                f'</div>'
            )

    # ── Sidebar: auto-refresh ───────────────────────────
    st.sidebar.html(f'<div style="height:20px"></div>')
    refresh_interval = st.sidebar.slider("REFRESH (s)", 1, 10, 2,
                                          label_visibility="visible")
    auto_refresh = st.sidebar.checkbox("AUTO-REFRESH", value=is_running)
    if st.sidebar.button("REFRESH NOW"):
        st.rerun()

    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
