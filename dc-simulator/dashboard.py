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


def fetch_json(path: str, base_url: str) -> dict | None:
    try:
        r = httpx.get(f"{base_url}{path}", timeout=2.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_json(path: str, base_url: str, json: dict | None = None) -> dict | None:
    try:
        r = httpx.post(f"{base_url}{path}", json=json or {}, timeout=5.0)
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
        f'.block-container {{ padding-top: 0 !important; max-width: 100% !important; }}'
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
        f'<div style="background:{C_CARD};border-bottom:1px solid {C_BORDER};padding:8px 20px;'
        f'display:flex;align-items:center;justify-content:space-between;margin:-1rem -1rem 12px -1rem;">'
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
    tab_overview, tab_infra, tab_fleet, tab_carbon_tab = st.tabs(
        ["OVERVIEW", "INFRASTRUCTURE", "FLEET", "CARBON"]
    )

    thermal_racks = thermal.get("racks", [])
    power_racks = power.get("racks", [])

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
