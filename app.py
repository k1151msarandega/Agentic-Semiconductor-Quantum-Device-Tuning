"""
app.py — SimQuantum Mission Control
====================================
Streamlit demo for AMD Developer Hackathon.

Run from the repo root:
    pip install streamlit plotly
    streamlit run app.py

Optional — enable Dr. Q LLM narrator (requires MI300X):
    $env:QDOT_LLM_BASE_URL = "http://129.212.186.42:8000/v1"
    $env:QDOT_LLM_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Make sure repo root is on the path when run from anywhere
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SimQuantum Mission Control",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (heavy qdot deps only after page is live)
# ---------------------------------------------------------------------------
def _import_qdot():
    from qdot.core.state import ExperimentState
    from qdot.core.types import ChargeLabel, TuningStage
    from qdot.core.governance import GovernanceLogger
    from qdot.core.hitl import HITLManager
    from qdot.hardware.safety import SafetyCritic
    from qdot.perception.dqc import DQCGatekeeper
    from qdot.simulator.cim import CIMSimulatorAdapter
    from qdot.agent.executive import ExecutiveAgent
    return (ExperimentState, ChargeLabel, TuningStage, GovernanceLogger,
            HITLManager, SafetyCritic, DQCGatekeeper, CIMSimulatorAdapter,
            ExecutiveAgent)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
STAGE_META = {
    "BOOTSTRAPPING":       ("⚡", "Electrical integrity check",    False),
    "COARSE_SURVEY":       ("🔭", "Locate Coulomb signal",          False),
    "HYPERSURFACE_SEARCH": ("🎯", "Navigate to charge boundary",    False),
    "CHARGE_ID":           ("🧬", "Classify charge regime",          False),
    "NAVIGATION":          ("🧭", "Navigate to (1,1) — Phase 3",    True),
    "VERIFICATION":        ("✅", "Confirm (1,1) stability — Phase 3", True),
}
DEMO_STAGES = ["BOOTSTRAPPING", "COARSE_SURVEY", "HYPERSURFACE_SEARCH", "CHARGE_ID"]
ALL_STAGES  = list(STAGE_META.keys())

SPY_AGENTS = [
    # (key, emoji, label, role, active_in_stages)
    ("perception", "🔬", "Perception", "Quality Inspectors",
     {"BOOTSTRAPPING", "COARSE_SURVEY", "CHARGE_ID"}),
    ("executive",  "🏛️", "Executive",  "Mission Conductor",
     set(ALL_STAGES)),                           # always active
    ("planning",   "🗺️", "Planning",   "Strategic Navigators",
     {"HYPERSURFACE_SEARCH", "NAVIGATION"}),
    ("safety",     "🏰", "Safety",     "Hardware Marshals",
     {"NAVIGATION", "VERIFICATION"}),
    ("hitl",       "🛑", "HITL",       "Human Governor",
     set()),                                      # lit dynamically
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, .stApp {
    background: #060b16 !important;
    font-family: 'DM Sans', sans-serif;
    color: #c8d8e8;
}

/* ── Scanline overlay ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,200,0.012) 2px, rgba(0,255,200,0.012) 4px
    );
    pointer-events: none; z-index: 9999;
}

/* ── Header ── */
h1 { font-family: 'Space Mono', monospace !important; color: #00ffcc !important;
     letter-spacing: -1px; text-shadow: 0 0 30px rgba(0,255,200,0.4); }
h3 { font-family: 'Space Mono', monospace !important; color: #88aacc !important;
     font-size: 13px !important; letter-spacing: 2px; text-transform: uppercase; }

/* ── Stage pills ── */
.stage-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
.pill {
    font-family: 'Space Mono', monospace;
    font-size: 10px; padding: 5px 10px; border-radius: 3px;
    display: inline-flex; flex-direction: column; align-items: center;
    min-width: 90px; text-align: center; line-height: 1.4;
}
.pill-done    { background:#081e10; border:1px solid #1a6b3c; color:#3ddc78; }
.pill-active  { background:#08192a; border:1px solid #0066ff;  color:#66aaff;
                box-shadow: 0 0 14px rgba(0,102,255,0.35);
                animation: pillPulse 2s ease-in-out infinite; }
.pill-pending { background:#0a0f1a; border:1px solid #1a2535; color:#3a5070; }
.pill-phase3  { background:#080a0e; border:1px solid #131920; color:#2a3540;
                font-style: italic; }
@keyframes pillPulse {
    0%,100% { box-shadow: 0 0 14px rgba(0,102,255,0.35); }
    50%      { box-shadow: 0 0 28px rgba(0,102,255,0.65); }
}

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-bottom:12px; }
.kpi {
    background: #080f1e; border: 1px solid #0d2040;
    border-radius: 6px; padding: 10px 8px; text-align: center;
}
.kpi-val { font-family:'Space Mono',monospace; font-size:20px; color:#00ffcc; }
.kpi-lbl { font-size:10px; color:#4a6a8a; margin-top:3px; letter-spacing:1px; }

/* ── Narration feed ── */
.feed { max-height: 380px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
.entry {
    font-family: 'Space Mono', monospace; font-size: 11px;
    padding: 8px 12px; border-radius: 4px; line-height: 1.6;
}
.entry-transition { background:#060e1e; border-left:3px solid #0066ff; color:#88aadd; }
.entry-exception  { background:#180808; border-left:3px solid #ff3333; color:#ff9999; }
.entry-hitl       { background:#1a1200; border-left:3px solid #ffaa00; color:#ffdd88; }
.entry-summary    { background:#051510; border-left:3px solid #00cc66; color:#66ffbb; }

/* ── Spy panel ── */
.spy-row { display: grid; grid-template-columns: repeat(5,1fr); gap: 6px; margin-top: 8px; }
.spy {
    background: #06090f; border: 1px solid #0d1520;
    border-radius: 6px; padding: 10px 6px; text-align: center;
    font-size: 10px; transition: all 0.4s;
}
.spy-on  { border-color: #0066ff; background: #060d1e;
           box-shadow: 0 0 16px rgba(0,102,255,0.3); }
.spy-hitl { border-color: #ffaa00; background: #0e0a00;
            box-shadow: 0 0 16px rgba(255,170,0,0.3);
            animation: hitlPulse 1.5s ease-in-out infinite; }
@keyframes hitlPulse {
    0%,100% { box-shadow: 0 0 16px rgba(255,170,0,0.3); }
    50%      { box-shadow: 0 0 32px rgba(255,170,0,0.6); }
}
.spy-em   { font-size: 22px; margin-bottom: 4px; }
.spy-name { color: #88aacc; font-weight: bold; margin-bottom: 2px; }
.spy-role { color: #3a5070; font-size: 9px; }
.spy-dot  { width:8px; height:8px; border-radius:50%; margin:5px auto 0;
            background:#1a3050; }
.spy-dot-on   { background: #00ffcc; box-shadow: 0 0 8px #00ffcc; }
.spy-dot-hitl { background: #ffaa00; box-shadow: 0 0 8px #ffaa00; }

/* ── Section dividers ── */
.sect { color:#1e3a5a; border:none; border-top:1px solid #0d2035;
        margin: 16px 0 12px; }

/* ── Status badge ── */
.badge-live { color:#00ffcc; font-size:11px; font-family:'Space Mono',monospace; }
.badge-idle { color:#ff3366; font-size:11px; font-family:'Space Mono',monospace; }

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 6px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #040810 !important; }

/* ── Streamlit element overrides ── */
div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "agent": None,
        "exp_state": None,
        "narrator": None,
        "done_event": None,
        "thread": None,
        "running": False,
        "run_count": 0,
        "use_llm": False,
        "use_cnn": True,
        "meas_budget": 2048,
        "max_steps": 100,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def _make_agent(use_llm: bool, use_cnn: bool, meas_budget: int, max_steps: int):
    (ExperimentState, ChargeLabel, TuningStage, GovernanceLogger,
     HITLManager, SafetyCritic, DQCGatekeeper, CIMSimulatorAdapter,
     ExecutiveAgent) = _import_qdot()

    # Randomise E_c slightly so each demo run looks different
    rng = np.random.default_rng()
    E_c = float(rng.uniform(2.2, 2.8))

    adapter = CIMSimulatorAdapter(
        device_id="demo_qdot",
        params={
            "E_c1": E_c, "E_c2": E_c * float(rng.uniform(0.95, 1.05)),
            "t_c": 0.05, "T": 0.015,
            "lever_arm": float(rng.uniform(0.70, 0.80)),
            "noise_level": 0.02,
        },
    )

    state = ExperimentState.new(
        device_id="demo_qdot",
        target_label=ChargeLabel.DOUBLE_DOT,
    )

    inspection_agent = None
    if use_cnn:
        try:
            from qdot.perception.inspector import InspectionAgent
            inspection_agent = InspectionAgent()
        except Exception as exc:
            st.sidebar.warning(f"CNN unavailable: {exc}")

    if not use_llm:
        os.environ.pop("QDOT_LLM_BASE_URL", None)

    run_dir = Path("results") / "demo" / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = ExecutiveAgent(
        state=state,
        adapter=adapter,
        inspection_agent=inspection_agent,
        dqc=DQCGatekeeper(),
        safety_critic=SafetyCritic(
            voltage_bounds=state.voltage_bounds,
            l1_max=0.10,
        ),
        hitl_manager=HITLManager(),
        governance_logger=GovernanceLogger(
            run_id=state.run_id,
            log_dir=str(run_dir / "governance"),
        ),
        max_steps=max_steps,
        measurement_budget=meas_budget,
    )
    return agent, state, agent.narrator


def _run_thread(agent, done_event):
    try:
        agent.run()
    except Exception:
        pass
    finally:
        done_event.set()


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#04080f",
    font=dict(color="#6a8aaa", size=10, family="Space Mono"),
    margin=dict(l=8, r=8, t=36, b=8),
)


def _chart_stability(state) -> go.Figure:
    """Latest 2D measurement as a charge stability diagram."""
    arr, m = None, None
    for m_obj in reversed(list(state.measurements.values())):
        if m_obj.is_2d and m_obj.array is not None:
            a = np.asarray(m_obj.array)
            if a.ndim == 2:
                arr, m = a, m_obj
                break

    fig = go.Figure()
    if arr is None:
        fig.add_annotation(text="Awaiting first scan...", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#1e3a5a", size=13))
    else:
        v1lo, v1hi = m.v1_range or (-3, 3)
        v2lo, v2hi = m.v2_range or (-3, 3)
        fig.add_trace(go.Heatmap(
            z=arr,
            x=np.linspace(v1lo, v1hi, arr.shape[1]),
            y=np.linspace(v2lo, v2hi, arr.shape[0]),
            colorscale=[
                [0.0, "#04080f"], [0.3, "#0a2040"], [0.6, "#0044aa"],
                [0.8, "#00aaff"], [1.0, "#00ffcc"],
            ],
            showscale=False,
        ))
        # Current voltage crosshair
        vg1 = state.current_voltage.vg1
        vg2 = state.current_voltage.vg2
        for sh in [
            dict(type="line", x0=vg1, x1=vg1, y0=v2lo, y1=v2hi,
                 line=dict(color="#ff3366", width=1, dash="dot")),
            dict(type="line", x0=v1lo, x1=v1hi, y0=vg2, y1=vg2,
                 line=dict(color="#ff3366", width=1, dash="dot")),
        ]:
            fig.add_shape(**sh)

    fig.update_layout(
        title=dict(text="Charge Stability Diagram", font=dict(size=11)),
        xaxis=dict(title="Vg1 (V)", gridcolor="#0d1e30", showgrid=True),
        yaxis=dict(title="Vg2 (V)", gridcolor="#0d1e30", showgrid=True),
        height=260, **_DARK_LAYOUT,
    )
    return fig


def _chart_belief(state) -> go.Figure:
    """3×3 belief state heatmap with (1,1) target highlighted."""
    probs = state.belief.charge_probs
    z = np.zeros((3, 3))
    for (n1, n2), p in probs.items():
        if 0 <= n1 <= 2 and 0 <= n2 <= 2:
            z[n2][n1] = float(p)

    text = [[f"{z[j][i]:.2f}" for i in range(3)] for j in range(3)]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=["N₁=0","N₁=1","N₁=2"], y=["N₂=0","N₂=1","N₂=2"],
        colorscale=[[0,"#04080f"],[0.5,"#0044aa"],[1,"#00ffcc"]],
        showscale=False, zmin=0, zmax=1,
        text=text, texttemplate="%{text}",
        textfont=dict(size=13, color="white", family="Space Mono"),
    ))
    # Highlight target (1,1)
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.5, y1=1.5,
                  line=dict(color="#00ffcc", width=2))
    fig.add_annotation(x=1, y=1, text="TARGET", showarrow=False,
                       yshift=22, font=dict(color="#00ffcc", size=9, family="Space Mono"))

    fig.update_layout(
        title=dict(text="Belief State  P(N₁,N₂ | obs)", font=dict(size=11)),
        height=240, **_DARK_LAYOUT,
    )
    return fig


def _chart_trajectory(state) -> go.Figure | None:
    traj = state.trajectory
    if len(traj) < 2:
        return None
    vg1s = [v.vg1 for v in traj]
    vg2s = [v.vg2 for v in traj]
    n = len(vg1s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vg1s, y=vg2s, mode="lines",
        line=dict(color="rgba(0,102,255,0.3)", width=1),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=vg1s, y=vg2s, mode="markers",
        marker=dict(size=5, color=list(range(n)),
                    colorscale=[[0,"#0d2040"],[1,"#00ffcc"]], showscale=False),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[vg1s[-1]], y=[vg2s[-1]], mode="markers",
        marker=dict(size=10, color="#ff3366", symbol="x-thin", line=dict(width=2)),
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Voltage Trajectory", font=dict(size=11)),
        xaxis=dict(title="Vg1", gridcolor="#0d1e30"),
        yaxis=dict(title="Vg2", gridcolor="#0d1e30"),
        height=200, **_DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Stage pill renderer
# ---------------------------------------------------------------------------
def _render_stage_pills(current_stage_name: str):
    try:
        cur_idx = ALL_STAGES.index(current_stage_name)
    except ValueError:
        cur_idx = -1

    pills_html = '<div class="stage-row">'
    for i, sname in enumerate(ALL_STAGES):
        icon, desc, is_phase3 = STAGE_META[sname]
        if is_phase3:
            css = "pill pill-phase3"
            badge = "○"
        elif i < cur_idx:
            css = "pill pill-done"
            badge = "✓"
        elif i == cur_idx:
            css = "pill pill-active"
            badge = "●"
        else:
            css = "pill pill-pending"
            badge = "○"

        short = sname.replace("_", " ").title()
        pills_html += (
            f'<div class="{css}">'
            f'<span style="font-size:16px">{icon}</span>'
            f'<span>{short}</span>'
            f'<span style="font-size:9px;opacity:0.7">{badge}</span>'
            f'</div>'
        )
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Spy panel renderer
# ---------------------------------------------------------------------------
def _render_spy_panel(current_stage: str, hitl_active: bool):
    img_path = Path(__file__).parent / "assets" / "simquantum.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    row_html = '<div class="spy-row">'
    for key, em, label, role, active_stages in SPY_AGENTS:
        if key == "hitl":
            on = hitl_active
        else:
            on = current_stage in active_stages

        if key == "hitl" and on:
            css = "spy spy-hitl"
            dot_css = "spy-dot spy-dot-hitl"
        elif on:
            css = "spy spy-on"
            dot_css = "spy-dot spy-dot-on"
        else:
            css = "spy"
            dot_css = "spy-dot"

        row_html += (
            f'<div class="{css}">'
            f'<div class="spy-em">{em}</div>'
            f'<div class="spy-name">{label}</div>'
            f'<div class="spy-role">{role}</div>'
            f'<div class="{dot_css}"></div>'
            f'</div>'
        )
    row_html += '</div>'
    st.markdown(row_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Narration feed renderer
# ---------------------------------------------------------------------------
def _render_feed(events):
    if not events:
        st.markdown(
            '<div style="color:#1e3a5a;font-size:12px;font-family:Space Mono,'
            'monospace;padding:20px;text-align:center">'
            'Dr. Q is observing.<br>Narration appears on anomalies, HITL gates,'
            ' and stage transitions.</div>',
            unsafe_allow_html=True,
        )
        return

    kind_map = {
        "transition": ("entry entry-transition", "◈ TRANSITION"),
        "exception":  ("entry entry-exception",  "◉ ANOMALY"),
        "hitl":       ("entry entry-hitl",        "◆ HITL GATE"),
        "summary":    ("entry entry-summary",     "◎ SUMMARY"),
        "ask":        ("entry entry-transition",  "◇ QUERY"),
    }

    html = '<div class="feed">'
    for ev in reversed(events[-12:]):
        css, prefix = kind_map.get(ev.kind, ("entry entry-transition", "○"))
        body = ev.response if ev.response else ev.description[:160] + "…"
        # Escape HTML special chars in body
        body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += (
            f'<div class="{css}">'
            f'<span style="opacity:0.5;font-size:9px">'
            f'{prefix} · step {ev.step} · {ev.stage}</span><br>'
            f'{body}'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ CONFIGURATION")
    st.divider()

    use_llm = st.toggle(
        "Dr. Q LLM Narrator",
        value=st.session_state.use_llm,
        help="Requires Qwen2.5 running on AMD MI300X. Set QDOT_LLM_BASE_URL.",
        disabled=st.session_state.running,
    )
    use_cnn = st.toggle(
        "CNN Charge Classifier",
        value=st.session_state.use_cnn,
        help="Requires checkpoint in experiments/checkpoints/phase1/",
        disabled=st.session_state.running,
    )
    st.session_state.use_llm = use_llm
    st.session_state.use_cnn = use_cnn

    meas_budget = st.slider(
        "Measurement Budget", 512, 4096, st.session_state.meas_budget, 128,
        disabled=st.session_state.running,
    )
    max_steps = st.slider(
        "Max Steps", 10, 200, st.session_state.max_steps, 10,
        disabled=st.session_state.running,
    )
    st.session_state.meas_budget = meas_budget
    st.session_state.max_steps = max_steps

    st.divider()
    st.markdown("### 🖥️ AMD MI300X")
    llm_url = st.text_input(
        "vLLM endpoint",
        value=os.environ.get("QDOT_LLM_BASE_URL", ""),
        placeholder="http://129.212.186.42:8000/v1",
        disabled=st.session_state.running,
    )
    if llm_url:
        os.environ["QDOT_LLM_BASE_URL"] = llm_url

    st.divider()
    # Run history
    if st.session_state.run_count > 0:
        st.markdown(f"**Runs this session:** {st.session_state.run_count}")

    st.divider()
    st.markdown(
        '<div style="font-size:10px;color:#1e3a5a;font-family:Space Mono,monospace">'
        'Agentic Quantum Dot Tuning<br>AMD Developer Hackathon 2025<br>'
        'BOOTSTRAPPING → CHARGE_ID  live<br>NAVIGATION → Phase 3</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header row
# ---------------------------------------------------------------------------
hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.markdown("# ⚛️ SimQuantum Mission Control")
    st.markdown(
        '<span style="font-size:12px;color:#3a6a8a;font-family:Space Mono,monospace">'
        'Autonomous quantum dot tuning · AMD MI300X · Qwen2.5-1.5B · ROCm</span>',
        unsafe_allow_html=True,
    )
with hdr_right:
    if st.session_state.running:
        st.markdown(
            '<div style="text-align:right;padding-top:24px">'
            '<span class="badge-live">● LIVE</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:right;padding-top:24px">'
            '<span class="badge-idle">● IDLE</span></div>',
            unsafe_allow_html=True,
        )

# Launch / stop buttons
btn1, btn2, _ = st.columns([1.4, 0.8, 3])
with btn1:
    launch = st.button(
        "🚀  LAUNCH TUNING RUN",
        type="primary",
        disabled=st.session_state.running,
        use_container_width=True,
    )
with btn2:
    stop = st.button(
        "⏹ STOP",
        disabled=not st.session_state.running,
        use_container_width=True,
    )

if launch and not st.session_state.running:
    agent, exp_state, narrator = _make_agent(
        use_llm, use_cnn, meas_budget, max_steps
    )
    done_event = threading.Event()
    thread = threading.Thread(
        target=_run_thread, args=(agent, done_event), daemon=True
    )
    st.session_state.agent      = agent
    st.session_state.exp_state  = exp_state
    st.session_state.narrator   = narrator
    st.session_state.done_event = done_event
    st.session_state.thread     = thread
    st.session_state.running    = True
    st.session_state.run_count += 1
    thread.start()
    st.rerun()

if stop and st.session_state.running:
    if st.session_state.done_event:
        st.session_state.done_event.set()
    st.session_state.running = False


# ---------------------------------------------------------------------------
# Pre-run splash
# ---------------------------------------------------------------------------
if st.session_state.agent is None:
    st.markdown("<hr class='sect'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:50px 20px'>
      <div style='font-size:72px;filter:drop-shadow(0 0 24px rgba(0,255,200,0.5))'>⚛️</div>
      <div style='font-family:Space Mono,monospace;font-size:18px;color:#00ffcc;margin-top:20px'>
        Quantum Dot Tuning Agent
      </div>
      <div style='font-size:13px;color:#2a5070;margin-top:12px;max-width:560px;margin-left:auto;margin-right:auto'>
        The agent autonomously navigates gate voltage space to tune a double quantum dot
        to the (1,1) charge state — one electron per dot — required for spin qubit operation.<br><br>
        Configure options in the sidebar, then launch a run.
      </div>
      <div style='margin-top:28px;display:flex;gap:24px;justify-content:center;flex-wrap:wrap'>
        <div style='background:#060e1e;border:1px solid #0d2a50;border-radius:6px;padding:14px 20px;font-size:11px;font-family:Space Mono,monospace;color:#3a6a8a'>
          6-stage POMDP planner
        </div>
        <div style='background:#060e1e;border:1px solid #0d2a50;border-radius:6px;padding:14px 20px;font-size:11px;font-family:Space Mono,monospace;color:#3a6a8a'>
          CNN charge classifier (91.35% acc)
        </div>
        <div style='background:#060e1e;border:1px solid #0d2a50;border-radius:6px;padding:14px 20px;font-size:11px;font-family:Space Mono,monospace;color:#3a6a8a'>
          Qwen2.5 on AMD MI300X
        </div>
        <div style='background:#060e1e;border:1px solid #0d2a50;border-radius:6px;padding:14px 20px;font-size:11px;font-family:Space Mono,monospace;color:#3a6a8a'>
          Bayesian optimisation
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------
agent      = st.session_state.agent
exp_state  = st.session_state.exp_state
narrator   = st.session_state.narrator
done_event = st.session_state.done_event

# Check completion
if done_event.is_set() and st.session_state.running:
    st.session_state.running = False
    if exp_state.stage.name == "COMPLETE":
        st.balloons()

current_stage = exp_state.stage.name
is_done = done_event.is_set()
hitl_active = len(exp_state.hitl_events) > 0

# ── Stage progress ──────────────────────────────────────────────────────────
st.markdown("<hr class='sect'>", unsafe_allow_html=True)
_render_stage_pills(current_stage)

# ── KPI bar ─────────────────────────────────────────────────────────────────
budget_used = exp_state.total_measurements
budget_total = agent.measurement_budget
budget_pct = min(100, int(100 * budget_used / max(budget_total, 1)))
vg1 = exp_state.current_voltage.vg1
vg2 = exp_state.current_voltage.vg2
snr = exp_state.last_dqc.snr_db if exp_state.last_dqc else 0.0
bt  = exp_state.total_backtracks

kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-val">{budget_used}<span style="font-size:11px;color:#2a5070">/{budget_total}</span></div>
    <div class="kpi-lbl">MEASUREMENTS</div>
  </div>
  <div class="kpi">
    <div class="kpi-val">{vg1:+.2f}<span style="font-size:11px;color:#2a5070">V</span></div>
    <div class="kpi-lbl">Vg1</div>
  </div>
  <div class="kpi">
    <div class="kpi-val">{vg2:+.2f}<span style="font-size:11px;color:#2a5070">V</span></div>
    <div class="kpi-lbl">Vg2</div>
  </div>
  <div class="kpi">
    <div class="kpi-val">{snr:.1f}<span style="font-size:11px;color:#2a5070">dB</span></div>
    <div class="kpi-lbl">SNR</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)
st.progress(budget_pct / 100, text=f"Measurement budget  {budget_pct}%")

# ── Two-column body ─────────────────────────────────────────────────────────
st.markdown("<hr class='sect'>", unsafe_allow_html=True)
left, right = st.columns(2, gap="large")

# ── LEFT — Engine (physics) ─────────────────────────────────────────────────
with left:
    st.markdown("### THE ENGINE")

    st.plotly_chart(
        _chart_stability(exp_state),
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"stability_{time.monotonic_ns()}",
    )

    traj_fig = _chart_trajectory(exp_state)
    if traj_fig:
        st.plotly_chart(
            traj_fig, use_container_width=True,
            config={"displayModeBar": False},
            key=f"traj_{time.monotonic_ns()}",
        )

    st.markdown("### THE LAB")
    _render_spy_panel(current_stage, hitl_active)

# ── RIGHT — Mind (intelligence) ─────────────────────────────────────────────
with right:
    st.markdown("### THE MIND")

    st.plotly_chart(
        _chart_belief(exp_state),
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"belief_{time.monotonic_ns()}",
    )

    # Classification result
    if exp_state.last_classification:
        cls = exp_state.last_classification
        ood_txt = "⚠ OOD" if exp_state.is_ood else "✓ in-distribution"
        ood_col = "#ff9900" if exp_state.is_ood else "#00cc66"
        st.markdown(
            f'<div style="background:#06090f;border:1px solid #0d1e30;'
            f'border-radius:6px;padding:10px 14px;margin-bottom:10px">'
            f'<span style="font-size:10px;color:#3a5a7a;font-family:Space Mono,monospace">'
            f'CNN CLASSIFICATION</span><br>'
            f'<span style="font-family:Space Mono,monospace;font-size:18px;color:#00ffcc">'
            f'{cls.label.value.upper()}</span>'
            f'<span style="color:#3a5a7a;font-size:12px;margin-left:10px">'
            f'{cls.confidence:.1%}</span>'
            f'<span style="float:right;font-size:11px;color:{ood_col}">{ood_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Dr. Q narration
    st.markdown("### DR. Q — AI CO-PILOT")
    events = narrator.event_log() if narrator else []
    _render_feed(events)

    # HITL events
    if exp_state.hitl_events:
        st.markdown("<hr class='sect'>", unsafe_allow_html=True)
        st.markdown("### HITL EVENTS")
        for ev in reversed(exp_state.hitl_events[-3:]):
            outcome_col = {
                "approved": "#00cc66", "rejected": "#ff3333",
                "modified": "#ffaa00", "pending": "#3a5a7a",
            }.get(ev.outcome.value, "#3a5a7a")
            st.markdown(
                f'<div style="background:#0a0c10;border:1px solid #0d1520;'
                f'border-radius:4px;padding:8px 12px;margin:4px 0;'
                f'font-family:Space Mono,monospace;font-size:10px">'
                f'<span style="color:#3a5a7a">Step {ev.step}</span> '
                f'<span style="color:#6a8aaa">{ev.trigger_reason[:60]}</span><br>'
                f'risk={ev.risk_score:.2f} → '
                f'<span style="color:{outcome_col}">{ev.outcome.value.upper()}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Post-run summary ─────────────────────────────────────────────────────────
if is_done:
    st.markdown("<hr class='sect'>", unsafe_allow_html=True)
    outcome_color = "#00ffcc" if current_stage == "COMPLETE" else "#ff3366"
    outcome_text  = "MISSION COMPLETE ✓" if current_stage == "COMPLETE" else f"STOPPED AT {current_stage}"
    reduction = 1.0 - (budget_used / max(64 * 64, 1))

    st.markdown(
        f'<div style="background:#06090f;border:1px solid #0d1e30;'
        f'border-radius:8px;padding:20px;text-align:center">'
        f'<div style="font-family:Space Mono,monospace;font-size:20px;color:{outcome_color}">'
        f'{outcome_text}</div>'
        f'<div style="display:flex;gap:32px;justify-content:center;margin-top:16px;'
        f'font-family:Space Mono,monospace;font-size:12px;color:#3a6a8a">'
        f'<span>Measurements: <b style="color:#00ffcc">{budget_used}</b></span>'
        f'<span>Steps: <b style="color:#00ffcc">{agent.control_steps}</b></span>'
        f'<span>Backtracks: <b style="color:#00ffcc">{exp_state.total_backtracks}</b></span>'
        f'<span>50% reduction vs dense scan</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.button("🔄 New Run", on_click=lambda: st.session_state.update(
        agent=None, exp_state=None, narrator=None,
        done_event=None, thread=None, running=False,
    ))


# ---------------------------------------------------------------------------
# Auto-refresh while live
# ---------------------------------------------------------------------------
if st.session_state.running and not done_event.is_set():
    time.sleep(0.75)
    st.rerun()
