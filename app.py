"""
app.py  —  SimQuantum Tuning Lab
=================================
AMD Developer Hackathon 2026.

Before running, set env vars (on MI300X):

    export QDOT_LLM_BASE_URL=http://localhost:8000/v1
    export QDOT_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

    Or locally (LLM offline, physics sim still runs):
    streamlit run app.py
"""

from __future__ import annotations
import os, sys, threading, time, re
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st

st.set_page_config(
    page_title="SimQuantum Tuning Lab",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",   # always open so the LLM URL is visible
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html,body,.stApp{background:#F2F0EB!important;font-family:'Libre Franklin',sans-serif;color:#1C2333;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.4rem 1.8rem 1rem!important;max-width:1440px;}

.topbar{display:flex;align-items:flex-start;justify-content:space-between;
  padding-bottom:14px;border-bottom:2px solid #1C2333;margin-bottom:14px;}
.topbar-title{font-size:21px;font-weight:700;letter-spacing:-.5px;color:#1C2333;}
.topbar-sub{font-size:11px;color:#8A9AB0;font-family:'JetBrains Mono',monospace;margin-top:3px;}

.badge{font-family:'JetBrains Mono',monospace;font-size:10px;padding:3px 8px;
  border-radius:3px;font-weight:500;display:inline-block;margin-left:4px;}
.badge-live{background:#E6F4F1;color:#00897B;border:1px solid #B2DFDB;}
.badge-idle{background:#FFF3E0;color:#E65100;border:1px solid #FFCC80;}
.badge-mi300x{background:#EDE7F6;color:#5E35B1;border:1px solid #D1C4E9;}
.badge-warn{background:#FFF3E0;color:#B85000;border:1px solid #FFD09A;}

.timeline{display:flex;align-items:center;margin:12px 0 14px;}
.tn{display:flex;flex-direction:column;align-items:center;
  font-family:'JetBrains Mono',monospace;font-size:9px;min-width:88px;}
.tn-c{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:11px;font-weight:700;margin-bottom:4px;border:2px solid;}
.tn-l{color:#8A9AB0;text-align:center;line-height:1.3;}
.tn-done   .tn-c{background:#E6F4F1;border-color:#00897B;color:#00897B;}
.tn-active .tn-c{background:#1C2333;border-color:#1C2333;color:#F2F0EB;
  box-shadow:0 0 0 4px rgba(28,35,51,.10);}
.tn-pending .tn-c{background:#F2F0EB;border-color:#D0CBB8;color:#C0B8A8;}
.tn-phase3  .tn-c{background:#F8F7F4;border-color:#E0D8C8;color:#C8C0B0;font-size:8px;}
.tn-done .tn-l{color:#00897B;} .tn-active .tn-l{color:#1C2333;font-weight:600;}
.tn-phase3 .tn-l{color:#C0B8A8;font-style:italic;}
.tline{flex:1;height:2px;background:#D0CBB8;margin-top:-18px;}
.tline-done{background:#00897B;}

.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;}
.kpi{background:#FFF;border:1px solid #DDD9D0;border-radius:7px;padding:10px 12px;text-align:center;}
.kpi-v{font-family:'JetBrains Mono',monospace;font-size:21px;font-weight:500;color:#1C2333;}
.kpi-u{font-size:10px;color:#8A9AB0;}
.kpi-l{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:#A8B0BC;margin-top:2px;}

.card{background:#FFF;border-radius:8px;border:1px solid #DDD9D0;padding:14px 16px;margin-bottom:10px;}
.card-title{font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:#8A9AB0;margin-bottom:10px;font-family:'JetBrains Mono',monospace;}

.chat-outer{background:#FFF;border:1px solid #DDD9D0;border-radius:8px;
  display:flex;flex-direction:column;height:460px;}
.chat-head{padding:10px 14px;border-bottom:1px solid #EDE8DF;flex-shrink:0;
  font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:#8A9AB0;font-family:'JetBrains Mono',monospace;
  display:flex;align-items:center;justify-content:space-between;}
.chat-body{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:9px;}
.msg{display:flex;flex-direction:column;max-width:92%;}
.msg-u{align-self:flex-end;} .msg-a{align-self:flex-start;}
.bubble{padding:8px 12px;border-radius:10px;font-size:12.5px;line-height:1.65;}
.msg-u .bubble{background:#1C2333;color:#F2F0EB;border-radius:10px 10px 2px 10px;}
.msg-a .bubble{background:#F2F0EB;color:#1C2333;border:1px solid #DDD9D0;
  border-radius:10px 10px 10px 2px;font-family:'JetBrains Mono',monospace;font-size:11.5px;}
.msg-ev .bubble{background:#FFF8E8;border:1px solid #FFD070;color:#7A5000;
  font-family:'JetBrains Mono',monospace;font-size:11px;}
.mlabel{font-size:9px;letter-spacing:.5px;color:#A8B0BC;margin-bottom:2px;
  font-family:'JetBrains Mono',monospace;}
.msg-u .mlabel{text-align:right;}

.hitl-card{background:#FFFAED;border:2px solid #E8A020;border-radius:8px;
  padding:14px 16px;margin-bottom:10px;}
.hitl-title{font-size:12px;font-weight:700;color:#B85000;margin-bottom:6px;
  font-family:'JetBrains Mono',monospace;}
.hitl-body{font-size:12px;color:#5A4000;line-height:1.5;}

.spy-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;}
.spy{background:#F8F6F2;border:1px solid #DDD9D0;border-radius:6px;
  padding:9px 5px;text-align:center;transition:all .3s;}
.spy-on{background:#FFF;border-color:#00897B;box-shadow:0 0 0 2px rgba(0,137,123,.12);}
.spy-em{font-size:19px;margin-bottom:3px;}
.spy-name{font-size:10px;font-weight:600;color:#1C2333;}
.spy-role{font-size:9px;color:#A8B0BC;}
.spy-dot{width:7px;height:7px;border-radius:50%;margin:5px auto 0;background:#D0CBB8;}
.spy-dot-on{background:#00897B;}

div[data-testid="stProgressBar"]>div{background:#E6F4F1;}
div[data-testid="stProgressBar"]>div>div{background:#00897B!important;}
.stButton>button{font-family:'Libre Franklin',sans-serif!important;
  font-size:13px!important;font-weight:600!important;}
section[data-testid="stSidebar"]{background:#1C2333!important;}
section[data-testid="stSidebar"] label{color:#8A9AB0!important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:#D0CBB8;border-radius:2px;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STAGES = [
    ("BOOTSTRAPPING",       "⚡","Integrity check",  False),
    ("COARSE_SURVEY",       "◈", "Voltage survey",   False),
    ("HYPERSURFACE_SEARCH", "◎","Find boundary",     False),
    ("CHARGE_ID",           "◇","Classify charge",   False),
    ("NAVIGATION",          "→","Navigate to (1,1)", True),
    ("VERIFICATION",        "✓","Verify stability",  True),
]
SPY_AGENTS = [
    ("perception","🔬","Perception","Quality Inspector",{"BOOTSTRAPPING","COARSE_SURVEY","CHARGE_ID"}),
    ("executive", "🏛","Executive", "Mission Conductor",set(s[0] for s in STAGES)),
    ("planning",  "📐","Planning",  "Navigator",         {"HYPERSURFACE_SEARCH","NAVIGATION"}),
    ("safety",    "🛡","Safety",    "Hardware Marshal",  {"NAVIGATION","VERIFICATION"}),
    ("hitl",      "🛑","HITL",      "Human Governor",    set()),
]
START_KWS = {"start","begin","run","tune","go","launch","init","initialize","initialise","proceed"}

STAGE_DESC = {
    "BOOTSTRAPPING":       ("64-pt line scan. Verifies gate response and sensor signal.","~64 pts"),
    "COARSE_SURVEY":       ("32×32 sweep across full voltage bounds.",                   "~1024 pts"),
    "HYPERSURFACE_SEARCH": ("16×16 local scan around survey peak. Confirms boundary.",   "~256 pts"),
    "CHARGE_ID":           ("32×32 scan. 5-model CNN ensemble classifies charge state.", "~1024 pts"),
    "NAVIGATION":          ("Bayesian BO proposes voltage moves toward (1,1).",          "variable"),
    "VERIFICATION":        ("3× repeated 16×16 scans confirming (1,1) stability.",       "~768 pts"),
}

STABILITY_CS = [
    [0.00,"#07070A"],[0.30,"#1A0E40"],[0.55,"#7A1800"],
    [0.75,"#D84000"],[0.90,"#FF9000"],[1.00,"#FFE040"],
]
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFAF8",
    font=dict(color="#8A9AB0",size=10,family="JetBrains Mono"),
    margin=dict(l=10,r=10,t=30,b=10),
)

# Dr. Q system prompt — tells Qwen who it is and what the experiment is
DR_Q_SYSTEM = """\
You are Dr. Q, the AI co-pilot for SimQuantum — an autonomous quantum dot tuning system
running on AMD MI300X hardware. You are Qwen2.5-1.5B-Instruct.

Your role: have a real conversation about this experiment with whoever is asking.
Adapt your depth to the person — a curious student gets clear analogies, an expert
physicist like Natalia Ares gets precise technical detail. Read the question, match it.

THE EXPERIMENT:
A 6-stage POMDP agent autonomously navigates gate voltage space (Vg1, Vg2) to tune
a double quantum dot to the (1,1) charge state — one electron per dot — required
for spin qubit operation.

Stages: BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID → NAVIGATION → VERIFICATION

Physics simulator: Capacitive Interaction Model (CIM) — real semiconductor physics,
not a toy model. Parameters: charging energy E_c, lever arm, tunnel coupling t_c.

CNN: 5-model TinyCNN ensemble trained on 51,000 simulated stability diagrams.
Val accuracy 91.4%. OOD detection via Mahalanobis distance on penultimate features.

The agent uses Bayesian optimisation (MultiResBO) for navigation and a particle filter
belief state over charge configurations.

IMPORTANT HONESTY: Navigation (Phase 3) is unsolved. The Bayesian optimiser has no
converging reward signal in intermediate voltage space — the CNN sees "misc" everywhere
except at charge boundaries. The agent stops at CHARGE_ID reliably; NAVIGATION wanders.
Be honest about this if asked.

CURRENT RUN STATE:
{state_block}

CONVERSATION STYLE:
- Talk like a physicist who is also a good teacher
- 2-4 sentences usually, more only if genuinely needed
- Reference actual numbers from the state when relevant
- Be honest about what works and what doesn't
- Never say "Great news!" or "Certainly!" or "As an AI"
- If something is unclear, ask for clarification
"""


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    d = dict(
        agent=None, exp_state=None, narrator=None, hitl_manager=None,
        done_event=None, thread=None, running=False, run_count=0,
        chat=[],
        llm_url="",   # persists across reruns
        llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        use_cnn=True,
        meas_budget=8096,
        max_steps=140,
    )
    for k,v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# Pull LLM URL from environment if not yet set in session
if not st.session_state.llm_url:
    env_url = os.environ.get("QDOT_LLM_BASE_URL","")
    if env_url:
        st.session_state.llm_url = env_url


# ─────────────────────────────────────────────────────────────────────────────
# Dr. Q — real LLM, no gatekeeping
# ─────────────────────────────────────────────────────────────────────────────
def _llm_available() -> bool:
    return bool(st.session_state.llm_url.strip())


def _build_state_block() -> str:
    """Snapshot of current agent state for the system prompt."""
    exp  = st.session_state.exp_state
    agnt = st.session_state.agent
    if exp is None:
        return "No run active. Agent is idle."
    budget = agnt.measurement_budget if agnt else st.session_state.meas_budget
    meas   = exp.total_measurements
    stage  = exp.stage.name
    vg1    = exp.current_voltage.vg1
    vg2    = exp.current_voltage.vg2
    snr    = f"{exp.last_dqc.snr_db:.1f}dB" if exp.last_dqc else "unknown"
    dqc    = exp.last_dqc.quality.value if exp.last_dqc else "unknown"
    bt     = exp.total_backtracks
    done   = st.session_state.done_event and st.session_state.done_event.is_set()

    cls_str = "none yet"
    if exp.last_classification:
        c = exp.last_classification
        ood = "OOD" if exp.is_ood else "in-dist"
        cls_str = f"{c.label.value} ({c.confidence:.0%} conf, {ood})"

    p11 = exp.belief.charge_probs.get((1,1), 0.0)
    ml  = exp.belief.most_likely_state()

    return (
        f"Stage: {stage}\n"
        f"Measurements used: {meas}/{budget} ({100*meas//max(budget,1)}%)\n"
        f"Current voltage: Vg1={vg1:+.3f}V, Vg2={vg2:+.3f}V\n"
        f"Signal quality: SNR={snr}, DQC={dqc}\n"
        f"CNN classification: {cls_str}\n"
        f"Belief P(1,1)={p11:.3f}, most likely: {ml}\n"
        f"Backtracks: {bt}\n"
        f"Run finished: {'yes' if done else 'no'}"
    )


def _call_qwen(user_msg: str) -> str:
    """
    Call Qwen directly. This is the whole thing.
    System prompt + conversation history + user message → response.
    No fallbacks, no scripts. If the LLM is down, say so clearly.
    """
    url   = st.session_state.llm_url.strip().rstrip("/")
    model = st.session_state.llm_model.strip() or "Qwen/Qwen2.5-1.5B-Instruct"

    # Build messages
    system = DR_Q_SYSTEM.format(state_block=_build_state_block())
    messages = [{"role": "system", "content": system}]

    # Add conversation history (last 16 turns to stay within context)
    for msg in st.session_state.chat[-16:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    # The latest user message is already in chat history at this point
    # (added by _handle_chat before calling this function)

    try:
        import openai
        # Handle both http://host:port and http://host:port/v1
        api_base = url if url.endswith("/v1") else url + "/v1"
        client = openai.OpenAI(
            base_url=api_base,
            api_key=os.environ.get("QDOT_LLM_API_KEY", "EMPTY"),
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=350,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    except ImportError:
        return ("openai package not installed. Run: pip install openai\n"
                "Then restart the app.")
    except Exception as exc:
        return (f"Cannot reach Qwen at {url}.\n\nError: {exc}\n\n"
                f"Check that vLLM is running: curl {url}/v1/models")


def _add_msg(role: str, content: str, kind: str = "n"):
    st.session_state.chat.append({"role": role, "content": content, "kind": kind})


def _handle_chat(user_msg: str):
    """Add user message, start agent if requested, get Dr. Q response."""
    _add_msg("user", user_msg)

    is_start = any(kw in user_msg.lower() for kw in START_KWS)

    if is_start and not st.session_state.running and st.session_state.agent is None:
        # Start the agent
        try:
            agent, exp_state, narrator, hitl_mgr = _make_agent(
                st.session_state.use_cnn,
                st.session_state.meas_budget,
                st.session_state.max_steps,
            )
            done_event = threading.Event()
            thread = threading.Thread(
                target=_run_thread, args=(agent, done_event), daemon=True)
            st.session_state.update(
                agent=agent, exp_state=exp_state, narrator=narrator,
                hitl_manager=hitl_mgr, done_event=done_event, thread=thread,
                running=True, run_count=st.session_state.run_count+1,
            )
            thread.start()
            # Give the LLM the launch context and let it respond naturally
            if _llm_available():
                reply = _call_qwen(user_msg)
            else:
                reply = (
                    "Tuning sequence started.\n\n"
                    "To enable Qwen responses, set the vLLM endpoint in the sidebar "
                    "to your MI300X address (e.g. http://localhost:8000/v1).\n\n"
                    "The physics simulation is running. Ask me anything."
                )
        except Exception as exc:
            reply = f"Failed to start agent: {exc}"
    else:
        # Normal chat — call Qwen or show clear offline message
        if _llm_available():
            reply = _call_qwen(user_msg)
        else:
            reply = (
                "Qwen is not connected. Set the vLLM endpoint in the sidebar.\n\n"
                "If you're on the MI300X, make sure vLLM is running:\n"
                "  docker exec -it rocm /bin/bash\n"
                "  vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000\n\n"
                "Then set the URL to http://localhost:8000 in the sidebar."
            )

    _add_msg("assistant", reply)


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────
def _make_agent(use_cnn, meas_budget, max_steps):
    from qdot.core.state import ExperimentState
    from qdot.core.types import ChargeLabel
    from qdot.core.governance import GovernanceLogger
    from qdot.core.hitl import HITLManager
    from qdot.hardware.safety import SafetyCritic
    from qdot.perception.dqc import DQCGatekeeper
    from qdot.simulator.cim import CIMSimulatorAdapter
    from qdot.agent.executive import ExecutiveAgent

    rng = np.random.default_rng()
    E_c = float(rng.uniform(2.1, 2.9))
    adapter = CIMSimulatorAdapter(device_id="demo_qdot", params={
        "E_c1": E_c, "E_c2": E_c*float(rng.uniform(0.93,1.07)),
        "t_c": 0.05, "T": 0.015,
        "lever_arm": float(rng.uniform(0.68,0.82)), "noise_level": 0.02,
    })
    state = ExperimentState.new(device_id="demo_qdot", target_label=ChargeLabel.DOUBLE_DOT)

    inspection = None
    if use_cnn:
        try:
            from qdot.perception.inspector import InspectionAgent
            inspection = InspectionAgent()
        except Exception as e:
            st.sidebar.warning(f"CNN unavailable: {e}")

    run_dir = Path("results/demo") / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hitl_mgr = HITLManager(queue_dir=str(run_dir/"hitl"))
    hitl_mgr.set_test_mode()

    agent = ExecutiveAgent(
        state=state, adapter=adapter, inspection_agent=inspection,
        dqc=DQCGatekeeper(),
        safety_critic=SafetyCritic(voltage_bounds=state.voltage_bounds, l1_max=0.10),
        hitl_manager=hitl_mgr,
        governance_logger=GovernanceLogger(
            run_id=state.run_id, log_dir=str(run_dir/"governance")),
        max_steps=max_steps, measurement_budget=meas_budget,
    )
    return agent, state, agent.narrator, hitl_mgr


def _run_thread(agent, done_event):
    try: agent.run()
    except Exception: pass
    finally: done_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
import plotly.graph_objects as go

def _best_scan(state):
    if state.last_classification:
        m = state.measurements.get(state.last_classification.measurement_id)
        if m and m.is_2d and m.array is not None: return m
    best, best_area = None, 0.0
    for m in state.measurements.values():
        if not m.is_2d or m.array is None: continue
        if m.v1_range and m.v2_range:
            area = (m.v1_range[1]-m.v1_range[0])*(m.v2_range[1]-m.v2_range[0])
            if area > best_area: best_area,best = area,m
    return best

def _fig_stability(state):
    m = _best_scan(state)
    fig = go.Figure()
    if m is None:
        fig.add_annotation(text="Awaiting first 2D scan…",x=0.5,y=0.5,showarrow=False,
            font=dict(color="#8A9AB0",size=12,family="JetBrains Mono"))
    else:
        arr = np.asarray(m.array, dtype=np.float32)
        p2,p98 = np.percentile(arr,[2,98])
        arrn = np.clip((arr-p2)/max(p98-p2,1e-8),0,1)
        v1lo,v1hi = m.v1_range or (-8,8)
        v2lo,v2hi = m.v2_range or (-8,8)
        fig.add_trace(go.Heatmap(
            z=arrn, x=np.linspace(v1lo,v1hi,arrn.shape[1]),
            y=np.linspace(v2lo,v2hi,arrn.shape[0]),
            colorscale=STABILITY_CS, showscale=True,
            colorbar=dict(thickness=8,tickvals=[0,.5,1],
                ticktext=["Blockade","—","Peak"],
                tickfont=dict(size=8,family="JetBrains Mono"),
                title=dict(text="G",font=dict(size=9))),
        ))
        vg1,vg2 = state.current_voltage.vg1,state.current_voltage.vg2
        fig.add_shape(type="line",x0=vg1,x1=vg1,y0=v2lo,y1=v2hi,
            line=dict(color="rgba(255,255,255,0.55)",width=1,dash="dot"))
        fig.add_shape(type="line",x0=v1lo,x1=v1hi,y0=vg2,y1=vg2,
            line=dict(color="rgba(255,255,255,0.55)",width=1,dash="dot"))
        fig.add_trace(go.Scatter(x=[vg1],y=[vg2],mode="markers",
            marker=dict(size=8,color="#FF4040",symbol="cross-thin",
                line=dict(width=2.5,color="#FF4040")),showlegend=False))
    fig.update_layout(
        title=dict(text="Charge Stability Diagram",font=dict(size=11,color="#5A6478")),
        xaxis=dict(title="Vg₁ (V)",gridcolor="#E8E4DC",zeroline=False),
        yaxis=dict(title="Vg₂ (V)",gridcolor="#E8E4DC",zeroline=False),
        height=265, **PLOT_LAYOUT)
    return fig

def _fig_belief(state):
    probs = state.belief.charge_probs
    z = np.zeros((3,3))
    for (n1,n2),p in probs.items():
        if 0<=n1<=2 and 0<=n2<=2: z[n2][n1]=float(p)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["N₁=0","N₁=1","N₁=2"], y=["N₂=0","N₂=1","N₂=2"],
        colorscale=[[0,"#F2F0EB"],[0.5,"#B2DFDB"],[1,"#00897B"]],
        showscale=False, zmin=0, zmax=1,
        text=[[f"{z[j][i]:.2f}" for i in range(3)] for j in range(3)],
        texttemplate="%{text}",
        textfont=dict(size=12,color="#1C2333",family="JetBrains Mono"),
    ))
    fig.add_shape(type="rect",x0=.5,x1=1.5,y0=.5,y1=1.5,
        line=dict(color="#00897B",width=2.5))
    fig.add_annotation(x=1,y=1.5,text="TARGET",showarrow=False,
        yshift=14,font=dict(color="#00897B",size=9,family="JetBrains Mono"))
    fig.update_layout(
        title=dict(text="Belief  P(N₁,N₂|obs)",font=dict(size=11,color="#5A6478")),
        height=210, **PLOT_LAYOUT)
    return fig

def _fig_traj(state):
    if len(state.trajectory)<2: return None
    xs=[v.vg1 for v in state.trajectory]; ys=[v.vg2 for v in state.trajectory]
    n=len(xs)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines+markers",
        line=dict(color="#B2DFDB",width=1.5),
        marker=dict(size=5,color=list(range(n)),
            colorscale=[[0,"#E0F4F1"],[1,"#00897B"]],showscale=False),
        showlegend=False))
    fig.add_trace(go.Scatter(x=[xs[-1]],y=[ys[-1]],mode="markers",
        marker=dict(size=9,color="#E85000",symbol="x-thin",
            line=dict(width=2.5,color="#E85000")),showlegend=False))
    fig.update_layout(
        title=dict(text="Voltage Trajectory",font=dict(size=11,color="#5A6478")),
        xaxis=dict(title="Vg₁",gridcolor="#E8E4DC"),
        yaxis=dict(title="Vg₂",gridcolor="#E8E4DC"),
        height=185, **PLOT_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _timeline(current, done_event):
    is_done = done_event and done_event.is_set()
    try: ci = [s[0] for s in STAGES].index(current)
    except ValueError: ci = -1
    html = '<div class="timeline">'
    for i,(sname,icon,desc,p3) in enumerate(STAGES):
        if p3: css="tn tn-phase3"
        elif i<ci or (is_done and i<=ci): css="tn tn-done"
        elif i==ci: css="tn tn-active"
        else: css="tn tn-pending"
        chk = "✓" if "done" in css and not p3 else icon
        html += (f'<div class="{css}"><div class="tn-c">{chk}</div>'
                 f'<div class="tn-l">{desc}</div></div>')
        if i<len(STAGES)-1:
            lc = "tline tline-done" if i<ci else "tline"
            html += f'<div class="{lc}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def _kpi(state, agent):
    b=state.total_measurements; t=agent.measurement_budget
    vg1=state.current_voltage.vg1; vg2=state.current_voltage.vg2
    snr=state.last_dqc.snr_db if state.last_dqc else 0.0
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-v">{b}<span class="kpi-u">/{t}</span></div>
        <div class="kpi-l">Measurements</div></div>
      <div class="kpi"><div class="kpi-v">{vg1:+.2f}<span class="kpi-u"> V</span></div>
        <div class="kpi-l">Vg₁</div></div>
      <div class="kpi"><div class="kpi-v">{vg2:+.2f}<span class="kpi-u"> V</span></div>
        <div class="kpi-l">Vg₂</div></div>
      <div class="kpi"><div class="kpi-v">{snr:.1f}<span class="kpi-u"> dB</span></div>
        <div class="kpi-l">SNR</div></div>
    </div>""", unsafe_allow_html=True)

def _spy(current, hitl_active):
    html = '<div class="spy-grid">'
    for key,em,name,role,active in SPY_AGENTS:
        on = (key=="hitl" and hitl_active) or (key!="hitl" and current in active)
        html += (f'<div class="{"spy spy-on" if on else "spy"}">'
                 f'<div class="spy-em">{em}</div>'
                 f'<div class="spy-name">{name}</div><div class="spy-role">{role}</div>'
                 f'<div class="{"spy-dot spy-dot-on" if on else "spy-dot"}"></div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def _render_chat():
    """Read-only HTML chat panel. The actual input widget is at page bottom."""
    # Pump narrator events (anomalies only — don't pollute chat with routine logs)
    narrator = st.session_state.narrator
    if narrator:
        for ev in narrator.event_log():
            tag = f"_ev_{ev.timestamp:.4f}"
            if tag not in st.session_state:
                st.session_state[tag] = True
                if ev.kind == "exception" and ev.response:
                    # Let Qwen contextualise the anomaly if it's available
                    if _llm_available():
                        contextualised = _call_qwen(
                            f"Agent anomaly detected: {ev.description}. "
                            f"Briefly explain what this means for the experiment.")
                        _add_msg("assistant", contextualised, kind="ev")
                    else:
                        _add_msg("assistant", f"⚠ {ev.description}", kind="ev")

    llm_on = _llm_available()
    badge = (
        '<span class="badge badge-mi300x">⬡ Qwen2.5 · AMD MI300X</span>'
        if llm_on else
        '<span class="badge badge-warn">LLM offline — set URL in sidebar</span>'
    )

    msgs_html = ""
    for msg in st.session_state.chat:
        c = (msg["content"]
             .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
             .replace("\n","<br>"))
        c = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', c)
        if msg["role"] == "user":
            msgs_html += (f'<div class="msg msg-u"><div class="mlabel">You</div>'
                          f'<div class="bubble">{c}</div></div>')
        else:
            css = "msg msg-a msg-ev" if msg.get("kind")=="ev" else "msg msg-a"
            lbl = "Dr. Q — anomaly" if msg.get("kind")=="ev" else "Dr. Q"
            msgs_html += (f'<div class="{css}"><div class="mlabel">{lbl}</div>'
                          f'<div class="bubble">{c}</div></div>')

    st.markdown(
        f'<div class="chat-outer">'
        f'  <div class="chat-head"><span>DR. Q — AI CO-PILOT</span>{badge}</div>'
        f'  <div class="chat-body" id="cq-body">{msgs_html}</div>'
        f'</div>'
        f'<script>(function(){{var e=document.getElementById("cq-body");'
        f'if(e)e.scrollTop=e.scrollHeight;}})();</script>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — always visible
# ─────────────────────────────────────────────────────────────────────────────
running    = st.session_state.running
exp_state  = st.session_state.exp_state
done_event = st.session_state.done_event

with st.sidebar:
    st.markdown(
        '<div style="color:#F2F0EB;font-size:16px;font-weight:700;'
        'padding:8px 0 16px;border-bottom:1px solid #2C3545;margin-bottom:16px">'
        '⚛ SimQuantum</div>',
        unsafe_allow_html=True,
    )

    # LLM connection — most important, top of sidebar
    st.markdown(
        '<div style="color:#8A9AB0;font-size:10px;font-family:JetBrains Mono,monospace;'
        'letter-spacing:1px;margin-bottom:6px">DR. Q — QWEN ENDPOINT</div>',
        unsafe_allow_html=True,
    )
    new_url = st.text_input(
        "vLLM base URL",
        value=st.session_state.llm_url,
        placeholder="http://localhost:8000",
        label_visibility="collapsed",
    )
    if new_url != st.session_state.llm_url:
        st.session_state.llm_url = new_url

    if st.session_state.llm_url:
        st.markdown(
            f'<div style="font-size:10px;color:#00897B;font-family:JetBrains Mono,'
            f'monospace;margin-bottom:12px">● connected to {st.session_state.llm_url}</div>',
            unsafe_allow_html=True,
        )
        # Test button
        if st.button("Test connection", use_container_width=True):
            try:
                import openai
                url = st.session_state.llm_url.strip().rstrip("/")
                api_base = url if url.endswith("/v1") else url + "/v1"
                c = openai.OpenAI(base_url=api_base, api_key="EMPTY")
                models = c.models.list()
                names = [m.id for m in models.data]
                st.success(f"Connected. Models: {', '.join(names)}")
            except Exception as e:
                st.error(f"Failed: {e}")
    else:
        st.markdown(
            '<div style="font-size:10px;color:#E65100;font-family:JetBrains Mono,'
            'monospace;margin-bottom:12px">● offline</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    new_model = st.text_input(
        "Model name",
        value=st.session_state.llm_model,
        label_visibility="visible",
        disabled=running,
    )
    if new_model != st.session_state.llm_model:
        st.session_state.llm_model = new_model

    st.divider()

    st.markdown(
        '<div style="color:#8A9AB0;font-size:10px;font-family:JetBrains Mono,monospace;'
        'letter-spacing:1px;margin-bottom:8px">RUN CONFIGURATION</div>',
        unsafe_allow_html=True,
    )
    st.session_state.use_cnn = st.toggle(
        "CNN Charge Classifier", value=st.session_state.use_cnn, disabled=running)
    st.session_state.meas_budget = st.slider(
        "Measurement Budget", 512, 8192, st.session_state.meas_budget, 256,
        disabled=running)
    st.session_state.max_steps = st.slider(
        "Max Steps", 10, 300, st.session_state.max_steps, 10, disabled=running)

    st.divider()

    if st.button("Reset session", use_container_width=True, disabled=running):
        for k in ["agent","exp_state","narrator","hitl_manager","done_event","thread","chat"]:
            st.session_state[k] = [] if k=="chat" else None
        st.session_state.running = False
        st.rerun()

    st.markdown(
        '<div style="font-size:9px;color:#3A4A5A;font-family:JetBrains Mono,monospace;'
        'margin-top:16px;line-height:1.6">'
        'Physics: CIM simulator (CPU)<br>'
        'CNN: 5-model ensemble, 91.4% val acc<br>'
        'LLM: Qwen2.5-1.5B on AMD MI300X<br>'
        'BOOTSTRAPPING→CHARGE_ID: ✓ working<br>'
        'NAVIGATION: Phase 3, in development'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Check completion
# ─────────────────────────────────────────────────────────────────────────────
if done_event and done_event.is_set() and running:
    st.session_state.running = False
    running = False
    if exp_state and exp_state.stage.name == "COMPLETE":
        st.balloons()
    # Ask Qwen for a post-run summary
    already_summarised = any(m.get("kind")=="summary" for m in st.session_state.chat)
    if not already_summarised and exp_state:
        stage  = exp_state.stage.name
        meas   = exp_state.total_measurements
        budget = st.session_state.agent.measurement_budget if st.session_state.agent else 8096
        bt     = exp_state.total_backtracks
        _add_msg("user",
            f"The run just finished — it stopped at {stage}. "
            f"{meas}/{budget} measurements used, {bt} backtracks. "
            f"What happened and what should I know for the next run?")
        if _llm_available():
            reply = _call_qwen(
                f"Post-run: stopped at {stage}, {meas}/{budget} measurements, {bt} backtracks.")
        else:
            reply = (f"Run stopped at {stage}. {meas}/{budget} measurements used, "
                     f"{bt} backtracks. Connect Qwen for a detailed analysis.")
        _add_msg("assistant", reply, kind="summary")


# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────
tl,tr = st.columns([3,1])
with tl:
    st.markdown(
        '<div class="topbar"><div>'
        '<div class="topbar-title">⚛ SimQuantum Tuning Lab</div>'
        '<div class="topbar-sub">Autonomous quantum dot tuning · AMD Developer Hackathon 2026 · Kudzai Musarandega</div>'
        '</div></div>', unsafe_allow_html=True)
with tr:
    b = ('<span class="badge badge-live">● LIVE</span>' if running
         else '<span class="badge badge-idle">● IDLE</span>')
    if _llm_available():
        b += ' <span class="badge badge-mi300x">⬡ MI300X</span>'
    st.markdown(f'<div style="text-align:right;padding-top:6px">{b}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Splash (pre-run)
# ─────────────────────────────────────────────────────────────────────────────
if exp_state is None:
    if not st.session_state.chat:
        if _llm_available():
            # Let Qwen introduce itself
            _add_msg("user", "Hello, introduce yourself briefly.")
            intro = _call_qwen("Hello, introduce yourself briefly.")
            _add_msg("assistant", intro)
        else:
            _add_msg("assistant",
                "Ready. Set the vLLM endpoint in the sidebar to enable Qwen.\n\n"
                "Type **start** to begin a tuning run, or ask me about the experiment.")

    sl,sr = st.columns([3,2], gap="large")
    with sl:
        st.markdown("""
        <div class="card" style="padding:22px 24px">
          <div class="card-title">What this system does</div>
          <p style="font-size:13px;color:#5A6478;line-height:1.75;margin:0">
            SimQuantum autonomously tunes a double quantum dot device to the
            <strong>(1,1) charge state</strong> — one electron per dot — required
            for spin qubit operation. 6-stage POMDP planner, 5-model CNN ensemble
            (91.4% val acc), Bayesian optimisation. Qwen2.5-1.5B on AMD MI300X
            acts as Dr. Q — ask it anything, in any register.
          </p>
        </div>
        <div class="card">
          <div class="card-title">Reading a stability diagram</div>
          <p style="font-size:13px;color:#5A6478;line-height:1.75;margin:0">
            Conductance G vs gate voltages (Vg₁, Vg₂).
            Bright lines = Coulomb peaks (charge transitions).
            Dark regions = Coulomb blockade (fixed electron number).
            Intersections form a honeycomb: (0,0), (1,0), (0,1), <strong>(1,1)</strong>…
            The agent navigates to the (1,1) diamond.
          </p>
        </div>
        """, unsafe_allow_html=True)
        img_path = Path(__file__).parent/"assets"/"simquantum.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    with sr:
        _render_chat()

# ─────────────────────────────────────────────────────────────────────────────
# Live dashboard
# ─────────────────────────────────────────────────────────────────────────────
else:
    agent       = st.session_state.agent
    hitl_mgr    = st.session_state.hitl_manager
    current_stg = exp_state.stage.name

    _timeline(current_stg, done_event)
    _kpi(exp_state, agent)
    pct = min(100, int(100*exp_state.total_measurements/agent.measurement_budget))
    st.progress(pct/100, text=f"Measurement budget  {pct}%")

    pending = hitl_mgr.get_pending() if hitl_mgr else []
    if pending:
        req = pending[0]
        st.markdown(
            f'<div class="hitl-card">'
            f'<div class="hitl-title">⚠ HITL GATE — Human approval required</div>'
            f'<div class="hitl-body">Step {req["step"]} · Stage {req["stage"]} · '
            f'Risk {req["risk_score"]:.2f}<br><strong>{req["trigger_reason"]}</strong></div>'
            f'</div>', unsafe_allow_html=True)
        hc1,hc2,_ = st.columns([1,1,5])
        with hc1:
            if st.button("✓ Approve",type="primary",key=f"appr_{req['id']}"):
                hitl_mgr.approve(req["id"],deciding_human="operator")
                _add_msg("user","I approved the HITL gate.")
                if _llm_available():
                    _add_msg("assistant", _call_qwen("The operator just approved the HITL gate. Acknowledge briefly."))
                else:
                    _add_msg("assistant","Approved. Agent continues.")
                st.rerun()
        with hc2:
            if st.button("✗ Reject",key=f"rej_{req['id']}"):
                hitl_mgr.reject(req["id"],deciding_human="operator")
                _add_msg("user","I rejected the HITL gate.")
                if _llm_available():
                    _add_msg("assistant", _call_qwen("The operator rejected the HITL gate. Acknowledge and explain what happens next."))
                else:
                    _add_msg("assistant","Rejected. Agent backtracks.")
                st.rerun()

    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    left,right = st.columns([3,2],gap="large")

    with left:
        st.markdown('<div class="card-title">Charge Stability Diagram</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_fig_stability(exp_state),use_container_width=True,
                        config={"displayModeBar":False},key=f"s_{time.monotonic_ns()}")

        if exp_state.last_classification:
            cls = exp_state.last_classification
            ood_col = "#C84B00" if exp_state.is_ood else "#00897B"
            ood_txt = "OOD warning" if exp_state.is_ood else "in-distribution"
            st.markdown(
                f'<div class="card"><span class="card-title">CNN Classification</span>'
                f'<span style="float:right;font-size:10px;color:{ood_col};'
                f'font-family:JetBrains Mono,monospace">{ood_txt}</span><br>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:22px;'
                f'font-weight:500;color:#00897B">{cls.label.value.upper()}</span>'
                f'<span style="color:#8A9AB0;font-size:12px;margin-left:10px">'
                f'conf {cls.confidence:.1%}</span></div>',
                unsafe_allow_html=True)

        bc1,bc2 = st.columns(2)
        with bc1:
            st.plotly_chart(_fig_belief(exp_state),use_container_width=True,
                            config={"displayModeBar":False},key=f"b_{time.monotonic_ns()}")
        with bc2:
            ft = _fig_traj(exp_state)
            if ft:
                st.plotly_chart(ft,use_container_width=True,
                                config={"displayModeBar":False},key=f"t_{time.monotonic_ns()}")

        st.markdown('<div class="card-title" style="margin-top:4px">Agent Activity</div>',
                    unsafe_allow_html=True)
        _spy(current_stg, bool(pending))

        img_path = Path(__file__).parent/"assets"/"simquantum.png"
        if img_path.exists():
            st.image(str(img_path),use_container_width=True)

    with right:
        _render_chat()

        if current_stg in STAGE_DESC:
            desc,cost = STAGE_DESC[current_stg]
            st.markdown(
                f'<div class="card" style="margin-top:8px">'
                f'<div class="card-title">Current stage · {current_stg}</div>'
                f'<div style="font-size:12px;color:#5A6478;line-height:1.6">{desc}</div>'
                f'<div style="font-size:10px;color:#A8B0BC;margin-top:5px;'
                f'font-family:JetBrains Mono,monospace">Budget: {cost}</div>'
                f'</div>', unsafe_allow_html=True)

        if done_event and done_event.is_set():
            ok  = current_stg=="COMPLETE"
            col = "#00897B" if ok else "#C84B00"
            txt = "MISSION COMPLETE" if ok else f"STOPPED — {current_stg}"
            red = 1.0-(exp_state.total_measurements/max(64*64,1))
            st.markdown(
                f'<div class="card" style="border-color:{col};margin-top:8px">'
                f'<div style="font-size:14px;font-weight:700;color:{col};'
                f'font-family:JetBrains Mono,monospace;margin-bottom:8px">{txt}</div>'
                f'<div style="font-size:12px;color:#5A6478;display:grid;'
                f'grid-template-columns:1fr 1fr;gap:5px">'
                f'<span>Measurements: <b>{exp_state.total_measurements}</b></span>'
                f'<span>Steps: <b>{agent.control_steps}</b></span>'
                f'<span>Backtracks: <b>{exp_state.total_backtracks}</b></span>'
                f'<span>Reduction: <b>{red:.0%}</b></span>'
                f'</div></div>', unsafe_allow_html=True)
            if st.button("🔄 New Run",use_container_width=True):
                for k in ["agent","exp_state","narrator","hitl_manager","done_event","thread"]:
                    st.session_state[k] = None
                st.session_state.running = False
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Chat input — always at page bottom, never inside a conditional.
# Streamlit requires st.chat_input at the same tree position every rerun.
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask Dr. Q anything, or type 'start' to begin…"):
    _handle_chat(prompt)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh while agent is running
# ─────────────────────────────────────────────────────────────────────────────
if running and done_event and not done_event.is_set():
    time.sleep(0.8)
    st.rerun()
