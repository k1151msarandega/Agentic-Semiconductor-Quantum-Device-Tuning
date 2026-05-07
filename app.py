"""
app.py  —  SimQuantum Tuning Lab
=================================
AMD Developer Hackathon 2025.

Run locally:
    pip install streamlit plotly openai
    streamlit run app.py

Full AMD MI300X stack:
    ./deploy.sh          # starts vLLM + Streamlit on the MI300X
    or manually:
    $env:QDOT_LLM_BASE_URL = "http://129.212.186.42:8000/v1"
    $env:QDOT_LLM_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
    streamlit run app.py
"""
from __future__ import annotations
import os, sys, threading, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st

st.set_page_config(
    page_title="SimQuantum Tuning Lab",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (same warm aesthetic — no changes here)
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

/* Timeline */
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

/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;}
.kpi{background:#FFF;border:1px solid #DDD9D0;border-radius:7px;padding:10px 12px;text-align:center;}
.kpi-v{font-family:'JetBrains Mono',monospace;font-size:21px;font-weight:500;color:#1C2333;}
.kpi-u{font-size:10px;color:#8A9AB0;} .kpi-l{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:#A8B0BC;margin-top:2px;}

/* Card */
.card{background:#FFF;border-radius:8px;border:1px solid #DDD9D0;padding:14px 16px;margin-bottom:10px;}
.card-title{font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:#8A9AB0;margin-bottom:10px;font-family:'JetBrains Mono',monospace;}

/* Chat display (read-only HTML) */
.chat-outer{background:#FFF;border:1px solid #DDD9D0;border-radius:8px;
  display:flex;flex-direction:column;height:480px;}
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
.msg-thinking .bubble{background:#F0F8FF;border:1px solid #B0D0F0;color:#1C4060;
  font-family:'JetBrains Mono',monospace;font-size:11px;font-style:italic;}
.mlabel{font-size:9px;letter-spacing:.5px;color:#A8B0BC;margin-bottom:2px;
  font-family:'JetBrains Mono',monospace;}
.msg-u .mlabel{text-align:right;}

/* HITL */
.hitl-card{background:#FFFAED;border:2px solid #E8A020;border-radius:8px;
  padding:14px 16px;margin-bottom:10px;}
.hitl-title{font-size:12px;font-weight:700;color:#B85000;margin-bottom:6px;
  font-family:'JetBrains Mono',monospace;}
.hitl-body{font-size:12px;color:#5A4000;line-height:1.5;}

/* Spy */
.spy-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;}
.spy{background:#F8F6F2;border:1px solid #DDD9D0;border-radius:6px;
  padding:9px 5px;text-align:center;transition:all .3s;}
.spy-on{background:#FFF;border-color:#00897B;box-shadow:0 0 0 2px rgba(0,137,123,.12);}
.spy-em{font-size:19px;margin-bottom:3px;} .spy-name{font-size:10px;font-weight:600;color:#1C2333;}
.spy-role{font-size:9px;color:#A8B0BC;}
.spy-dot{width:7px;height:7px;border-radius:50%;margin:5px auto 0;background:#D0CBB8;}
.spy-dot-on{background:#00897B;}

div[data-testid="stProgressBar"]>div{background:#E6F4F1;}
div[data-testid="stProgressBar"]>div>div{background:#00897B!important;}
.stButton>button{font-family:'Libre Franklin',sans-serif!important;font-size:13px!important;font-weight:600!important;}
section[data-testid="stSidebar"]{background:#1C2333!important;}
::-webkit-scrollbar{width:4px;} ::-webkit-scrollbar-thumb{background:#D0CBB8;border-radius:2px;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STAGES = [
    ("BOOTSTRAPPING",       "⚡","Integrity check",  False),
    ("COARSE_SURVEY",       "◈", "Voltage survey",    False),
    ("HYPERSURFACE_SEARCH", "◎","Find boundary",     False),
    ("CHARGE_ID",           "◇","Classify charge",   False),
    ("NAVIGATION",          "→","Navigate to (1,1)", True),
    ("VERIFICATION",        "✓","Verify stability",  True),
]
SPY_AGENTS = [
    ("perception","🔬","Perception","Quality Inspector", {"BOOTSTRAPPING","COARSE_SURVEY","CHARGE_ID"}),
    ("executive", "🏛", "Executive", "Mission Conductor", set(s[0] for s in STAGES)),
    ("planning",  "📐","Planning",  "Navigator",          {"HYPERSURFACE_SEARCH","NAVIGATION"}),
    ("safety",    "🛡", "Safety",    "Hardware Marshal",   {"NAVIGATION","VERIFICATION"}),
    ("hitl",      "🛑","HITL",      "Human Governor",     set()),
]
START_KWS = {"start","begin","run","tune","go","launch","init","initialize","initialise","proceed"}

STAGE_DESC = {
    "BOOTSTRAPPING":       ("64-point line scan. Verifies gate response and charge sensor signal.",   "~64 pts"),
    "COARSE_SURVEY":       ("32×32 systematic 2D sweep across full voltage bounds.",                   "~1024 pts"),
    "HYPERSURFACE_SEARCH": ("16×16 local scan centred on survey peak. Confirms boundary visibility.", "~256 pts"),
    "CHARGE_ID":           ("32×32 scan. CNN 5-model ensemble classifies charge state.",              "~1024 pts"),
    "NAVIGATION":          ("Bayesian BO proposes voltage moves. Local 8×8 belief scans per step.",  "variable"),
    "VERIFICATION":        ("3× repeated 16×16 scans confirming (1,1) stability.",                    "~768 pts"),
}

# Physics-standard colorscale: Coulomb blockade = black, peaks = yellow
STABILITY_CS = [
    [0.00,"#07070A"],[0.30,"#1A0E40"],[0.55,"#7A1800"],
    [0.75,"#D84000"],[0.90,"#FF9000"],[1.00,"#FFE040"],
]
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFAF8",
    font=dict(color="#8A9AB0",size=10,family="JetBrains Mono"),
    margin=dict(l=10,r=10,t=30,b=10),
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    d = dict(
        agent=None, exp_state=None, narrator=None, hitl_manager=None,
        done_event=None, thread=None, running=False, run_count=0,
        chat=[],            # [{role, content, kind}]
        use_llm=False, use_cnn=True, meas_budget=2048, max_steps=100,
    )
    for k,v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ─────────────────────────────────────────────────────────────────────────────
# Dr. Q — REAL LLM chat  (calls Qwen directly, not via narrator)
# ─────────────────────────────────────────────────────────────────────────────
DR_Q_SYSTEM = """\
You are Dr. Q, the AI co-pilot for SimQuantum — an autonomous quantum dot tuning system \
running on AMD MI300X hardware. You run as Qwen2.5-1.5B-Instruct.

Your job: answer any question about this experiment in the right register. \
A curious 17-year-old asks "what even is a quantum dot?" — give them a clear analogy. \
Natalia Ares asks about the CNN OOD detector's Mahalanobis distance threshold — give \
her the exact technical detail. Read the question, match the depth.

WHAT THIS SYSTEM DOES:
- 6-stage POMDP agent (BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID → NAVIGATION → VERIFICATION)
- Autonomously navigates gate voltage space (Vg1, Vg2) to reach the (1,1) charge state
- Physics simulator: Capacitive Interaction Model (CIM) for double quantum dot
- CNN: 5-model TinyCNN ensemble trained on 51k simulated stability diagrams, 91.4% val accuracy
- OOD detector: Mahalanobis distance on CNN penultimate features
- HITL safety critic: triggers on risk score ≥ 0.70
- LLM narrator (you): Qwen2.5-1.5B on AMD MI300X via vLLM

KEY PHYSICS:
- A stability diagram maps conductance G vs (Vg1, Vg2). Bright lines = Coulomb peaks (charge transitions)
- The (1,1) state = one electron per dot = required starting point for spin qubit operations
- E_c = charging energy (determines spacing of Coulomb peaks in voltage)
- Lever arm = conversion between gate voltage and electrochemical potential

CURRENT RUN STATE:
{state_block}

STYLE RULES:
- 2–4 sentences unless a longer answer is genuinely needed
- Reference real numbers from the state above when they're relevant
- Never start with "I " or "As Dr. Q"
- Never say "Great news!" or "We've achieved" or "Certainly!"
- If you don't know something, say so directly
- The NAVIGATION stage currently has no converging reward signal — be honest about this if asked
"""

def _build_system_prompt() -> str:
    """Inject current agent state into the Dr. Q system prompt."""
    exp_state = st.session_state.exp_state
    if exp_state is None:
        state_block = "No run active. Agent is idle."
    else:
        stage = exp_state.stage.name
        meas  = exp_state.total_measurements
        budget = st.session_state.agent.measurement_budget if st.session_state.agent else 2048
        pct   = int(100*meas/budget)
        vg1   = exp_state.current_voltage.vg1
        vg2   = exp_state.current_voltage.vg2
        snr   = exp_state.last_dqc.snr_db if exp_state.last_dqc else None
        dqc   = exp_state.last_dqc.quality.value if exp_state.last_dqc else "unknown"
        bt    = exp_state.total_backtracks
        cls   = exp_state.last_classification
        belief = exp_state.belief.charge_probs
        p11   = belief.get((1,1), 0.0)
        most_likely = exp_state.belief.most_likely_state()

        cls_str = "none yet"
        if cls:
            ood_str = f", OOD={'yes' if exp_state.is_ood else 'no'}"
            cls_str = f"{cls.label.value} ({cls.confidence:.0%} conf{ood_str})"

        state_block = (
            f"Stage: {stage}\n"
            f"Measurements: {meas}/{budget} ({pct}% used)\n"
            f"Current voltage: Vg1={vg1:+.3f}V, Vg2={vg2:+.3f}V\n"
            f"SNR: {snr:.1f}dB, DQC quality: {dqc}\n" if snr else
            f"Current voltage: Vg1={vg1:+.3f}V, Vg2={vg2:+.3f}V\n"
            f"DQC quality: {dqc}\n"
        ) + (
            f"CNN classification: {cls_str}\n"
            f"Belief P(1,1)={p11:.3f}, most likely state: {most_likely}\n"
            f"Backtracks so far: {bt}\n"
            f"Run complete: {'yes' if st.session_state.done_event and st.session_state.done_event.is_set() else 'no'}"
        )
    return DR_Q_SYSTEM.format(state_block=state_block)


def _call_drq_llm(user_msg: str) -> str:
    """
    Call Qwen directly with full chat history + dynamic state.
    This is Dr. Q's actual intelligence — separate from the narrator.
    Falls back to an informative offline response when LLM is unavailable.
    """
    base_url = os.environ.get("QDOT_LLM_BASE_URL", "").rstrip("/")
    model    = os.environ.get("QDOT_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    if not base_url:
        return _offline_drq(user_msg)

    # Build message list: system + last N turns + new user message
    system = _build_system_prompt()
    messages = [{"role": "system", "content": system}]
    for msg in st.session_state.chat[-12:]:   # last 12 keeps context manageable
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    # The new user message is already appended to chat by the caller,
    # so it's included in the loop above.

    try:
        import openai
        client = openai.OpenAI(
            base_url=base_url + "/v1" if not base_url.endswith("/v1") else base_url,
            api_key=os.environ.get("QDOT_LLM_API_KEY", "EMPTY"),
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM connection error: {exc}]\n\nFalling back: {_offline_drq(user_msg)}"


def _offline_drq(user_msg: str) -> str:
    """
    Context-aware offline response when the LLM is not connected.
    Tries to be genuinely useful rather than a canned script.
    """
    exp_state = st.session_state.exp_state
    stage = exp_state.stage.name if exp_state else None
    q = user_msg.lower()

    # Physics questions
    if any(w in q for w in ["what is","what's","explain","how does","why","stability diagram","quantum dot","charge state","coulomb","electron"]):
        if "stability diagram" in q or "plot" in q or "graph" in q or "diagram" in q:
            return ("A stability diagram maps electrical conductance G as both gate voltages Vg₁ and Vg₂ "
                    "are swept. The bright lines you see are Coulomb peaks — points where an electron "
                    "tunnels onto or off a dot. Where two sets of lines intersect, you're at a triple "
                    "point — the boundary between three charge states. The honeycomb pattern of those "
                    "intersections maps out regions of fixed electron occupation: (0,0), (1,0), (0,1), "
                    "(1,1), etc. The agent's job is to find and navigate to the (1,1) diamond.")
        if "quantum dot" in q:
            return ("A quantum dot is a tiny island of semiconductor material — so small (~100nm) that "
                    "electrons are confined and can only exist in discrete energy levels, like an artificial "
                    "atom. In this experiment, two dots are connected: gate voltages Vg₁ and Vg₂ control "
                    "how many electrons sit on each dot. The (1,1) state — one electron per dot — is the "
                    "starting configuration needed for spin qubit operations.")
        if "1,1" in q or "target" in q:
            return ("The (1,1) charge state means exactly one electron occupies each of the two quantum dots. "
                    "It matters because spin qubits — the basis of this class of quantum computers — are "
                    "encoded in the spin of single electrons. You can't run a spin qubit experiment if you "
                    "have 0 or 2 electrons on a dot. So (1,1) is the required starting configuration.")
        if "cnn" in q or "classifier" in q or "neural" in q:
            return ("The charge classifier is a 5-model ensemble of TinyCNNs — each a small conv net with "
                    "4 convolutional blocks and global average pooling. Each model independently classifies "
                    "a 64×64 stability diagram as double-dot, single-dot, or misc. The ensemble's "
                    "disagreement (max pairwise L∞ distance between softmax vectors) drives the uncertainty "
                    "estimate. An OOD detector using Mahalanobis distance on the penultimate features "
                    "catches diagrams outside the training distribution. Val accuracy: 91.4% on 51k simulated diagrams.")
        if "navigation" in q or "why did it stop" in q or "failed" in q:
            return ("Navigation is Phase 3 — it's the known unsolved part. The Bayesian optimiser proposes "
                    "voltage moves, but in intermediate voltage space the CNN almost always says 'misc' "
                    "because the device isn't near a charge boundary. There's no useful reward signal, so "
                    "the BO wanders randomly. The BOOTSTRAPPING → CHARGE_ID pipeline works reliably; "
                    "Navigation needs a proper reward signal redesign before it can converge.")

    # State questions
    if stage and any(w in q for w in ["status","happening","going","what stage","where"]):
        if exp_state:
            meas = exp_state.total_measurements
            budget = st.session_state.agent.measurement_budget if st.session_state.agent else 2048
            return (f"Currently in {stage} — {meas}/{budget} measurements used. "
                    + STAGE_DESC.get(stage, ("",""))[0])

    # Start intent
    if any(kw in q for kw in START_KWS):
        return ("Type 'start' to launch a tuning run. I'll narrate what the agent is doing "
                "at each stage and flag anything unusual. Ask me anything during the run.")

    # Generic fallback
    ctx = f" (current stage: {stage})" if stage else " (no run active)"
    return (f"[LLM offline{ctx}] — Connect the AMD MI300X to get real answers. "
            f"Set QDOT_LLM_BASE_URL in the sidebar or via environment variable. "
            f"In the meantime, try asking me about the stability diagram, quantum dots, "
            f"the CNN classifier, or why navigation stops.")


def _add_msg(role: str, content: str, kind: str = "n"):
    st.session_state.chat.append({"role": role, "content": content, "kind": kind})


def _handle_chat(user_msg: str):
    """Process a user message: add to chat, decide whether to start agent, call Dr. Q."""
    _add_msg("user", user_msg)
    is_start = any(kw in user_msg.lower() for kw in START_KWS)

    if is_start and not st.session_state.running and st.session_state.agent is None:
        # Launch the agent
        agent, exp_state, narrator, hitl_mgr = _make_agent(
            st.session_state.use_llm, st.session_state.use_cnn,
            st.session_state.meas_budget, st.session_state.max_steps,
        )
        done_event = threading.Event()
        thread = threading.Thread(target=_run_thread, args=(agent, done_event), daemon=True)
        st.session_state.update(
            agent=agent, exp_state=exp_state, narrator=narrator,
            hitl_manager=hitl_mgr, done_event=done_event, thread=thread,
            running=True, run_count=st.session_state.run_count+1,
        )
        thread.start()
        # Immediate acknowledgement (doesn't call LLM — agent is just starting)
        _add_msg("assistant",
            "Tuning sequence initiated.\n\n"
            "Physics sim: CIM double quantum dot (local CPU)\n"
            "LLM narrator: " +
            ("Qwen2.5-1.5B on AMD MI300X" if st.session_state.use_llm
             else "offline — connect via sidebar") +
            "\n\nStarting BOOTSTRAPPING — electrical integrity check.\n\n"
            "Ask me anything while the agent runs.")
    else:
        # Full LLM call (or smart offline fallback)
        # Show thinking indicator in chat temporarily
        reply = _call_drq_llm(user_msg)
        _add_msg("assistant", reply)


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────
def _make_agent(use_llm, use_cnn, meas_budget, max_steps):
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
        except Exception:
            pass

    if not use_llm:
        os.environ.pop("QDOT_LLM_BASE_URL", None)

    run_dir = Path("results/demo") / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hitl_mgr = HITLManager(queue_dir=str(run_dir/"hitl"))
    hitl_mgr.set_test_mode()

    agent = ExecutiveAgent(
        state=state, adapter=adapter, inspection_agent=inspection,
        dqc=DQCGatekeeper(),
        safety_critic=SafetyCritic(voltage_bounds=state.voltage_bounds, l1_max=0.10),
        hitl_manager=hitl_mgr,
        governance_logger=GovernanceLogger(run_id=state.run_id, log_dir=str(run_dir/"governance")),
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
        arr = np.asarray(m.array,dtype=np.float32)
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
            marker=dict(size=8,color="#FF4040",symbol="cross-thin",line=dict(width=2.5,color="#FF4040")),
            showlegend=False))
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
        showscale=False,zmin=0,zmax=1,
        text=[[f"{z[j][i]:.2f}" for i in range(3)] for j in range(3)],
        texttemplate="%{text}",textfont=dict(size=12,color="#1C2333",family="JetBrains Mono"),
    ))
    fig.add_shape(type="rect",x0=.5,x1=1.5,y0=.5,y1=1.5,line=dict(color="#00897B",width=2.5))
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
            colorscale=[[0,"#E0F4F1"],[1,"#00897B"]],showscale=False),showlegend=False))
    fig.add_trace(go.Scatter(x=[xs[-1]],y=[ys[-1]],mode="markers",
        marker=dict(size=9,color="#E85000",symbol="x-thin",line=dict(width=2.5,color="#E85000")),
        showlegend=False))
    fig.update_layout(
        title=dict(text="Voltage Trajectory",font=dict(size=11,color="#5A6478")),
        xaxis=dict(title="Vg₁",gridcolor="#E8E4DC"),
        yaxis=dict(title="Vg₂",gridcolor="#E8E4DC"),
        height=185, **PLOT_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI components
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
      <div class="kpi"><div class="kpi-v">{b}<span class="kpi-u">/{t}</span></div><div class="kpi-l">Measurements</div></div>
      <div class="kpi"><div class="kpi-v">{vg1:+.2f}<span class="kpi-u"> V</span></div><div class="kpi-l">Vg₁</div></div>
      <div class="kpi"><div class="kpi-v">{vg2:+.2f}<span class="kpi-u"> V</span></div><div class="kpi-l">Vg₂</div></div>
      <div class="kpi"><div class="kpi-v">{snr:.1f}<span class="kpi-u"> dB</span></div><div class="kpi-l">SNR</div></div>
    </div>""", unsafe_allow_html=True)


def _spy(current, hitl_active):
    html = '<div class="spy-grid">'
    for key,em,name,role,active in SPY_AGENTS:
        on = (key=="hitl" and hitl_active) or (key!="hitl" and current in active)
        cc = "spy spy-on" if on else "spy"
        dc = "spy-dot spy-dot-on" if on else "spy-dot"
        html += (f'<div class="{cc}"><div class="spy-em">{em}</div>'
                 f'<div class="spy-name">{name}</div><div class="spy-role">{role}</div>'
                 f'<div class="{dc}"></div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _chat_display():
    """Render the chat history as a read-only HTML panel."""
    # Pump narrator async events (anomalies, summary) into chat
    narrator = st.session_state.narrator
    if narrator:
        for ev in narrator.event_log():
            tag = f"_ev_{ev.timestamp:.4f}"
            if tag not in st.session_state:
                st.session_state[tag] = True
                if ev.kind == "exception" and ev.response:
                    _add_msg("assistant", f"⚠ {ev.response}", kind="ev")
                elif ev.kind == "summary" and ev.response:
                    _add_msg("assistant", ev.response)

    llm_url = os.environ.get("QDOT_LLM_BASE_URL","")
    use_llm = st.session_state.use_llm
    if llm_url and use_llm:
        badge = '<span class="badge badge-mi300x">⬡ Qwen2.5 · AMD MI300X</span>'
    else:
        badge = '<span class="badge badge-idle">LLM offline</span>'

    msgs_html = ""
    for msg in st.session_state.chat:
        c = (msg["content"]
             .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
             .replace("\n","<br>"))
        # Bold markdown
        import re
        c = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', c)
        if msg["role"] == "user":
            msgs_html += (f'<div class="msg msg-u"><div class="mlabel">You</div>'
                          f'<div class="bubble">{c}</div></div>')
        else:
            kind = msg.get("kind","n")
            css  = "msg msg-a msg-ev" if kind=="ev" else "msg msg-a"
            lbl  = "Dr. Q — anomaly" if kind=="ev" else "Dr. Q"
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
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
running    = st.session_state.running
exp_state  = st.session_state.exp_state
done_event = st.session_state.done_event
llm_url    = os.environ.get("QDOT_LLM_BASE_URL","")

with st.sidebar:
    st.markdown('<div style="color:#8A9AB0;font-size:10px;font-family:JetBrains Mono,monospace;'
                'padding:8px 0 12px;border-bottom:1px solid #2C3545;margin-bottom:14px;'
                'letter-spacing:1px">CONFIGURATION</div>', unsafe_allow_html=True)
    st.session_state.use_llm  = st.toggle("Dr. Q on AMD MI300X",  value=st.session_state.use_llm,  disabled=running)
    st.session_state.use_cnn  = st.toggle("CNN Charge Classifier", value=st.session_state.use_cnn,  disabled=running)
    st.session_state.meas_budget = st.slider("Measurement Budget",512,4096,st.session_state.meas_budget,128,disabled=running)
    st.session_state.max_steps   = st.slider("Max Steps",10,200,st.session_state.max_steps,10,disabled=running)
    new_url = st.text_input("vLLM endpoint",value=llm_url,
                            placeholder="http://129.212.186.42:8000/v1",disabled=running)
    if new_url:
        os.environ["QDOT_LLM_BASE_URL"] = new_url
        llm_url = new_url
    st.divider()
    if st.button("Reset session",use_container_width=True,disabled=running):
        for k in ["agent","exp_state","narrator","hitl_manager","done_event","thread","chat"]:
            st.session_state[k] = [] if k=="chat" else None
        st.session_state.running = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Check completion
# ─────────────────────────────────────────────────────────────────────────────
if done_event and done_event.is_set() and running:
    st.session_state.running = False
    running = False
    if exp_state and exp_state.stage.name == "COMPLETE":
        st.balloons()
    # Post-run Dr. Q summary via LLM
    if exp_state and not any(m.get("kind")=="summary" for m in st.session_state.chat):
        stage = exp_state.stage.name
        meas  = exp_state.total_measurements
        budget = st.session_state.agent.measurement_budget if st.session_state.agent else 2048
        bt    = exp_state.total_backtracks
        _add_msg("user", f"The run just finished at stage {stage}. Give me a post-run summary.")
        summary = _call_drq_llm(f"The run finished at stage {stage}. "
                                 f"{meas}/{budget} measurements used, {bt} backtracks. "
                                 f"Give a concise post-run summary.")
        _add_msg("assistant", summary, kind="summary")


# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────
tl,tr = st.columns([3,1])
with tl:
    st.markdown('<div class="topbar"><div>'
                '<div class="topbar-title">⚛ SimQuantum Tuning Lab</div>'
                '<div class="topbar-sub">Autonomous quantum dot tuning · AMD Developer Hackathon 2025</div>'
                '</div></div>', unsafe_allow_html=True)
with tr:
    b = ('<span class="badge badge-live">● LIVE</span>' if running
         else '<span class="badge badge-idle">● IDLE</span>')
    if llm_url and st.session_state.use_llm:
        b += ' <span class="badge badge-mi300x">⬡ MI300X</span>'
    st.markdown(f'<div style="text-align:right;padding-top:6px">{b}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Splash (pre-run)
# ─────────────────────────────────────────────────────────────────────────────
if exp_state is None:
    if not st.session_state.chat:
        _add_msg("assistant",
            "Ready to tune.\n\n"
            "Type **start** to begin a run, or ask me about the physics — "
            "I can explain stability diagrams, the CNN classifier, why (1,1) matters, "
            "or anything else about this experiment.")

    sl,sr = st.columns([3,2],gap="large")
    with sl:
        st.markdown("""
        <div class="card" style="padding:22px 24px">
          <div class="card-title">What this system does</div>
          <p style="font-size:13px;color:#5A6478;line-height:1.75;margin:0">
            SimQuantum autonomously tunes a double quantum dot device to the
            <strong>(1,1) charge state</strong> — one electron per dot — required
            for spin qubit operation. It uses a <strong>6-stage POMDP planner</strong>,
            a <strong>5-model CNN ensemble</strong> (91.4% val acc, 51k training diagrams),
            and <strong>Bayesian optimisation</strong>.
            <strong>Qwen2.5-1.5B on AMD MI300X</strong> acts as Dr. Q — an AI
            co-pilot you can ask anything, in plain English or technical detail.
          </p>
        </div>
        <div class="card">
          <div class="card-title">How to read a stability diagram</div>
          <p style="font-size:13px;color:#5A6478;line-height:1.75;margin:0">
            The chart maps conductance G as both gate voltages are swept.
            <strong>Bright lines</strong> are Coulomb peaks — charge transitions.
            Their intersections form a honeycomb of fixed electron states:
            (0,0), (1,0), (0,1), <strong>(1,1)</strong>, etc.
            Coulomb blockade (dark regions) = electrons can't move. Peaks = they can.
            Ask Dr. Q to explain anything in more depth.
          </p>
        </div>
        """, unsafe_allow_html=True)
        img_path = Path(__file__).parent/"assets"/"simquantum.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    with sr:
        _chat_display()


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

    # HITL gate
    pending = hitl_mgr.get_pending() if hitl_mgr else []
    if pending:
        req = pending[0]
        st.markdown(
            f'<div class="hitl-card">'
            f'  <div class="hitl-title">⚠ HITL GATE — Human approval required</div>'
            f'  <div class="hitl-body">Step {req["step"]} · Stage {req["stage"]} · '
            f'    Risk {req["risk_score"]:.2f}<br><strong>{req["trigger_reason"]}</strong></div>'
            f'</div>', unsafe_allow_html=True)
        hc1,hc2,_ = st.columns([1,1,5])
        with hc1:
            if st.button("✓ Approve",type="primary",key=f"appr_{req['id']}"):
                hitl_mgr.approve(req["id"],deciding_human="operator")
                _add_msg("assistant","Approved. The agent continues.")
                st.rerun()
        with hc2:
            if st.button("✗ Reject",key=f"rej_{req['id']}"):
                hitl_mgr.reject(req["id"],deciding_human="operator")
                _add_msg("assistant","Rejected. The agent backtracks.")
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
            ft=_fig_traj(exp_state)
            if ft: st.plotly_chart(ft,use_container_width=True,
                                   config={"displayModeBar":False},key=f"t_{time.monotonic_ns()}")

        st.markdown('<div class="card-title" style="margin-top:4px">Agent Activity</div>',
                    unsafe_allow_html=True)
        _spy(current_stg, bool(pending))

        img_path = Path(__file__).parent/"assets"/"simquantum.png"
        if img_path.exists():
            st.image(str(img_path),use_container_width=True)

    with right:
        _chat_display()

        if current_stg in STAGE_DESC:
            desc,cost = STAGE_DESC[current_stg]
            st.markdown(
                f'<div class="card" style="margin-top:8px">'
                f'  <div class="card-title">Current stage · {current_stg}</div>'
                f'  <div style="font-size:12px;color:#5A6478;line-height:1.6">{desc}</div>'
                f'  <div style="font-size:10px;color:#A8B0BC;margin-top:5px;'
                f'    font-family:JetBrains Mono,monospace">Budget: {cost}</div>'
                f'</div>', unsafe_allow_html=True)

        if done_event and done_event.is_set():
            ok  = current_stg=="COMPLETE"
            col = "#00897B" if ok else "#C84B00"
            txt = "MISSION COMPLETE" if ok else f"STOPPED — {current_stg}"
            red = 1.0-(exp_state.total_measurements/max(64*64,1))
            st.markdown(
                f'<div class="card" style="border-color:{col};margin-top:8px">'
                f'  <div style="font-size:14px;font-weight:700;color:{col};'
                f'    font-family:JetBrains Mono,monospace;margin-bottom:8px">{txt}</div>'
                f'  <div style="font-size:12px;color:#5A6478;display:grid;'
                f'    grid-template-columns:1fr 1fr;gap:5px">'
                f'    <span>Measurements: <b>{exp_state.total_measurements}</b></span>'
                f'    <span>Steps: <b>{agent.control_steps}</b></span>'
                f'    <span>Backtracks: <b>{exp_state.total_backtracks}</b></span>'
                f'    <span>Reduction: <b>{red:.0%}</b></span>'
                f'  </div></div>', unsafe_allow_html=True)
            if st.button("🔄 New Run",use_container_width=True):
                for k in ["agent","exp_state","narrator","hitl_manager","done_event","thread"]:
                    st.session_state[k] = None
                st.session_state.running = False
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Chat input — ALWAYS at the bottom, never inside a column or conditional.
# This is the fix for "can't type after 2 messages" — Streamlit requires
# st.chat_input to be called at the same script position on every rerun.
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask Dr. Q anything, or type 'start' to begin…",
    key="drq_input",
):
    _handle_chat(prompt)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh while live
# ─────────────────────────────────────────────────────────────────────────────
if running and done_event and not done_event.is_set():
    time.sleep(0.8)
    st.rerun()
