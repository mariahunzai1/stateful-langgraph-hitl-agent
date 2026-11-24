import importlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Safeguard 1: Insert Repo Root at sys.path[0] before local imports
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

# Safeguard 1: Sync secrets into os.environ before backend initialization
try:
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if key in st.secrets and isinstance(st.secrets[key], str):
            os.environ[key] = st.secrets[key]
except Exception:
    pass

import httpx
import uvicorn

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# --- Background Server Self-Launcher for Streamlit Community Cloud ---

class NoSignalServer(uvicorn.Server):
    """Custom uvicorn server subclass with disabled signal handlers for background execution."""
    def install_signal_handlers(self) -> None:
        pass


@st.cache_resource
def ensure_backend_running(base_url: str) -> bool:
    """Ensure FastAPI backend is running; self-launch in background daemon thread if running locally."""
    # 1. Quick probe
    try:
        resp = httpx.get(f"{base_url}/health", timeout=1.5)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    # 2. If target is localhost, launch daemon server
    if "127.0.0.1" in base_url or "localhost" in base_url:
        try:
            api_module = importlib.import_module("api.main")
            app = getattr(api_module, "app")
            port = 8000
            if ":" in base_url:
                port_str = base_url.split(":")[-1].split("/")[0]
                if port_str.isdigit():
                    port = int(port_str)

            config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
            server = NoSignalServer(config=config)

            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()

            # Poll until ready
            start_time = time.time()
            while time.time() - start_time < 12.0:
                try:
                    probe = httpx.get(f"{base_url}/health", timeout=1.0)
                    if probe.status_code == 200:
                        return True
                except Exception:
                    time.sleep(0.3)
        except Exception as exc:
            st.error(f"Failed to auto-launch backend server: {exc}")
            return False

    return False


# --- HTTP Client API Helpers (Strictly Decoupled) ---

def api_get_health() -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def api_run_agent(thread_id: str, message: str) -> dict[str, Any] | None:
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/agent/run",
            json={"thread_id": thread_id, "message": message},
            timeout=120.0,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as exc:
        st.error(f"API Error during run: {exc}")
        return None


def api_resume_agent(thread_id: str, decision: dict[str, Any]) -> dict[str, Any] | None:
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/agent/resume",
            json={"thread_id": thread_id, "decision": decision},
            timeout=120.0,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as exc:
        st.error(f"API Error during resume: {exc}")
        return None


def api_get_thread_state(thread_id: str) -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{API_BASE_URL}/threads/{thread_id}", timeout=120.0)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def api_get_ops_snapshot() -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{API_BASE_URL}/ops/snapshot", timeout=120.0)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def api_reset_ops_data() -> bool:
    try:
        resp = httpx.post(f"{API_BASE_URL}/ops/reset", timeout=120.0)
        return resp.status_code == 200
    except Exception:
        return False


# --- Streamlit Page Configuration ---

st.set_page_config(
    page_title="LangGraph HITL Operations Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"thread_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "current_response" not in st.session_state:
    st.session_state["current_response"] = None
if "queued_prompt" not in st.session_state:
    st.session_state["queued_prompt"] = None


def queue_demo_prompt(prompt_text: str) -> None:
    """Callback to stage a 1-click demo prompt for immediate turn execution."""
    st.session_state["queued_prompt"] = prompt_text


# Ensure Backend Availability
backend_online = ensure_backend_running(API_BASE_URL)
health_info = api_get_health() if backend_online else None


# --- Sidebar (Placed Above Main Workspace & Tabs) ---

with st.sidebar:
    st.title("🛡️ Operator Console")
    st.caption("Human-in-the-Loop Governance & Audit Engine")

    # Status Badge
    if health_info:
        st.success(f"🟢 **Backend Online**\nModel: `{health_info.get('model', 'unknown')}`")
    else:
        st.error("🔴 **Backend Offline** — attempting connection...")

    st.divider()

    # Thread Manager
    st.subheader("Thread Management")
    st.text_input("Active Thread ID", key="thread_id_input", value=st.session_state["thread_id"])
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("New Thread", use_container_width=True):
            new_id = f"thread_{uuid.uuid4().hex[:8]}"
            st.session_state["thread_id"] = new_id
            st.session_state["chat_history"] = []
            st.session_state["current_response"] = None
            st.session_state["queued_prompt"] = None
            st.rerun()
    with col_t2:
        if st.button("Sync State", use_container_width=True):
            state = api_get_thread_state(st.session_state["thread_id"])
            if state:
                st.session_state["current_response"] = state
                st.toast("Thread state recovered from SQLite checkpoint!")
            else:
                st.warning("No checkpoint found for this thread.")
            st.rerun()

    st.divider()

    # 1-Click Demo Scenarios (Callback-based Immediate Execution)
    st.subheader("🎯 1-Click Demo Scenarios")

    st.button(
        "1. Read-Only Customer Lookup",
        use_container_width=True,
        on_click=queue_demo_prompt,
        args=("What is the status of Alice Chen's recent order (cust_001)?",),
    )

    st.button(
        "2. Eligible Refund (Triggers HITL)",
        use_container_width=True,
        on_click=queue_demo_prompt,
        args=("Please issue a full refund for order ord_101 ($150.00) due to damaged merchandise.",),
    )

    st.button(
        "3. Partial Refund Edit Scenario",
        use_container_width=True,
        on_click=queue_demo_prompt,
        args=("Issue a $150.00 refund for order ord_101.",),
    )

    st.button(
        "4. Expired Order Policy Check",
        use_container_width=True,
        on_click=queue_demo_prompt,
        args=("Check the refund policy and refund order ord_102 ($45.00).",),
    )

    st.divider()

    # Reset Database
    st.subheader("Database Governance")
    if st.button("🔄 Reset Demo Database", use_container_width=True, type="secondary"):
        if api_reset_ops_data():
            st.session_state["chat_history"] = []
            st.session_state["current_response"] = None
            st.session_state["queued_prompt"] = None
            st.toast("Operational database reset and re-seeded!")
            st.rerun()
        else:
            st.error("Failed to reset database.")


# --- Main Workspace ---

st.title("Stateful LangGraph Operations Agent")
st.markdown(
    "**100% $0 Stack** • Stateful LangGraph • SQLite Checkpointer • Deterministic Post-Tool SQL Verification"
)

tab_chat, tab_ledger = st.tabs(["💬 Agent & HITL Console", "📊 Live Operations Ledger & Audit Trail"])

# --- TAB 1: Chat & HITL Approval Console ---

with tab_chat:
    chat_container = st.container()
    with chat_container:
        # Render Conversation History
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "tool_trace" in msg and msg["tool_trace"]:
                    with st.expander(f"Tool Trace ({len(msg['tool_trace'])} hops)", expanded=False):
                        for hop in msg["tool_trace"]:
                            t_name = hop.get("tool", "unknown")
                            is_high_risk = t_name in ("issue_refund", "update_subscription_tier")
                            risk_badge = "🔴 HIGH_RISK" if is_high_risk else "🟢 READ_ONLY"
                            st.markdown(f"**`{t_name}`** `{risk_badge}`")
                            st.json({"arguments": hop.get("args"), "result": hop.get("result")})

        # Render Current Turn / Paused Approval Gate
        cur = st.session_state["current_response"]

        if cur and cur.get("status") == "awaiting_approval":
            pending = cur.get("pending_approval") or {}
            tool_name = pending.get("tool_name", "high_risk_tool")
            tool_args = pending.get("tool_args", {})
            policy_summary = pending.get("policy_summary", "")

            st.warning("⚠️ **Human Supervisor Authorization Required**")
            with st.container(border=True):
                st.subheader(f"Approval Gate: `{tool_name}`")
                st.markdown(f"**Policy Summary:** {policy_summary}")
                st.caption(f"Interrupt ID: `{pending.get('interrupt_id')}` • Created: `{pending.get('created_at')}`")

                # Form for editing arguments or providing reviewer note
                with st.form("approval_form"):
                    st.markdown("#### Review Arguments")
                    args_json = st.text_area(
                        "Proposed Arguments (Edit JSON to modify parameters):",
                        value=json.dumps(tool_args, indent=2),
                        height=120,
                    )
                    reviewer_note = st.text_input(
                        "Reviewer Note / Justification:",
                        placeholder="e.g. Approved partial return per customer retention policy",
                    )

                    col_a, col_e, col_r = st.columns(3)
                    with col_a:
                        approve_btn = st.form_submit_button("✅ Approve", use_container_width=True, type="primary")
                    with col_e:
                        edit_btn = st.form_submit_button("✏️ Edit & Approve", use_container_width=True)
                    with col_r:
                        reject_btn = st.form_submit_button("❌ Reject", use_container_width=True)

                    if approve_btn:
                        with st.spinner("Submitting approval..."):
                            decision = {"decision": "approve", "reviewer_note": reviewer_note or "Approved by operator"}
                            resp = api_resume_agent(st.session_state["thread_id"], decision)
                            if resp:
                                st.session_state["current_response"] = resp
                                st.session_state["chat_history"].append({
                                    "role": "assistant",
                                    "content": resp.get("assistant_message", "Operation approved and executed."),
                                    "tool_trace": resp.get("tool_trace", []),
                                })
                                st.rerun()

                    if edit_btn:
                        try:
                            edited_dict = json.loads(args_json)
                            with st.spinner("Submitting edited approval..."):
                                decision = {
                                    "decision": "edit",
                                    "edited_args": edited_dict,
                                    "reviewer_note": reviewer_note or "Edited and approved by operator",
                                }
                                resp = api_resume_agent(st.session_state["thread_id"], decision)
                                if resp:
                                    st.session_state["current_response"] = resp
                                    st.session_state["chat_history"].append({
                                        "role": "assistant",
                                        "content": resp.get("assistant_message", "Operation modified and executed."),
                                        "tool_trace": resp.get("tool_trace", []),
                                    })
                                    st.rerun()
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format in edited arguments.")

                    if reject_btn:
                        with st.spinner("Submitting rejection..."):
                            decision = {"decision": "reject", "reviewer_note": reviewer_note or "Rejected by supervisor"}
                            resp = api_resume_agent(st.session_state["thread_id"], decision)
                            if resp:
                                st.session_state["current_response"] = resp
                                st.session_state["chat_history"].append({
                                    "role": "assistant",
                                    "content": resp.get("assistant_message", "Operation rejected by supervisor."),
                                    "tool_trace": resp.get("tool_trace", []),
                                })
                                st.rerun()

        # Render Post-Tool Verification Card
        if cur and cur.get("verification"):
            v = cur["verification"]
            if v.get("verified"):
                st.success(f"🔍 **Deterministic Post-Tool Verification Passed** (`{v.get('tool_name')}`)")
            else:
                st.error(f"❌ **Verification Failed**: {v.get('discrepancy')}")


# --- TAB 2: Live Operations Ledger & Audit Trail ---

with tab_ledger:
    col_lh1, col_lh2 = st.columns([4, 1])
    with col_lh1:
        st.subheader("Live Operational Database Snapshot")
        st.caption("Queried directly from SQLite (`ops_data.sqlite`) via `GET /ops/snapshot`")
    with col_lh2:
        if st.button("🔄 Refresh Snapshot", use_container_width=True):
            st.rerun()

    snapshot = api_get_ops_snapshot()
    if snapshot:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Customers (`customers`)")
            st.dataframe(snapshot.get("customers", []), use_container_width=True, hide_index=True)

        with col_c2:
            st.markdown("#### Orders (`orders`)")
            st.dataframe(snapshot.get("orders", []), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("#### Refund Ledger (`refund_ledger`)")
        refunds = snapshot.get("refund_ledger", [])
        if refunds:
            st.dataframe(refunds, use_container_width=True, hide_index=True)
        else:
            st.info("No refunds recorded yet.")

        st.divider()

        st.markdown("#### Append-Only Audit Trail (`audit_logs`)")
        audit = snapshot.get("audit_logs", [])
        if audit:
            st.dataframe(audit, use_container_width=True, hide_index=True)
        else:
            st.info("No audit logs recorded yet.")
    else:
        st.warning("Could not fetch operational database snapshot from backend.")


# --- Unified Input & Execution Flow ---

typed_prompt = st.chat_input("Enter customer request or command...")
active_prompt = typed_prompt or st.session_state.get("queued_prompt")

if active_prompt:
    st.session_state["queued_prompt"] = None
    st.session_state["chat_history"].append({"role": "user", "content": active_prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(active_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Agent deliberating and inspecting operational policy..."):
                resp = api_run_agent(st.session_state["thread_id"], active_prompt)
                if resp:
                    st.session_state["current_response"] = resp
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": resp.get("assistant_message", ""),
                        "tool_trace": resp.get("tool_trace", []),
                    })
                    st.markdown(resp.get("assistant_message", ""))
    st.rerun()
