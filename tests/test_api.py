import ast
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from api.main import app, get_agent_service, get_settings_dep
from core.config import Settings
from core.database import init_db, seed_demo_data, get_order
from core.graph import AgentService


class FakeApiChatModel(BaseChatModel):
    """Deterministic scripted mock LLM for API integration testing."""
    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-api-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.index < len(self.responses):
            resp = self.responses[self.index]
            self.index += 1
        else:
            resp = self.responses[-1]
        return ChatResult(generations=[ChatGeneration(message=resp)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@pytest.fixture
def api_test_env():
    """Create isolated temporary files and configure FastAPI dependency overrides."""
    fd_ops, ops_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd_ops)
    fd_chk, chk_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd_chk)

    init_db(ops_path)
    seed_demo_data(ops_path, force=True)

    test_settings = Settings(
        business_db_path=ops_path,
        checkpoint_db_path=chk_path,
        auto_seed_demo_data=True,
    )

    yield {"ops_path": ops_path, "chk_path": chk_path, "settings": test_settings}

    app.dependency_overrides.clear()
    for path in (ops_path, chk_path):
        try:
            if os.path.exists(path):
                os.remove(path)
            for ext in ("-wal", "-shm"):
                if os.path.exists(path + ext):
                    os.remove(path + ext)
        except OSError:
            pass


def test_architectural_isolation_ast():
    """
    Safeguard 3: Verify that ui/streamlit_app.py NEVER imports 'core' or 'sqlite3' directly.
    Ensures strict process isolation and HTTP-only communication boundary.
    """
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py"
    assert ui_path.exists(), "ui/streamlit_app.py must exist"

    with open(ui_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(ui_path))

    disallowed_modules = {"core", "sqlite3"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                assert root_module not in disallowed_modules, (
                    f"Architectural Violation: ui/streamlit_app.py imports '{alias.name}'. "
                    f"The UI must never import directly from {disallowed_modules}."
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                assert root_module not in disallowed_modules, (
                    f"Architectural Violation: ui/streamlit_app.py imports from '{node.module}'. "
                    f"The UI must never import directly from {disallowed_modules}."
                )


def test_health_endpoint(api_test_env):
    """Verify /health endpoint returns online status and LLM configuration."""
    app.dependency_overrides[get_settings_dep] = lambda: api_test_env["settings"]
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "base_url" in data


def test_ops_snapshot_and_reset_endpoints(api_test_env, monkeypatch):
    """Verify live database snapshot inspection and clean thread-safe demo reset."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: api_test_env["ops_path"])
    app.dependency_overrides[get_settings_dep] = lambda: api_test_env["settings"]
    client = TestClient(app)

    # 1. Test /ops/snapshot
    resp_snap = client.get("/ops/snapshot")
    assert resp_snap.status_code == 200
    data = resp_snap.json()
    assert len(data["customers"]) == 4
    assert len(data["orders"]) == 6
    assert len(data["refund_ledger"]) == 1

    # 2. Test /ops/reset
    resp_reset = client.post("/ops/reset")
    assert resp_reset.status_code == 200
    reset_data = resp_reset.json()
    assert reset_data["status"] == "reset_successful"

    # Verify counts post-reset
    resp_snap2 = client.get("/ops/snapshot")
    data2 = resp_snap2.json()
    assert len(data2["customers"]) == 4
    assert len(data2["orders"]) == 6


def test_agent_run_readonly_endpoint(api_test_env, monkeypatch):
    """Verify /agent/run handles multi-turn read-only flow cleanly over HTTP."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: api_test_env["ops_path"])

    llm = FakeApiChatModel(responses=[
        AIMessage(
            content="",
            id="api_ro_1",
            tool_calls=[{"name": "lookup_customer", "args": {"customer_id": "cust_001"}, "id": "call_ro_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="Alice Chen has 2 orders on file.",
            id="api_ro_2",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=api_test_env["chk_path"],
        db_path=api_test_env["ops_path"],
    )

    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[get_settings_dep] = lambda: api_test_env["settings"]
    client = TestClient(app)

    resp = client.post(
        "/agent/run",
        json={"thread_id": "thread_api_ro_01", "message": "Look up cust_001"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["pending_approval"] is None
    assert "Alice Chen" in data["assistant_message"]
    assert len(data["tool_trace"]) == 1


def test_agent_hitl_run_and_approve_endpoint(api_test_env, monkeypatch):
    """Verify /agent/run pauses on high-risk tool and /agent/resume executes on approval."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: api_test_env["ops_path"])

    llm = FakeApiChatModel(responses=[
        AIMessage(
            content="",
            id="api_hitl_1",
            tool_calls=[{
                "name": "issue_refund",
                "args": {"order_id": "ord_101", "amount": 150.00, "reason": "Damaged goods"},
                "id": "call_hitl_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="The refund of $150.00 has been executed.",
            id="api_hitl_2",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=api_test_env["chk_path"],
        db_path=api_test_env["ops_path"],
    )

    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[get_settings_dep] = lambda: api_test_env["settings"]
    client = TestClient(app)

    # 1. Run turn -> pauses at HITL gate
    resp_run = client.post(
        "/agent/run",
        json={"thread_id": "thread_api_hitl_01", "message": "Refund ord_101 for $150"},
    )
    assert resp_run.status_code == 200
    data_run = resp_run.json()
    assert data_run["status"] == "awaiting_approval"
    assert data_run["pending_approval"]["tool_name"] == "issue_refund"

    # 2. Check /threads/{thread_id} recovery
    resp_state = client.get("/threads/thread_api_hitl_01")
    assert resp_state.status_code == 200
    data_state = resp_state.json()
    assert data_state["status"] == "awaiting_approval"
    assert data_state["pending_approval"]["tool_name"] == "issue_refund"

    # 3. Resume with approve
    resp_resume = client.post(
        "/agent/resume",
        json={
            "thread_id": "thread_api_hitl_01",
            "decision": {"decision": "approve", "reviewer_note": "Approved by supervisor via API"},
        },
    )
    assert resp_resume.status_code == 200
    data_resume = resp_resume.json()
    assert data_resume["status"] == "completed"
    assert data_resume["verification"]["verified"] is True

    # Confirm DB update
    order = get_order("ord_101", db_path=api_test_env["ops_path"])
    assert order["status"] == "refunded"


def test_agent_resume_edit_endpoint(api_test_env, monkeypatch):
    """Verify /agent/resume with decision='edit' patches arguments and issues edited refund."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: api_test_env["ops_path"])

    llm = FakeApiChatModel(responses=[
        AIMessage(
            content="",
            id="api_edit_1",
            tool_calls=[{
                "name": "issue_refund",
                "args": {"order_id": "ord_101", "amount": 150.00, "reason": "Full refund request"},
                "id": "call_edit_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="Partial refund of $50.00 processed.",
            id="api_edit_2",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=api_test_env["chk_path"],
        db_path=api_test_env["ops_path"],
    )

    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[get_settings_dep] = lambda: api_test_env["settings"]
    client = TestClient(app)

    # 1. Run turn -> pauses
    client.post(
        "/agent/run",
        json={"thread_id": "thread_api_edit_01", "message": "Refund ord_101"},
    )

    # 2. Resume with edit decision
    resp_resume = client.post(
        "/agent/resume",
        json={
            "thread_id": "thread_api_edit_01",
            "decision": {
                "decision": "edit",
                "edited_args": {"order_id": "ord_101", "amount": 50.00, "reason": "Partial compensation"},
                "reviewer_note": "Reduced from $150 to $50",
            },
        },
    )
    assert resp_resume.status_code == 200
    data_resume = resp_resume.json()
    assert data_resume["status"] == "completed"
    assert data_resume["verification"]["verified"] is True

    order = get_order("ord_101", db_path=api_test_env["ops_path"])
    assert order["status"] == "partially_refunded"


def test_agent_resume_reject_endpoint(api_test_env, monkeypatch):
    """Verify /agent/resume with decision='reject' prevents mutation."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: api_test_env["ops_path"])

    llm = FakeApiChatModel(responses=[
        AIMessage(
            content="",
            id="api_rej_1",
            tool_calls=[{
                "name": "issue_refund",
                "args": {"order_id": "ord_102", "amount": 45.00, "reason": "Expired item return"},
                "id": "call_rej_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="Refund was rejected by supervisor.",
            id="api_rej_2",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=api_test_env["chk_path"],
        db_path=api_test_env["ops_path"],
    )

    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[get_settings_dep] = lambda: api_test_env["settings"]
    client = TestClient(app)

    client.post(
        "/agent/run",
        json={"thread_id": "thread_api_rej_01", "message": "Refund ord_102"},
    )

    resp_resume = client.post(
        "/agent/resume",
        json={
            "thread_id": "thread_api_rej_01",
            "decision": {"decision": "reject", "reviewer_note": "Expired past 30 days"},
        },
    )
    assert resp_resume.status_code == 200
    data_resume = resp_resume.json()
    assert data_resume["status"] == "rejected"

    order = get_order("ord_102", db_path=api_test_env["ops_path"])
    assert order["status"] == "delivered"


def test_threads_not_found_404(api_test_env):
    """Verify unknown thread ID returns 404 HTTPException."""
    app.dependency_overrides[get_settings_dep] = lambda: api_test_env["settings"]
    client = TestClient(app)

    response = client.get("/threads/completely_unknown_thread_id")
    assert response.status_code == 404
