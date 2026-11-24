import os
import tempfile
from typing import Any, List, Optional
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.database import (
    get_customer,
    get_order,
    get_order_refunds,
    get_audit_logs,
    init_db,
    seed_demo_data,
)
from core.graph import AgentService
from core.schemas import ApprovalDecision, ToolRiskLevel


class FakeScriptedChatModel(BaseChatModel):
    """Deterministic scripted mock LLM for 100% offline testing with zero network requests."""
    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

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
def test_env():
    """Create isolated temporary database files for ops_data and checkpoints."""
    fd_ops, ops_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd_ops)
    fd_chk, chk_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd_chk)

    init_db(ops_path)
    seed_demo_data(ops_path, force=True)

    yield {"ops_path": ops_path, "chk_path": chk_path}

    for path in (ops_path, chk_path):
        try:
            if os.path.exists(path):
                os.remove(path)
            for ext in ("-wal", "-shm"):
                if os.path.exists(path + ext):
                    os.remove(path + ext)
        except OSError:
            pass


def test_readonly_query_flow(test_env, monkeypatch):
    """Verify that a multi-step read-only query executes autonomously without triggering HITL interrupt."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: test_env["ops_path"])

    # Scripted LLM conversation:
    # Turn 1: Call lookup_customer
    # Turn 2: Call get_order_details
    # Turn 3: Synthesize final answer (no tool calls)
    llm = FakeScriptedChatModel(responses=[
        AIMessage(
            content="",
            id="ai_1",
            tool_calls=[{"name": "lookup_customer", "args": {"customer_id": "cust_001"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            id="ai_2",
            tool_calls=[{"name": "get_order_details", "args": {"order_id": "ord_101"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(
            content="Alice Chen has 2 orders. Order ord_101 is $150.00 and was delivered 10 days ago.",
            id="ai_3",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=test_env["chk_path"],
        db_path=test_env["ops_path"],
    )

    resp = service.run_turn("thread_ro_01", "What is the status of Alice Chen's recent order?")

    assert resp.status == "completed"
    assert resp.pending_approval is None
    assert "Alice Chen" in resp.assistant_message
    assert len(resp.tool_trace) == 2
    assert resp.tool_trace[0]["tool"] == "lookup_customer"
    assert resp.tool_trace[1]["tool"] == "get_order_details"


def test_high_risk_refund_pause_and_approve_flow(test_env, monkeypatch):
    """Verify high-risk refund pauses at human_approval_gate and completes on 'approve' with verification."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: test_env["ops_path"])

    llm = FakeScriptedChatModel(responses=[
        AIMessage(
            content="",
            id="ai_refund_1",
            tool_calls=[{
                "name": "issue_refund",
                "args": {"order_id": "ord_101", "amount": 150.00, "reason": "Product arrived broken"},
                "id": "call_ref_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="The refund of $150.00 for order ord_101 has been approved and issued.",
            id="ai_refund_2",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=test_env["chk_path"],
        db_path=test_env["ops_path"],
    )

    # 1. Run turn -> Should pause at human_approval_gate
    resp1 = service.run_turn("thread_refund_01", "Please refund order ord_101 for $150.00.")
    assert resp1.status == "awaiting_approval"
    assert resp1.pending_approval is not None
    assert resp1.pending_approval.tool_name == "issue_refund"
    assert resp1.pending_approval.tool_args["amount"] == 150.00
    assert "ord_101" in resp1.pending_approval.policy_summary
    assert "Eligible" in resp1.pending_approval.policy_summary

    # Order should NOT be modified yet in DB
    order_before = get_order("ord_101", db_path=test_env["ops_path"])
    assert order_before["status"] == "delivered"

    # 2. Resume with approval
    decision = ApprovalDecision(decision="approve", reviewer_note="Supervisor verified return tracking")
    resp2 = service.resume_turn("thread_refund_01", decision)

    assert resp2.status == "completed"
    assert resp2.verification is not None
    assert resp2.verification.verified is True
    assert resp2.verification.tool_name == "issue_refund"

    # Verify state mutation in database
    order_after = get_order("ord_101", db_path=test_env["ops_path"])
    assert order_after["status"] == "refunded"
    refunds = get_order_refunds("ord_101", db_path=test_env["ops_path"])
    assert len(refunds) == 1
    assert refunds[0]["amount"] == 150.00


def test_high_risk_refund_edit_flow(test_env, monkeypatch):
    """Verify operator can edit arguments (e.g. partial refund) and execute patched arguments in-place."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: test_env["ops_path"])

    llm = FakeScriptedChatModel(responses=[
        AIMessage(
            content="",
            id="ai_edit_1",
            tool_calls=[{
                "name": "issue_refund",
                "args": {"order_id": "ord_101", "amount": 150.00, "reason": "Customer requested full refund"},
                "id": "call_edit_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="A partial refund of $50.00 has been successfully issued as modified by the supervisor.",
            id="ai_edit_2",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=test_env["chk_path"],
        db_path=test_env["ops_path"],
    )

    # 1. Run turn -> pauses
    resp1 = service.run_turn("thread_edit_01", "Issue full refund for ord_101.")
    assert resp1.status == "awaiting_approval"

    # 2. Resume with edited arguments: partial refund of $50 instead of $150
    decision = ApprovalDecision(
        decision="edit",
        edited_args={"order_id": "ord_101", "amount": 50.00, "reason": "Partial 33% courtesy refund approved"},
        reviewer_note="Full refund rejected; partial $50 approved",
    )
    resp2 = service.resume_turn("thread_edit_01", decision)

    assert resp2.status == "completed"
    assert resp2.verification is not None
    assert resp2.verification.verified is True

    # Verify only $50 was refunded and status is partially_refunded
    order_after = get_order("ord_101", db_path=test_env["ops_path"])
    assert order_after["status"] == "partially_refunded"
    refunds = get_order_refunds("ord_101", db_path=test_env["ops_path"])
    assert len(refunds) == 1
    assert refunds[0]["amount"] == 50.00

    # Verify audit trail contains edited action
    audit = get_audit_logs("thread_edit_01", db_path=test_env["ops_path"])
    actions = [a["action"] for a in audit]
    assert "APPROVAL_EDITED" in actions
    assert "REFUND_ISSUED" in actions


def test_high_risk_refund_reject_flow(test_env, monkeypatch):
    """Verify operator rejection prevents database mutation and returns 'rejected' status."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: test_env["ops_path"])

    llm = FakeScriptedChatModel(responses=[
        AIMessage(
            content="",
            id="ai_rej_1",
            tool_calls=[{
                "name": "issue_refund",
                "args": {"order_id": "ord_102", "amount": 45.00, "reason": "Expired item return"},
                "id": "call_rej_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="I cannot issue the refund because the request was rejected by our operational supervisor.",
            id="ai_rej_2",
            tool_calls=[],
        ),
    ])

    service = AgentService(
        llm=llm,
        checkpoint_db_path=test_env["chk_path"],
        db_path=test_env["ops_path"],
    )

    # 1. Run turn -> pauses
    resp1 = service.run_turn("thread_rej_01", "Refund order ord_102.")
    assert resp1.status == "awaiting_approval"

    # 2. Resume with rejection
    decision = ApprovalDecision(
        decision="reject",
        reviewer_note="Policy violation: order is 45 days old, exceeds strict 30-day window",
    )
    resp2 = service.resume_turn("thread_rej_01", decision)

    assert resp2.status == "rejected"
    assert "rejected" in resp2.assistant_message

    # Verify database was NOT mutated
    order = get_order("ord_102", db_path=test_env["ops_path"])
    assert order["status"] == "delivered"
    refunds = get_order_refunds("ord_102", db_path=test_env["ops_path"])
    assert len(refunds) == 0

    # Verify rejection audit log
    audit = get_audit_logs("thread_rej_01", db_path=test_env["ops_path"])
    assert any(a["action"] == "APPROVAL_REJECTED" for a in audit)


def test_crash_restart_durability(test_env, monkeypatch):
    """Verify state survives complete process restart: pause in service 1, destroy it, resume in service 2."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: test_env["ops_path"])

    # First service instance creates the pause
    llm1 = FakeScriptedChatModel(responses=[
        AIMessage(
            content="",
            id="ai_crash_1",
            tool_calls=[{
                "name": "issue_refund",
                "args": {"order_id": "ord_104", "amount": 25.00, "reason": "Item arrived broken"},
                "id": "call_crash_1",
                "type": "tool_call",
            }],
        ),
    ])

    service1 = AgentService(
        llm=llm1,
        checkpoint_db_path=test_env["chk_path"],
        db_path=test_env["ops_path"],
    )

    thread_id = "thread_crash_durability"
    resp1 = service1.run_turn(thread_id, "Please refund ord_104 for $25.00.")
    assert resp1.status == "awaiting_approval"
    assert resp1.pending_approval is not None

    # SIMULATE SERVER CRASH / RESTART:
    # Explicitly destroy service1 and close its connection
    service1._conn.close()
    del service1

    # Instantiate brand new AgentService connected to the same SQLite files
    llm2 = FakeScriptedChatModel(responses=[
        AIMessage(
            content="The refund for ord_104 was successfully approved and completed after system restart.",
            id="ai_crash_2",
            tool_calls=[],
        ),
    ])

    service2 = AgentService(
        llm=llm2,
        checkpoint_db_path=test_env["chk_path"],
        db_path=test_env["ops_path"],
    )

    # Check that service2 reconstructs the pending state directly from SQLite disk
    recovered_state = service2.get_thread_state(thread_id)
    assert recovered_state is not None
    assert recovered_state.status == "awaiting_approval"
    assert recovered_state.pending_approval is not None
    assert recovered_state.pending_approval.tool_name == "issue_refund"
    assert recovered_state.pending_approval.tool_args["amount"] == 25.00

    # Resume the turn on the recovered service
    decision = ApprovalDecision(decision="approve", reviewer_note="Approved post-restart by supervisor")
    resp_resumed = service2.resume_turn(thread_id, decision)

    assert resp_resumed.status == "completed"
    assert resp_resumed.verification is not None
    assert resp_resumed.verification.verified is True

    # Verify DB state was updated
    order = get_order("ord_104", db_path=test_env["ops_path"])
    assert order["status"] == "refunded"
    refunds = get_order_refunds("ord_104", db_path=test_env["ops_path"])
    assert len(refunds) == 1
    assert refunds[0]["amount"] == 25.00
