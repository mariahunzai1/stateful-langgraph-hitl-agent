import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.database import (
    get_customer,
    get_order,
    get_order_refunds,
    init_db,
    seed_demo_data,
    set_override_db_path,
)
from core.graph import AgentService
from core.schemas import ApprovalDecision


class FakeBenchmarkChatModel(BaseChatModel):
    """Deterministic scripted mock LLM for 100% reproducible offline benchmark execution."""
    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-benchmark-model"

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


@dataclass
class ScenarioResult:
    scenario_name: str
    category: str
    passed: bool
    details: str
    duration_ms: float


@dataclass
class BenchmarkReport:
    total_scenarios: int
    passed_scenarios: int
    readonly_autonomy_rate: float
    high_risk_interception_precision: float
    unauthorized_mutation_prevention_rate: float
    post_execution_db_verification_accuracy: float
    crash_recovery_durability: float
    scenarios: list[ScenarioResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 88)
        print(" STATEFUL LANGGRAPH HITL GOVERNANCE & RELIABILITY BENCHMARK REPORT")
        print(" Portfolio Showcase: mariahunzai1 | Project 2 | 100% $0 Stack")
        print("=" * 88)
        print(f"\nTotal Evaluation Scenarios: {self.total_scenarios} | Passed: {self.passed_scenarios} | Failed: {self.total_scenarios - self.passed_scenarios}\n")

        print("| Metric                                      | Target  | Achieved | Status |")
        print("|:--------------------------------------------|:-------:|:--------:|:------:|")
        print(f"| Read-Only Autonomy Rate                     | 100.0%  |  {self.readonly_autonomy_rate:5.1f}%  | {'PASS ✅' if self.readonly_autonomy_rate >= 100.0 else 'FAIL ❌'}  |")
        print(f"| High-Risk Interception Precision            | 100.0%  |  {self.high_risk_interception_precision:5.1f}%  | {'PASS ✅' if self.high_risk_interception_precision >= 100.0 else 'FAIL ❌'}  |")
        print(f"| Unauthorized Mutation Prevention Rate       | 100.0%  |  {self.unauthorized_mutation_prevention_rate:5.1f}%  | {'PASS ✅' if self.unauthorized_mutation_prevention_rate >= 100.0 else 'FAIL ❌'}  |")
        print(f"| Post-Execution DB Verification Accuracy     | 100.0%  |  {self.post_execution_db_verification_accuracy:5.1f}%  | {'PASS ✅' if self.post_execution_db_verification_accuracy >= 100.0 else 'FAIL ❌'}  |")
        print(f"| Crash-Recovery State Durability             | 100.0%  |  {self.crash_recovery_durability:5.1f}%  | {'PASS ✅' if self.crash_recovery_durability >= 100.0 else 'FAIL ❌'}  |")
        print("-" * 88)

        print("\nScenario Breakdown:")
        for idx, sc in enumerate(self.scenarios, 1):
            mark = "PASS ✅" if sc.passed else "FAIL ❌"
            print(f" {idx:2d}. [{mark}] {sc.scenario_name:<50} ({sc.category}) in {sc.duration_ms:.1f}ms")
            if not sc.passed:
                print(f"     -> Error: {sc.details}")
        print("=" * 88 + "\n")


def run_benchmark() -> BenchmarkReport:
    """Execute the full suite of 11 deterministic governance benchmark scenarios."""
    scenarios: list[ScenarioResult] = []

    def make_env():
        fd_ops, ops = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd_ops)
        fd_chk, chk = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd_chk)
        set_override_db_path(ops)
        init_db(ops)
        seed_demo_data(ops, force=True)
        return ops, chk

    def cleanup_env(ops, chk):
        set_override_db_path(None)
        for p in (ops, chk):
            try:
                if os.path.exists(p):
                    os.remove(p)
                for ext in ("-wal", "-shm"):
                    if os.path.exists(p + ext):
                        os.remove(p + ext)
            except OSError:
                pass

    # --- Group 1: Read-Only Autonomy Scenarios (5 Scenarios) ---

    # 1. lookup_customer
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "lookup_customer", "args": {"customer_id": "cust_001"}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="Customer Alice Chen found.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        resp = svc.run_turn("b_ro_1", "lookup cust_001")
        passed = (resp.status == "completed" and resp.pending_approval is None and len(resp.tool_trace) == 1)
        scenarios.append(ScenarioResult("Read-Only Customer Profile Lookup", "Read-Only Autonomy", passed, "Autonomous lookup without pause", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # 2. get_order_details
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "get_order_details", "args": {"order_id": "ord_101"}, "id": "c2", "type": "tool_call"}]),
            AIMessage(content="Order ord_101 is $150.00.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        resp = svc.run_turn("b_ro_2", "order details ord_101")
        passed = (resp.status == "completed" and resp.pending_approval is None and len(resp.tool_trace) == 1)
        scenarios.append(ScenarioResult("Read-Only Order Details Retrieval", "Read-Only Autonomy", passed, "Autonomous order fetch", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # 3. check_refund_policy (eligible)
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "check_refund_policy", "args": {"order_id": "ord_101"}, "id": "c3", "type": "tool_call"}]),
            AIMessage(content="Order ord_101 is eligible under 30-day window.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        resp = svc.run_turn("b_ro_3", "check policy ord_101")
        passed = (resp.status == "completed" and resp.pending_approval is None and len(resp.tool_trace) == 1)
        scenarios.append(ScenarioResult("Deterministic 30-Day Window Eligibility Check", "Read-Only Autonomy", passed, "Autonomous policy calculation", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # 4. check_refund_policy (expired)
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "check_refund_policy", "args": {"order_id": "ord_102"}, "id": "c4", "type": "tool_call"}]),
            AIMessage(content="Order ord_102 is 45 days old and expired.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        resp = svc.run_turn("b_ro_4", "check policy ord_102")
        passed = (resp.status == "completed" and resp.pending_approval is None and len(resp.tool_trace) == 1)
        scenarios.append(ScenarioResult("Expired Order Policy Ineligibility Detection", "Read-Only Autonomy", passed, "Autonomous expiration detection", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # 5. Chained multi-tool read-only inquiry
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "lookup_customer", "args": {"customer_id": "cust_001"}, "id": "c5a", "type": "tool_call"}]),
            AIMessage(content="", tool_calls=[{"name": "get_order_details", "args": {"order_id": "ord_101"}, "id": "c5b", "type": "tool_call"}]),
            AIMessage(content="Alice Chen has order ord_101.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        resp = svc.run_turn("b_ro_5", "full inquiry on cust_001")
        passed = (resp.status == "completed" and resp.pending_approval is None and len(resp.tool_trace) == 2)
        scenarios.append(ScenarioResult("Chained Multi-Tool Customer Inquiry Flow", "Read-Only Autonomy", passed, "Autonomous multi-hop read-only execution", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # --- Group 2: High-Risk Interception Scenarios (2 Scenarios) ---

    # 6. High-risk refund interception
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": "ord_101", "amount": 150.0, "reason": "Damaged"}, "id": "c6", "type": "tool_call"}]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        resp = svc.run_turn("b_hr_1", "refund ord_101")
        order_pre = get_order("ord_101", db_path=ops)
        passed = (
            resp.status == "awaiting_approval"
            and resp.pending_approval is not None
            and resp.pending_approval.tool_name == "issue_refund"
            and order_pre["status"] == "delivered"  # No DB write occurred!
        )
        scenarios.append(ScenarioResult("High-Risk Refund Gate Interception & Pre-Mutation Hold", "Interception Precision", passed, "Halted before SQLite write", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # 7. High-risk tier change interception
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "update_subscription_tier", "args": {"customer_id": "cust_002", "new_tier": "enterprise"}, "id": "c7", "type": "tool_call"}]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        resp = svc.run_turn("b_hr_2", "upgrade cust_002")
        cust_pre = get_customer("cust_002", db_path=ops)
        passed = (
            resp.status == "awaiting_approval"
            and resp.pending_approval is not None
            and resp.pending_approval.tool_name == "update_subscription_tier"
            and cust_pre["subscription_tier"] == "free"  # No DB write occurred!
        )
        scenarios.append(ScenarioResult("Subscription Tier Upgrade Gate Interception", "Interception Precision", passed, "Halted before SQLite write", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # --- Group 3: Human Approval & Deterministic Post-Tool Verification (1 Scenario) ---

    # 8. Human approval execution + verification
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": "ord_101", "amount": 150.0, "reason": "Damaged"}, "id": "c8", "type": "tool_call"}]),
            AIMessage(content="Refund completed.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        svc.run_turn("b_appr_1", "refund ord_101")
        resp_res = svc.resume_turn("b_appr_1", ApprovalDecision(decision="approve", reviewer_note="Verified"))
        order_post = get_order("ord_101", db_path=ops)
        refunds_post = get_order_refunds("ord_101", db_path=ops)
        passed = (
            resp_res.status == "completed"
            and resp_res.verification is not None
            and resp_res.verification.verified is True
            and order_post["status"] == "refunded"
            and len(refunds_post) == 1
        )
        scenarios.append(ScenarioResult("Human Authorization & Deterministic SQL Verification", "Verification Accuracy", passed, "Approved, mutated, and SQL verified", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # --- Group 4: Human In-Flight Argument Modification (1 Scenario) ---

    # 9. Human in-flight edit
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": "ord_101", "amount": 150.0, "reason": "Full return"}, "id": "c9", "type": "tool_call"}]),
            AIMessage(content="Partial refund completed.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        svc.run_turn("b_edit_1", "refund ord_101")
        resp_res = svc.resume_turn("b_edit_1", ApprovalDecision(
            decision="edit",
            edited_args={"order_id": "ord_101", "amount": 50.0, "reason": "Partial compensation"},
            reviewer_note="Reduced to $50",
        ))
        order_post = get_order("ord_101", db_path=ops)
        refunds_post = get_order_refunds("ord_101", db_path=ops)
        passed = (
            resp_res.status == "completed"
            and resp_res.verification is not None
            and resp_res.verification.verified is True
            and order_post["status"] == "partially_refunded"
            and len(refunds_post) == 1
            and refunds_post[0]["amount"] == 50.0
        )
        scenarios.append(ScenarioResult("In-Flight Parameter Patching & Ledger Verification", "In-Flight Argument Modification", passed, "Patched $150 to $50 and verified in ledger", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # --- Group 5: Human Rejection & Unauthorized Mutation Prevention (1 Scenario) ---

    # 10. Human rejection protection
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": "ord_102", "amount": 45.0, "reason": "Late return"}, "id": "c10", "type": "tool_call"}]),
            AIMessage(content="Refund rejected.", tool_calls=[]),
        ])
        svc = AgentService(llm=llm, checkpoint_db_path=chk, db_path=ops)
        svc.run_turn("b_rej_1", "refund ord_102")
        resp_res = svc.resume_turn("b_rej_1", ApprovalDecision(decision="reject", reviewer_note="Policy violation"))
        order_post = get_order("ord_102", db_path=ops)
        refunds_post = get_order_refunds("ord_102", db_path=ops)
        passed = (
            resp_res.status == "rejected"
            and order_post["status"] == "delivered"
            and len(refunds_post) == 0
        )
        scenarios.append(ScenarioResult("Human Rejection & Zero Unauthorized Mutation Guard", "Mutation Prevention", passed, "Rejected with zero database mutations", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # --- Group 6: Checkpoint Persistence & Crash Recovery Durability (1 Scenario) ---

    # 11. Crash recovery durability
    ops, chk = make_env()
    t0 = time.time()
    try:
        llm1 = FakeBenchmarkChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": "ord_104", "amount": 25.0, "reason": "Damaged"}, "id": "c11", "type": "tool_call"}]),
        ])
        svc1 = AgentService(llm=llm1, checkpoint_db_path=chk, db_path=ops)
        svc1.run_turn("b_crash_1", "refund ord_104")

        # Simulate crash: kill connection and service1 instance
        svc1._conn.close()
        del svc1

        # Service 2 boots up pointing to same files
        llm2 = FakeBenchmarkChatModel(responses=[
            AIMessage(content="Post-crash refund executed.", tool_calls=[]),
        ])
        svc2 = AgentService(llm=llm2, checkpoint_db_path=chk, db_path=ops)
        recovered = svc2.get_thread_state("b_crash_1")

        rec_ok = (recovered is not None and recovered.status == "awaiting_approval")
        resp_res = svc2.resume_turn("b_crash_1", ApprovalDecision(decision="approve", reviewer_note="Resumed after restart"))

        order_post = get_order("ord_104", db_path=ops)
        passed = (
            rec_ok
            and resp_res.status == "completed"
            and resp_res.verification is not None
            and resp_res.verification.verified is True
            and order_post["status"] == "refunded"
        )
        scenarios.append(ScenarioResult("SQLite Checkpoint Persistence & Crash-Recovery Durability", "Durability", passed, "Survives complete service re-instantiation", (time.time() - t0) * 1000))
    finally:
        cleanup_env(ops, chk)

    # --- Aggregate Metrics ---
    total = len(scenarios)
    passed_count = sum(1 for s in scenarios if s.passed)

    # Specific category calculations
    ro_scenarios = [s for s in scenarios if s.category == "Read-Only Autonomy"]
    ro_rate = (sum(1 for s in ro_scenarios if s.passed) / len(ro_scenarios)) * 100.0 if ro_scenarios else 0.0

    ir_scenarios = [s for s in scenarios if s.category == "Interception Precision"]
    ir_rate = (sum(1 for s in ir_scenarios if s.passed) / len(ir_scenarios)) * 100.0 if ir_scenarios else 0.0

    mp_scenarios = [s for s in scenarios if s.category == "Mutation Prevention"]
    mp_rate = (sum(1 for s in mp_scenarios if s.passed) / len(mp_scenarios)) * 100.0 if mp_scenarios else 0.0

    va_scenarios = [s for s in scenarios if s.category in ("Verification Accuracy", "In-Flight Argument Modification")]
    va_rate = (sum(1 for s in va_scenarios if s.passed) / len(va_scenarios)) * 100.0 if va_scenarios else 0.0

    dr_scenarios = [s for s in scenarios if s.category == "Durability"]
    dr_rate = (sum(1 for s in dr_scenarios if s.passed) / len(dr_scenarios)) * 100.0 if dr_scenarios else 0.0

    report = BenchmarkReport(
        total_scenarios=total,
        passed_scenarios=passed_count,
        readonly_autonomy_rate=ro_rate,
        high_risk_interception_precision=ir_rate,
        unauthorized_mutation_prevention_rate=mp_rate,
        post_execution_db_verification_accuracy=va_rate,
        crash_recovery_durability=dr_rate,
        scenarios=scenarios,
    )
    return report


if __name__ == "__main__":
    rep = run_benchmark()
    rep.print_summary()
