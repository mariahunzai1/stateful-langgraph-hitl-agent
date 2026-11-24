import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from core.config import Settings, get_settings
from core.database import (
    get_customer,
    get_order,
    get_order_refunds,
    insert_audit_log,
    get_audit_logs,
)
from core.llm import build_chat_model
from core.schemas import (
    AgentTurnResponse,
    ApprovalDecision,
    PendingApproval,
    ToolRiskLevel,
    VerificationResult,
)
from core.tools import ALL_TOOLS, TOOL_MAP, get_tool_risk

OPERATIONS_SYSTEM_PROMPT = (
    "You are the production-grade Customer & Refund Operations Agent for Maria Ali's e-commerce platform. "
    "You handle customer inquiries, order status lookups, policy verification, refunds, and subscription changes. "
    "Always remain polite, helpful, grounded, and concise. "
    "Strict operational business policy mandates that refunds may only be issued within 30 days of order placement "
    "and cannot exceed the order's remaining balance. High-risk operations (refunds, tier changes) will be halted "
    "for human supervisor governance approval before being committed."
)

MAX_TOOL_HOPS = 6


class AgentState(TypedDict):
    """LangGraph state schema for operations agent."""
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    pending_approval: dict[str, Any] | None
    verification: dict[str, Any] | None
    tool_trace: list[dict[str, Any]]
    last_status: Literal["completed", "awaiting_approval", "rejected"]
    tool_hop_count: int


def build_graph(
    llm: BaseChatModel | None = None,
    checkpointer: SqliteSaver | None = None,
    settings: Settings | None = None,
    db_path: str | None = None,
) -> CompiledStateGraph:
    """Build and compile the stateful LangGraph operations workflow."""
    cfg = settings or get_settings()
    active_llm = llm or build_chat_model(cfg)
    model_with_tools = active_llm.bind_tools(ALL_TOOLS)
    ops_db_path = db_path or str(cfg.business_db_abs_path)

    # 1. Agent Node
    def agent_node(state: AgentState) -> dict[str, Any]:
        hop_count = state.get("tool_hop_count", 0)
        if hop_count >= MAX_TOOL_HOPS:
            safety_msg = AIMessage(
                content="Operational guardrail reached: maximum tool execution hops (6) exceeded for this turn."
            )
            return {
                "messages": [safety_msg],
                "last_status": "completed",
            }

        convo_messages = state.get("messages", [])
        prompt_messages = [SystemMessage(content=OPERATIONS_SYSTEM_PROMPT)] + list(convo_messages)
        response = model_with_tools.invoke(prompt_messages)

        new_hop_count = hop_count + 1 if response.tool_calls else hop_count
        return {
            "messages": [response],
            "tool_hop_count": new_hop_count,
        }

    # Conditional Routing After Agent
    def route_after_agent(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            risk = get_tool_risk(tool_name)
            if risk == ToolRiskLevel.HIGH_RISK:
                # Check auto refund limit exemption
                if tool_name == "issue_refund" and cfg.max_auto_refund_limit > 0:
                    amount = float(tc["args"].get("amount", float("inf")))
                    if amount <= cfg.max_auto_refund_limit:
                        continue
                return "human_approval_gate"

        return "execute_tools"

    # 2. Human Approval Gate Node (Side-effect free before interrupt)
    def human_approval_gate_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_message = messages[-1]
        tool_calls = last_message.tool_calls if isinstance(last_message, AIMessage) else []
        target_tool = tool_calls[0] if tool_calls else {"name": "unknown", "args": {}, "id": "unknown"}

        # Build policy summary (pure query, zero side-effects before interrupt)
        tool_name = target_tool["name"]
        tool_args = target_tool["args"]
        policy_summary = "Operation requires human supervisor authorization."

        if tool_name == "issue_refund":
            order_id = tool_args.get("order_id", "")
            order = get_order(order_id, db_path=ops_db_path)
            if order:
                refunds = get_order_refunds(order_id, db_path=ops_db_path)
                already_refunded = sum(r["amount"] for r in refunds)
                remaining = max(0.0, order["amount"] - already_refunded)
                order_time = datetime.fromisoformat(order["created_at"])
                days_since = (datetime.now(timezone.utc) - order_time).days
                eligibility = "Eligible" if (days_since <= cfg.refund_window_days and remaining > 0) else "Policy Exception / Ineligible"
                policy_summary = (
                    f"Refund Request for Order '{order_id}': Proposed amount ${tool_args.get('amount')}. "
                    f"Order total: ${order['amount']} (Status: {order['status']}). Already refunded: ${already_refunded:.2f}. "
                    f"Remaining balance: ${remaining:.2f}. Placed {days_since} days ago. Policy status: {eligibility}."
                )
            else:
                policy_summary = f"Refund Request for unknown Order '{order_id}'."
        elif tool_name == "update_subscription_tier":
            cust_id = tool_args.get("customer_id", "")
            cust = get_customer(cust_id, db_path=ops_db_path)
            current_tier = cust["subscription_tier"] if cust else "unknown"
            policy_summary = (
                f"Subscription Tier Update for Customer '{cust_id}': Proposed new tier: '{tool_args.get('new_tier')}'. "
                f"Current tier: '{current_tier}'."
            )

        pending_payload = PendingApproval(
            interrupt_id=f"int_{uuid.uuid4().hex[:8]}",
            thread_id=state["thread_id"],
            tool_name=tool_name,
            tool_args=tool_args,
            policy_summary=policy_summary,
            created_at=datetime.now(timezone.utc).isoformat(),
        ).model_dump()

        # Interrupt state machine (returns only when resumed via Command(resume=...))
        decision_raw = interrupt(pending_payload)

        # Runtime Safeguard 2: Side effects executed ONLY AFTER interrupt returns
        if isinstance(decision_raw, dict):
            decision = ApprovalDecision.model_validate(decision_raw)
        elif isinstance(decision_raw, ApprovalDecision):
            decision = decision_raw
        else:
            decision = ApprovalDecision(decision="approve")

        if decision.decision == "reject":
            # Write rejection audit log
            insert_audit_log(
                thread_id=state["thread_id"],
                action="APPROVAL_REJECTED",
                tool_name=tool_name,
                payload=tool_args,
                status="REJECTED",
                reviewer_note=decision.reviewer_note or "Rejected by operator",
                db_path=ops_db_path,
            )
            # Create synthetic ToolMessages for rejection
            rejection_messages = [
                ToolMessage(
                    content=f"Operation rejected by human operator. Rationale: {decision.reviewer_note or 'No rationale provided.'}",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in tool_calls
            ]
            return {
                "messages": rejection_messages,
                "pending_approval": None,
                "last_status": "rejected",
            }

        if decision.decision == "edit":
            # Write edit audit log
            insert_audit_log(
                thread_id=state["thread_id"],
                action="APPROVAL_EDITED",
                tool_name=tool_name,
                payload={"original_args": tool_args, "edited_args": decision.edited_args},
                status="EDITED",
                reviewer_note=decision.reviewer_note or "Edited by operator",
                db_path=ops_db_path,
            )
            # Runtime Safeguard 3: Patch AIMessage preserving exact original message id and tool_call_id
            patched_tool_calls = []
            for tc in tool_calls:
                if tc["id"] == target_tool["id"]:
                    patched_tool_calls.append({
                        "name": tc["name"],
                        "args": decision.edited_args if decision.edited_args is not None else tc["args"],
                        "id": tc["id"],
                        "type": tc.get("type", "tool_call"),
                    })
                else:
                    patched_tool_calls.append(tc)

            updated_ai_msg = AIMessage(
                content=last_message.content,
                id=last_message.id,
                tool_calls=patched_tool_calls,
                additional_kwargs=dict(last_message.additional_kwargs or {}),
            )
            return {
                "messages": [updated_ai_msg],
                "pending_approval": None,
                "last_status": "completed",
            }

        # Approve
        insert_audit_log(
            thread_id=state["thread_id"],
            action="APPROVAL_GRANTED",
            tool_name=tool_name,
            payload=tool_args,
            status="APPROVED",
            reviewer_note=decision.reviewer_note or "Approved by operator",
            db_path=ops_db_path,
        )
        return {
            "pending_approval": None,
            "last_status": "completed",
        }

    # Conditional Routing After Approval
    def route_after_approval(state: AgentState) -> str:
        if state.get("last_status") == "rejected":
            return "agent"
        return "execute_tools"

    # 3. Tool Execution Node
    def execute_tools_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_message = messages[-1]
        tool_calls = last_message.tool_calls if isinstance(last_message, AIMessage) else []

        new_tool_messages = []
        new_trace_entries = []

        for tc in tool_calls:
            tname = tc["name"]
            targs = tc["args"]
            tool_obj = TOOL_MAP.get(tname)

            if not tool_obj:
                res = {"error": f"Tool '{tname}' not recognized."}
            else:
                try:
                    # Provide thread_id if accepted
                    call_kwargs = dict(targs)
                    if "thread_id" in getattr(tool_obj.args_schema, "model_fields", {}):
                        call_kwargs["thread_id"] = state["thread_id"]
                    res = tool_obj.invoke(call_kwargs)
                except Exception as exc:
                    res = {"error": str(exc)}

            new_tool_messages.append(
                ToolMessage(
                    content=json.dumps(res),
                    tool_call_id=tc["id"],
                    name=tname,
                )
            )
            new_trace_entries.append({
                "tool": tname,
                "args": targs,
                "result": res,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        existing_trace = state.get("tool_trace", []) or []
        return {
            "messages": new_tool_messages,
            "tool_trace": existing_trace + new_trace_entries,
        }

    # Conditional Routing After Tool Execution
    def route_after_tools(state: AgentState) -> str:
        messages = state.get("messages", [])
        # Inspect recently added tool messages to see if any high risk tool was run
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if get_tool_risk(msg.name) == ToolRiskLevel.HIGH_RISK:
                    return "verify_execution"
            elif isinstance(msg, AIMessage):
                break
        return "agent"

    # 4. Deterministic Post-Tool Verification Node (SQL inspection, 0 LLM)
    def verify_execution_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        executed_high_risk = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if get_tool_risk(msg.name) == ToolRiskLevel.HIGH_RISK:
                    try:
                        content_data = json.loads(msg.content)
                    except Exception:
                        content_data = {}
                    executed_high_risk.append((msg.name, content_data))
            elif isinstance(msg, AIMessage):
                break

        verification_result = None

        for tool_name, result_data in executed_high_risk:
            if tool_name == "issue_refund":
                order_id = result_data.get("order_id")
                order = get_order(order_id, db_path=ops_db_path)
                refunds = get_order_refunds(order_id, db_path=ops_db_path)
                verified = (
                    order is not None
                    and order["status"] in ("refunded", "partially_refunded")
                    and len(refunds) > 0
                )
                verification_result = VerificationResult(
                    verified=verified,
                    tool_name="issue_refund",
                    db_snapshot={"order": order, "refunds_count": len(refunds)},
                    discrepancy=None if verified else f"Verification failed: order {order_id} state mismatch.",
                )
                break
            elif tool_name == "update_subscription_tier":
                customer_id = result_data.get("customer_id")
                target_tier = result_data.get("new_tier")
                cust = get_customer(customer_id, db_path=ops_db_path)
                verified = cust is not None and cust["subscription_tier"] == target_tier
                verification_result = VerificationResult(
                    verified=verified,
                    tool_name="update_subscription_tier",
                    db_snapshot={"customer": cust},
                    discrepancy=None if verified else f"Verification failed: customer {customer_id} tier is not {target_tier}.",
                )
                break

        return {
            "verification": verification_result.model_dump() if verification_result else None,
        }

    # Graph Assembly
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("human_approval_gate", human_approval_gate_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("verify_execution", verify_execution_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "end": END,
            "human_approval_gate": "human_approval_gate",
            "execute_tools": "execute_tools",
        },
    )
    builder.add_conditional_edges(
        "human_approval_gate",
        route_after_approval,
        {
            "agent": "agent",
            "execute_tools": "execute_tools",
        },
    )
    builder.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "verify_execution": "verify_execution",
            "agent": "agent",
        },
    )
    builder.add_edge("verify_execution", "agent")

    return builder.compile(checkpointer=checkpointer)


class AgentService:
    """Production service wrapping the compiled LangGraph engine with SQLite persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        checkpoint_db_path: str | None = None,
        db_path: str | None = None,
        conn: sqlite3.Connection | None = None,
    ):
        self.settings = settings or get_settings()
        self.ops_db_path = db_path or str(self.settings.business_db_abs_path)
        
        # Initialize SQLite checkpointer with multi-thread safety
        if conn is not None:
            self._conn = conn
        else:
            chk_path = checkpoint_db_path or str(self.settings.checkpoint_db_abs_path)
            Path(chk_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(chk_path, check_same_thread=False)

        self.checkpointer = SqliteSaver(self._conn)
        # Runtime Safeguard 1: Call checkpointer.setup() upfront
        self.checkpointer.setup()

        self.graph = build_graph(
            llm=llm,
            checkpointer=self.checkpointer,
            settings=self.settings,
            db_path=self.ops_db_path,
        )

    def run_turn(self, thread_id: str, user_message: str) -> AgentTurnResponse:
        """Run an agent turn with the provided user message."""
        config = {"configurable": {"thread_id": thread_id}}
        input_state: dict[str, Any] = {
            "messages": [HumanMessage(content=user_message)],
            "thread_id": thread_id,
            "tool_hop_count": 0,
        }

        self.graph.invoke(input_state, config=config)
        return self._build_turn_response(thread_id)

    def resume_turn(self, thread_id: str, decision: ApprovalDecision) -> AgentTurnResponse:
        """Resume an agent execution paused at a human approval gate."""
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke(Command(resume=decision), config=config)
        return self._build_turn_response(thread_id)

    def get_thread_state(self, thread_id: str) -> AgentTurnResponse | None:
        """Reconstruct current state directly from SQLite checkpoint, surviving server restarts."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot or not snapshot.values:
            return None
        return self._build_turn_response(thread_id)

    def _build_turn_response(self, thread_id: str) -> AgentTurnResponse:
        """Helper to construct AgentTurnResponse from the checkpointer state."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        values = snapshot.values if snapshot else {}

        # 1. Check for active interrupts (awaiting approval)
        pending_approval = None
        status: Literal["completed", "awaiting_approval", "rejected"] = "completed"

        if snapshot and snapshot.tasks:
            for task in snapshot.tasks:
                if task.interrupts:
                    raw_interrupt = task.interrupts[0].value
                    if isinstance(raw_interrupt, dict):
                        pending_approval = PendingApproval.model_validate(raw_interrupt)
                    status = "awaiting_approval"
                    break

        if status != "awaiting_approval":
            last_status = values.get("last_status", "completed")
            status = "rejected" if last_status == "rejected" else "completed"

        # 2. Extract assistant response
        messages = values.get("messages", [])
        assistant_message = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                assistant_message = str(msg.content)
                break

        if not assistant_message:
            if status == "awaiting_approval":
                assistant_message = (
                    f"A high-risk action ({pending_approval.tool_name if pending_approval else 'operation'}) "
                    f"requires human supervisor authorization before proceeding."
                )
            elif status == "rejected":
                assistant_message = "The requested high-risk operation was rejected by the operator."
            else:
                assistant_message = "Turn completed."

        # 3. Extract verification
        raw_verification = values.get("verification")
        verification = VerificationResult.model_validate(raw_verification) if raw_verification else None

        return AgentTurnResponse(
            thread_id=thread_id,
            status=status,
            assistant_message=assistant_message,
            tool_trace=values.get("tool_trace", []) or [],
            pending_approval=pending_approval,
            verification=verification,
        )
