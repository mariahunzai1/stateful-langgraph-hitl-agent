from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class ToolRiskLevel(str, Enum):
    """Deterministic risk classification for tools."""
    READ_ONLY = "read_only"
    HIGH_RISK = "high_risk"


class PendingApproval(BaseModel):
    """Payload representing a paused state machine awaiting human approval."""
    interrupt_id: str = Field(description="Unique identifier for the HITL pause point")
    thread_id: str = Field(description="Conversation thread identifier")
    tool_name: str = Field(description="Target tool awaiting authorization")
    tool_args: dict[str, Any] = Field(description="Arguments proposed by the agent")
    policy_summary: str = Field(description="Human-readable operational policy summary")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when the interrupt was generated"
    )


class ApprovalDecision(BaseModel):
    """Payload submitted by the human operator."""
    decision: Literal["approve", "reject", "edit"] = Field(
        description="Operator action: approve as-is, reject operation, or edit arguments"
    )
    edited_args: dict[str, Any] | None = Field(
        default=None,
        description="Modified tool arguments when decision is 'edit'"
    )
    reviewer_note: str = Field(
        default="",
        description="Auditable reviewer explanation or rationale"
    )


class VerificationResult(BaseModel):
    """Deterministic post-tool verification output querying the operational database."""
    verified: bool = Field(description="Whether the DB state matches the expected tool mutation")
    tool_name: str = Field(description="Name of the verified tool")
    db_snapshot: dict[str, Any] = Field(description="Snapshot of the operational DB state after execution")
    discrepancy: str | None = Field(
        default=None,
        description="Explanation of discrepancy if verification failed"
    )


class AgentRunRequest(BaseModel):
    """Inbound request to run an agent turn."""
    thread_id: str = Field(description="Unique thread identifier for conversation state")
    message: str = Field(description="User prompt or customer query")


class AgentResumeRequest(BaseModel):
    """Inbound request to resume an agent paused at a HITL interrupt."""
    thread_id: str = Field(description="Thread identifier corresponding to the paused run")
    approval: ApprovalDecision | None = Field(default=None, description="Operator decision and parameters")
    decision: ApprovalDecision | None = Field(default=None, description="Alternative field for operator decision")

    @model_validator(mode="after")
    def resolve_decision(self) -> "AgentResumeRequest":
        if self.approval is None and self.decision is None:
            raise ValueError("Either 'approval' or 'decision' must be provided.")
        if self.approval is None:
            self.approval = self.decision
        if self.decision is None:
            self.decision = self.approval
        return self


class AgentTurnResponse(BaseModel):
    """Outbound response representing the culmination of an agent turn."""
    thread_id: str = Field(description="Active thread identifier")
    status: Literal["completed", "awaiting_approval", "rejected"] = Field(
        description="Current state of the agent run"
    )
    assistant_message: str = Field(description="Synthesized assistant response or notification")
    tool_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Audit trace of tools executed during this turn"
    )
    pending_approval: PendingApproval | None = Field(
        default=None,
        description="Approval details if the agent is paused at a high-risk gate"
    )
    verification: VerificationResult | None = Field(
        default=None,
        description="Post-tool database verification result if a state mutation occurred"
    )


class AuditLogEntry(BaseModel):
    """Immutable audit record for governance."""
    id: int | None = None
    thread_id: str
    action: str
    tool_name: str
    payload: dict[str, Any]
    status: str
    reviewer_note: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# --- Tool Input/Output Schemas ---

class CustomerLookupInput(BaseModel):
    customer_id: str = Field(description="Customer identifier (e.g. cust_001)")


class OrderDetailsInput(BaseModel):
    order_id: str = Field(description="Order identifier (e.g. ord_101)")


class CheckRefundPolicyInput(BaseModel):
    order_id: str = Field(description="Order identifier to evaluate against refund rules")


class RefundPolicyOutput(BaseModel):
    eligible: bool
    reason: str
    order_id: str
    order_amount: float
    total_refunded: float
    remaining_refundable_amount: float
    days_since_order: int
    policy_max_days: int = 30


class IssueRefundInput(BaseModel):
    order_id: str = Field(description="Order identifier to issue refund for")
    amount: float = Field(gt=0, description="Refund amount in USD (must be positive)")
    reason: str = Field(min_length=3, description="Operational justification for the refund")
    thread_id: str = Field(default="direct", description="Conversation thread identifier for auditing")


class IssueRefundOutput(BaseModel):
    success: bool
    refund_id: str
    order_id: str
    customer_id: str
    amount_refunded: float
    new_order_status: str
    remaining_balance: float
    timestamp: str


class UpdateSubscriptionTierInput(BaseModel):
    customer_id: str = Field(description="Customer identifier")
    new_tier: Literal["free", "pro", "enterprise"] = Field(
        description="Target subscription tier"
    )
    thread_id: str = Field(default="direct", description="Conversation thread identifier for auditing")


class UpdateSubscriptionTierOutput(BaseModel):
    success: bool
    customer_id: str
    old_tier: str
    new_tier: str
    timestamp: str
