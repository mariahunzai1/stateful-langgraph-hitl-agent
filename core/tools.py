from datetime import datetime, timezone
from typing import Any, Callable
from langchain_core.tools import StructuredTool, BaseTool
from core.config import get_settings
from core.database import (
    get_customer,
    get_order,
    list_customer_orders,
    get_order_refunds,
    execute_refund as db_execute_refund,
    execute_subscription_update as db_execute_subscription_update,
)
from core.schemas import (
    ToolRiskLevel,
    CustomerLookupInput,
    OrderDetailsInput,
    CheckRefundPolicyInput,
    RefundPolicyOutput,
    IssueRefundInput,
    IssueRefundOutput,
    UpdateSubscriptionTierInput,
    UpdateSubscriptionTierOutput,
)


class RiskTieredTool(StructuredTool):
    """Subclass of StructuredTool with programmatic risk tiering."""
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY


def risk_tool(risk_level: ToolRiskLevel, args_schema: type | None = None):
    """Decorator to define a risk-tiered tool with programmatic metadata."""
    def decorator(fn: Callable[..., Any]) -> RiskTieredTool:
        return RiskTieredTool.from_function(
            func=fn,
            name=fn.__name__,
            description=fn.__doc__ or "",
            args_schema=args_schema,
            risk_level=risk_level,
            metadata={"risk_level": risk_level.value}
        )
    return decorator


# --- Read-Only Tools ---

@risk_tool(risk_level=ToolRiskLevel.READ_ONLY, args_schema=CustomerLookupInput)
def lookup_customer(customer_id: str) -> dict[str, Any]:
    """Retrieve customer profile and past order history by customer ID."""
    customer = get_customer(customer_id)
    if not customer:
        return {"error": f"Customer '{customer_id}' not found."}
    orders = list_customer_orders(customer_id)
    return {
        "customer": customer,
        "orders": orders,
        "total_orders": len(orders)
    }


@risk_tool(risk_level=ToolRiskLevel.READ_ONLY, args_schema=OrderDetailsInput)
def get_order_details(order_id: str) -> dict[str, Any]:
    """Retrieve order details including item amount, status, date, and historical refund records."""
    order = get_order(order_id)
    if not order:
        return {"error": f"Order '{order_id}' not found."}
    
    refunds = get_order_refunds(order_id)
    total_refunded = round(sum(r["amount"] for r in refunds), 2)
    remaining_balance = round(max(0.0, order["amount"] - total_refunded), 2)
    
    order_time = datetime.fromisoformat(order["created_at"])
    now = datetime.now(timezone.utc)
    days_elapsed = (now - order_time).days

    return {
        "order": order,
        "refund_history": refunds,
        "total_refunded": total_refunded,
        "remaining_refundable_amount": remaining_balance,
        "days_since_order": days_elapsed
    }


@risk_tool(risk_level=ToolRiskLevel.READ_ONLY, args_schema=CheckRefundPolicyInput)
def check_refund_policy(order_id: str) -> dict[str, Any]:
    """Evaluate whether an order is eligible for a refund according to operational policy (30-day window and balance check)."""
    settings = get_settings()
    order = get_order(order_id)
    if not order:
        return {"eligible": False, "reason": f"Order '{order_id}' does not exist.", "order_id": order_id}

    refunds = get_order_refunds(order_id)
    total_refunded = round(sum(r["amount"] for r in refunds), 2)
    remaining_balance = round(max(0.0, order["amount"] - total_refunded), 2)

    order_time = datetime.fromisoformat(order["created_at"])
    now = datetime.now(timezone.utc)
    days_elapsed = (now - order_time).days

    if order["status"] == "refunded" or remaining_balance <= 0.001:
        output = RefundPolicyOutput(
            eligible=False,
            reason=f"Order '{order_id}' has already been fully refunded ($0.00 remaining balance).",
            order_id=order_id,
            order_amount=order["amount"],
            total_refunded=total_refunded,
            remaining_refundable_amount=0.0,
            days_since_order=days_elapsed,
            policy_max_days=settings.refund_window_days
        )
        return output.model_dump()

    if days_elapsed > settings.refund_window_days:
        output = RefundPolicyOutput(
            eligible=False,
            reason=f"Order '{order_id}' was placed {days_elapsed} days ago, exceeding the strict {settings.refund_window_days}-day return policy window.",
            order_id=order_id,
            order_amount=order["amount"],
            total_refunded=total_refunded,
            remaining_refundable_amount=remaining_balance,
            days_since_order=days_elapsed,
            policy_max_days=settings.refund_window_days
        )
        return output.model_dump()

    output = RefundPolicyOutput(
        eligible=True,
        reason=f"Order '{order_id}' is within the {settings.refund_window_days}-day policy window ({days_elapsed} days old) and has ${remaining_balance:.2f} refundable balance.",
        order_id=order_id,
        order_amount=order["amount"],
        total_refunded=total_refunded,
        remaining_refundable_amount=remaining_balance,
        days_since_order=days_elapsed,
        policy_max_days=settings.refund_window_days
    )
    return output.model_dump()


# --- High-Risk Mutating Tools ---

@risk_tool(risk_level=ToolRiskLevel.HIGH_RISK, args_schema=IssueRefundInput)
def issue_refund(order_id: str, amount: float, reason: str, thread_id: str = "direct") -> dict[str, Any]:
    """Execute an operational refund for an order. HIGH-RISK state-mutating tool requiring policy compliance."""
    try:
        result = db_execute_refund(
            order_id=order_id,
            amount=amount,
            reason=reason,
            approved_by="agent_execution",
            thread_id=thread_id
        )
        output = IssueRefundOutput(**result)
        return output.model_dump()
    except Exception as exc:
        return {"success": False, "error": str(exc), "order_id": order_id}


@risk_tool(risk_level=ToolRiskLevel.HIGH_RISK, args_schema=UpdateSubscriptionTierInput)
def update_subscription_tier(customer_id: str, new_tier: str, thread_id: str = "direct") -> dict[str, Any]:
    """Upgrade or change a customer's subscription tier. HIGH-RISK state-mutating tool."""
    try:
        result = db_execute_subscription_update(
            customer_id=customer_id,
            new_tier=new_tier,
            approved_by="agent_execution",
            thread_id=thread_id
        )
        output = UpdateSubscriptionTierOutput(**result)
        return output.model_dump()
    except Exception as exc:
        return {"success": False, "error": str(exc), "customer_id": customer_id}


# --- Tool Registry & Risk Inspection Helpers ---

ALL_TOOLS: list[RiskTieredTool] = [
    lookup_customer,
    get_order_details,
    check_refund_policy,
    issue_refund,
    update_subscription_tier,
]

TOOL_MAP: dict[str, RiskTieredTool] = {t.name: t for t in ALL_TOOLS}


def get_all_tools() -> list[RiskTieredTool]:
    """Return all available operations tools."""
    return ALL_TOOLS


def get_tool_risk(tool_name: str) -> ToolRiskLevel:
    """Deterministically determine the risk tier of a tool by name."""
    tool_obj = TOOL_MAP.get(tool_name)
    if not tool_obj:
        raise KeyError(f"Unknown tool name: '{tool_name}'")
    return getattr(tool_obj, "risk_level", ToolRiskLevel.HIGH_RISK)
