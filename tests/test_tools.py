import pytest
from pydantic import ValidationError
from core.schemas import ToolRiskLevel, IssueRefundInput, UpdateSubscriptionTierInput
from core.database import get_order, get_customer
from core.tools import (
    lookup_customer,
    get_order_details,
    check_refund_policy,
    issue_refund,
    update_subscription_tier,
    get_tool_risk,
    ALL_TOOLS,
)


def test_tool_risk_level_metadata():
    """Verify that all operations tools have explicit risk level metadata attached."""
    assert lookup_customer.risk_level == ToolRiskLevel.READ_ONLY
    assert get_order_details.risk_level == ToolRiskLevel.READ_ONLY
    assert check_refund_policy.risk_level == ToolRiskLevel.READ_ONLY
    assert issue_refund.risk_level == ToolRiskLevel.HIGH_RISK
    assert update_subscription_tier.risk_level == ToolRiskLevel.HIGH_RISK


def test_get_tool_risk_helper():
    """Verify get_tool_risk helper returns correct level and raises on unknown tools."""
    assert get_tool_risk("issue_refund") == ToolRiskLevel.HIGH_RISK
    assert get_tool_risk("lookup_customer") == ToolRiskLevel.READ_ONLY
    
    with pytest.raises(KeyError, match="Unknown tool name"):
        get_tool_risk("non_existent_tool")


def test_pydantic_contract_validation():
    """Verify Pydantic input schemas enforce constraints."""
    # Amount must be > 0
    with pytest.raises(ValidationError):
        IssueRefundInput(order_id="ord_101", amount=-5.0, reason="Invalid negative")

    # Reason must have min_length 3
    with pytest.raises(ValidationError):
        IssueRefundInput(order_id="ord_101", amount=10.0, reason="ab")

    # Tier must be valid literal
    with pytest.raises(ValidationError):
        UpdateSubscriptionTierInput(customer_id="cust_001", new_tier="diamond")


def test_lookup_customer_tool(temp_db_path, monkeypatch):
    """Verify lookup_customer tool output structure."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = lookup_customer.invoke({"customer_id": "cust_001"})
    assert "customer" in result
    assert result["customer"]["id"] == "cust_001"
    assert result["customer"]["subscription_tier"] == "pro"
    assert result["total_orders"] == 2


def test_lookup_customer_nonexistent(temp_db_path, monkeypatch):
    """Verify lookup_customer handles non-existent customer gracefully."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = lookup_customer.invoke({"customer_id": "cust_999"})
    assert "error" in result
    assert "not found" in result["error"]


def test_get_order_details_tool(temp_db_path, monkeypatch):
    """Verify get_order_details returns complete order and refund statistics."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = get_order_details.invoke({"order_id": "ord_101"})
    assert "order" in result
    assert result["order"]["amount"] == 150.00
    assert result["total_refunded"] == 0.0
    assert result["remaining_refundable_amount"] == 150.00
    assert result["days_since_order"] <= 11


def test_check_refund_policy_eligible_recent_order(temp_db_path, monkeypatch):
    """Verify check_refund_policy allows refund for order within 30 days."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = check_refund_policy.invoke({"order_id": "ord_101"})
    assert result["eligible"] is True
    assert result["remaining_refundable_amount"] == 150.00
    assert "policy window" in result["reason"]


def test_check_refund_policy_expired_order(temp_db_path, monkeypatch):
    """Verify check_refund_policy denies refund for order older than 30 days."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = check_refund_policy.invoke({"order_id": "ord_102"})  # 45 days old
    assert result["eligible"] is False
    assert "exceeding the strict 30-day" in result["reason"]


def test_check_refund_policy_already_refunded_order(temp_db_path, monkeypatch):
    """Verify check_refund_policy denies refund for already refunded order."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = check_refund_policy.invoke({"order_id": "ord_103"})  # Already refunded
    assert result["eligible"] is False
    assert "already been fully refunded" in result["reason"]


def test_issue_refund_tool_success(temp_db_path, monkeypatch):
    """Verify issue_refund tool executes state mutation and produces schema-valid output."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = issue_refund.invoke({
        "order_id": "ord_104",
        "amount": 25.00,
        "reason": "Customer received damaged packaging",
        "thread_id": "thread_tool_test"
    })
    
    assert result["success"] is True
    assert result["amount_refunded"] == 25.00
    assert result["new_order_status"] == "refunded"
    assert result["remaining_balance"] == 0.0
    
    # Check DB state
    order = get_order("ord_104", db_path=temp_db_path)
    assert order["status"] == "refunded"


def test_issue_refund_tool_double_refund_failure(temp_db_path, monkeypatch):
    """Verify issue_refund catches database errors and returns graceful failure dict."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = issue_refund.invoke({
        "order_id": "ord_103",  # Already refunded
        "amount": 10.00,
        "reason": "Attempting double refund",
        "thread_id": "thread_tool_test"
    })
    
    assert result["success"] is False
    assert "already fully refunded" in result["error"]


def test_update_subscription_tier_tool_success(temp_db_path, monkeypatch):
    """Verify update_subscription_tier tool mutates customer record."""
    monkeypatch.setattr("core.database.get_db_path", lambda path=None: temp_db_path)
    
    result = update_subscription_tier.invoke({
        "customer_id": "cust_004",
        "new_tier": "enterprise",
        "thread_id": "thread_tier_test"
    })
    
    assert result["success"] is True
    assert result["old_tier"] == "free"
    assert result["new_tier"] == "enterprise"
    
    # Check DB state
    cust = get_customer("cust_004", db_path=temp_db_path)
    assert cust["subscription_tier"] == "enterprise"
