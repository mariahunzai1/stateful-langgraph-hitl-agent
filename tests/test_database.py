import pytest
import sqlite3
from core.database import (
    get_db_connection,
    init_db,
    seed_demo_data,
    get_customer,
    get_order,
    list_customer_orders,
    get_order_refunds,
    execute_refund,
    execute_subscription_update,
    insert_audit_log,
    get_audit_logs,
)


def test_init_db_creates_tables(temp_db_path):
    """Verify that all required tables and foreign keys are created."""
    with get_db_connection(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert {"customers", "orders", "refund_ledger", "audit_logs"}.issubset(tables)


def test_foreign_key_enforcement(temp_db_path):
    """Verify foreign key constraint rejects orphan orders."""
    with pytest.raises(sqlite3.IntegrityError):
        with get_db_connection(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO orders (id, customer_id, amount, status, created_at)
                VALUES ('ord_invalid', 'non_existent_cust', 100.0, 'delivered', '2026-01-01T00:00:00Z');
            """)
            conn.commit()


def test_seed_demo_data_is_idempotent(temp_db_path):
    """Verify that calling seed_demo_data multiple times does not duplicate records."""
    # First seed is run in conftest; run it again without force
    seed_demo_data(temp_db_path, force=False)
    
    with get_db_connection(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers;")
        assert cursor.fetchone()[0] == 4
        
        cursor.execute("SELECT COUNT(*) FROM orders;")
        assert cursor.fetchone()[0] == 6
        
        cursor.execute("SELECT COUNT(*) FROM refund_ledger;")
        assert cursor.fetchone()[0] == 1


def test_get_customer_and_orders(temp_db_path):
    """Verify retrieval of customer and their related orders."""
    cust = get_customer("cust_001", db_path=temp_db_path)
    assert cust is not None
    assert cust["name"] == "Alice Chen"
    assert cust["subscription_tier"] == "pro"
    
    orders = list_customer_orders("cust_001", db_path=temp_db_path)
    assert len(orders) == 2
    order_ids = {o["id"] for o in orders}
    assert order_ids == {"ord_101", "ord_102"}


def test_execute_refund_partial(temp_db_path):
    """Verify executing a partial refund transitions order status to partially_refunded."""
    result = execute_refund(
        order_id="ord_101",
        amount=50.00,
        reason="Partial item return",
        approved_by="operator_test",
        thread_id="thread_001",
        db_path=temp_db_path
    )
    
    assert result["success"] is True
    assert result["amount_refunded"] == 50.00
    assert result["new_order_status"] == "partially_refunded"
    assert result["remaining_balance"] == 100.00

    # Verify order state in DB
    order = get_order("ord_101", db_path=temp_db_path)
    assert order["status"] == "partially_refunded"

    # Verify refund ledger
    refunds = get_order_refunds("ord_101", db_path=temp_db_path)
    assert len(refunds) == 1
    assert refunds[0]["amount"] == 50.00

    # Verify audit log
    logs = get_audit_logs("thread_001", db_path=temp_db_path)
    assert len(logs) == 1
    assert logs[0]["action"] == "REFUND_ISSUED"
    assert logs[0]["payload"]["amount"] == 50.00


def test_execute_refund_full(temp_db_path):
    """Verify full refund transitions order status to refunded."""
    result = execute_refund(
        order_id="ord_104",
        amount=25.00,
        reason="Complete return",
        approved_by="operator_test",
        thread_id="thread_002",
        db_path=temp_db_path
    )
    assert result["success"] is True
    assert result["new_order_status"] == "refunded"
    assert result["remaining_balance"] == 0.0

    order = get_order("ord_104", db_path=temp_db_path)
    assert order["status"] == "refunded"


def test_prevent_double_refund(temp_db_path):
    """Verify that an already refunded order cannot be refunded again."""
    with pytest.raises(ValueError, match="already fully refunded"):
        execute_refund(
            order_id="ord_103",
            amount=10.00,
            reason="Illegal second refund",
            approved_by="operator_test",
            db_path=temp_db_path
        )


def test_prevent_over_refund(temp_db_path):
    """Verify that refund amounts greater than remaining balance are rejected."""
    with pytest.raises(ValueError, match="exceeds remaining refundable balance"):
        execute_refund(
            order_id="ord_101",
            amount=200.00,  # Order 101 is only $150.00
            reason="Over refund attempt",
            approved_by="operator_test",
            db_path=temp_db_path
        )


def test_execute_subscription_update(temp_db_path):
    """Verify upgrading a customer subscription tier and writing an audit trail."""
    result = execute_subscription_update(
        customer_id="cust_002",
        new_tier="pro",
        approved_by="operator_test",
        thread_id="thread_003",
        db_path=temp_db_path
    )
    assert result["success"] is True
    assert result["old_tier"] == "free"
    assert result["new_tier"] == "pro"

    # Verify in DB
    cust = get_customer("cust_002", db_path=temp_db_path)
    assert cust["subscription_tier"] == "pro"

    # Verify audit log
    logs = get_audit_logs("thread_003", db_path=temp_db_path)
    assert len(logs) == 1
    assert logs[0]["action"] == "SUBSCRIPTION_UPDATED"
    assert logs[0]["payload"]["new_tier"] == "pro"


def test_subscription_update_invalid_tier(temp_db_path):
    """Verify rejection of non-existent subscription tier."""
    with pytest.raises(ValueError, match="Invalid subscription tier"):
        execute_subscription_update(
            customer_id="cust_002",
            new_tier="platinum_unlimited",
            approved_by="operator_test",
            db_path=temp_db_path
        )


def test_audit_log_immutability_and_retrieval(temp_db_path):
    """Verify audit log insertion and chronological retrieval."""
    insert_audit_log(
        thread_id="thread_audit",
        action="TEST_ACTION",
        tool_name="test_tool",
        payload={"key": "value1"},
        status="SUCCESS",
        reviewer_note="Note 1",
        db_path=temp_db_path
    )
    insert_audit_log(
        thread_id="thread_audit",
        action="TEST_ACTION_2",
        tool_name="test_tool_2",
        payload={"key": "value2"},
        status="SUCCESS",
        reviewer_note="Note 2",
        db_path=temp_db_path
    )
    
    logs = get_audit_logs("thread_audit", db_path=temp_db_path)
    assert len(logs) == 2
    assert logs[0]["action"] == "TEST_ACTION"
    assert logs[0]["payload"]["key"] == "value1"
    assert logs[1]["action"] == "TEST_ACTION_2"
    assert logs[1]["payload"]["key"] == "value2"
