import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from core.config import get_settings


_OVERRIDE_DB_PATH: str | None = None


def set_override_db_path(path: str | None) -> None:
    """Set global override for database path (useful for testing and evaluation harnesses)."""
    global _OVERRIDE_DB_PATH
    _OVERRIDE_DB_PATH = path


def get_db_path(db_path: str | None = None) -> str:
    """Resolve database path, falling back to configured settings."""
    if db_path is not None:
        return db_path
    if _OVERRIDE_DB_PATH is not None:
        return _OVERRIDE_DB_PATH
    settings = get_settings()
    return str(settings.business_db_abs_path)


def get_db_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Create a SQLite connection configured with WAL mode and foreign key enforcement."""
    path = get_db_path(db_path)
    
    # Ensure directory exists if not in-memory
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: str | None = None) -> None:
    """Initialize operational database schema."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Customers Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                subscription_tier TEXT NOT NULL CHECK(subscription_tier IN ('free', 'pro', 'enterprise')),
                created_at TEXT NOT NULL
            );
        """)
        
        # Orders Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount >= 0),
                status TEXT NOT NULL CHECK(status IN ('delivered', 'shipped', 'refunded', 'partially_refunded')),
                created_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE RESTRICT
            );
        """)
        
        # Refund Ledger Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS refund_ledger (
                id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                reason TEXT NOT NULL,
                approved_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE RESTRICT,
                FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE RESTRICT
            );
        """)
        
        # Audit Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                action TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                reviewer_note TEXT,
                created_at TEXT NOT NULL
            );
        """)
        
        conn.commit()


def seed_demo_data(db_path: str | None = None, force: bool = False) -> None:
    """Idempotently seed the operational database with realistic demo accounts and orders."""
    init_db(db_path)
    now = datetime.now(timezone.utc)
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        if not force:
            cursor.execute("SELECT COUNT(*) FROM customers;")
            if cursor.fetchone()[0] > 0:
                return  # Data already seeded
        else:
            cursor.execute("DELETE FROM refund_ledger;")
            cursor.execute("DELETE FROM orders;")
            cursor.execute("DELETE FROM customers;")
            cursor.execute("DELETE FROM audit_logs;")

        # 4 Customer accounts
        customers = [
            ("cust_001", "Alice Chen", "alice@example.com", "pro", (now - timedelta(days=60)).isoformat()),
            ("cust_002", "Bob Smith", "bob@example.com", "free", (now - timedelta(days=90)).isoformat()),
            ("cust_003", "Charlie Davis", "charlie@enterprise.com", "enterprise", (now - timedelta(days=120)).isoformat()),
            ("cust_004", "Diana Prince", "diana@example.com", "free", (now - timedelta(days=40)).isoformat()),
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO customers (id, name, email, subscription_tier, created_at)
            VALUES (?, ?, ?, ?, ?);
        """, customers)

        # 6 Orders representing operational edge cases:
        # ord_101: Recent within 10 days, delivered, eligible for refund
        # ord_102: 45 days ago, delivered, expired past 30-day window
        # ord_103: 20 days ago, already fully refunded
        # ord_104: 5 days ago, delivered, eligible for refund
        # ord_105: 12 days ago, enterprise account, eligible for refund
        # ord_106: 60 days ago, delivered, expired past 30-day window
        orders = [
            ("ord_101", "cust_001", 150.00, "delivered", (now - timedelta(days=10)).isoformat()),
            ("ord_102", "cust_001", 45.00, "delivered", (now - timedelta(days=45)).isoformat()),
            ("ord_103", "cust_002", 89.99, "refunded", (now - timedelta(days=20)).isoformat()),
            ("ord_104", "cust_002", 25.00, "delivered", (now - timedelta(days=5)).isoformat()),
            ("ord_105", "cust_003", 1200.00, "delivered", (now - timedelta(days=12)).isoformat()),
            ("ord_106", "cust_004", 60.00, "delivered", (now - timedelta(days=60)).isoformat()),
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO orders (id, customer_id, amount, status, created_at)
            VALUES (?, ?, ?, ?, ?);
        """, orders)

        # Pre-seed existing refund for ord_103
        refunds = [
            ("ref_001", "ord_103", "cust_002", 89.99, "Product damaged upon arrival", "system_policy", (now - timedelta(days=18)).isoformat()),
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO refund_ledger (id, order_id, customer_id, amount, reason, approved_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, refunds)

        # Pre-seed audit log entry
        cursor.execute("""
            INSERT INTO audit_logs (thread_id, action, tool_name, payload, status, reviewer_note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            "system_init",
            "DEMO_SEED",
            "seed_demo_data",
            json.dumps({"customers_count": len(customers), "orders_count": len(orders)}),
            "SUCCESS",
            "Automated demo data initialization",
            now.isoformat()
        ))

        conn.commit()


# --- Query Helpers ---

def get_customer(customer_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    """Retrieve customer record by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE id = ?;", (customer_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_order(order_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    """Retrieve order record by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE id = ?;", (order_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_customer_orders(customer_id: str, db_path: str | None = None) -> list[dict[str, Any]]:
    """List all orders associated with a customer."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE customer_id = ? ORDER BY created_at DESC;", (customer_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_order_refunds(order_id: str, db_path: str | None = None) -> list[dict[str, Any]]:
    """List all refunds recorded for a given order."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM refund_ledger WHERE order_id = ? ORDER BY created_at ASC;", (order_id,))
        return [dict(row) for row in cursor.fetchall()]


# --- Mutation Operations with Transactional Integrity ---

def execute_refund(
    order_id: str,
    amount: float,
    reason: str,
    approved_by: str,
    thread_id: str = "direct",
    db_path: str | None = None
) -> dict[str, Any]:
    """
    Atomically issue a refund, record in refund_ledger, update orders status, and write to audit_logs.
    Enforces double-refund and over-refund guards.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    refund_id = f"ref_{uuid.uuid4().hex[:8]}"

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Fetch order
        cursor.execute("SELECT * FROM orders WHERE id = ?;", (order_id,))
        order_row = cursor.fetchone()
        if not order_row:
            raise ValueError(f"Order '{order_id}' does not exist.")
        order = dict(order_row)

        # 2. Fetch existing refunds
        cursor.execute("SELECT COALESCE(SUM(amount), 0.0) FROM refund_ledger WHERE order_id = ?;", (order_id,))
        already_refunded = float(cursor.fetchone()[0])
        remaining_balance = round(order["amount"] - already_refunded, 2)

        # 3. Double-refund guard
        if order["status"] == "refunded" or remaining_balance <= 0.001:
            raise ValueError(f"Order '{order_id}' is already fully refunded. Remaining refundable balance is $0.00.")

        # 4. Over-refund guard
        if amount > remaining_balance + 1e-4:
            raise ValueError(
                f"Requested refund amount (${amount:.2f}) exceeds remaining refundable balance (${remaining_balance:.2f}) for order '{order_id}'."
            )

        new_total_refunded = already_refunded + amount
        new_remaining = round(order["amount"] - new_total_refunded, 2)
        new_status = "refunded" if new_remaining <= 0.001 else "partially_refunded"

        # 5. Insert refund ledger entry
        cursor.execute("""
            INSERT INTO refund_ledger (id, order_id, customer_id, amount, reason, approved_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (refund_id, order_id, order["customer_id"], round(amount, 2), reason, approved_by, now_str))

        # 6. Update order status
        cursor.execute("""
            UPDATE orders
            SET status = ?
            WHERE id = ?;
        """, (new_status, order_id))

        # 7. Write to audit_logs
        audit_payload = {
            "refund_id": refund_id,
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "amount": amount,
            "previous_refunds": already_refunded,
            "new_status": new_status,
            "reason": reason,
            "approved_by": approved_by
        }
        cursor.execute("""
            INSERT INTO audit_logs (thread_id, action, tool_name, payload, status, reviewer_note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            thread_id,
            "REFUND_ISSUED",
            "issue_refund",
            json.dumps(audit_payload),
            "SUCCESS",
            f"Approved by {approved_by}",
            now_str
        ))

        conn.commit()

        return {
            "success": True,
            "refund_id": refund_id,
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "amount_refunded": amount,
            "new_order_status": new_status,
            "remaining_balance": max(0.0, new_remaining),
            "timestamp": now_str
        }


def execute_subscription_update(
    customer_id: str,
    new_tier: str,
    approved_by: str,
    thread_id: str = "direct",
    db_path: str | None = None
) -> dict[str, Any]:
    """
    Atomically update customer subscription tier and record to audit_logs.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    if new_tier not in ("free", "pro", "enterprise"):
        raise ValueError(f"Invalid subscription tier '{new_tier}'. Allowed tiers: free, pro, enterprise.")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM customers WHERE id = ?;", (customer_id,))
        cust_row = cursor.fetchone()
        if not cust_row:
            raise ValueError(f"Customer '{customer_id}' does not exist.")
        old_tier = cust_row["subscription_tier"]

        cursor.execute("""
            UPDATE customers
            SET subscription_tier = ?
            WHERE id = ?;
        """, (new_tier, customer_id))

        audit_payload = {
            "customer_id": customer_id,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "approved_by": approved_by
        }
        cursor.execute("""
            INSERT INTO audit_logs (thread_id, action, tool_name, payload, status, reviewer_note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            thread_id,
            "SUBSCRIPTION_UPDATED",
            "update_subscription_tier",
            json.dumps(audit_payload),
            "SUCCESS",
            f"Tier change approved by {approved_by}",
            now_str
        ))

        conn.commit()

        return {
            "success": True,
            "customer_id": customer_id,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "timestamp": now_str
        }


def insert_audit_log(
    thread_id: str,
    action: str,
    tool_name: str,
    payload: dict[str, Any],
    status: str,
    reviewer_note: str | None = None,
    db_path: str | None = None
) -> int:
    """Insert an auditable governance record."""
    now_str = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (thread_id, action, tool_name, payload, status, reviewer_note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            thread_id,
            action,
            tool_name,
            json.dumps(payload),
            status,
            reviewer_note,
            now_str
        ))
        conn.commit()
        return cursor.lastrowid


def get_audit_logs(thread_id: str | None = None, db_path: str | None = None) -> list[dict[str, Any]]:
    """Query audit logs, optionally filtered by thread_id."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if thread_id:
            cursor.execute("SELECT * FROM audit_logs WHERE thread_id = ? ORDER BY id ASC;", (thread_id,))
        else:
            cursor.execute("SELECT * FROM audit_logs ORDER BY id ASC;")
        
        results = []
        for row in cursor.fetchall():
            entry = dict(row)
            try:
                entry["payload"] = json.loads(entry["payload"])
            except Exception:
                pass
            results.append(entry)
        return results
