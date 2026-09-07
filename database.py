# database.py — SQLite database: connection, seeding, and queries

import json
import sqlite3
import os
from seed_data import CUSTOMERS, TRANSACTIONS

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neobank.db")


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(conn: sqlite3.Connection):
    """Create tables and seed data. Safe to call on every app start."""
    cur = conn.cursor()

    # ── Create tables ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            user_id     TEXT PRIMARY KEY,
            name        TEXT,
            tier        TEXT,
            balance     REAL,
            status      TEXT,
            email       TEXT,
            phone       TEXT,
            dob         TEXT,
            address     TEXT,
            account_opened TEXT,
            kyc_status  TEXT,
            fraud_flags TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT,
            date        TEXT,
            type        TEXT,
            amount      REAL,
            description TEXT,
            recipient   TEXT,
            status      TEXT,
            FOREIGN KEY (user_id) REFERENCES customers(user_id)
        )
    """)

    # ── Seed customers (upsert) ──
    for c in CUSTOMERS:
        cur.execute(
            """INSERT INTO customers
               (user_id, name, tier, balance, status, email, phone, dob,
                address, account_opened, kyc_status, fraud_flags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 name=excluded.name, tier=excluded.tier, balance=excluded.balance,
                 status=excluded.status, email=excluded.email, phone=excluded.phone,
                 dob=excluded.dob, address=excluded.address,
                 account_opened=excluded.account_opened, kyc_status=excluded.kyc_status,
                 fraud_flags=excluded.fraud_flags
            """,
            (
                c["user_id"], c["name"], c["tier"], c["balance"],
                c["status"], c["email"], c["phone"], c["dob"],
                c["address"], c["account_opened"], c["kyc_status"],
                json.dumps(c.get("fraud_flags", [])),
            ),
        )

    # ── Seed transactions (only if table is empty) ──
    count = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    if count == 0:
        for t in TRANSACTIONS:
            cur.execute(
                """INSERT INTO transactions
                   (user_id, date, type, amount, description, recipient, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    t["user_id"], t["date"], t["type"], t["amount"],
                    t.get("description"), t.get("recipient"), t["status"],
                ),
            )

    conn.commit()


def get_customer(conn: sqlite3.Connection, user_id: str) -> dict | None:
    """Retrieve a single customer by user_id."""
    row = conn.execute(
        "SELECT * FROM customers WHERE user_id = ?", (user_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["fraud_flags"] = json.loads(d.get("fraud_flags") or "[]")
    return d


def get_transactions(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    """Retrieve all transactions for a user_id, newest first."""
    rows = conn.execute(
        "SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def authenticate_user(conn: sqlite3.Connection, user_id: str, name: str) -> dict | None:
    """Authenticate a user by matching user_id and name (case-insensitive)."""
    row = conn.execute(
        "SELECT * FROM customers WHERE user_id = ? AND LOWER(name) = LOWER(?)",
        (user_id, name),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["fraud_flags"] = json.loads(d.get("fraud_flags") or "[]")
    return d


def authenticate_by_name(conn: sqlite3.Connection, name: str) -> dict | None:
    """Authenticate a user by name only (case-insensitive)."""
    row = conn.execute(
        "SELECT * FROM customers WHERE LOWER(name) = LOWER(?)",
        (name.strip(),),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["fraud_flags"] = json.loads(d.get("fraud_flags") or "[]")
    return d


def format_account_details(customer: dict, transactions: list[dict]) -> str:
    """Format customer account details and transactions into a readable string."""
    if not customer:
        return "Account not found."

    lines = [
        f"Account Details for {customer['name']}",
        f"  User ID: {customer['user_id']}",
        f"  Tier: {customer['tier']}",
        f"  Balance: ${customer['balance']:,.2f}",
        f"  Status: {customer['status']}",
        f"  Email: {customer['email']}",
        f"  Phone: {customer['phone']}",
        f"  Account Opened: {customer['account_opened']}",
        f"  KYC Status: {customer['kyc_status']}",
    ]

    flags = customer.get("fraud_flags", [])
    if flags:
        lines.append(f"  Fraud Flags: {', '.join(flags)}")

    if transactions:
        lines.append("\nRecent Transactions:")
        for txn in transactions:
            amount_str = f"${txn['amount']:,.2f}" if txn["amount"] else "—"
            desc = txn.get("description") or ""
            recipient = txn.get("recipient")
            detail = f"{recipient} ({desc})" if recipient else desc
            lines.append(
                f"  [{txn['date']}] {txn['type']} | {amount_str} | {detail} | {txn['status']}"
            )

    return "\n".join(lines)
