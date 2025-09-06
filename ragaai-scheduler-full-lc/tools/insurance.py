
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import sqlite3, uuid
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "store.db"

@dataclass
class Insurance:
    insurance_id: str
    patient_id: str
    carrier: str
    member_id: str
    group_number: Optional[str]
    effective_from: Optional[str]
    status: str
    created_at: str

def _ensure_tables():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS insurances (
            insurance_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            carrier TEXT NOT NULL,
            member_id TEXT NOT NULL,
            group_number TEXT,
            effective_from TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS insurance_audit_log (
            log_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            old_insurance_id TEXT,
            new_insurance_id TEXT,
            changed_at TEXT NOT NULL,
            changed_by TEXT NOT NULL
        )
    """)
    conn.commit(); conn.close()

def get_active_insurance(patient_id: str) -> Optional[Insurance]:
    _ensure_tables()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        SELECT insurance_id, patient_id, carrier, member_id, group_number, effective_from, status, created_at
        FROM insurances WHERE patient_id=? AND status='ACTIVE' ORDER BY datetime(created_at) DESC LIMIT 1
    """, (patient_id,))
    row = cur.fetchone(); conn.close()
    if not row: return None
    return Insurance(*row)

def create_initial_insurance(patient_id: str, carrier: str, member_id: str, group_number: str | None, effective_from: str | None) -> dict:
    _ensure_tables()
    iid = f"I{uuid.uuid4().hex[:10]}"; now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        cur.execute("""UPDATE insurances SET status='INACTIVE' WHERE patient_id=? AND status='ACTIVE'""", (patient_id,))
        cur.execute("""
            INSERT INTO insurances (insurance_id, patient_id, carrier, member_id, group_number, effective_from, status, created_at)
            VALUES (?,?,?,?,?,?, 'ACTIVE', ?)
        """, (iid, patient_id, carrier, member_id, group_number, effective_from, now))
        cur.execute("""
            INSERT INTO insurance_audit_log (log_id, patient_id, old_insurance_id, new_insurance_id, changed_at, changed_by)
            VALUES (?,?,?,?,?,?)
        """, (f"LG{uuid.uuid4().hex[:10]}", patient_id, None, iid, now, "admin_ui"))
        conn.commit()
    finally:
        conn.close()
    return {"insurance_id": iid, "patient_id": patient_id, "carrier": carrier, "member_id": member_id, "group_number": group_number, "effective_from": effective_from, "status": "ACTIVE"}

def change_insurance(patient_id: str, carrier: str, member_id: str, group_number: str | None, effective_from: str | None, actor: str="assistant") -> dict:
    _ensure_tables()
    old = get_active_insurance(patient_id)
    iid = f"I{uuid.uuid4().hex[:10]}"; now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        cur.execute("""UPDATE insurances SET status='INACTIVE' WHERE patient_id=? AND status='ACTIVE'""", (patient_id,))
        cur.execute("""
            INSERT INTO insurances (insurance_id, patient_id, carrier, member_id, group_number, effective_from, status, created_at)
            VALUES (?,?,?,?,?,?, 'ACTIVE', ?)
        """, (iid, patient_id, carrier, member_id, group_number, effective_from, now))
        cur.execute("""
            INSERT INTO insurance_audit_log (log_id, patient_id, old_insurance_id, new_insurance_id, changed_at, changed_by)
            VALUES (?,?,?,?,?,?)
        """, (f"LG{uuid.uuid4().hex[:10]}", patient_id, old.insurance_id if old else None, iid, now, actor))
        conn.commit()
    finally:
        conn.close()
    return {"insurance_id": iid, "patient_id": patient_id, "carrier": carrier, "member_id": member_id, "group_number": group_number, "effective_from": effective_from, "status": "ACTIVE"}

def list_insurances(patient_id: str) -> List[Insurance]:
    _ensure_tables()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        SELECT insurance_id, patient_id, carrier, member_id, group_number, effective_from, status, created_at
        FROM insurances WHERE patient_id=? ORDER BY datetime(created_at) DESC
    """, (patient_id,))
    rows = [Insurance(*r) for r in cur.fetchall()]
    conn.close(); return rows
