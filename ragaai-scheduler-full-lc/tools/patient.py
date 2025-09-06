# tools/patient.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pandas as pd
import sqlite3, json
from uuid import uuid4

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PATIENTS_CSV = DATA_DIR / "patients.csv"
DB_PATH = DATA_DIR / "store.db"


# ---------------------------
# CSV helpers
# ---------------------------
REQUIRED_COLS = [
    "patient_id", "first_name", "last_name", "dob",
    "phone", "email", "is_returning", "checked_out",
]

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLS)

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # ensure required columns exist
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = "" if c not in ("is_returning", "checked_out") else 0
    # keep only required cols, correct order
    df = df[REQUIRED_COLS].copy()
    # normalize booleans
    df["is_returning"] = df["is_returning"].apply(lambda v: 1 if str(v).strip().lower() in ("1", "true", "yes") else 0)
    df["checked_out"]  = df["checked_out"].apply(lambda v: 1 if str(v).strip().lower() in ("1", "true", "yes") else 0)
    return df

def _load_df() -> pd.DataFrame:
    if PATIENTS_CSV.exists():
        df = pd.read_csv(PATIENTS_CSV)
        return _normalize_df(df)
    return _empty_df()

def _save_df(df: pd.DataFrame) -> None:
    df = _normalize_df(df)
    df.to_csv(PATIENTS_CSV, index=False)

def _next_patient_id(df: pd.DataFrame) -> str:
    nums = []
    if "patient_id" in df.columns:
        for x in df["patient_id"].astype(str).tolist():
            if x.startswith("P") and x[1:].isdigit():
                nums.append(int(x[1:]))
    nxt = (max(nums) if nums else 1000) + 1
    return f"P{nxt}"


# ---------------------------
# Public API
# ---------------------------
@dataclass
class Patient:
    patient_id: str
    first_name: str
    last_name: str
    dob: str
    phone: str
    email: str
    is_returning: bool = False
    # checked_out is tracked in CSV but not required in the dataclass here


def ensure_checked_out_column() -> None:
    """
    Keep the name for compatibility with app.py, but implement it for CSV.
    Ensures `checked_out` column exists in data/patients.csv.
    """
    df = _load_df()
    if "checked_out" not in df.columns:
        df["checked_out"] = 0
    _save_df(df)


def set_patient_checked_out(patient_id: str, checked: bool = True) -> None:
    """Mark a patient as checked out (1) or active (0) in the CSV."""
    df = _load_df()
    if df.empty or patient_id not in set(df["patient_id"]):
        raise ValueError(f"Patient {patient_id} not found")
    df.loc[df["patient_id"] == patient_id, "checked_out"] = 1 if checked else 0
    _save_df(df)

    # best-effort audit in SQLite (table may or may not exist)
    try:
        con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS admin_audit_log (
                log_id TEXT PRIMARY KEY,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details TEXT,
                ts TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cur.execute(
            "INSERT INTO admin_audit_log (log_id, actor, action, target_type, target_id, details, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (str(uuid4()), "admin", "CHECKOUT" if checked else "UNCHECKOUT",
             "patient", patient_id, f"checked_out={int(checked)}")
        )
        con.commit()
    except Exception:
        pass
    finally:
        try: con.close()
        except Exception: pass


def get_patients_df(active_only: bool | None = None) -> pd.DataFrame:
    """
    Return patients as DataFrame from CSV, optionally filtered by checked_out.
      - active_only=True  -> checked_out == 0
      - active_only=False -> checked_out == 1
      - active_only=None  -> all
    """
    df = _load_df()
    if df.empty:
        return _empty_df()
    if active_only is True:
        df = df[df["checked_out"] != 1]
    elif active_only is False:
        df = df[df["checked_out"] == 1]
    # nice ordering
    return df.sort_values(by=["last_name", "first_name"], na_position="last")


def update_patient_fields(
    patient_id: str,
    first_name: str | None = None,
    last_name: str | None = None,
    dob: str | None = None,
    phone: str | None = None,
    email: str | None = None,
) -> None:
    """
    Update any subset of patient fields in the CSV.
    Only provided fields change; others remain as-is.
    """
    df = _load_df()
    if df.empty or patient_id not in set(df["patient_id"]):
        raise ValueError(f"Patient {patient_id} not found")

    mask = df["patient_id"] == patient_id
    if first_name is not None:
        df.loc[mask, "first_name"] = first_name
    if last_name is not None:
        df.loc[mask, "last_name"] = last_name
    if dob is not None:
        df.loc[mask, "dob"] = dob
    if phone is not None:
        df.loc[mask, "phone"] = phone
    if email is not None:
        df.loc[mask, "email"] = email

    _save_df(df)


def add_patient(first_name: str, last_name: str, dob: str, phone: str, email: str,
                is_returning: bool = False) -> Patient:
    df = _load_df()
    pid = _next_patient_id(df)
    row = {
        "patient_id": pid,
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "dob": dob.strip(),
        "phone": phone.strip(),
        "email": email.strip(),
        "is_returning": 1 if is_returning else 0,
        "checked_out": 0,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_df(df)
    return Patient(
        patient_id=row["patient_id"], first_name=row["first_name"],
        last_name=row["last_name"], dob=row["dob"], phone=row["phone"],
        email=row["email"], is_returning=bool(row["is_returning"])
    )


def list_patients(limit: int = 500) -> List[Patient]:
    df = _load_df()
    if df.empty:
        return []
    sub = df.head(limit)
    out: List[Patient] = []
    for r in sub.itertuples(index=False):
        out.append(Patient(
            patient_id=r.patient_id,
            first_name=r.first_name,
            last_name=r.last_name,
            dob=r.dob,
            phone=r.phone,
            email=r.email,
            is_returning=bool(r.is_returning),
        ))
    return out


def get_patient_by_id(patient_id: str) -> Optional[Patient]:
    df = _load_df()
    row = df.loc[df["patient_id"] == patient_id]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    return Patient(
        patient_id=r["patient_id"], first_name=r["first_name"], last_name=r["last_name"],
        dob=r["dob"], phone=r["phone"], email=r["email"],
        is_returning=bool(r.get("is_returning", 0))
    )


def get_patient_by_email(email: str) -> Optional[Patient]:
    df = _load_df()
    if df.empty:
        return None
    row = df.loc[df["email"].astype(str).str.lower() == str(email).strip().lower()]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    return Patient(
        patient_id=r["patient_id"], first_name=r["first_name"], last_name=r["last_name"],
        dob=r["dob"], phone=r["phone"], email=r["email"],
        is_returning=bool(r.get("is_returning", 0))
    )


# ---------------------------
# Delete patient (CSV) + related SQLite cleanup (optional)
# ---------------------------
def _ensure_admin_audit_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            log_id TEXT PRIMARY KEY,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            details TEXT,
            ts TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def delete_patient(patient_id: str, cascade: bool = False) -> dict:
    df = _load_df()
    row = df.loc[df["patient_id"] == patient_id]
    if row.empty:
        raise ValueError(f"Patient {patient_id} not found")

    # related cleanup (appointments/insurance) if those tables exist in SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    appt_count = ins_count = audit_count = 0
    try:
        if _table_exists(conn, "appointments"):
            cur.execute("SELECT COUNT(*) FROM appointments WHERE patient_id=?", (patient_id,))
            appt_count = cur.fetchone()[0] or 0
        if _table_exists(conn, "insurances"):
            cur.execute("SELECT COUNT(*) FROM insurances WHERE patient_id=?", (patient_id,))
            ins_count = cur.fetchone()[0] or 0
        if _table_exists(conn, "insurance_audit_log"):
            cur.execute("SELECT COUNT(*) FROM insurance_audit_log WHERE patient_id=?", (patient_id,))
            audit_count = cur.fetchone()[0] or 0

        if not cascade and (appt_count > 0 or ins_count > 0 or audit_count > 0):
            raise ValueError(
                f"Patient has related data (appointments={appt_count}, insurances={ins_count}, audit={audit_count}). "
                "Enable cascade to delete."
            )

        deleted = {"appointments": 0, "insurances": 0, "audit": 0, "patient": 0}
        if cascade:
            if _table_exists(conn, "appointments"):
                cur.execute("DELETE FROM appointments WHERE patient_id=?", (patient_id,))
                deleted["appointments"] = cur.rowcount if (cur.rowcount or 0) > 0 else appt_count
            if _table_exists(conn, "insurances"):
                cur.execute("DELETE FROM insurances WHERE patient_id=?", (patient_id,))
                deleted["insurances"] = cur.rowcount if (cur.rowcount or 0) > 0 else ins_count
            if _table_exists(conn, "insurance_audit_log"):
                cur.execute("DELETE FROM insurance_audit_log WHERE patient_id=?", (patient_id,))
                deleted["audit"] = cur.rowcount if (cur.rowcount or 0) > 0 else audit_count
            conn.commit()
    finally:
        conn.close()

    # remove from CSV
    df = df.loc[df["patient_id"] != patient_id].copy()
    _save_df(df)
    deleted["patient"] = 1

    # audit (best-effort)
    conn2 = sqlite3.connect(DB_PATH)
    try:
        _ensure_admin_audit_table(conn2)
        cur2 = conn2.cursor()
        cur2.execute(
            "INSERT INTO admin_audit_log (log_id, actor, action, target_type, target_id, details) VALUES (?,?,?,?,?,?)",
            (f"AD{uuid4().hex[:10]}", "admin_ui", "DELETE_PATIENT", "patient", patient_id, json.dumps(deleted))
        )
        conn2.commit()
    finally:
        conn2.close()

    return deleted
