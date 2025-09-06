from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import os, sqlite3
from uuid import uuid4
from datetime import datetime as _dt

# Keep DB path consistent with tools/doctors.py
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "store.db"))

# -------------------------
# Bootstrap tables
# -------------------------
# tools/scheduler.py
def _ensure_tables():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    PRAGMA foreign_keys=ON;

    CREATE TABLE IF NOT EXISTS appointments (
        appointment_id TEXT PRIMARY KEY,
        patient_id     TEXT NOT NULL,
        doctor_id      TEXT NOT NULL,
        start          TEXT NOT NULL,   -- ISO8601 (YYYY-MM-DDTHH:MM)
        end            TEXT NOT NULL,
        status         TEXT NOT NULL DEFAULT 'BOOKED',
        created_at     TEXT NOT NULL DEFAULT (datetime('now'))
        -- newer columns added below if missing
    );

    CREATE TABLE IF NOT EXISTS slots (
        slot_id   TEXT PRIMARY KEY,
        doctor_id TEXT NOT NULL,
        start     TEXT NOT NULL,
        end       TEXT NOT NULL,
        is_booked INTEGER NOT NULL DEFAULT 0,
        UNIQUE(doctor_id, start, end)
    );
    """)

    # helper: add a column if it doesn't exist
    def _add_col_if_missing(table: str, col: str, decl: str):
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        if col not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    # migrate older DBs
    _add_col_if_missing("appointments", "insurance_id", "TEXT")
    _add_col_if_missing("appointments", "cancel_reason", "TEXT")
    _add_col_if_missing("appointments", "notes", "TEXT")

    con.commit()
    con.close()


_ensure_tables()

# -------------------------
# Public API
# -------------------------
@dataclass
class Slot:
    start: str
    end: str

def _fetch_open_slots(doctor_id: str, limit: int = 200, after_iso: Optional[str] = None) -> List[Slot]:
    """Fetch raw 30-min open slots for a doctor."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    if after_iso:
        cur.execute(
            "SELECT start, end FROM slots WHERE doctor_id=? AND is_booked=0 AND datetime(start)>=datetime(?) ORDER BY datetime(start) LIMIT ?",
            (doctor_id, after_iso, limit),
        )
    else:
        cur.execute(
            "SELECT start, end FROM slots WHERE doctor_id=? AND is_booked=0 AND datetime(start)>=datetime('now') ORDER BY datetime(start) LIMIT ?",
            (doctor_id, limit),
        )
    rows = cur.fetchall()
    con.close()
    return [Slot(r[0], r[1]) for r in rows]

def find_slots(doctor_id: str, limit: int = 8, duration_minutes: Optional[int] = None) -> List[Slot]:
    """
    Return upcoming available starts for the given doctor.
    If duration_minutes is 60, only return starts that have 2 contiguous 30-min slots.
    If 30 (or None), return single 30-min starts.
    """
    _ensure_tables()
    need = 1 if (duration_minutes in (None, 30)) else max(1, duration_minutes // 30)

    # Get a generous window to scan for contiguity
    raw = _fetch_open_slots(doctor_id, limit=400)
    if not raw:
        return []

    if need == 1:
        return raw[:limit]

    # need >= 2: find contiguous sequences
    out: List[Slot] = []
    for i in range(len(raw) - (need - 1)):
        ok = True
        # every consecutive slot must chain end == next.start
        for k in range(need - 1):
            if raw[i + k].end != raw[i + k + 1].start:
                ok = False
                break
        if ok:
            out.append(raw[i])  # present the *first* 30-min start of the block
            if len(out) >= limit:
                break
    return out

def book_appointment_with_duration(
    patient_id: str,
    doctor_id: str,
    start_iso: str,
    duration_minutes: int,
    insurance_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Book a visit starting at start_iso if there are enough contiguous 30-min slots.
    Marks those slots as booked and inserts an appointment spanning the whole duration.
    """
    _ensure_tables()
    need = max(1, duration_minutes // 30)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Load the candidate 'need' slots starting at start_iso
    cur.execute(
        "SELECT start, end FROM slots WHERE doctor_id=? AND is_booked=0 AND datetime(start)>=datetime(?) ORDER BY datetime(start) LIMIT ?",
        (doctor_id, start_iso, need),
    )
    seq = cur.fetchall()
    if len(seq) < need or seq[0][0] != start_iso:
        con.close()
        raise ValueError("Not enough contiguous 30-minute slots available for the requested duration.")

    # Check contiguity
    for i in range(need - 1):
        if seq[i][1] != seq[i + 1][0]:
            con.close()
            raise ValueError("Not enough contiguous 30-minute slots available for the requested duration.")

    start = seq[0][0]
    end = seq[-1][1]
    appt_id = f"A{uuid4().hex[:10]}"

    try:
        # Mark those specific slots as booked
        for i in range(need):
            cur.execute(
                "UPDATE slots SET is_booked=1 WHERE doctor_id=? AND start=? AND is_booked=0",
                (doctor_id, seq[i][0]),
            )
            if cur.rowcount == 0:
                raise ValueError("Slot was taken while booking. Please choose another.")

        # Create appointment
        cur.execute(
            "INSERT INTO appointments (appointment_id, patient_id, doctor_id, start, end, status, insurance_id) VALUES (?,?,?,?,?,'BOOKED',?)",
            (appt_id, patient_id, doctor_id, start, end, insurance_id),
        )
        con.commit()
    finally:
        con.close()

    return {
        "appointment_id": appt_id,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "start": start,
        "end": end,
        "status": "BOOKED",
        "insurance_id": insurance_id,
    }

def cancel_appointment(appointment_id: str, free_slots: bool = True) -> None:
    """Cancel an appointment and optionally free the underlying slots."""
    _ensure_tables()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT doctor_id, start, end FROM appointments WHERE appointment_id=?", (appointment_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise ValueError("Appointment not found.")
    doctor_id, start_iso, end_iso = row[0], row[1], row[2]

    # cancel appointment
    cur.execute(
        "UPDATE appointments SET status='CANCELLED', cancel_reason=COALESCE(cancel_reason,'User request') WHERE appointment_id=?",
        (appointment_id,),
    )

    if free_slots:
        # Free any 30-min slots fully inside the appointment window
        cur.execute(
            "UPDATE slots SET is_booked=0 WHERE doctor_id=? AND datetime(start)>=datetime(?) AND datetime(end)<=datetime(?)",
            (doctor_id, start_iso, end_iso),
        )
    con.commit()
    con.close()

def list_appointments(patient_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    _ensure_tables()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    if patient_id:
        cur.execute(
            "SELECT appointment_id, patient_id, doctor_id, start, end, status, insurance_id FROM appointments WHERE patient_id=? ORDER BY datetime(start) DESC LIMIT ?",
            (patient_id, limit),
        )
    else:
        cur.execute(
            "SELECT appointment_id, patient_id, doctor_id, start, end, status, insurance_id FROM appointments ORDER BY datetime(start) DESC LIMIT ?",
            (limit,),
        )
    rows = cur.fetchall()
    con.close()
    return [
        {
            "appointment_id": r[0],
            "patient_id": r[1],
            "doctor_id": r[2],
            "start": r[3],
            "end": r[4],
            "status": r[5],
            "insurance_id": r[6],
        }
        for r in rows
    ]
