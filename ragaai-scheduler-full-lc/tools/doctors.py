from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import os
import sqlite3
import pandas as pd
import uuid
from uuid import uuid4
from datetime import datetime as _dt, date as _date, time as _time, timedelta as _td

# ----------------------------
# Paths / storage
# ----------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "store.db"))

DOCTORS_CSV = DATA_DIR / "doctors.csv"
CLINICS_CSV = DATA_DIR / "clinics.csv"

# ----------------------------
# DB bootstrap
# ----------------------------
def _ensure_tables() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript(
        """
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS clinics (
            location_id TEXT PRIMARY KEY,
            name        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            specialty   TEXT,
            location_id TEXT,
            FOREIGN KEY(location_id) REFERENCES clinics(location_id)
        );

        CREATE TABLE IF NOT EXISTS slots (
            slot_id   TEXT PRIMARY KEY,
            doctor_id TEXT NOT NULL,
            start     TEXT NOT NULL,   -- ISO8601: YYYY-MM-DDTHH:MM
            end       TEXT NOT NULL,
            is_booked INTEGER NOT NULL DEFAULT 0,
            UNIQUE(doctor_id, start, end),
            FOREIGN KEY(doctor_id) REFERENCES doctors(doctor_id)
        );
        """
    )
    con.commit()
    con.close()

# call once at import to be safe
_ensure_tables()

# ----------------------------
# CSV helpers
# ----------------------------
def _df_or_empty(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        # backfill missing columns
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def _next_id(prefix: str, existing_ids: List[str], base: int = 100) -> str:
    nums = []
    for x in (existing_ids or []):
        if isinstance(x, str) and x.startswith(prefix):
            tail = x[len(prefix):]
            if tail.isdigit():
                nums.append(int(tail))
    nxt = (max(nums) if nums else base) + 1
    return f"{prefix}{nxt}"

# ----------------------------
# Models
# ----------------------------
@dataclass
class Doctor:
    doctor_id: str
    name: str
    specialty: str
    location_id: str

@dataclass
class Clinic:
    location_id: str
    name: str

# ----------------------------
# Public list APIs (CSV-backed)
# ----------------------------
def list_doctors_df() -> pd.DataFrame:
    return _df_or_empty(DOCTORS_CSV, ["doctor_id", "name", "specialty", "location_id"])

def list_clinics_df() -> pd.DataFrame:
    return _df_or_empty(CLINICS_CSV, ["location_id", "name"])

# ----------------------------
# Add clinic / doctor (CSV + SQLite)
# ----------------------------
def add_location(name: str) -> Clinic:
    _ensure_tables()
    cdf = list_clinics_df()
    new_id = _next_id("L", cdf["location_id"].tolist() if not cdf.empty else [], base=0)
    row = {"location_id": new_id, "name": name.strip()}
    cdf = pd.concat([cdf, pd.DataFrame([row])], ignore_index=True)
    cdf.to_csv(CLINICS_CSV, index=False)

    # mirror to SQLite
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO clinics (location_id, name) VALUES (?,?)",
        (row["location_id"], row["name"]),
    )
    con.commit()
    con.close()

    return Clinic(**row)

def add_doctor(name: str, specialty: str, location_id: str) -> Doctor:
    _ensure_tables()
    ddf = list_doctors_df()
    new_id = _next_id("D", ddf["doctor_id"].tolist() if not ddf.empty else [], base=0)
    row = {
        "doctor_id": new_id,
        "name": name.strip(),
        "specialty": specialty.strip(),
        "location_id": location_id.strip(),
    }
    # CSV
    ddf = pd.concat([ddf, pd.DataFrame([row])], ignore_index=True)
    ddf.to_csv(DOCTORS_CSV, index=False)

    # ensure clinic exists in SQLite
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO clinics (location_id, name) VALUES (?, COALESCE((SELECT name FROM clinics WHERE location_id=?), 'Clinic'))",
        (row["location_id"], row["location_id"]),
    )
    # mirror doctor into SQLite
    cur.execute(
        "INSERT OR REPLACE INTO doctors (doctor_id, name, specialty, location_id) VALUES (?,?,?,?)",
        (row["doctor_id"], row["name"], row["specialty"], row["location_id"]),
    )
    con.commit()
    con.close()

    return Doctor(**row)

# ----------------------------
# Time parsing
# ----------------------------
def _parse_hhmm(val) -> _time:
    """
    Accepts '09:00', '9:00', '9.00', '0900', or a datetime.time.
    Returns datetime.time, raises ValueError on bad input.
    """
    if isinstance(val, _time):
        return val
    if val is None:
        raise ValueError("start_time/end_time cannot be None")

    s = str(val).strip()
    s = (s.replace("．", ":").replace("。", ":").replace("·", ":").replace(".", ":")
           .replace("：", ":"))
    if ":" not in s:
        if len(s) == 4 and s.isdigit():
            s = s[:2] + ":" + s[2:]
        else:
            raise ValueError(f"Invalid time string '{val}', expected 'HH:MM'")

    hh, mm = s.split(":", 1)
    try:
        hh_i, mm_i = int(hh), int(mm)
    except Exception:
        raise ValueError(f"Invalid time string '{val}', expected 'HH:MM'")
    if not (0 <= hh_i < 24 and 0 <= mm_i < 60):
        raise ValueError(f"Time out of range '{val}'")
    return _time(hour=hh_i, minute=mm_i)

# ----------------------------
# Slot generation
# ----------------------------
def generate_slots_for_doctor(
    doctor_id: str,
    days: int | None = None,
    slots_count: int | None = None,
    start_time: str | _time = "09:00",
    end_time: str | _time = "17:00",
    step_minutes: int = 30,
) -> int:
    """
    Generate 30-min slots either by:
      - number of DAYS (fill each day from start->end)
      - EXACT number of slots (spill into following days).
    Returns the number of slots inserted.
    """
    if (days is None and slots_count is None) or (days is not None and slots_count is not None):
        raise ValueError("Pass either 'days' or 'slots_count' (but not both).")

    _ensure_tables()

    # make sure this doctor exists in SQLite (read from CSV for details)
    ddf = list_doctors_df()
    if ddf.empty or doctor_id not in ddf["doctor_id"].values:
        raise ValueError(f"Doctor '{doctor_id}' not found in CSV.")
    drow = ddf.loc[ddf["doctor_id"] == doctor_id].iloc[0].to_dict()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # ensure mirrored rows
    cur.execute(
        "INSERT OR IGNORE INTO clinics (location_id, name) VALUES (?, COALESCE((SELECT name FROM clinics WHERE location_id=?), 'Clinic'))",
        (drow["location_id"], drow["location_id"]),
    )
    cur.execute(
        "INSERT OR REPLACE INTO doctors (doctor_id, name, specialty, location_id) VALUES (?,?,?,?)",
        (drow["doctor_id"], drow["name"], drow["specialty"], drow["location_id"]),
    )
    con.commit()
    con.close()

    start_t = _parse_hhmm(start_time)
    end_t   = _parse_hhmm(end_time)
    if (end_t.hour, end_t.minute) <= (start_t.hour, start_t.minute):
        raise ValueError("end_time must be after start_time")
    step = _td(minutes=int(step_minutes))

    created = 0
    today = _date.today()

    with sqlite3.connect(DB_PATH) as con2:
        cur2 = con2.cursor()

        def _insert_slot(start_dt: _dt, end_dt: _dt) -> int:
            start_iso = start_dt.replace(second=0, microsecond=0).isoformat(timespec="minutes")
            end_iso   = end_dt.replace(second=0, microsecond=0).isoformat(timespec="minutes")
            cur2.execute(
                "INSERT OR IGNORE INTO slots (slot_id, doctor_id, start, end, is_booked) VALUES (?,?,?,?,0)",
                (f"S{uuid.uuid4().hex[:10]}", doctor_id, start_iso, end_iso),
            )
            return 1 if (cur2.rowcount or 0) > 0 else 0

        if days is not None:
            for d in range(int(days)):
                day = today + _td(days=d)
                t   = _dt.combine(day, start_t)
                end = _dt.combine(day, end_t)
                while t + step <= end:
                    created += _insert_slot(t, t + step)
                    t += step
        else:
            remaining = int(slots_count)
            day = today
            t = _dt.combine(day, start_t)
            end_day = _dt.combine(day, end_t)
            while remaining > 0:
                if t + step > end_day:
                    day = day + _td(days=1)
                    t = _dt.combine(day, start_t)
                    end_day = _dt.combine(day, end_t)
                    continue
                created += _insert_slot(t, t + step)
                if (cur2.rowcount or 0) > 0:
                    remaining -= 1
                t += step

    return created

# ----------------------------
# Helpers for delete_doctor
# ----------------------------
def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def _ensure_admin_audit_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            log_id TEXT PRIMARY KEY,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            details TEXT,
            ts TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )

def _column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())

# ----------------------------
# Delete doctor (with cascade)
# ----------------------------
def delete_doctor(doctor_id: str, cascade: bool = False) -> dict:
    """
    Delete a doctor.

    If cascade=True:
      - Cancel FUTURE BOOKED appointments for this doctor
      - Delete FUTURE slots for this doctor
    Otherwise, raise if there are future booked appointments.
    Returns a summary dict.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    pending = 0
    if _table_exists(cur, "appointments"):
        cur.execute(
            "SELECT COUNT(*) FROM appointments "
            "WHERE doctor_id=? AND status='BOOKED' AND datetime(start) > datetime('now')",
            (doctor_id,),
        )
        pending = cur.fetchone()[0] or 0

    if pending > 0 and not cascade:
        con.close()
        raise ValueError("Doctor has future booked appointments. Enable cascade to cancel them.")

    cancelled = 0
    slots_deleted = 0

    if cascade:
        if _table_exists(cur, "appointments"):
            set_parts = ["status='CANCELLED'"]
            if _column_exists(cur, "appointments", "cancel_reason"):
                set_parts.append("cancel_reason='Doctor removed by admin'")
            set_clause = ", ".join(set_parts)
            cur.execute(
                f"UPDATE appointments SET {set_clause} "
                "WHERE doctor_id=? AND status='BOOKED' AND datetime(start) > datetime('now')",
                (doctor_id,),
            )
            cancelled = cur.rowcount or 0

        if _table_exists(cur, "slots"):
            cur.execute(
                "DELETE FROM slots WHERE doctor_id=? AND datetime(start) >= datetime('now')",
                (doctor_id,),
            )
            slots_deleted = cur.rowcount or 0

    removed = 0
    if _table_exists(cur, "doctors"):
        cur.execute("DELETE FROM doctors WHERE doctor_id=?", (doctor_id,))
        removed = cur.rowcount or 0

    con.commit()

    # audit (best-effort)
    try:
        _ensure_admin_audit_table(cur)
        cur.execute(
            "INSERT INTO admin_audit_log (log_id, actor, action, target_type, target_id, details, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                str(uuid4()),
                "admin",
                "DELETE",
                "doctor",
                doctor_id,
                f"cascade={cascade}; cancelled={cancelled}; slots_deleted={slots_deleted}; removed={removed}; pending_found={pending}",
            ),
        )
        con.commit()
    except Exception:
        pass
    finally:
        con.close()

    return {
        "removed": removed,
        "cancelled": cancelled,
        "slots_deleted": slots_deleted,
        "pending_found": pending,
    }
