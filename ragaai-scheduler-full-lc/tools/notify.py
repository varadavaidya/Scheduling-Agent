
from __future__ import annotations
import os, sqlite3, uuid
from pathlib import Path
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formatdate
import smtplib

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "store.db"
ASSETS = Path(__file__).resolve().parents[1] / "assets"
OUTBOX = Path(__file__).resolve().parents[1] / "outbox"

def _ensure_tables():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            reminder_id TEXT PRIMARY KEY,
            appointment_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            reminder_time TEXT NOT NULL,
            stage INTEGER NOT NULL,
            sent_flag INTEGER NOT NULL DEFAULT 0,
            sent_via TEXT,
            sent_at TEXT
        )
    """)
    conn.commit(); conn.close()

def schedule_reminders_for(appt: dict, patient_email: str | None = None) -> int:
    _ensure_tables()
    start = datetime.fromisoformat(appt["start"])
    stages = [(1, start - timedelta(hours=24)), (2, start - timedelta(hours=3)), (3, start - timedelta(minutes=30))]
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    added = 0
    for stage, when in stages:
        rid = f"R{uuid.uuid4().hex[:10]}"
        cur.execute("""
            INSERT INTO reminders (reminder_id, appointment_id, patient_id, reminder_time, stage, sent_flag)
            VALUES (?,?,?,?,?,0)
        """, (rid, appt["appointment_id"], appt["patient_id"], when.isoformat(), stage))
        added += 1
    conn.commit(); conn.close()
    return added

def due_reminders(now: datetime | None = None) -> list[dict]:
    _ensure_tables()
    if now is None: from datetime import datetime as _dt; now = _dt.now()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        SELECT reminder_id, appointment_id, patient_id, reminder_time, stage, sent_flag, sent_via, sent_at
        FROM reminders WHERE sent_flag=0 AND datetime(reminder_time) <= datetime(?)
        ORDER BY datetime(reminder_time) ASC
    """, (now.isoformat(),))
    cols = [d[0] for d in cur.description]; rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close(); return rows

def mark_sent(reminder_id: str, via: str):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("UPDATE reminders SET sent_flag=1, sent_via=?, sent_at=datetime('now') WHERE reminder_id=?", (via, reminder_id))
    conn.commit(); conn.close()

def _smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"))

def _send_via_smtp(to_addr: str, msg: EmailMessage) -> bool:
    host = os.getenv("SMTP_HOST"); port = int(os.getenv("SMTP_PORT") or "587")
    user = os.getenv("SMTP_USER"); pwd  = os.getenv("SMTP_PASS"); sender = os.getenv("SMTP_FROM")
    if not (host and sender): return False
    with smtplib.SMTP(host, port) as server:
        server.starttls()
        if user and pwd: server.login(user, pwd)
        server.sendmail(sender, [to_addr], msg.as_string())
    return True

def _compose_email(to_addr: str, subject: str, body: str, attachments: list[tuple[str, bytes, str]] | None = None) -> EmailMessage:
    msg = EmailMessage()
    msg["To"] = to_addr; msg["From"] = os.getenv("SMTP_FROM", "no-reply@example.com")
    msg["Date"] = formatdate(localtime=True); msg["Subject"] = subject
    msg.set_content(body)
    for filename, data, mime in (attachments or []):
        maintype, subtype = mime.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return msg

def _save_eml(msg: EmailMessage, basename: str) -> str:
    OUTBOX.mkdir(exist_ok=True, parents=True)
    path = OUTBOX / f"{basename}.eml"
    with open(path, "wb") as f: f.write(msg.as_bytes())
    return str(path)

def send_confirmation(patient_email: str, patient_name: str, appt: dict, doctor_name: str, location_name: str, attach_intake: bool=True) -> dict:
    subject = f"Appointment Confirmation — {appt['start'].replace('T',' ')}"
    body = (f"Hi {patient_name},\n\nYour appointment is confirmed.\n\n"
            f"- Doctor: {doctor_name}\n- Location: {location_name}\n"
            f"- When: {appt['start'].replace('T',' ')} to {appt['end'].replace('T',' ')}\n"
            f"- Appointment ID: {appt['appointment_id']}\n\nPlease arrive 10 minutes early.\n\nRegards,\nRagaAI Clinic Scheduling\n")
    attachments = []
    intake = ASSETS / "intake_form.pdf"
    if attach_intake and intake.exists():
        data = intake.read_bytes(); attachments.append(("New Patient Intake Form.pdf", data, "application/pdf"))
    msg = _compose_email(patient_email, subject, body, attachments)
    if _smtp_configured():
        try:
            ok = _send_via_smtp(patient_email, msg)
            if ok: return {"status": "sent", "path": ""}
        except Exception: pass
    eml_path = _save_eml(msg, f"confirm_{appt['appointment_id']}"); return {"status": "saved", "path": eml_path}

def send_reminder(patient_email: str, patient_name: str, appt: dict, stage: int, doctor_name: str) -> dict:
    when = appt['start'].replace('T',' ')
    if stage == 1:
        subject = "Reminder: Appointment in 24 hours"
        body = f"Hi {patient_name},\n\nReminder: your appointment with {doctor_name} is tomorrow at {when}.\n\n— RagaAI Clinic"
    elif stage == 2:
        subject = "Reminder: Appointment in 3 hours"
        body = f"Hi {patient_name},\n\nReminder: your appointment with {doctor_name} is today at {when}.\n\n— RagaAI Clinic"
    else:
        subject = "Final Reminder: Appointment in 30 minutes"
        body = f"Hi {patient_name},\n\nFinal reminder: your appointment with {doctor_name} is in 30 minutes at {when}.\n\n— RagaAI Clinic"
    msg = _compose_email(patient_email, subject, body, attachments=None)
    if _smtp_configured():
        try:
            ok = _send_via_smtp(patient_email, msg)
            if ok: return {"status":"sent","path":""}
        except Exception: pass
    eml_path = _save_eml(msg, f"reminder_{stage}_{appt['appointment_id']}"); return {"status":"saved","path":eml_path}

def _get_appt(appt_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT appointment_id, patient_id, doctor_id, start, end, status, active_insurance_id FROM appointments WHERE appointment_id=?", (appt_id,))
    row = cur.fetchone(); conn.close()
    if not row: return None
    cols = ["appointment_id","patient_id","doctor_id","start","end","status","active_insurance_id"]; return dict(zip(cols, row))

def resend_confirmation(appt_id: str) -> dict:
    appt = _get_appt(appt_id)
    if not appt: raise ValueError(f"Appointment {appt_id} not found")
    from tools.patient import get_patient_by_id
    from tools.doctors import list_doctors_df, list_clinics_df
    p = get_patient_by_id(appt["patient_id"]); ddf = list_doctors_df(); cdf = list_clinics_df()
    doc_row = ddf[ddf["doctor_id"] == appt["doctor_id"]].iloc[0].to_dict() if not ddf.empty else {"name": appt["doctor_id"], "location_id": ""}
    loc_name = ""
    if doc_row.get("location_id") and not cdf.empty:
        match = cdf[cdf["location_id"] == doc_row["location_id"]]
        if not match.empty: loc_name = match.iloc[0]["name"]
    return send_confirmation(p.email if p else "patient@example.com", f"{p.first_name} {p.last_name}" if p else "Patient", appt, doc_row.get("name","Doctor"), loc_name, attach_intake=True)

def send_reminder_now(appt_id: str, stage: int) -> dict:
    appt = _get_appt(appt_id)
    if not appt: raise ValueError(f"Appointment {appt_id} not found")
    from tools.patient import get_patient_by_id
    from tools.doctors import list_doctors_df
    p = get_patient_by_id(appt["patient_id"]); ddf = list_doctors_df()
    docname = appt["doctor_id"]
    if not ddf.empty:
        m = ddf[ddf["doctor_id"] == appt["doctor_id"]]
        if not m.empty: docname = m.iloc[0]["name"]
    return send_reminder(p.email if p else "patient@example.com", f"{p.first_name} {p.last_name}" if p else "Patient", appt, int(stage), docname)

def cancel_reminders_for(appointment_id: str) -> int:
    """Mark all pending reminders for this appointment as cancelled."""
    _ensure_tables()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE reminders
            SET sent_flag=1, sent_via='cancelled', sent_at=datetime('now')
            WHERE appointment_id=? AND sent_flag=0
        """, (appointment_id,))
        conn.commit()
        return cur.rowcount if cur.rowcount != -1 else 0
    finally:
        conn.close()
