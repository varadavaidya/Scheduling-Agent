
#!/usr/bin/env python3
import os, sys
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from tools.notify import due_reminders, mark_sent, send_reminder
from tools.scheduler import list_appointments
from tools.patient import get_patient_by_id
from tools.doctors import list_doctors_df

def main():
    due = due_reminders()
    if not due:
        print("No reminders due."); return 0
    ddf = list_doctors_df()
    dmap = {r["doctor_id"]: r["name"] for _, r in ddf.iterrows()} if not ddf.empty else {}
    sent = 0
    for r in due:
        appts = list_appointments(patient_id=r['patient_id'])
        appt = next((a for a in appts if a['appointment_id']==r['appointment_id']), None)
        if not appt: continue
        p = get_patient_by_id(r['patient_id'])
        docname = dmap.get(appt['doctor_id'], appt['doctor_id'])
        result = send_reminder(p.email if p else 'patient@example.com', f"{p.first_name} {p.last_name}" if p else 'Patient', appt, r['stage'], docname)
        mark_sent(r['reminder_id'], 'smtp' if result['status']=='sent' else 'eml')
        sent += 1
        print(f"Sent reminder {r['reminder_id']} (stage {r['stage']}) for {appt['appointment_id']} via {result['status']}")
    print(f"Done. Processed {sent} reminder(s)."); return 0

if __name__ == "__main__":
    raise SystemExit(main())
