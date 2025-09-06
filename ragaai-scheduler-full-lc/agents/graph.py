
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools import scheduler as sch
from tools import insurance as ins
from tools import doctors as docs
from tools import notify as notify

@dataclass
class ConvState:
    messages: List[str] = field(default_factory=list)
    patient_id: Optional[str] = "P1001"
    intent: Optional[str] = None
    temp: Dict[str, Any] = field(default_factory=dict)

class SimpleGraph:
    def invoke(self, state: ConvState) -> ConvState:
        if not state.messages:
            return state
        last = (state.messages[-1] or "").strip()

        # infer or continue intent
        if state.intent is None:
            low = last.lower()
            if "book" in low or "appointment" in low:
                state.intent = "BOOK_APPT"
                state.temp["desired"] = {}
                return handle_book_appt(state)
            if "insurance" in low and ("change" in low or "updated" in low or "switch" in low):
                state.intent = "CHANGE_INS"
                state.messages.append("Sure — please provide new insurance as: Carrier, MemberID, GroupNumber(optional), EffectiveFrom YYYY-MM-DD")
                return state
            # default small talk
            state.messages.append("Hi! I can help you book appointments or update insurance. Try: 'book appointment' or 'I changed my insurance'.")
            return state

        if state.intent == "BOOK_APPT":
            return handle_book_appt(state)

        if state.intent == "CHANGE_INS":
            # parse a simple CSV-like line
            parts = [p.strip() for p in last.split(",")]
            if len(parts) < 2:
                state.messages.append("Please provide at least Carrier and MemberID, optionally GroupNumber and EffectiveFrom (YYYY-MM-DD).")
                return state
            carrier = parts[0]
            member_id = parts[1]
            group_number = parts[2] if len(parts) >= 3 and parts[2] else None
            eff = parts[3] if len(parts) >= 4 and parts[3] else None
            try:
                res = ins.change_insurance(state.patient_id or "P1001", carrier, member_id, group_number, eff, actor="assistant")
                state.messages.append(f"Updated insurance to {carrier} (member ****{member_id[-4:]}). It's now active.")
                state.intent = None
            except Exception as e:
                state.messages.append(f"Could not update insurance: {e}")
            return state

        # fallback
        state.messages.append("I'm not sure — try 'book appointment' or 'I changed my insurance'.")
        return state

def handle_book_appt(state: ConvState) -> ConvState:
    desired = state.temp.get("desired", {})

    # Step 1: pick doctor
    if not desired.get("doctor_id"):
        ddf = docs.list_doctors_df()
        if ddf.empty:
            state.messages.append("No doctors available yet. Ask an admin to add doctors in the Admin tab.")
            return state
        choices = [f"{i+1}. {row['name']} — {row['specialty']} ({row['location_id']})" for i, row in ddf.reset_index(drop=True).iterrows()]
        state.temp["doctor_choices"] = ddf.to_dict(orient="records")
        state.messages.append("Please choose a doctor by number:\n" + "\n".join(choices))
        return state

    # Step 2: list slots
    if not desired.get("slot_index"):
        slots = sch.find_slots(doctor_id=desired["doctor_id"], limit=6)
        if not slots:
            state.messages.append("No available slots for this doctor. Try another day or doctor.")
            desired.pop("doctor_id", None)
            state.temp["desired"] = desired
            return state
        pretty = "\n".join(f"{i+1}. {s.start[0:16].replace('T',' ')}" for i,s in enumerate(slots))
        state.temp["slot_cache"] = [(s.start, s.end) for s in slots]
        state.messages.append("Here are available slots, reply with a number:\n" + pretty)
        return state

    # Step 3: book
    idx = desired["slot_index"] - 1
    slot_cache = state.temp.get("slot_cache", [])
    if idx < 0 or idx >= len(slot_cache):
        state.messages.append("Invalid choice. Please pick a valid number.")
        return state
    start, end = slot_cache[idx]
    active = ins.get_active_insurance(state.patient_id or "P1001")
    appt = sch.book_appointment(state.patient_id or "P1001", desired["doctor_id"], start, end, active.insurance_id if active else None)
    state.messages.append(f"Booked! Appointment {appt['appointment_id']} on {appt['start'].replace('T',' ')} with {desired['doctor_id']}.")

    # After booking: schedule reminders + confirmation with intake form
    try:
        from tools.patient import get_patient_by_id
        from tools.doctors import list_doctors_df, list_clinics_df
        p = get_patient_by_id(state.patient_id or "P1001")
        ddf = list_doctors_df(); cdf = list_clinics_df()
        doc_row = ddf[ddf["doctor_id"] == desired["doctor_id"]].iloc[0].to_dict() if not ddf.empty else {"name": desired["doctor_id"], "location_id": ""}
        loc_name = ""
        if doc_row.get("location_id") and not cdf.empty:
            match = cdf[cdf["location_id"] == doc_row["location_id"]]
            if not match.empty: loc_name = match.iloc[0]["name"]
        notify.schedule_reminders_for(appt, patient_email=p.email if p else "")
        conf = notify.send_confirmation(p.email if p else "patient@example.com",
                                        f"{p.first_name} {p.last_name}" if p else "Patient",
                                        appt, doc_row.get("name","Doctor"), loc_name, attach_intake=True)
        if conf.get("status") == "sent":
            state.messages.append("A confirmation email has been sent to your email.")
        else:
            state.messages.append("Confirmation saved as an email file you can download.")
        state.temp["last_confirmation_eml"] = conf.get("path")
    except Exception as e:
        state.messages.append("Booked, but could not queue reminders or confirmation email.")

    # clear temp
    state.temp.pop("slot_cache", None)
    state.temp.pop("doctor_choices", None)
    state.temp.pop("desired", None)
    state.intent = None
    return state

def build_graph():
    return SimpleGraph()
