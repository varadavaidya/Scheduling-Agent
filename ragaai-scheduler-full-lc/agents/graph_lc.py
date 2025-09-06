# agents/graph_lc.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

# LangChain / LangGraph
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    # If langchain community isn't available, we'll guard calls at runtime
    ChatOllama = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore

from langgraph.graph import StateGraph, END

# Project tools
from tools import doctors as docs
from tools import scheduler as sch
from tools import insurance as ins
from tools.patient import (
    get_patient_by_email,
    add_patient,
    get_patient_by_id,
    update_patient_fields,  # used by edit identity
)

# =========================
# Receptionist rewriter (Ollama)
# =========================
RECEPTIONIST_PROMPT = """You are 'Asha', the friendly front-desk receptionist for Raga Clinic.
Rewrite the ASSISTANT_MESSAGE in a warm, concise, and professional tone.

CRITICAL RULES:
- Do NOT invent facts or availability. Do not change any data.
- Preserve ALL dates/times, appointment IDs, phone numbers, and emails EXACTLY as given.
- If there is a numbered list or doctor options, KEEP the numbering and wording exactly. You may add a brief, friendly intro/outro only.
- Keep it short (2–4 sentences) unless the original text is longer.
Return ONLY the rewritten text.
"""

def _as_receptionist(text: str) -> str:
    """Use local Ollama (llama3) to rewrite the assistant message in a receptionist tone.
    Falls back to the original text on any error or if disabled via env."""
    if os.getenv("RECEPTIONIST_TONE", "1").lower() not in ("1", "true", "yes", "on"):
        return text
    try:
        if ChatOllama is None or HumanMessage is None or SystemMessage is None:
            return text
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        sys = SystemMessage(content=RECEPTIONIST_PROMPT)
        msg = HumanMessage(content=f"ASSISTANT_MESSAGE:\n```\n{text}\n```")
        out = llm.invoke([sys, msg])
        new_text = (out.content or "").strip()
        return new_text or text
    except Exception:
        return text

def say(state: "GraphState", raw: str) -> "GraphState":
    """Append one assistant message, rewritten by the receptionist."""
    text = _as_receptionist(raw)
    state["messages"] = state.get("messages", []) + [text]
    return state


# =========================
# Conversation state types
# =========================
@dataclass
class ConvState:
    messages: List[str] = field(default_factory=list)
    patient_id: Optional[str] = None
    intent: Optional[str] = None
    temp: Dict[str, Any] = field(default_factory=dict)

class GraphState(TypedDict, total=False):
    messages: List[str]
    patient_id: Optional[str]
    intent: Optional[str]
    temp: Dict[str, Any]

def _to_graph_state(s: ConvState) -> GraphState:
    return {
        "messages": list(s.messages),
        "patient_id": s.patient_id,
        "intent": s.intent,
        "temp": dict(s.temp),
    }

def _from_graph_state(gs: GraphState) -> ConvState:
    return ConvState(
        messages=list(gs.get("messages", [])),
        patient_id=gs.get("patient_id"),
        intent=gs.get("intent"),
        temp=dict(gs.get("temp", {})),
    )


# =========================
# Router / intent detection
# =========================
def router_node(state: GraphState) -> GraphState:
    # If UI set an intent via buttons, honor it immediately
    if state.get("intent"):
        return state

    if not state.get("messages"):
        return state

    last = (state["messages"][-1] or "").strip()
    low = last.lower()

    # High-priority intents that should work anytime
    if "cancel" in low and ("appointment" in low or "booking" in low):
        state["intent"] = "CANCEL_APPT"; return state
    if ("different doctor" in low) or ("switch doctor" in low) or ("change doctor" in low):
        state["intent"] = "CHANGE_DOC"; return state
    if ("edit details" in low) or ("change details" in low) or ("update details" in low) \
       or ("edit my" in low and ("phone" in low or "email" in low or "name" in low or "dob" in low)):
        state["intent"] = "EDIT_ID"; return state
    if "edit insurance" in low or ("insurance" in low and ("change" in low or "update" in low)):
        state["intent"] = "CHANGE_INS"; return state

    # If we’re mid-flow awaiting input, keep booking
    awaiting = state.get("temp", {}).get("awaiting")
    if awaiting:
        state["intent"] = state.get("intent") or "BOOK_APPT"
        return state

    # Obvious explicit
    if "book" in low or "appointment" in low:
        state["intent"] = "BOOK_APPT"; return state

    # LLM fallback classification (optional)
    try:
        if ChatOllama is not None:
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
            sys = SystemMessage(content="Classify intent as one of: BOOK_APPT, CHANGE_INS, CHANGE_DOC, EDIT_ID, CANCEL_APPT, VIEW_APPTS, NONE. Answer only the label.")
            out = llm.invoke([sys, HumanMessage(content=last)]).content.strip().upper()
            if "BOOK" in out or "APPT" in out: state["intent"] = "BOOK_APPT"
            elif "INS" in out: state["intent"] = "CHANGE_INS"
            elif "CHANGE_DOC" in out or ("CHANGE" in out and "DOC" in out): state["intent"] = "CHANGE_DOC"
            elif "EDIT_ID" in out or "EDIT" in out: state["intent"] = "EDIT_ID"
            elif "CANCEL" in out: state["intent"] = "CANCEL_APPT"
            elif "VIEW" in out: state["intent"] = "VIEW_APPTS"
            else: state["intent"] = None
        else:
            state["intent"] = None
    except Exception:
        state["intent"] = None

    return state


# =========================
# Identity capture
# =========================
def ask_identity_node(state: GraphState) -> GraphState:
    state.setdefault("temp", {})["awaiting"] = "identity"
    msg = (
        "Before we book, please share your details as:\n\n"
        "**Full Name | DOB (YYYY-MM-DD) | Email | Phone**\n\n"
        "Example: `Alex Kumar | 1992-04-18 | alex@example.com | +91-98765-43210`"
    )
    return say(state, msg)

def capture_identity_node(state: GraphState) -> GraphState:
    last = (state.get("messages") or [""])[-1]
    parts = [p.strip() for p in last.split("|")]
    if len(parts) != 4:
        return say(state, "Please use the format: Full Name | DOB YYYY-MM-DD | Email | Phone")
    full, dob, email, phone = parts
    if "@" not in email:
        return say(state, "That email doesn't look right. Try again (use the `|` separators).")

    first, lastn = (full.split(" ", 1) + [""])[:2]
    existing = get_patient_by_email(email)
    if existing:
        state["patient_id"] = existing.patient_id
        returning = True
        name_for_greet = existing.first_name or first or "there"
    else:
        p = add_patient(first or "Patient", lastn or "", dob or "1990-01-01", phone or "", email, is_returning=False)
        state["patient_id"] = p.patient_id
        returning = False
        name_for_greet = first or p.first_name or "there"

    t = state.setdefault("temp", {})
    t["identity_complete"] = True
    t["awaiting"] = None
    t["identity"] = {
        "full": full, "first": first, "last": lastn, "dob": dob,
        "email": email, "phone": phone, "returning": returning,
        "name_for_greet": name_for_greet
    }
    return say(state, f"Thanks, {first or 'there'}!")


# =========================
# Insurance capture / update
# =========================
def ask_insurance_node(state: GraphState) -> GraphState:
    state.setdefault("temp", {})["awaiting"] = "insurance"
    mode = state["temp"].get("ins_mode") or ("update" if state.get("intent") == "CHANGE_INS" else "initial")
    state["temp"]["ins_mode"] = mode
    msg = (
        "Please provide your insurance as: **Carrier, Member ID, Group**.\n"
        "Example: `CareHealth, ABCD1234, GRP42`\n"
        "If you don't have insurance, reply `skip`."
    )
    return say(state, msg)

def capture_insurance_node(state: GraphState) -> GraphState:
    last = (state.get("messages") or [""])[-1].strip()
    t = state.setdefault("temp", {})
    mode = t.get("ins_mode", "initial")

    if last.lower() == "skip":
        t["insurance_complete"] = True
        t["awaiting"] = None
        return say(state, "No problem. We can add insurance later.")

    parts = [p.strip() for p in last.split(",")]
    if len(parts) < 2:
        return say(state, "Please send: Carrier, Member ID, Group (comma-separated). Or `skip`.")

    carrier = parts[0]
    member = parts[1]
    group = parts[2] if len(parts) >= 3 else None
    pid = state.get("patient_id") or "P1001"
    try:
        active = ins.get_active_insurance(pid)
        if mode == "update":
            ins.change_insurance(pid, carrier, member, group, None, actor="assistant")
            state = say(state, f"Updated insurance to {carrier} (****{member[-4:]}).")
        else:
            if active:
                ins.change_insurance(pid, carrier, member, group, None, actor="assistant")
                state = say(state, f"Updated insurance to {carrier} (****{member[-4:]}).")
            else:
                ins.create_initial_insurance(pid, carrier, member, group, None)
                state = say(state, f"Saved insurance with {carrier} (****{member[-4:]}).")
        t["insurance_complete"] = True
        t["awaiting"] = None
        if state.get("intent") == "CHANGE_INS":
            state["intent"] = None
    except Exception as e:
        state = say(state, f"Could not save insurance: {e}")
    return state


# =========================
# Doctor selection & slot pick
# =========================
def ask_doctor_node(state: GraphState) -> GraphState:
    ddf = docs.list_doctors_df()
    if ddf is None or ddf.empty:
        return say(state, "No doctors available yet. Ask an admin to add doctors in the Admin tab.")
    choices = [
        f"{i+1}. {row['name']} — {row['specialty']} ({row['location_id']})"
        for i, (_, row) in enumerate(ddf.reset_index(drop=True).iterrows())
    ]
    state.setdefault("temp", {})["doctor_choices"] = ddf.to_dict(orient="records")
    state["temp"]["desired"] = state["temp"].get("desired", {})
    state["temp"]["awaiting"] = "doctor"
    return say(state, "Choose a doctor by number:\n" + "\n".join(choices))

def ask_slot_node(state: GraphState) -> GraphState:
    desired = state.get("temp", {}).get("desired", {})
    doctor_id = desired.get("doctor_id")
    if not doctor_id:
        return state

    # duration: 60 for new, 30 for returning
    identity = state.get("temp", {}).get("identity", {})
    returning = bool(identity.get("returning"))
    duration = 30 if returning else 60

    # Ask scheduler for starts that already have enough contiguous slots
    slots = sch.find_slots(doctor_id=doctor_id, limit=8, duration_minutes=duration)
    if not slots:
        state["messages"] = state.get("messages", []) + [
            "I'm so sorry — I couldn't find openings that fit this visit length with that doctor right now. "
            "You can pick a different doctor, or try again later."
        ]
        # Let user re-pick doctor
        state["temp"]["desired"].pop("doctor_id", None)
        return state

    pretty = "\n".join(f"{i+1}. {s.start[0:16].replace('T',' ')}" for i, s in enumerate(slots))
    state["temp"]["slot_cache"] = [(s.start, s.end) for s in slots]
    state["temp"]["awaiting"] = "slot"
    state["messages"] = state.get("messages", []) + [
        "Here are the next available times. Reply with a number:\n" + pretty
    ]
    return state


def try_book_node(state: GraphState) -> GraphState:
    t = state.setdefault("temp", {})
    desired = t.get("desired", {})
    idx = desired.get("slot_index")
    cache = t.get("slot_cache", [])
    doctor_id = desired.get("doctor_id")
    if not (doctor_id and isinstance(idx, int) and 1 <= idx <= len(cache)):
        return say(state, "Invalid choice. Please pick a valid number.")

    start, _ = cache[idx-1]
    pid = state.get("patient_id") or "P1001"

    # decide duration: 60 for new, 30 for returning
    identity = t.get("identity", {})
    returning = bool(identity.get("returning"))
    duration = 30 if returning else 60

    active = ins.get_active_insurance(pid)
    try:
        appt = sch.book_appointment_with_duration(
            pid, doctor_id, start, duration,
            active.insurance_id if active else None
        )
    except Exception as e:
        t["awaiting"] = "slot"
        return say(state, f"Couldn't book that slot: {e}. Please choose another.")

    # greet & confirm
    name = (identity.get("first") or "there")
    when = appt['start'].replace('T',' ')
    state = say(state, f"🎉 Hi {name}! Your appointment is booked.")
    state = say(state, f"Appointment **{appt['appointment_id']}** on **{when}** with **{doctor_id}**.")
    state = say(state, f"(Duration: {duration} minutes — {'returning' if returning else 'new'} patient.)")

    # queue reminders + send confirmation to the captured email
    try:
        p = get_patient_by_id(pid)
        ddf = docs.list_doctors_df()
        cdf = docs.list_clinics_df()
        doc_row = ddf[ddf["doctor_id"] == doctor_id].iloc[0].to_dict() if (ddf is not None and not ddf.empty) else {"name": doctor_id, "location_id": ""}
        loc_name = ""
        if doc_row.get("location_id") and cdf is not None and not cdf.empty:
            match = cdf[cdf["location_id"] == doc_row["location_id"]]
            if not match.empty:
                loc_name = match.iloc[0]["name"]

        import tools.notify as notify
        notify.schedule_reminders_for(appt, patient_email=p.email if p else "")
        conf = notify.send_confirmation(
            p.email if p else "patient@example.com",
            f"{p.first_name} {p.last_name}" if p else "Patient",
            appt, doc_row.get("name","Doctor"), loc_name, attach_intake=True
        )
        if conf.get("status") == "sent":
            state = say(state, "A confirmation email has been sent to your email.")
        else:
            state = say(state, "Confirmation saved as an email file you can download.")
        t["last_confirmation_eml"] = conf.get("path")
    except Exception:
        state = say(state, "Booked, but could not queue reminders or confirmation email.")

    # cleanup / stop
    t.pop("slot_cache", None); t.pop("doctor_choices", None); t.pop("desired", None)
    t["awaiting"] = None
    state["intent"] = None
    return state


# =========================
# Cancel flow
# =========================
def ask_cancel_node(state: GraphState) -> GraphState:
    """List upcoming appointments for this patient and ask which one to cancel."""
    pid = state.get("patient_id")
    if not pid:
        state.setdefault("temp", {})["awaiting"] = "identity"
        state["intent"] = "BOOK_APPT"  # route into identity flow
        return say(state, "I need your details first. Please share: Full Name | DOB | Email | Phone")

    from datetime import datetime as _dt
    appts = sch.list_appointments(patient_id=pid, limit=500) or []
    now = _dt.now()
    upcoming = [a for a in appts if a.get("status") == "BOOKED" and _dt.fromisoformat(a["start"]) >= now]

    if not upcoming:
        state["intent"] = None
        return say(state, "You have no upcoming appointments to cancel.")

    # map doctor ids to names
    ddf = docs.list_doctors_df()
    dmap = {}
    if ddf is not None and not ddf.empty:
        for _, r in ddf.iterrows():
            dmap[r["doctor_id"]] = r["name"]

    lines = []
    for i, a in enumerate(upcoming, start=1):
        when = a["start"].replace("T", " ")
        docn = dmap.get(a["doctor_id"], a["doctor_id"])
        lines.append(f"{i}. {when} — {docn} (ID {a['appointment_id']})")

    t = state.setdefault("temp", {})
    t["cancel_choices"] = upcoming
    t["awaiting"] = "cancel_select"
    return say(state, "Which appointment do you want to cancel? Reply with a number:\n" + "\n".join(lines))

def do_cancel_node(state: GraphState) -> GraphState:
    """Cancel the chosen appointment, free slots, and stop reminders."""
    t = state.setdefault("temp", {})
    idx = int(t.get("cancel_index") or 0)
    choices = t.get("cancel_choices") or []
    if not (1 <= idx <= len(choices)):
        return say(state, "Please reply with a valid number from the list.")

    appt = choices[idx - 1]
    appt_id = appt["appointment_id"]
    try:
        sch.cancel_appointment(appt_id, free_slots=True)
        try:
            from tools.notify import cancel_reminders_for
            cancel_reminders_for(appt_id)
        except Exception:
            pass
        when = appt["start"].replace("T", " ")
        state = say(state, f"✅ Your appointment {appt_id} on {when} has been cancelled.")
    except Exception as e:
        state = say(state, f"Couldn't cancel: {e}")

    # cleanup & finish
    t.pop("cancel_choices", None)
    t.pop("cancel_index", None)
    t["awaiting"] = None
    state["intent"] = None
    return state


# =========================
# Edit identity (profile)
# =========================
def ask_edit_identity_node(state: GraphState) -> GraphState:
    state.setdefault("temp", {})["awaiting"] = "identity_edit"
    msg = (
        "What would you like to update?\n"
        "Please resend your details as:\n\n"
        "**Full Name | DOB (YYYY-MM-DD) | Email | Phone**\n\n"
        "Example: `Alex Kumar | 1992-04-18 | alex@example.com | +91-98765-43210`"
    )
    return say(state, msg)

def capture_edit_identity_node(state: GraphState) -> GraphState:
    last = (state.get("messages") or [""])[-1]
    parts = [p.strip() for p in last.split("|")]
    if len(parts) != 4:
        return say(state, "Please use the format: Full Name | DOB YYYY-MM-DD | Email | Phone")

    full, dob, email, phone = parts
    first, lastn = (full.split(" ", 1) + [""])[:2]
    pid = state.get("patient_id")

    if not pid:
        # If we don't have a patient yet, treat this like initial identity capture
        state.setdefault("temp", {})["awaiting"] = "identity"
        state["intent"] = "BOOK_APPT"
        return say(state, "Thanks! We'll use these details for your bookings.")

    try:
        update_patient_fields(
            pid,
            first_name=first or None,
            last_name=lastn or None,
            dob=dob or None,
            phone=phone or None,
            email=email or None,
        )
        p = get_patient_by_id(pid)
        name = (p.first_name if p else first) or "there"
        t = state.setdefault("temp", {})
        t["identity"] = {
            "full": full, "first": first, "last": lastn, "dob": dob,
            "email": email, "phone": phone, "returning": True, "name_for_greet": name
        }
        t["awaiting"] = None
        state["intent"] = None
        return say(state, f"✅ Updated your details, {name}.")
    except Exception as e:
        return say(state, f"Couldn't update details: {e}")


# =========================
# Change / pick different doctor
# =========================
def reset_doctor_node(state: GraphState) -> GraphState:
    t = state.setdefault("temp", {})
    desired = t.get("desired", {})
    desired.pop("doctor_id", None)
    desired.pop("slot_index", None)
    t["desired"] = desired
    t["awaiting"] = None
    return ask_doctor_node(state)


# =========================
# View upcoming appointments
# =========================
def view_appts_node(state: GraphState) -> GraphState:
    pid = state.get("patient_id")
    if not pid:
        state.setdefault("temp", {})["awaiting"] = "identity"
        state["intent"] = "BOOK_APPT"
        return say(state, "I need your details first. Please share: **Full Name | DOB | Email | Phone**")

    from datetime import datetime as _dt
    appts = sch.list_appointments(patient_id=pid, limit=200) or []
    upcoming = [a for a in appts if a.get("status") == "BOOKED" and _dt.fromisoformat(a["start"]) >= _dt.now()]
    if not upcoming:
        state["intent"] = None
        return say(state, "You have no upcoming appointments.")

    ddf = docs.list_doctors_df()
    dmap = {r["doctor_id"]: r["name"] for _, r in ddf.iterrows()} if (ddf is not None and not ddf.empty) else {}
    lines = []
    for a in upcoming:
        when = a["start"].replace("T"," ")
        lines.append(f"- {when} — {dmap.get(a['doctor_id'], a['doctor_id'])} (ID {a['appointment_id']})")
    state["intent"] = None
    return say(state, "Your upcoming appointments:\n" + "\n".join(lines))


# =========================
# Change insurance (force update)
# =========================
def change_insurance_node(state: GraphState) -> GraphState:
    t = state.setdefault("temp", {})
    t["ins_mode"] = "update"
    return ask_insurance_node(state)


# =========================
# Fallback help
# =========================
def fallback_node(state: GraphState) -> GraphState:
    return say(state,
        "I can help you book or manage appointments:\n"
        "• Type **book appointment** or use the buttons.\n"
        "• To change details: **edit details**\n"
        "• To change insurer: **edit insurance**\n"
        "• To cancel: **cancel appointment**\n"
        "• To switch doctor: **different doctor**\n"
        "• To view bookings: **view my appointments**"
    )


# =========================
# Graph wiring
# =========================
def build_graph():
    g = StateGraph(GraphState)

    # nodes
    g.add_node("router", router_node)

    g.add_node("ask_identity", ask_identity_node)
    g.add_node("capture_identity", capture_identity_node)

    g.add_node("ask_insurance", ask_insurance_node)
    g.add_node("capture_insurance", capture_insurance_node)

    g.add_node("ask_doctor", ask_doctor_node)
    g.add_node("ask_slot", ask_slot_node)
    g.add_node("try_book", try_book_node)

    g.add_node("ask_cancel", ask_cancel_node)
    g.add_node("do_cancel", do_cancel_node)

    g.add_node("ask_edit_identity", ask_edit_identity_node)
    g.add_node("capture_edit_identity", capture_edit_identity_node)

    g.add_node("reset_doctor", reset_doctor_node)
    g.add_node("view_appts", view_appts_node)

    g.add_node("change_insurance", change_insurance_node)
    g.add_node("fallback", fallback_node)

    # routing logic
    def route_after_router(state: GraphState):
        intent = state.get("intent")
        t = state.get("temp", {})
        desired = t.get("desired", {})

        # Booking flow
        if intent == "BOOK_APPT":
            if not t.get("identity_complete"):
                return "capture_identity" if t.get("awaiting") == "identity" else "ask_identity"
            if not t.get("insurance_complete"):
                return "capture_insurance" if t.get("awaiting") == "insurance" else "ask_insurance"
            if not desired.get("doctor_id"):
                return "ask_doctor"
            if not desired.get("slot_index"):
                return "ask_slot"
            return "try_book"

        # Insurance edit flow
        if intent == "CHANGE_INS":
            return "capture_insurance" if t.get("awaiting") == "insurance" else "ask_insurance"

        # Cancel flow
        if intent == "CANCEL_APPT":
            return "do_cancel" if t.get("awaiting") == "cancel_select" else "ask_cancel"

        # Edit personal details
        if intent == "EDIT_ID":
            return "capture_edit_identity" if t.get("awaiting") == "identity_edit" else "ask_edit_identity"

        # Change doctor mid-flow
        if intent == "CHANGE_DOC":
            return "reset_doctor"

        # View upcoming appointments
        if intent == "VIEW_APPTS":
            return "view_appts"

        # Default help
        return "fallback"

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "ask_identity": "ask_identity",
            "capture_identity": "capture_identity",
            "ask_insurance": "ask_insurance",
            "capture_insurance": "capture_insurance",
            "ask_doctor": "ask_doctor",
            "ask_slot": "ask_slot",
            "try_book": "try_book",
            "ask_cancel": "ask_cancel",
            "do_cancel": "do_cancel",
            "ask_edit_identity": "ask_edit_identity",
            "capture_edit_identity": "capture_edit_identity",
            "reset_doctor": "reset_doctor",
            "view_appts": "view_appts",
            "fallback": "fallback",
        },
    )

    # Stop after each step; wait for next user turn
    for node in [
        "ask_identity",
        "capture_identity",
        "ask_insurance",
        "capture_insurance",
        "ask_doctor",
        "ask_slot",
        "try_book",
        "ask_cancel",
        "do_cancel",
        "ask_edit_identity",
        "capture_edit_identity",
        "reset_doctor",
        "view_appts",
        "fallback",
    ]:
        g.add_edge(node, END)

    app = g.compile()

    class Wrapper:
        def invoke(self, conv: ConvState) -> GraphState:
            return app.invoke(_to_graph_state(conv))

    return Wrapper()
