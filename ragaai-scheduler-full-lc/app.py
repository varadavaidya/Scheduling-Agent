# app.py
import os
import io
import sqlite3
from pathlib import Path as _Path

import streamlit as st
from dotenv import load_dotenv

# ---- Agents (LangGraph) ----
try:
    from agents.graph_lc import build_graph, ConvState  # LangGraph version
except Exception:
    from agents.graph import build_graph, ConvState      # Fallback (simple agent)

# ---- Patients bootstrap (checked_out column) ----
from tools.patient import ensure_checked_out_column as _patient_boot
from tools.scheduler import _ensure_tables as _sched_boot
_sched_boot()


# ---- Env & Page ----
load_dotenv()
_patient_boot()
st.set_page_config(page_title="RagaAI Scheduler", page_icon="🩺", layout="wide")

# ---- Force Light Theme (no dark switch) ----
pal = {
    "bg": "#f8fafc", "panel": "#ffffff", "card": "#ffffff",
    "text": "#0f172a", "muted": "#334155", "accent": "#0ea5e9",
    "accentSoft": "rgba(14,165,233,0.12)", "shadow": "0 2px 10px rgba(0,0,0,0.08)"
}
st.markdown(f"""
<style>
:root {{
  --bg: {pal['bg']}; --panel: {pal['panel']}; --card: {pal['card']};
  --text: {pal['text']}; --muted: {pal['muted']}; --accent: {pal['accent']};
  --accent-soft: {pal['accentSoft']}; --shadow: {pal['shadow']};
}}
html, body, [data-testid="stAppViewContainer"] {{ background: var(--bg); color: var(--text); }}
section[data-testid="stSidebar"] {{ background: var(--panel); }}
div[data-testid="stChatMessage"] {{
  background: var(--card); border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px; box-shadow: var(--shadow);
}}
.stButton>button {{
  background: linear-gradient(90deg, var(--accent-soft), transparent);
  color: var(--text); border: 1px solid var(--accent); border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

st.title("🩺 AI Scheduling Agent — RagaAI Case Study")

# ---- Session init ----
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "state" not in st.session_state:
    st.session_state.state = ConvState(messages=[], patient_id=None, intent=None, temp={})
if "history" not in st.session_state:
    # store both user and assistant messages so chat replays correctly
    st.session_state.history = []

# Safety: if earlier code stored a dict instead of ConvState, coerce it
if isinstance(st.session_state.get("state"), dict):
    d = st.session_state.state
    st.session_state.state = ConvState(
        messages=list(d.get("messages", [])),
        patient_id=d.get("patient_id"),
        intent=d.get("intent"),
        temp=dict(d.get("temp", {})),
    )

# Sidebar helpers
with st.sidebar:
    st.markdown("### Tools")
    if st.button("↩️ Reset chat"):
        st.session_state.state = ConvState(messages=[], patient_id=None, intent=None, temp={})
        st.session_state.history = []
        st.rerun()

# =========================
# Tabs
# =========================
tab_patient, tab_admin = st.tabs(["🧑 Patient", "👨‍⚕️ Doctor/Admin"])

# =========================
# Patient Tab (Chatbot)
# =========================
with tab_patient:
    # Always-visible guidance
    st.info(
        "Quick start:\n"
        "• Click a button below **or** type naturally (e.g., *book appointment*, *different doctor*, *edit details*).\n"
        "• Identity format: `Full Name | 1995-01-01 | you@example.com | +91-...`\n"
        "• Insurance format: `Carrier, MEMBER1234, GRP42` (or type `skip`).\n"
        "• When you see a **numbered list**, reply with the **number**."
    )

    # --------------------------
    # Quick actions (no typing)
    # --------------------------
    action = None
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🩺 Book appointment"):
            action = "BOOK_APPT"
        if st.button("🔁 Different doctor"):
            action = "CHANGE_DOC"
    with c2:
        if st.button("📝 Edit details"):
            action = "EDIT_ID"
        if st.button("🏷️ Edit insurance"):
            action = "CHANGE_INS"
    with c3:
        if st.button("❌ Cancel appointment"):
            action = "CANCEL_APPT"
        if st.button("📅 View my appointments"):
            action = "VIEW_APPTS"

    # Replay full chat history
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    # If a quick action was chosen, trigger the graph without needing a user message
    if action:
        after_user_len = len(st.session_state.state.messages or [])
        st.session_state.state.intent = action

        # small prep when changing doctor: clear previous choice
        if action == "CHANGE_DOC":
            t = st.session_state.state.temp or {}
            desired = t.get("desired", {})
            desired.pop("doctor_id", None)
            desired.pop("slot_index", None)
            t["desired"] = desired
            t["awaiting"] = None
            st.session_state.state.temp = t

        # invoke graph
        result = st.session_state.graph.invoke(st.session_state.state)
        if isinstance(result, dict):
            st.session_state.state.messages = result.get("messages", []) or st.session_state.state.messages
            st.session_state.state.patient_id = result.get("patient_id", st.session_state.state.patient_id)
            st.session_state.state.intent = result.get("intent", None)
            st.session_state.state.temp = result.get("temp", st.session_state.state.temp)
        else:
            st.session_state.state = result

        # render only new assistant replies
        msgs = st.session_state.state.messages or []
        new_assistant = msgs[after_user_len:]
        for m in new_assistant:
            with st.chat_message("assistant"):
                st.markdown(m)
            st.session_state.history.append(("assistant", m))

    # --------------------------
    # Freeform chat input
    # --------------------------
    user_inp = st.chat_input("Type here (or use the buttons above)…")
    if user_inp:
        # show and keep user's message (so it replays on rerun)
        with st.chat_message("user"):
            st.markdown(user_inp)
        st.session_state.history.append(("user", user_inp))

        # Add user's text so router can read it
        st.session_state.state.messages.append(user_inp)

        # boundary to avoid echo of user as assistant
        after_user_len = len(st.session_state.state.messages)

        # numeric replies for doctor / slot / cancel
        stripped = user_inp.strip()
        if stripped.isdigit():
            idx = int(stripped)
            awaiting = (st.session_state.state.temp or {}).get("awaiting")
            if awaiting == "doctor" and st.session_state.state.temp.get("doctor_choices"):
                choices = st.session_state.state.temp.get("doctor_choices", [])
                if 1 <= idx <= len(choices):
                    chosen = choices[idx - 1]
                    desired = st.session_state.state.temp.get("desired", {})
                    desired["doctor_id"] = chosen["doctor_id"]
                    st.session_state.state.temp["desired"] = desired
                    st.toast(f"Selected doctor: {chosen['name']} ({chosen.get('specialty','')})")
                else:
                    st.toast("Invalid doctor choice", icon="⚠️")
            elif awaiting == "slot":
                desired = st.session_state.state.temp.get("desired", {})
                desired["slot_index"] = idx
                st.session_state.state.temp["desired"] = desired
            elif awaiting == "cancel_select":
                st.session_state.state.temp["cancel_index"] = idx

        # run graph once
        result = st.session_state.graph.invoke(st.session_state.state)

        # merge back
        if isinstance(result, dict):
            st.session_state.state.messages = result.get("messages", []) or st.session_state.state.messages
            st.session_state.state.patient_id = result.get("patient_id", st.session_state.state.patient_id)
            st.session_state.state.intent = result.get("intent", None)
            st.session_state.state.temp = result.get("temp", st.session_state.state.temp)
        else:
            st.session_state.state = result

        # render only new assistant messages (no echo)
        msgs = st.session_state.state.messages or []
        new_assistant = msgs[after_user_len:]
        for m in new_assistant:
            with st.chat_message("assistant"):
                st.markdown(m)
            st.session_state.history.append(("assistant", m))

        # confirmation .eml download if available
        eml_path = (st.session_state.state.temp or {}).get("last_confirmation_eml")
        if eml_path and os.path.exists(eml_path):
            try:
                with open(eml_path, "rb") as f:
                    st.download_button(
                        "📩 Download confirmation email (.eml)",
                        f.read(),
                        file_name=os.path.basename(eml_path),
                        mime="message/rfc822"
                    )
            except Exception:
                pass

# =========================
# Admin Tab
# =========================
with tab_admin:
    # ---- Register Walk-in Patient ----
    st.subheader("Register Walk-in Patient")
    with st.form("walkin_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            first_name = st.text_input("First Name *")
            dob = st.text_input("Date of Birth * (YYYY-MM-DD)")
            phone = st.text_input("Phone *", value="+91-")
        with c2:
            last_name = st.text_input("Last Name *")
            email = st.text_input("Email *")
            is_returning = st.checkbox("Returning patient?", value=False)

        st.markdown("**Optional — Initial Insurance**")
        c3, c4, c5 = st.columns(3)
        with c3:
            carrier = st.text_input("Carrier")
        with c4:
            member_id = st.text_input("Member ID")
        with c5:
            group_number = st.text_input("Group Number")
        eff = st.text_input("Effective From (YYYY-MM-DD)")

        # --- NEW: choose doctor & immediate booking ---
        st.markdown("**Book appointment now (optional)**")
        book_now = st.checkbox("Book appointment now", value=True)

        chosen_doc_id = None
        chosen_start_iso = None

        if book_now:
            from tools.doctors import list_doctors_df, list_clinics_df
            from tools import scheduler as sch
            ddf = list_doctors_df()
            if ddf is None or ddf.empty:
                st.info("No doctors available yet. Add one in 'Manage Doctors'.")
            else:
                # doctor pick
                ddf = ddf.reset_index(drop=True)
                doc_labels = [f"{r['name']} — {r['specialty']} ({r['doctor_id']})" for _, r in ddf.iterrows()]
                doc_map = {doc_labels[i]: ddf.iloc[i]['doctor_id'] for i in range(len(doc_labels))}
                chosen_label = st.selectbox("Doctor", options=doc_labels)
                chosen_doc_id = doc_map[chosen_label]

                # compute eligible start times (contiguous blocks)
                need_blocks = 1 if is_returning else 2   # 30m vs 60m
                raw_slots = sch.find_slots(doctor_id=chosen_doc_id, limit=200) or []
                raw_slots = sorted(raw_slots, key=lambda s: s.start)

                eligible_starts = []
                for i in range(0, max(0, len(raw_slots) - (need_blocks - 1))):
                    ok = True
                    prev_end = raw_slots[i].end
                    # check contiguity for k-1 next slots
                    for j in range(1, need_blocks):
                        s_next = raw_slots[i + j]
                        if s_next.start != prev_end:
                            ok = False
                            break
                        prev_end = s_next.end
                    if ok:
                        eligible_starts.append(raw_slots[i].start)

                if eligible_starts:
                    pretty = [s.replace("T", " ") for s in eligible_starts]
                    pretty_map = {pretty[i]: eligible_starts[i] for i in range(len(pretty))}
                    pretty_choice = st.selectbox("Available start times", options=pretty)
                    chosen_start_iso = pretty_map[pretty_choice]
                else:
                    st.warning(
                    "No suitable contiguous slot found for the selected doctor. "
                    "Tip: ensure their schedule has contiguous 30-min blocks (new patients need 60 min)."
                    )

        submitted = st.form_submit_button("Add Patient" + (" & Book" if book_now else ""))

    if submitted:
        errs = []
        if not first_name.strip(): errs.append("First name required")
        if not last_name.strip(): errs.append("Last name required")
        if not dob.strip(): errs.append("DOB required")
        if not phone.strip(): errs.append("Phone required")
        if not email.strip(): errs.append("Email required")
        if book_now and (not chosen_doc_id or not chosen_start_iso):
            errs.append("Pick a doctor and a start time for booking")

        if errs:
            st.error("; ".join(errs))
        else:
            from tools.patient import add_patient, get_patient_by_id
            from tools.insurance import create_initial_insurance
            p = add_patient(first_name, last_name, dob, phone, email, is_returning)
            st.success(f"Patient created: {p.patient_id} — {p.first_name} {p.last_name}")
            # optional initial insurance
            if carrier and member_id:
                try:
                    res = create_initial_insurance(p.patient_id, carrier, member_id, group_number or None, eff or None)
                    st.info(f"Initial insurance set: {res['carrier']} (ID ending {res['member_id'][-4:]})")
                except Exception as e:
                    st.warning(f"Patient created but insurance failed: {e}")

            # immediate booking (optional)
            if book_now and chosen_doc_id and chosen_start_iso:
                from tools import scheduler as sch
                from tools.insurance import get_active_insurance
                from tools.doctors import list_doctors_df, list_clinics_df
                import tools.notify as notify

                # duration rule: 60m new, 30m returning
                duration = 30 if is_returning else 60
                try:
                    active = get_active_insurance(p.patient_id)
                    appt = sch.book_appointment_with_duration(
                        p.patient_id, chosen_doc_id, chosen_start_iso, duration,
                        active.insurance_id if active else None
                    )
                    # labels for email
                    ddf = list_doctors_df(); cdf = list_clinics_df()
                    doc_row = ddf[ddf["doctor_id"] == chosen_doc_id].iloc[0].to_dict() if ddf is not None and not ddf.empty else {"name": chosen_doc_id, "location_id": ""}
                    loc_name = ""
                    if doc_row.get("location_id") and cdf is not None and not cdf.empty:
                        m = cdf[cdf["location_id"] == doc_row["location_id"]]
                        if not m.empty: loc_name = m.iloc[0]["name"]

                    # reminders + confirmation (attach intake)
                    notify.schedule_reminders_for(appt, patient_email=p.email or "")
                    sent = notify.send_confirmation(
                        p.email or "patient@example.com",
                        f"{p.first_name} {p.last_name}",
                        appt, doc_row.get("name","Doctor"), loc_name,
                        attach_intake=True
                    )

                    when = appt['start'].replace("T"," ")
                    st.success(
                        f"📅 Booked {when} with {doc_row.get('name','Doctor')} "
                        f"({duration} min, {'returning' if is_returning else 'new'} patient)."
                    )
                    if sent.get("status") == "sent":
                        st.info("Confirmation email sent to the patient.")
                    else:
                        st.info("Confirmation saved as .eml (no SMTP configured). See Admin → Reminders/Outbox.")
                except Exception as e:
                    st.error(f"Booking failed: {e}")


    # ---- Manage Doctors / Locations ----
    st.divider()
    st.subheader("Manage Doctors")

    from tools.doctors import (
        list_doctors_df, list_clinics_df,
        add_location, add_doctor, generate_slots_for_doctor,
        delete_doctor as _delete_doctor,
    )

    # ========= Actions first (Add / Delete) =========

    # Add Location
    st.markdown("### Add Location")
    loc_name = st.text_input("Location name", key="locname")
    if st.button("Add Location"):
        if not loc_name.strip():
            st.error("Please enter a location name.")
        else:
            loc = add_location(loc_name.strip())
            st.success(f"Added location {loc.location_id}: {loc.name}")
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

    # Add Doctor
    st.markdown("### Add Doctor")
    doc_name = st.text_input("Doctor Name", key="docname")
    doc_spec = st.text_input("Specialty", key="docspec", value="General Medicine")

    _cdf_now = list_clinics_df()
    if _cdf_now is None or _cdf_now.empty:
        st.warning("Add a location first.")
    else:
        loc_choice = st.selectbox(
            "Location",
            options=_cdf_now["location_id"].tolist(),
            format_func=lambda lid: _cdf_now.loc[_cdf_now["location_id"] == lid, "name"].iat[0]
        )
    
    # --- Slots Inspector / Debug ---
    st.markdown("### 🔎 Slots Inspector (Debug)")
    from pathlib import Path as _Path
    import sqlite3, pandas as _pd
    from tools.doctors import list_doctors_df, generate_slots_for_doctor

    # Local DB path (self-contained for this block)
    DATA_DIR = _Path(__file__).resolve().parents[0] / "data"
    DB_PATH = DATA_DIR / "store.db"

    _ddf_latest = list_doctors_df()
    if _ddf_latest is not None and not _ddf_latest.empty:
        # build “Doctor — Name (Specialty)” labels
        _labels = [f"{row['doctor_id']} — {row['name']} ({row['specialty']})"
                for _, row in _ddf_latest.iterrows()]
        _id_map = { _labels[i]: _ddf_latest.iloc[i]['doctor_id'] for i in range(len(_labels)) }

        _picked = st.selectbox("Doctor", options=_labels, key="slots_debug_doc")

        def _count_slots(doctor_id: str) -> int:
            with sqlite3.connect(DB_PATH) as con:
                cur = con.cursor()
                cur.execute("SELECT COUNT(*) FROM slots WHERE doctor_id=?", (doctor_id,))
                return cur.fetchone()[0]

        def _list_slots_df(doctor_id: str) -> _pd.DataFrame:
            with sqlite3.connect(DB_PATH) as con:
                return _pd.read_sql_query(
                    "SELECT slot_id, doctor_id, start, end, is_booked "
                    "FROM slots WHERE doctor_id=? ORDER BY start",
                    con, params=(doctor_id,)
                )

        if _picked:
            _doc_id = _id_map[_picked]
            st.write(f"Total slots in DB for this doctor: **{_count_slots(_doc_id)}**")

            if st.button("Show first 100 slots"):
                _df = _list_slots_df(_doc_id)
                st.dataframe(_df.head(100))

            colA, colB, colC = st.columns(3)
            with colA:
                _days = st.number_input("Generate for next N days", min_value=1, max_value=45, value=7, step=1, key="dbg_days")
            with colB:
                _start = st.text_input("Start (HH:MM)", value="09:00", key="dbg_start")
            with colC:
                _end   = st.text_input("End (HH:MM)", value="17:00", key="dbg_end")

            if st.button("➕ Generate by days"):
                try:
                    _created = generate_slots_for_doctor(
                        _doc_id,
                        days=int(_days),
                        start_time=_start,
                        end_time=_end,
                        step_minutes=30
                    )
                    st.success(f"Created {_created} slot(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate: {e}")

            if st.button("🗑️ Delete ALL slots for this doctor"):
                try:
                    with sqlite3.connect(DB_PATH) as con:
                        cur = con.cursor()
                        cur.execute("DELETE FROM slots WHERE doctor_id=?", (_doc_id,))
                        con.commit()
                    st.success("All slots deleted for this doctor.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
    else:
        st.caption("No doctors yet.")


    # --- NEW: generation mode & schedule controls ---
        gen_mode = st.radio(
            "Generate availability by",
            ["Days", "Exact slots"],
            horizontal=True,
            key="gen_mode"
        )
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            start_time = st.text_input("Start time (HH:MM)", value="09:00", key="slot_start")
        with c2:
            end_time = st.text_input("End time (HH:MM)", value="17:00", key="slot_end")
        with c3:
            step_minutes = st.number_input("Slot size (min)", min_value=15, max_value=60, value=30, step=15, key="slot_step",
                                       help="Keep 30 min to allow 60-min windows for new patients (2 blocks).")

        if gen_mode == "Days":
            gen_days = st.number_input("Days to generate", min_value=1, max_value=30, value=10, step=1, key="gen_days")
        else:
            gen_slots = st.number_input("Exact number of 30-min slots", min_value=2, max_value=500, value=40, step=2, key="gen_slots",
                                    help="New patients need 60 min (2 contiguous slots).")

        if st.button("Add Doctor"):
            if not doc_name.strip():
                st.error("Please enter doctor name.")
            else:
                d = add_doctor(doc_name.strip(), doc_spec.strip(), loc_choice)

            # Call generator in the chosen mode. Fallback gracefully if slots_count not supported yet.
                created = 0
                try:
                    if gen_mode == "Days":
                        created = generate_slots_for_doctor(
                            d.doctor_id,
                            days=int(gen_days),
                            start_time=start_time,
                            end_time=end_time,
                            step_minutes=int(step_minutes),
                        )
                    else:
                        created = generate_slots_for_doctor(
                            d.doctor_id,
                            slots_count=int(gen_slots),
                            start_time=start_time,
                            end_time=end_time,
                            step_minutes=int(step_minutes),
                        )
                except TypeError:
                    # Legacy generator that only supports `days`
                    if gen_mode == "Days":
                        created = generate_slots_for_doctor(d.doctor_id, days=int(gen_days))
                        st.warning("Using legacy slot generator (days-only). Consider updating tools/doctors.py to support slots_count/start/end/step.")
                    else:
                        st.error("Your generate_slots_for_doctor() doesn’t support slots_count yet. Update tools/doctors.py per earlier instructions.")
                        created = 0

                st.success(
                    f"Added {d.name} ({d.doctor_id}) at {loc_choice} with {created} slot(s). "
                    "Patients can now see & book."
                )
            # refresh UI
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass


    # Delete Doctor
    st.markdown("### Delete Doctor")
    _ddf_for_del = list_doctors_df()  # fetch fresh for the picker
    if _ddf_for_del is not None and not _ddf_for_del.empty:
        labels = [
            f"{row['doctor_id']} — {row['name']} ({row.get('specialty','')})"
            for _, row in _ddf_for_del.iterrows()
        ]
        id_map = {labels[i]: _ddf_for_del.iloc[i]['doctor_id'] for i in range(len(labels))}
        chosen_label = st.selectbox("Choose doctor to remove", options=labels, key="doc_del_label")
        cascade_doc = st.checkbox("Cancel future appts & delete future slots (cascade)", value=False, key="doc_del_cascade")
        if st.button("🗑️ Delete Doctor", type="primary"):
            try:
                res = _delete_doctor(id_map[chosen_label], cascade=cascade_doc)
                st.success(f"Doctor deleted. Summary: {res}")

                # 🔄 Immediate in-pass refresh of the “Doctors” table
                st.session_state["_force_doc_refresh"] = True

                # Clear cache & hard rerun so every panel is fresh
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Failed to delete doctor: {e}")
    else:
        st.info("No doctors to delete.")

    # ========= Then render the current lists (after any action) =========

    cols = st.columns([3, 2])

    with cols[0]:
        st.markdown("**Doctors**")
        ddf = list_doctors_df()  # re-fetch after potential actions
        if st.session_state.get("_force_doc_refresh"):
            # just to be explicit, re-load once more
            ddf = list_doctors_df()
            st.session_state["_force_doc_refresh"] = False
        if ddf is not None and not ddf.empty:
            st.dataframe(ddf, use_container_width=True)
        else:
            st.info("No doctors yet. Add one above.")

    with cols[1]:
        st.markdown("**Locations**")
        cdf = list_clinics_df()  # re-fetch after potential actions
        if cdf is not None and not cdf.empty:
            st.dataframe(cdf, use_container_width=True)
        else:
            st.info("No locations yet. Add one above.")


    # ---- Appointments Dashboard ----
    st.divider()
    st.subheader("Appointments Dashboard")
    from tools.scheduler import list_appointments
    appts = list_appointments(limit=200)
    if appts:
        import pandas as _pd
        st.dataframe(_pd.DataFrame(appts), use_container_width=True)
    else:
        st.info("No appointments yet.")

    # ---- Reminders ----
    st.divider()
    st.subheader("📬 Reminders")
    from tools.notify import due_reminders, mark_sent, send_reminder, resend_confirmation, send_reminder_now
    from tools.scheduler import list_appointments as _list_appts
    from tools.patient import get_patient_by_id
    from tools.doctors import list_doctors_df as _list_docs_df

    if st.button("Send due reminders"):
        due = due_reminders()
        if not due:
            st.info("No reminders due right now.")
        else:
            ddf2 = _list_docs_df()
            dmap = {r["doctor_id"]: r["name"] for _, r in ddf2.iterrows()} if not ddf2.empty else {}
            sent_count = 0
            for r in due:
                appts_for = _list_appts(patient_id=r['patient_id'])
                appt = next((a for a in appts_for if a['appointment_id'] == r['appointment_id']), None)
                if not appt:
                    continue
                p = get_patient_by_id(r['patient_id'])
                docname = dmap.get(appt['doctor_id'], appt['doctor_id'])
                result = send_reminder(
                    p.email if p else 'patient@example.com',
                    f"{p.first_name} {p.last_name}" if p else 'Patient',
                    appt, r['stage'], docname
                )
                mark_sent(r['reminder_id'], 'smtp' if result.get('status') == 'sent' else 'eml')
                sent_count += 1
            st.success(f"Processed {sent_count} reminder(s).")

    st.markdown("**Send reminder now (for testing)**")
    _apts2 = _list_appts(limit=200)
    if _apts2:
        _ids2 = [a["appointment_id"] for a in _apts2]
        _chosen2 = st.selectbox("Appointment ID", options=_ids2, key="rem_now_appt")
        _stage = st.selectbox("Stage", options=[1, 2, 3], format_func=lambda s: {1: "24h", 2: "3h", 3: "30m"}[s], key="rem_now_stage")
        if st.button("Send reminder now"):
            try:
                _res2 = send_reminder_now(_chosen2, _stage)
                st.success("Reminder sent via SMTP." if _res2.get("status") == "sent" else "Reminder saved as .eml (no SMTP).")
            except Exception as e:
                st.error(f"Failed to send reminder: {e}")
    else:
        st.caption("No appointments yet to send reminders.")

    st.markdown("**Resend confirmation**")
    _apts = _list_appts(limit=200)
    if _apts:
        _ids = [a["appointment_id"] for a in _apts]
        _chosen = st.selectbox("Select appointment ID", options=_ids)
        if st.button("Resend confirmation email"):
            try:
                _res = resend_confirmation(_chosen)
                if _res.get("status") == "sent":
                    st.success("Confirmation sent via SMTP.")
                else:
                    st.success("Confirmation saved as .eml (no SMTP).")
                    path = _res.get("path")
                    if path and os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                st.download_button("📩 Download .eml", f.read(), file_name=os.path.basename(path), mime="message/rfc822")
                        except Exception:
                            pass
            except Exception as e:
                st.error(f"Failed to resend: {e}")
    else:
        st.caption("No appointments yet to resend.")

    # ---- Export Appointments ----
    st.divider()
    st.subheader("Export Appointments (Excel)")
    import pandas as pd, pandas as _pd
    from tools.doctors import list_doctors_df as _docs_df
    ddf3 = _docs_df()
    dmap = {r["doctor_id"]: r["name"] for _, r in ddf3.iterrows()} if not ddf3.empty else {}
    appts_all = list_appointments(limit=1000)
    if appts_all:
        df = _pd.DataFrame(appts_all)
        if not df.empty:
            df["doctor_name"] = df["doctor_id"].map(dmap).fillna(df["doctor_id"])
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="appointments")
            st.download_button("⬇️ Download Excel", data=buffer.getvalue(),
                               file_name="appointments_export.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("No appointments to export.")
    else:
        st.info("No appointments to export.")

    # ---- Manage Patients ----
    st.divider()
    st.subheader("Manage Patients")

    from tools.patient import (
        ensure_checked_out_column,
        get_patients_df,
        set_patient_checked_out,
        delete_patient,
    )

    ensure_checked_out_column()

    col_active, col_co = st.columns(2)

    with col_active:
        st.markdown("**Active / In-care Patients**")
        _active_df = get_patients_df(active_only=True)
        if _active_df is not None and not _active_df.empty:
            st.dataframe(_active_df, use_container_width=True)
            pid_to_checkout = st.selectbox(
                "Mark patient as checked out",
                options=_active_df["patient_id"].tolist(),
                key="pid_checkout",
            )
            if st.button("✅ Check out selected"):
                try:
                    set_patient_checked_out(pid_to_checkout, True)
                    st.success(f"Patient {pid_to_checkout} checked out.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to check out: {e}")
        else:
            st.info("No active patients.")

    with col_co:
        st.markdown("**Checked-out Patients**")
        _co_df = get_patients_df(active_only=False)
        if _co_df is not None and not _co_df.empty:
            st.dataframe(_co_df, use_container_width=True)
            pid_to_uncheckout = st.selectbox(
                "Move checked-out patient back to active",
                options=_co_df["patient_id"].tolist(),
                key="pid_uncheckout",
            )
            if st.button("↩️ Mark active again"):
                try:
                    set_patient_checked_out(pid_to_uncheckout, False)
                    st.success(f"Patient {pid_to_uncheckout} moved back to active.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to revert: {e}")
        else:
            st.info("No checked-out patients yet.")

    st.markdown("**Delete a patient**")
    _all_df = get_patients_df(active_only=None)
    if _all_df is not None and not _all_df.empty:
        pid = st.selectbox("Select Patient ID to delete", options=list(_all_df["patient_id"]), key="del_pid")
        cascade = st.checkbox("Also delete appointments/insurances/audit (cascade)", value=False, key="del_cascade")
        confirm_text = st.text_input("Type DELETE to confirm", key="del_confirm")
        disabled_btn = (confirm_text.strip().upper() != "DELETE")
        if st.button("🗑️ Delete Patient", type="primary", disabled=disabled_btn):
            try:
                res = delete_patient(pid, cascade=cascade)
                st.success(f"Deleted patient {pid}. Summary: {res}")
                st.rerun()
            except Exception as e:
                st.error(f"Unable to delete patient {pid}: {e}")
    else:
        st.caption("No patients found.")

    # ---- Insurance Audit Log ----
    st.divider()
    st.subheader("Insurance Audit Log")

    from tools.insurance import _ensure_tables as _ins_ensure
    DATA_DIR = _Path(__file__).resolve().parents[0] / "data"
    DB_PATH = DATA_DIR / "store.db"
    _ins_ensure()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT log_id, patient_id, old_insurance_id, new_insurance_id, changed_at, changed_by
            FROM insurance_audit_log
            ORDER BY datetime(changed_at) DESC
            LIMIT 200
        """)
        rows = cur.fetchall()
        if rows:
            import pandas as _pd3
            st.dataframe(_pd3.DataFrame(rows, columns=[
                "log_id", "patient_id", "old_insurance_id", "new_insurance_id", "changed_at", "changed_by"
            ]), use_container_width=True)
        else:
            st.info("No audit events yet.")
    except sqlite3.OperationalError:
        st.info("No audit events yet.")
    finally:
        conn.close()

    # ---- Admin Actions Audit Log ----
    st.subheader("Admin Actions Audit Log")
    try:
        conn3 = sqlite3.connect(DB_PATH)
        cur3 = conn3.cursor()
        cur3.execute("""
            SELECT log_id, actor, action, target_type, target_id, details, ts
            FROM admin_audit_log
            ORDER BY datetime(ts) DESC
            LIMIT 200
        """)
        rows3 = cur3.fetchall()
        if rows3:
            import pandas as _pd4
            st.dataframe(_pd4.DataFrame(rows3, columns=[
                "log_id","actor","action","target_type","target_id","details","ts"
            ]), use_container_width=True)
        else:
            st.info("No admin actions yet.")
    except Exception as e:
        st.caption(f"(No admin audit table yet) {e}")
    finally:
        try:
            conn3.close()
        except Exception:
            pass
    # ---- Factory reset (Danger zone) ----
    st.divider()
    with st.expander("🧨 Danger zone: Factory reset (wipe ALL data)"):
        st.warning(
            "This will delete ALL doctors, clinics, slots, appointments, reminders, "
            "patients (CSV), and generated emails/exports.",
            icon="⚠️"
        )
        confirm = st.text_input("Type ERASE ALL to confirm")
        if st.button("🗑️ Wipe database & data files", type="primary",
                    disabled=(confirm.strip().upper() != "ERASE ALL")):
            from pathlib import Path
            import os, shutil

            base = Path(__file__).resolve().parents[0]
            data = base / "data"
            outbox = base / "outbox"
            exports = base / "exports"

            # delete db + patients.csv (ignore if already gone)
            for p in [data / "store.db", data / "patients.csv"]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception as e:
                    st.error(f"Failed removing {p.name}: {e}")

            # clear folders (outbox, exports, assets/generated if present)
            for folder in [outbox, exports, base / "assets" / "generated"]:
                try:
                    if folder.exists():
                        for f in folder.glob("*"):
                            try:
                                f.unlink()
                            except IsADirectoryError:
                                shutil.rmtree(f, ignore_errors=True)
                except Exception as e:
                    st.error(f"Failed clearing {folder}: {e}")

            # (re)create empty schema where initializers exist
            try:
                from tools.insurance import _ensure_tables as ins_ensure
                ins_ensure()
            except Exception:
                pass
            try:
                from tools.scheduler import _ensure_tables as sch_ensure
                sch_ensure()
            except Exception:
                pass
            try:
                from tools.doctors import _ensure_tables as doc_ensure
                doc_ensure()
            except Exception:
                pass
            try:
                from tools.patient import ensure_checked_out_column as pat_boot
                pat_boot()
            except Exception:
                pass

            # clear any cached dataframes and reload app
            try:
                st.cache_data.clear()
            except Exception:
                pass

            st.success("Factory reset complete. Reloading…")
            st.rerun()

