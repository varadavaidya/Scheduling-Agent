
# AI Scheduler — Full Build

Includes:
- Patient chatbot + Admin
- Dynamic doctors/locations, slot generation
- Booking (SQLite) + 3-stage reminders (24h/3h/30m)
- Email confirmations (SMTP or .eml to outbox/) with Intake Form attachment
- Insurance versioning + audit log
- Export appointments to Excel
- Delete patients (with cascade option) + Admin audit log

## Overview of the AGENT

<img width="1536" height="1024" alt="ChatGPT Image Sep 6, 2025, 01_50_22 PM" src="https://github.com/user-attachments/assets/2c64e8dd-50d7-4fe9-9faf-60591c9a4a70" />



## LangGraph and LangChain 

They help ntegrate several tools such as doctors side, patient side, 
scheduler, reminders, and database calling. specifically: 

LangChain gives you the building blocks to talk to the LLM and tools—prompts,
memory, retrieval, and tool-calling—so you can chain steps or run an “agent” 
that books slots, looks up data, sends emails, etc.

LangGraph wraps those steps in a stateful graph (nodes = actions, edges = routing)
so the conversation flow is reliable, resumable, and controllable—preventing loops,
handling branches (identity → insurance → doctor → slot), and managing retries.

## Patient Side

The patient tab is a chatbot which acts like a receptionist 
of the hospital. One can chat and book appointments and edit
details of insurance and change doctors and other functionalities.

## Admin Side

Here the assumptions is that , the doctors are managed by the admin 
at the clinic. So admin can add a doctor for corresponding locations of clinics.
He can manually generate appointments for walk-ins and Checkout a patient

## Run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env   # set LLM_PROVIDER=ollama, OLLAMA_MODEL=llama3
# (Optional) assets\intake_form.pdf
streamlit run app.py
```

## To install OLLAMA's LLAMA3 (PowerShell)

The llama3 has to run in system globally and will listen to port 11434.
Its programmed to act like a receptionist. The parsing and replies take 
sometime due to computations, have patience.

``` winget install Ollama.Ollama
ollama pull llama3
ollama run llama3
ollama list
```
