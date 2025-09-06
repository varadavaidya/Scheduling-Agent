
# RagaAI Scheduler — Full Build

Includes:
- Patient chatbot + Admin
- Dynamic doctors/locations, slot generation
- Booking (SQLite) + 3-stage reminders (24h/3h/30m)
- Email confirmations (SMTP or .eml to outbox/) with Intake Form attachment
- Insurance versioning + audit log
- Export appointments to Excel
- Delete patients (with cascade option) + Admin audit log
- Theming (Teal/Light/Dark)

## Overview of the AGENT

<img width="1536" height="1024" alt="ChatGPT Image Sep 6, 2025, 01_50_22 PM" src="https://github.com/user-attachments/assets/51201eeb-2597-48f4-8cb3-0d922dce8321" />



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
