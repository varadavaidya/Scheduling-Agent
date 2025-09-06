
# AI Scheduler — Full Build

Includes:
- Patient chatbot + Admin
- Dynamic doctors/locations, slot generation
- Booking (SQLite) + 3-stage reminders (24h/3h/30m)
- Email confirmations (SMTP or .eml to outbox/) with Intake Form attachment
- Insurance versioning + audit log
- Export appointments to Excel
- Delete patients (with cascade option) + Admin audit log

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

``` winget install Ollama.Ollama
ollama pull llama3
ollama run llama3
ollama list
```
