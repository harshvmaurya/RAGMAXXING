# RAGMAXXING

Upload text-based PDFs, index them for the current session, and ask questions
with answers grounded in the documents and their page references.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GROQ_API_KEY` to your Groq API key.
   Keep `.env` private; it is excluded from Git.
4. Start the app:

   ```powershell
   streamlit run app.py
   ```

## Deploy with Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set
   `app.py` as the main file.
3. Add `GROQ_API_KEY` in the app's **Settings > Secrets**. Do not commit the
   key or paste it into source files.

The app's uploads and search index are session-only. Document text retrieved
for a question is sent to Groq to generate the answer; do not upload documents
unless that processing is appropriate for their contents.

## Repository contents

- `app.py` — Streamlit document Q&A application.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — RAGMAXXING theme.
- `.env.example` — Local environment-variable template without credentials.

The local `.venv/`, secrets, uploaded/source documents, and generated vector
indexes are intentionally excluded from the repository.
