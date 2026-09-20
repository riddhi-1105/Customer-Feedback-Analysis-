# Deploy (Streamlit app -> use Streamlit Community Cloud, free)

Vercel/Netlify CANNOT run Streamlit (needs a long-running Python server).

## Option 1: Streamlit Community Cloud (easiest, 5 min)
1. Create a GitHub repo, upload ALL files of this folder (do NOT upload any `.env`).
2. Go to https://share.streamlit.io -> New app -> pick repo, branch `main`, main file `app.py`.
3. Advanced settings -> Python version 3.11 (or 3.12).
4. (Optional) Settings -> Secrets: `GROQ_API_KEY = "gsk_..."`.
5. Deploy. Click "Load Sample Dataset & Run Analysis" on the home page to test.

## Option 2: Render.com
New Web Service -> connect repo -> it reads `render.yaml`. Add GROQ_API_KEY env var if wanted.

## Option 3: Docker / Hugging Face Spaces (Docker) / any VPS
`docker build -t feedback . && docker run -p 8501:8501 feedback`

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
The Groq key is optional; all analysis works offline. Never commit keys.
If you set a server-side key on a public app, anyone can use your quota.
