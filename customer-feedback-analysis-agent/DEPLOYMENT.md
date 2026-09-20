# 🚀 Deployment Guide — Vercel

This project now runs in **two modes from the same codebase**:

| Mode | Entry point | Command | Use for |
|------|-------------|---------|---------|
| **Web (production)** | `api/index.py` (Flask) | `python api/index.py` | Vercel deployment, public URL |
| **Streamlit (local)** | `app.py` | `streamlit run app.py` | Local development / demos |

Both modes call the **exact same** `agents/` + `tools/` pipeline. No analysis
logic is duplicated.

---

## 1. Why Vercel was failing before

Vercel's Python runtime looks for a **WSGI/ASGI callable** named `app`,
`application`, or `handler` at the top level of the Python entry file.

The old root `app.py` was a **Streamlit script**. Streamlit is a long-running
server process with a WebSocket connection — it is not a WSGI app and it
**cannot run on serverless functions at all**. The file defined no `app`
variable, so Vercel reported:

```
Found app.py but it does not export a top-level 'app', 'application', or 'handler' variable.
```

**Fix applied:** a real Flask WSGI application now lives in `api/index.py` and
exports `app` (plus `application` and `handler` aliases). `vercel.json` points
every route at it, and `.vercelignore` excludes the Streamlit `app.py` from the
deployment bundle so it can never be auto-detected again.

---

## 2. Push to GitHub

```bash
cd customer-feedback-analysis-agent

git init
git add .
git commit -m "Vercel-ready: Flask serverless entry point + web UI"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
git push -u origin main
```

> ⚠️ **Before pushing:** confirm `.env` is *not* in the repo.
> `git status` must not list it — `.gitignore` already excludes it.
> ```bash
> git ls-files | grep -c "^\.env$"   # must print 0
> ```

---

## 3. Deploy on Vercel

1. Go to **https://vercel.com/new** and sign in with GitHub.
2. Click **Import** next to your repository.
3. **Framework Preset:** leave as **Other** (`vercel.json` handles everything).
4. **Root Directory:** if your repo root *is* `customer-feedback-analysis-agent`,
   leave it as `./`. If the folder sits inside the repo, click **Edit** and
   select `customer-feedback-analysis-agent`.
5. Leave Build/Output/Install commands **empty**.
6. Click **Deploy** and wait ~2 minutes.

You get one public URL, e.g. `https://your-project.vercel.app`.

---

## 4. Environment variables (optional — Groq LLM)

The app works fully without any keys. To enable the **AI Deep Dive** and
**AI Executive Summary** features:

**Vercel → your project → Settings → Environment Variables → Add New**

| Key | Value | Environments |
|-----|-------|--------------|
| `GROQ_API_KEY` | your key from https://console.groq.com/keys | Production, Preview, Development |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Production, Preview, Development |
| `MAX_ROWS` | `1500` *(optional)* | Production |

Then **Deployments → ⋯ → Redeploy** so the new variables are picked up.

Nothing is hard-coded: `tools/groq_client.py` reads these with `os.getenv()`.

---

## 5. Run locally

**Web version (identical to production):**
```bash
pip install -r requirements.txt
python api/index.py
# → http://127.0.0.1:5000
```

**Streamlit version (unchanged):**
```bash
pip install -r requirements-local.txt
streamlit run app.py
# → http://localhost:8501
```

**Tests:**
```bash
python -m pytest tests/ -q     # 62 tests
```

---

## 6. Serverless constraints to be aware of

| Constraint | Handling |
|------------|----------|
| Upload body ≤ ~4.5 MB | Enforced at 4 MB, returns a friendly 413 message |
| 10 s execution (Hobby plan) | `MAX_ROWS` caps analysis at 1500 rows; the UI says when data was truncated |
| No writable disk / no session state | Results are returned as JSON and held in the browser (`sessionStorage`); reports are regenerated on demand |
| 250 MB bundle limit | `requirements.txt` excludes `streamlit`, `plotly`, `scikit-learn`, `nltk`, `textblob` (~206 MB deployed) |

**Removed heavy libraries and their replacements — no feature was lost:**

| Removed from Vercel build | Replacement |
|---------------------------|-------------|
| `streamlit` | Flask + the HTML/CSS/JS UI in `web/` |
| `plotly` (Python) | Plotly.js via CDN — charts render in the browser |
| `scikit-learn` (TF-IDF + DBSCAN) | Built-in keyword-similarity fallback already present in `tools/recurring_issue_detector.py` |
| `nltk` / `textblob` | `vaderSentiment` — ships its own lexicon, so no runtime download is needed on a read-only filesystem |

All of these still install locally via `requirements-local.txt`, so the
Streamlit app keeps the full TF-IDF/DBSCAN clustering path.

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `does not export a top-level 'app'` | `vercel.json` missing or Root Directory wrong — make sure `vercel.json` and `api/index.py` sit in the selected root |
| 404 on every page | Check the `routes` block in `vercel.json` is present |
| `FUNCTION_INVOCATION_TIMEOUT` | Dataset too big — lower `MAX_ROWS` to e.g. `600` |
| "Groq LLM not configured" | `GROQ_API_KEY` not set, or you didn't redeploy after adding it |
| 413 on upload | CSV over 4 MB — trim rows or use the sample dataset |


---

## 8. Deploying on Render instead of Vercel

Render is a good alternative for this project: it runs a normal long-lived
process, so there is **no 10-second timeout** and you can raise `MAX_ROWS`.

Create the service as a **Web Service** (NOT a Static Site — a static site has
no Python runtime and the agent would never run).

**Render → New → Web Service → connect your repo**, then:

| Field | Value |
|-------|-------|
| Language / Runtime | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn api.index:app --bind 0.0.0.0:$PORT` |
| Root Directory | leave blank, or `customer-feedback-analysis-agent` if it's a subfolder |
| Health Check Path | `/api/health` |

Longer start command with sensible tuning (what `render.yaml` uses):

```
gunicorn api.index:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

Environment variables (Render → Environment): `GROQ_API_KEY`, `GROQ_MODEL`,
`MAX_ROWS`. Same names as Vercel — nothing is hard-coded.

**Or use the Blueprint:** this repo includes `render.yaml`. Render → New →
Blueprint → pick the repo, and everything above is filled in automatically.

### Render gotchas
| Symptom | Fix |
|---------|-----|
| Asked for a "Publish Directory" | You created a **Static Site**. Delete it and create a **Web Service**. |
| `ModuleNotFoundError: api` | Root Directory is wrong — it must be the folder containing `api/` |
| First request takes ~50 s | Free tier sleeps after 15 min idle. Normal. |
| Build runs out of memory | Free tier has 512 MB; the trimmed `requirements.txt` fits, but don't add `scikit-learn` back |
