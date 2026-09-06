# Automated Job Search Pipeline

Scrapes jobs daily, scores them against your resume with AI, pushes matches to Google Sheets. Runs free on GitHub Actions.

## What it does

1. Scrape fresh jobs (last 24h) from LinkedIn, Indeed, Arbeitnow, Arbeitsagentur (Bundesagentur für Arbeit), Adzuna, Jooble
2. Scores each against your resume using Groq (free Llama 3.1)
3. Pushes jobs scoring ≥7 to a Google Sheet
4. Runs every weekday at 6 AM automatically

**Cost: $0/month**

## Setup

### 1. Get free API keys

- **Groq** — [console.groq.com](https://console.groq.com) — required
- **Adzuna** — [developer.adzuna.com](https://developer.adzuna.com) — optional, adds ~50 jobs/day
- **Jooble** — [jooble.org/api/about](https://jooble.org/api/about) — optional. Free tier is a **500-request lifetime cap, not monthly** — the scraper only queries 5 broad terms (`JOOBLE_SEARCH_TERMS` in `scraper.py`) instead of the full search list to make it last ~3+ months of daily runs.
- **Arbeitsagentur (Bundesagentur für Arbeit)** — no signup needed, it's a free public API, on by default.

### 2. Clone and install

```bash
git clone https://github.com/Moravapalli/job-scrape-and-score.git
cd job-scrape-and-score
pip install -r requirements.txt
```

### 3. Local config

Create `.env`:
```env
GROQ_API_KEY=gsk_your_key
ADZUNA_APP_ID=your_id
ADZUNA_APP_KEY=your_key
JOOBLE_API_KEY=your_key
```

Create `my_resume.txt` with your resume as plain text. Gitignored.

### 4. Test locally

```bash
python scraper.py   # ~2 min
python scorer.py    # ~30 min for ~300 jobs
```

### 5. Push to GitHub

```bash
git init
git add .
git commit -m "initial"
gh repo create job-scrape-and-score --private --push --source=.
```

### 6. Add GitHub Secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret**.

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `MY_RESUME` | paste your full resume text |
| `ADZUNA_APP_ID` | optional |
| `ADZUNA_APP_KEY` | optional |
| `JOOBLE_API_KEY` | optional — 500-request lifetime cap, see above |
| `SHEET_NAME` | optional, e.g. `Job Tracker` |
| `GOOGLE_SERVICE_ACCOUNT` | optional, paste entire `service_account.json` |

Secret names: letters, numbers, underscores only. No spaces.

### 7. Run

**Actions** tab → **Daily Job Search Pipeline** → **Run workflow**.

After first successful run, the schedule is active.

## Google Sheets (optional)

1. [Google Cloud Console](https://console.cloud.google.com) → new project
2. Enable **Google Sheets API**
3. Create a **Service Account** → **Keys → Create new JSON key** → download
4. Create a Google Sheet named `Job Tracker`
5. Share it with the service account email (from the JSON) as **Editor**
6. Paste the full JSON into the `GOOGLE_SERVICE_ACCOUNT` GitHub Secret

## Configuration

In `scraper.py`:
```python
SEARCH_TERMS = ["Data Scientist", "AI Engineer", "Data Engineer"]
HOURS_WINDOW = 24
```

In `scorer.py`:
```python
MIN_SCORE            = 7        # threshold for shortlist
RESUME_MAX_CHARS     = 2000     # truncation for rate limit
JD_MAX_CHARS         = 800
SLEEP_BETWEEN_CALLS  = 2
DAILY_TOKEN_BUDGET   = 190_000  # stays under Groq's 200K tokens/day free-tier cap
TITLE_INCLUDE_KEYWORDS = ["ai", "machine learning", "computer vision", ...]  # pre-filter
TITLE_EXCLUDE_KEYWORDS = ["sales", "marketing", "recruiter", ...]           # pre-filter
```

### Quota-efficient scoring

Groq's free tier caps `openai/gpt-oss-20b` (and the other free models) at **200,000 tokens/day** — enough for roughly 250 scoring calls. To stay inside that as job volume grows, `scorer.py`:

- **Pre-filters on job title only** (not description — descriptions are full of generic "AI-powered" marketing text that let irrelevant roles slip through).
- **Skips obvious German-language hard-rejects** (e.g. `muttersprachlich`, `C2 Deutsch`) without an API call — the LLM would cap the score at 3 anyway.
- **Caches every scored job by URL** in `cache/scored_cache.json`, persisted across GitHub Actions runs via `actions/cache` — a job is never sent to Groq twice.
- **Tracks today's token spend** in `cache/token_usage.json` and stops gracefully before exceeding `DAILY_TOKEN_BUDGET`, marking any leftover jobs "not scored — will retry next run" instead of wasting calls on retries a daily cap can't fix.

In `.github/workflows/pipeline.yml`:
```yaml
on:
  schedule:
    - cron: '0 4 * * 1-5'    # 6am CEST (summer)
    - cron: '0 5 * * 1-5'    # 6am CET  (winter)
```

## Project structure
```text
job-scrape-and-score/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── scraper.py
├── scorer.py
├── sheets_logger.py
├── requirements.txt
├── README.md
├── .gitignore
```

