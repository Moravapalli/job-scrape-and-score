"""
Stage 2 — Score each job against the candidate's resume using Groq.

Features:
  • Loads resume from env var (CI) or file (local)
  • Truncates resume + JD to fit Groq's 6k TPM free-tier limit
  • Handles 429 rate limits with smart back-off (parses retry-after)
  • Optional keyword pre-filter to skip obviously irrelevant jobs
  • Saves all scored jobs + shortlist to output/
"""

from dotenv import load_dotenv
load_dotenv()

import os, json, time, re, pandas as pd
from datetime import date
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL              = "openai/gpt-oss-20b"
MIN_SCORE          = 7
RESUME_MAX_CHARS   = 2000      # truncate resume to control token usage
JD_MAX_CHARS       = 800       # truncate job description
SLEEP_BETWEEN_CALLS = 2        # seconds — stays under 6k TPM
ENABLE_PREFILTER   = True      # quick keyword check before API call

# Title-only pre-filter (not description — descriptions are full of generic
# "AI-powered" marketing boilerplate that let irrelevant roles like sales/BD
# jobs slip through when matched against description text). A job must match
# an include term and no exclude term to reach the LLM. Customize to your
# target roles/resume.
TITLE_INCLUDE_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "ml engineer",
    "ml systems", "deep learning", "computer vision", "cv engineer",
    "machine vision", "vision engineer", "embedded vision", "mlops",
    "data scientist", "data engineer", "nlp", "llm", "generative ai",
    "perception", "robotics", "autonomous", "inference engineer",
    "ai software engineer", "applied scientist", "research scientist",
    "research engineer",
]

TITLE_EXCLUDE_KEYWORDS = [
    "sales", "account executive", "account manager", "business development",
    "marketing", "recruiter", "recruiting", "talent acquisition",
    "human resources", "finance", "accountant", "bookkeeper", "legal",
    "procurement", "warehouse", "logistics", "customer success",
    "community manager", "social media", "content", "copywriter",
    "video editor", "chief of staff", "general manager", "operations manager",
    "presenter", "host", "onboarding manager", "billing manager",
]

# Unambiguous hard-reject language phrases — same "cap at 3" rule the system
# prompt already applies, but caught here for free (no API call) instead of
# spending tokens to have the LLM re-derive the same verdict.
HARD_REJECT_LANGUAGE = [
    "muttersprachlich", "muttersprachliche", "native german speaker",
    "c2 deutsch", "c2-niveau deutsch", "verhandlungssicheres deutsch",
    "verhandlungssicher deutsch", "verhandlungssichere deutschkenntnisse",
]

# ── Load resume — env var (CI) or file (local) ────────────────────────
MY_RESUME = os.environ.get("MY_RESUME", "").strip()

if MY_RESUME:
    print(f"✓ Resume loaded from MY_RESUME env var ({len(MY_RESUME)} chars)", flush=True)
elif os.path.exists("my_resume.txt"):
    with open("my_resume.txt", encoding="utf-8") as f:
        MY_RESUME = f.read().strip()
    print(f"✓ Resume loaded from my_resume.txt ({len(MY_RESUME)} chars)", flush=True)
else:
    print("✗ Resume not found.")
    print("  → Local:           create my_resume.txt with your resume")
    print("  → GitHub Actions:  add MY_RESUME repository secret")
    exit(1)

if "[PASTE YOUR" in MY_RESUME or len(MY_RESUME) < 100:
    print("✗ Resume looks like placeholder or too short — paste your real resume")
    exit(1)

# Truncate to fit rate limit
if len(MY_RESUME) > RESUME_MAX_CHARS:
    MY_RESUME = MY_RESUME[:RESUME_MAX_CHARS] + "\n[...truncated for scoring efficiency]"
    print(f"  → Truncated resume to {RESUME_MAX_CHARS} chars to fit rate limit", flush=True)

# ── Scoring prompt ────────────────────────────────────────────────────
# Kept intentionally short: this is sent on every single scoring call, so its
# token cost multiplies by job count against Groq's 200K-tokens/day free-tier
# cap. See LANGUAGE_RULES below for the full logic this compresses.
SYSTEM = """Expert technical recruiter. Score job listings vs. a resume, honestly \
(bad match = 2-3, not 5-6). JD may be English or German.

Candidate: English native/fluent. German B1 (conversational, not business-fluent).
Language rules — apply first:
- Cap score at 3 if: native/muttersprachlich German, C2 German, verhandlungssicher \
Deutsch, or a client-facing role (sales/support/HR/legal) needs fluent German.
- -3 if C1 German required. -2 if German clearly required (level unspecified) or \
fully-German JD + heavy communication role. -1 if fully-German JD but mostly technical.
- No penalty if: English is the working language, German is optional/a plus, \
B1/B2 is enough, remote/international team, or language isn't mentioned.

Respond with valid JSON only — no markdown, no text outside the JSON."""

def build_prompt(row):
    return f"""Score this job listing for the candidate below. Be honest — a bad match should score 2-3, not 5-6.

CANDIDATE RESUME:
{MY_RESUME}

JOB LISTING:
Title:       {row.get('title', 'N/A')}
Company:     {row.get('company', 'N/A')}
Location:    {row.get('location', 'N/A')}
Description: {str(row.get('description', ''))[:JD_MAX_CHARS]}

SCORING CRITERIA:
- Skills match (40%): Tech stack alignment with JD keywords
- Seniority match (20%): Level appropriate for their experience?
- Domain match (20%): Industry relevant to background?
- Location/remote (10%): Does location work?
- Growth potential (10%): Career progression?

Return ONLY this JSON — nothing else:
{{
  "score": <integer 1-10>,
  "skills_match": <integer 1-10>,
  "seniority_match": <integer 1-10>,
  "highlights": ["top reason 1", "top reason 2"],
  "concerns": ["concern 1"],
  "reason": "<2 sentence summary of fit>",
  "apply": <true if score >= 7, else false>
}}"""

class DailyQuotaExhausted(Exception):
    """Raised when Groq reports the daily token quota (TPD) is exhausted —
    unlike a per-minute rate limit, retrying within seconds/minutes can't
    fix this, so the caller should stop the whole run rather than retry."""


# ── Scorer with smart 429 handling ────────────────────────────────────
def score_job(row, retries=4):
    """Returns (result_dict, tokens_used_this_call)."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": build_prompt(row)},
                ],
                temperature=0.1,
                max_tokens=600,
                reasoning_effort="low",
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            # Extract JSON block
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            raw   = raw[start:end]

            result = json.loads(raw)
            tokens = response.usage.total_tokens if response.usage else 0
            return {
                "score":           result.get("score", 0),
                "skills_match":    result.get("skills_match", 0),
                "seniority_match": result.get("seniority_match", 0),
                "highlights":      " | ".join(result.get("highlights", [])),
                "concerns":        " | ".join(result.get("concerns", [])),
                "reason":          result.get("reason", ""),
                "apply":           result.get("apply", False),
            }, tokens

        except json.JSONDecodeError:
            print(f"  ✗ Bad JSON (attempt {attempt+1}) — retrying...", flush=True)
            time.sleep(2)

        except Exception as e:
            error_msg = str(e)
            if "tokens per day" in error_msg.lower() or "(tpd)" in error_msg.lower():
                # Daily cap — a 25s/1m retry can't help, don't waste more calls.
                raise DailyQuotaExhausted(error_msg)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                match = re.search(r"try again in ([\d.]+)s", error_msg)
                wait  = float(match.group(1)) + 2 if match else 25
                print(f"  ⏳ Rate limited — waiting {wait:.0f}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  ✗ Error: {error_msg[:120]}", flush=True)
                time.sleep(5)

    return _empty_score(), 0


def _empty_score():
    return {
        "score": 0, "skills_match": 0, "seniority_match": 0,
        "highlights": "", "concerns": "scoring failed",
        "reason": "Could not score this listing", "apply": False,
    }


def _unscored_quota():
    return {
        "score": None, "skills_match": None, "seniority_match": None,
        "highlights": "", "concerns": "not scored — daily token quota reached, will retry next run",
        "reason": "", "apply": False,
    }


def quick_filter(row):
    """Title-only keyword check — skips irrelevant jobs without an API call.
    Title-only (not description) to avoid buzzword false-positives; must hit
    an include term and no exclude term."""
    title = str(row.get("title", "")).lower()
    if any(kw in title for kw in TITLE_EXCLUDE_KEYWORDS):
        return False
    return any(kw in title for kw in TITLE_INCLUDE_KEYWORDS)


def language_hard_reject(row):
    """Catches the same 'cap at 3' verdict the system prompt would reach for
    unambiguous native/C2/verhandlungssicher German requirements — but for
    free, without spending an API call to have the LLM re-derive it."""
    text = (str(row.get("title", "")) + " " + str(row.get("description", ""))).lower()
    return any(kw in text for kw in HARD_REJECT_LANGUAGE)


def _hard_reject_score():
    return {
        "score": 3, "skills_match": 0, "seniority_match": 0,
        "highlights": "",
        "concerns": "Requires native/C2/verhandlungssicher German — exceeds candidate's B1 level",
        "reason": "Language requirement (native/C2/business-fluent German) rules this out regardless of skills fit.",
        "apply": False,
    }


# ── Persistent cache + daily token budget ───────────────────────────────
# Lives in cache/ (not output/), which persists across GitHub Actions runs
# via actions/cache in pipeline.yml — output/ is per-run artifact only.
CACHE_DIR          = "cache"
SCORED_CACHE_PATH  = os.path.join(CACHE_DIR, "scored_cache.json")
TOKEN_USAGE_PATH   = os.path.join(CACHE_DIR, "token_usage.json")
DAILY_TOKEN_BUDGET  = 190_000   # stay under Groq's 200K TPD free-tier cap, with margin
os.makedirs(CACHE_DIR, exist_ok=True)


def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_token_usage():
    state = load_json(TOKEN_USAGE_PATH)
    today = str(date.today())
    if state.get("date") != today:
        return {"date": today, "tokens_used": 0}
    return state


# ── Main ──────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

# Auto-detect CSV location
if os.path.exists("output/jobs_today.csv"):
    INPUT_CSV = "output/jobs_today.csv"
elif os.path.exists("jobs_today.csv"):
    INPUT_CSV = "jobs_today.csv"
else:
    print("✗ jobs_today.csv not found. Run scraper.py first.")
    exit(1)

print(f"✓ Reading {INPUT_CSV}", flush=True)
jobs = pd.read_csv(INPUT_CSV)
jobs = jobs[jobs["title"].notna()].reset_index(drop=True)
total_raw = len(jobs)
print(f"  → {total_raw} jobs in CSV", flush=True)

# Pre-filter
if ENABLE_PREFILTER:
    jobs = jobs[jobs.apply(quick_filter, axis=1)].reset_index(drop=True)
    skipped = total_raw - len(jobs)
    print(f"  → Skipped {skipped} obviously irrelevant jobs ({len(jobs)} to score)\n", flush=True)

total = len(jobs)
if total == 0:
    print("✗ No jobs to score after filtering")
    exit(0)

scored_cache = load_json(SCORED_CACHE_PATH)
token_state  = load_token_usage()
tokens_used  = token_state["tokens_used"]

cache_hits    = 0
hard_rejects  = 0
quota_skipped = 0
api_calls     = 0

print(f"  → Token budget today: {tokens_used:,} / {DAILY_TOKEN_BUDGET:,} used", flush=True)
print(f"Scoring {total} jobs with Groq ({MODEL})...\n", flush=True)

results    = []
start_time = time.time()

for i, row in jobs.iterrows():
    title   = str(row.get("title",   "?"))[:50]
    company = str(row.get("company", "?"))[:30]
    job_url = str(row.get("job_url", "") or "")
    print(f"[{i+1}/{total}] {title} @ {company}", end=" ... ", flush=True)

    if job_url and job_url in scored_cache:
        result = scored_cache[job_url]
        cache_hits += 1
        results.append(result)
        print(f"score: {result['score']}/10 (cached)", flush=True)
        continue

    if language_hard_reject(row):
        result = _hard_reject_score()
        hard_rejects += 1
        results.append(result)
        if job_url:
            scored_cache[job_url] = result
            save_json(SCORED_CACHE_PATH, scored_cache)
        print(f"score: {result['score']}/10 (language hard-reject, no API call)", flush=True)
        continue

    if tokens_used >= DAILY_TOKEN_BUDGET:
        quota_skipped += 1
        results.append(_unscored_quota())
        print("skipped — daily token budget reached", flush=True)
        continue

    try:
        result, tokens = score_job(row)
    except DailyQuotaExhausted:
        remaining = total - i
        quota_skipped += remaining
        results.extend(_unscored_quota() for _ in range(remaining))
        print(f"\n  ⚠ Daily token quota hit — stopping. {remaining} remaining job(s) will be retried next run.", flush=True)
        break

    tokens_used += tokens
    api_calls += 1
    token_state["tokens_used"] = tokens_used
    save_json(TOKEN_USAGE_PATH, token_state)
    if job_url:
        scored_cache[job_url] = result
        save_json(SCORED_CACHE_PATH, scored_cache)

    results.append(result)
    print(f"score: {result['score']}/10", flush=True)
    time.sleep(SLEEP_BETWEEN_CALLS)

# Combine & save
scored    = pd.concat([jobs, pd.DataFrame(results)], axis=1)
shortlist = scored[scored["score"] >= MIN_SCORE].sort_values("score", ascending=False)

scored.to_csv("output/jobs_scored.csv", index=False)
shortlist.to_csv("output/shortlist.csv", index=False)

elapsed = int(time.time() - start_time)
mins    = elapsed // 60
secs    = elapsed % 60
print(f"""
╔══════════════════════════════════╗
  Groq scoring complete ({mins}m {secs}s)
  Model:          {MODEL}
  Pre-filtered:   {total_raw} → {total}
  API calls:      {api_calls}  (cached: {cache_hits}, hard-reject: {hard_rejects}, quota-skipped: {quota_skipped})
  Tokens today:   {tokens_used:,} / {DAILY_TOKEN_BUDGET:,}
  Shortlisted:    {len(shortlist)}  (score >= {MIN_SCORE})
  Cost:           $0.00
╚══════════════════════════════════╝
""", flush=True)

if not shortlist.empty:
    cols      = ["title", "company", "location", "score", "reason", "job_url"]
    available = [c for c in cols if c in shortlist.columns]
    print("Top 10 matches:\n")
    print(shortlist[available].head(10).to_string(index=False))
else:
    print("No jobs scored above the threshold today.")




# # scorer.py — production-ready AI job scorer
# from dotenv import load_dotenv
# import os, json, time, pandas as pd
# from groq import Groq
# load_dotenv()


# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# MODEL     = "llama-3.1-8b-instant"
# MIN_SCORE = 7

# # ── Load resume — env var (CI/GitHub Actions) or file (local) ─────────
# MY_RESUME = os.environ.get("MY_RESUME", "").strip()

# if MY_RESUME:
#     print(f"✓ Resume loaded from MY_RESUME env var ({len(MY_RESUME)} chars)")
# elif os.path.exists("my_resume.txt"):
#     with open("my_resume.txt", encoding="utf-8") as f:
#         MY_RESUME = f.read().strip()
#     print(f"✓ Resume loaded from my_resume.txt ({len(MY_RESUME)} chars)")
# else:
#     print("✗ Resume not found.")
#     print("  → Local:           create my_resume.txt with your resume text")
#     print("  → GitHub Actions:  add MY_RESUME repository secret")
#     exit(1)

# if "[PASTE YOUR" in MY_RESUME or len(MY_RESUME) < 100:
#     print("✗ Resume looks like placeholder text or is too short — paste your real resume")
#     exit(1)

# # ── Scoring prompt ────────────────────────────────────────────────────
# SYSTEM  = """You are an expert technical recruiter and career coach.
# You evaluate job listings against a candidate's resume with precision and honesty. 
# The resume and job description may be in English or German.
# Understand both languages and map equivalent skills across languages.
# You always respond with valid JSON only — no preamble, no markdown, no explanation outside the JSON."""

# def build_prompt(row):
#     return f"""Score this job listing for the candidate below. Be honest — a bad match should score 2-3, not 5-6.

# CANDIDATE RESUME:
# {MY_RESUME}

# JOB LISTING:
# Title:       {row.get('title', 'N/A')}
# Company:     {row.get('company', 'N/A')}
# Location:    {row.get('location', 'N/A')}
# Description: {str(row.get('description', ''))[:2000]}

# SCORING CRITERIA:
# - Skills match (40%): Do their tech stack and tools match the JD keywords?
# - Seniority match (20%): Is the level appropriate for their experience?
# - Domain match (20%): Is the industry/domain relevant to their background?
# - Location/remote (10%): Does the location work for the candidate?
# - Growth potential (10%): Does this role offer career progression?

# Return ONLY this JSON — nothing else:
# {{
#   "score": <integer 1-10>,
#   "skills_match": <integer 1-10>,
#   "seniority_match": <integer 1-10>,
#   "highlights": ["top reason 1", "top reason 2"],
#   "concerns": ["concern 1"],
#   "reason": "<2 sentence summary of fit>",
#   "apply": <true if score >= 7, else false>
# }}"""

# # ── Scorer with retry + error handling ───────────────────────────────
# # def score_job(row, retries=2):
# #     for attempt in range(retries + 1):
# #         try:
# #             msg = client.messages.create(
# #                 model="claude-haiku-4-5-20251001",  # haiku = fast + cheap for scoring
# #                 max_tokens=300,
# #                 system=SYSTEM_PROMPT,
# #                 messages=[{"role": "user", "content": build_prompt(row)}]
# #             )
# #             raw = msg.content[0].text.strip()

# #             # Strip markdown fences if Claude adds them
# #             if raw.startswith("```"):
# #                 raw = raw.split("```")[1]
# #                 if raw.startswith("json"):
# #                     raw = raw[4:]
# #             raw = raw.strip()

# #             result = json.loads(raw)

# #             return {
# #                 "score":            result.get("score", 0),
# #                 "skills_match":     result.get("skills_match", 0),
# #                 "seniority_match":  result.get("seniority_match", 0),
# #                 "highlights":       " | ".join(result.get("highlights", [])),
# #                 "concerns":         " | ".join(result.get("concerns", [])),
# #                 "reason":           result.get("reason", ""),
# #                 "apply":            result.get("apply", False),
# #                 "input_tokens":     msg.usage.input_tokens,
# #                 "output_tokens":    msg.usage.output_tokens,
# #             }

# #         except json.JSONDecodeError:
# #             print(f"  ✗ JSON parse error on attempt {attempt+1} for {row.get('title','?')}")
# #             if attempt == retries:
# #                 return _empty_score()
# #             time.sleep(2)

# #         except Exception as e:
# #             print(f"  ✗ Error: {e}")
# #             return _empty_score()


# def score_job(row, retries=3):
#     for attempt in range(retries):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "system", "content": SYSTEM},
#                     {"role": "user",   "content": build_prompt(row)},
#                 ],
#                 temperature=0.1,
#                 max_tokens=300,
#             )
#             raw = response.choices[0].message.content.strip()

#             # Strip markdown fences
#             if "```" in raw:
#                 raw = raw.split("```")[1]
#                 if raw.startswith("json"):
#                     raw = raw[4:]

#             # Extract JSON block
#             start = raw.find("{")
#             end   = raw.rfind("}") + 1
#             raw   = raw[start:end]

#             result = json.loads(raw)
#             return {
#                 "score":        result.get("score", 0),
#                 "skills_match": result.get("skills_match", 0),
#                 "highlights":   " | ".join(result.get("highlights", [])),
#                 "concerns":     " | ".join(result.get("concerns", [])),
#                 "reason":       result.get("reason", ""),
#                 "apply":        result.get("apply", False),
#             }

#         except json.JSONDecodeError:
#             print(f"  ✗ Bad JSON (attempt {attempt+1}) — retrying...")
#             time.sleep(2)
#         except Exception as e:
#             print(f"  ✗ Error: {e}")
#             time.sleep(5)

#     return {
#         "score": 0, "skills_match": 0,
#         "highlights": "", "concerns": "scoring failed",
#         "reason": "error", "apply": False,
#     }


# def _empty_score():
#     return {
#         "score": 0, "skills_match": 0, "seniority_match": 0,
#         "highlights": "", "concerns": "scoring failed",
#         "reason": "Could not score this listing", "apply": False,
#         "input_tokens": 0, "output_tokens": 0
#     }

# # ── Main ──────────────────────────────────────────────────────────────
# jobs  = pd.read_csv("jobs_today.csv")
# jobs  = jobs[jobs["title"].notna()].reset_index(drop=True)
# total = len(jobs)
# print(f"Scoring {total} jobs with Groq ({MODEL})...\n")

# results    = []
# start_time = time.time()

# for i, row in jobs.iterrows():
#     title   = str(row.get("title",   "?"))[:50]
#     company = str(row.get("company", "?"))[:30]
#     print(f"[{i+1}/{total}] {title} @ {company}", end=" ... ", flush=True)

#     result = score_job(row)
#     results.append(result)
#     print(f"score: {result['score']}/10")
#     time.sleep(0.3)

# scored    = pd.concat([jobs, pd.DataFrame(results)], axis=1)
# shortlist = scored[scored["score"] >= MIN_SCORE].sort_values("score", ascending=False)

# scored.to_csv("output/jobs_scored.csv", index=False)
# shortlist.to_csv("output/shortlist.csv", index=False)

# elapsed = int(time.time() - start_time)
# print(f"""
# ╔══════════════════════════════════╗
#   Groq scoring complete ({elapsed}s)
#   Model:        {MODEL}
#   Total scored: {total}
#   Shortlisted:  {len(shortlist)}  (score >= {MIN_SCORE})
#   Cost:         $0.00
# ╚══════════════════════════════════╝
# """)

# cols      = ["title", "company", "location", "score", "reason", "job_url"]
# available = [c for c in cols if c in shortlist.columns]
# print(shortlist[available].head(10).to_string(index=False))
