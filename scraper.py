"""
Germany AI/Data Job Scraper — GitHub Actions optimized.
Forces stdout flush so logs appear live, not buffered.
"""
import sys
# Force unbuffered output on GitHub Actions
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import os, time, requests, pandas as pd, dateutil.parser
from datetime import date, datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from jobspy import scrape_jobs

SEARCH_TERMS = ["Data Scientist", "AI Engineer", "Data Engineer"]
HOURS_WINDOW = 24
ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID",  "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")
CUTOFF = datetime.now(timezone.utc) - timedelta(hours=HOURS_WINDOW)


def log(msg):
    """Force flush so GitHub Actions shows it immediately."""
    print(msg, flush=True)


def is_recent(date_str):
    if not date_str:
        return True
    try:
        dt = dateutil.parser.parse(str(date_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= CUTOFF
    except Exception:
        return True


def is_recent_unix(ts):
    if not ts:
        return True
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc) >= CUTOFF
    except Exception:
        return True


# ── Scrapers ─────────────────────────────────────────────────

def scrape_jobspy_sites(term):
    t0 = time.time()
    try:
        df = scrape_jobs(
            site_name=["linkedin", "indeed"],
            search_term=term,
            location="Germany",
            results_wanted=50,
            hours_old=HOURS_WINDOW,
            country_indeed="Germany",
            linkedin_fetch_description=False,
            verbose=0,
        )
        if not df.empty and "site" in df.columns:
            for site, count in df["site"].value_counts().items():
                log(f"  ✓ {site}: {count} jobs for '{term}'  ({int(time.time()-t0)}s)")
        else:
            log(f"  ✗ LinkedIn/Indeed: 0 jobs for '{term}'")
        return df
    except Exception as e:
        log(f"  ✗ LinkedIn/Indeed error: {e}")
        return pd.DataFrame()


def scrape_adzuna(term):
    if not (ADZUNA_APP_ID and ADZUNA_APP_KEY):
        return pd.DataFrame()
    
    # On GitHub Actions Adzuna often blocks. Fail fast.
    t0 = time.time()
    rows = []
    
    for page in range(1, 4):
        try:
            r = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/de/search/{page}",
                params={
                    "app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY,
                    "what": term, "results_per_page": 25,
                    "max_days_old": 1, "sort_by": "date",
                },
                timeout=8,                # short timeout
                headers={"User-Agent": "Mozilla/5.0"},
            )
            for j in r.json().get("results", []):
                if not is_recent(j.get("created", "")):
                    continue
                rows.append({
                    "title":       j.get("title", ""),
                    "company":     j.get("company", {}).get("display_name", ""),
                    "location":    j.get("location", {}).get("display_name", "Germany"),
                    "job_url":     j.get("redirect_url", ""),
                    "description": j.get("description", ""),
                    "date_posted": j.get("created", ""),
                    "site":        "adzuna",
                })
        except requests.exceptions.ConnectionError:
            log(f"  ⚠ Adzuna unreachable (GitHub Actions network block) — skipping")
            return pd.DataFrame()         # bail immediately, no retries
        except Exception as e:
            log(f"  ✗ Adzuna error: {type(e).__name__}")
            break
    
    log(f"  ✓ Adzuna: {len(rows)} jobs for '{term}'  ({int(time.time()-t0)}s)")
    return pd.DataFrame(rows)


def scrape_arbeitnow(term):
    t0 = time.time()
    try:
        r = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            params={"search": term},
            timeout=10,
        )
        rows = []
        for j in r.json().get("data", []):
            if not is_recent_unix(j.get("created_at", 0)):
                continue
            rows.append({
                "title":       j.get("title", ""),
                "company":     j.get("company_name", ""),
                "location":    j.get("location", "Germany"),
                "job_url":     j.get("url", ""),
                "description": (j.get("description", "") or "")[:1500],
                "date_posted": str(j.get("created_at", "")),
                "site":        "arbeitnow",
            })
        log(f"  ✓ Arbeitnow: {len(rows)} jobs for '{term}'  ({int(time.time()-t0)}s)")
        return pd.DataFrame(rows)
    except Exception as e:
        log(f"  ✗ Arbeitnow error: {e}")
        return pd.DataFrame()


# ── Pipeline ─────────────────────────────────────────────────

def process_term(term):
    log(f"\n[Scraping: '{term}']")
    results = [scrape_jobspy_sites(term)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(api, term) for api in [scrape_adzuna, scrape_arbeitnow]]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def main():
    start = time.time()
    log(f"⏱ Cutoff: {CUTOFF.strftime('%Y-%m-%d %H:%M UTC')} ({HOURS_WINDOW}h window)")
    log(f"⏱ Adzuna keys: {'✓ set' if (ADZUNA_APP_ID and ADZUNA_APP_KEY) else '✗ missing'}")

    all_jobs = []
    for i, term in enumerate(SEARCH_TERMS):
        all_jobs.extend(process_term(term))
        if i < len(SEARCH_TERMS) - 1:
            log("  ⏳ Cooling down 10s before next term...")
            time.sleep(10)   # reduced from 15s

    combined = pd.concat(all_jobs, ignore_index=True)
    combined = combined[combined["title"].notna() & (combined["title"] != "")]
    combined["company"] = combined["company"].fillna("").astype(str)
    combined.drop_duplicates(subset=["title", "company"], keep="first", inplace=True)
    if "job_url" in combined.columns:
        combined = combined[combined["job_url"].notna() & (combined["job_url"] != "")]
        combined.drop_duplicates(subset=["job_url"], keep="first", inplace=True)
    combined["scraped_date"] = str(date.today())

    os.makedirs("output", exist_ok=True)
    combined.to_csv("output/jobs_today.csv", index=False)

    elapsed = int(time.time() - start)
    log(f"\n{'─'*50}")
    log(f"✓ {len(combined)} unique jobs saved → output/jobs_today.csv  ({elapsed}s)")
    if "site" in combined.columns and not combined.empty:
        log("\nBreakdown by site:")
        log(combined.groupby("site")["title"].count().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()






# """
# Germany AI/Data Job Scraper
# Scrapes LinkedIn, Indeed, StepStone, and Bundesagentur für Arbeit.
# Searches both "Germany" and "Deutschland" location strings to maximise coverage.
# Output: jobs_today.csv

# Install deps first:
#     pip install jobspy pandas requests
# """

# from jobspy import scrape_jobs
# import pandas as pd
# import requests
# import xml.etree.ElementTree as ET
# import time
# from datetime import date

# # ── Search terms ─────────────────────────────────────────────
# # SEARCH_TERMS = [
# #     # General AI / ML
# #     "Artificial Intelligence Engineer",
# #     "AI Engineer",
# #     "Machine Learning Engineer",
# #     "ML Engineer",
# #     "Deep Learning Engineer",
# #     "Generative AI Engineer",
# #     "LLM Engineer",
# #     "Applied AI Engineer",
# #     "AI Research Engineer",
# #     "AI Research Scientist",
# #     "AI Scientist",
# #     "AI Developer",
# #     "AI Software Engineer",
# #     "AI Architect",
# #     "AI Consultant",
# #     # Data Science
# #     "Data Scientist",
# #     "Senior Data Scientist",
# #     "Junior Data Scientist",
# #     "Lead Data Scientist",
# #     "Principal Data Scientist",
# #     "Applied Scientist",
# #     "Research Scientist",
# #     "Quantitative Data Scientist",
# #     # Data Analysis
# #     "Data Analyst",
# #     "Business Data Analyst",
# #     "BI Analyst",
# #     "Business Intelligence Analyst",
# #     "Analytics Engineer",
# #     "Product Analyst",
# #     "Marketing Analyst",
# #     "Financial Analyst",
# #     "Reporting Analyst",
# #     # Data Engineering
# #     "Data Engineer",
# #     "Big Data Engineer",
# #     "ETL Developer",
# #     "ETL Engineer",
# #     "Data Platform Engineer",
# #     "Data Warehouse Engineer",
# #     "Cloud Data Engineer",
# #     "Pipeline Engineer",
# #     # Machine Learning Specializations
# #     "Machine Learning Scientist",
# #     "Machine Learning Researcher",
# #     "MLOps Engineer",
# #     "ML Ops Engineer",
# #     "AI Ops Engineer",
# #     "Model Engineer",
# #     "Inference Engineer",
# #     "Recommendation Systems Engineer",
# #     # NLP / LLM
# #     "NLP Engineer",
# #     "Natural Language Processing Engineer",
# #     "Computational Linguist",
# #     "LLM Developer",
# #     "Prompt Engineer",
# #     "Conversational AI Engineer",
# #     "Chatbot Developer",
# #     # Computer Vision
# #     "Computer Vision Engineer",
# #     "Computer Vision Scientist",
# #     "Image Processing Engineer",
# #     "Vision AI Engineer",
# #     # Robotics / Autonomous Systems
# #     "Robotics Engineer",
# #     "Autonomous Systems Engineer",
# #     "Perception Engineer",
# #     # Deep Learning
# #     "Deep Learning Researcher",
# #     "Neural Network Engineer",
# #     # AI Infrastructure
# #     "AI Infrastructure Engineer",
# #     "GPU Engineer",
# #     "Distributed ML Engineer",
# #     # Statistics / Quant
# #     "Statistician",
# #     "Quantitative Analyst",
# #     "Quant Analyst",
# #     "Decision Scientist",
# #     # Cloud + AI
# #     "Azure AI Engineer",
# #     "AWS Machine Learning Engineer",
# #     "GCP AI Engineer",
# #     # Specialized Domains
# #     "Fraud Detection Scientist",
# #     "Recommendation Engineer",
# #     "Speech Engineer",
# #     "Speech Recognition Engineer",
# #     "Audio ML Engineer",
# #     "Time Series Forecasting Engineer",
# #     # Entry-level / internship
# #     "Junior Machine Learning Engineer",
# #     "Junior Data Scientist",
# #     # Related software roles
# #     "Python Developer",
# #     "Backend AI Engineer",
# #     "Software Engineer AI",
# #     "AI Platform Engineer",
# # ]

# SEARCH_TERMS = [
#     "Data Scientist",
#     "AI Engineer",
#     "Data Engineer",
# ]

# # Location strings to try for LinkedIn/Indeed and StepStone
# LOCATIONS = ["Germany", "Deutschland"]


# # ── Helpers ───────────────────────────────────────────────────
# def scrape_stepstone(keyword: str, limit_per_location: int = 30) -> pd.DataFrame:
#     """Query StepStone with both 'Deutschland' and 'Germany' location strings,
#     deduplicating by job URL across both queries."""
#     kw = keyword.replace(" ", "+")
#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/124.0.0.0 Safari/537.36"
#         ),
#         "Accept-Language": "de-DE,de;q=0.9",
#         "Accept": "application/rss+xml, application/xml, text/xml, */*",
#     }
#     rows = []
#     seen_urls: set = set()

#     for where in LOCATIONS:
#         url = (
#             f"https://www.stepstone.de/5/ergebnisliste.html"
#             f"?what={kw}&where={where}&rssfeeds=1"
#         )
#         try:
#             r = requests.get(url, headers=headers, timeout=30)
#             r.raise_for_status()
#             root = ET.fromstring(r.content)
#             new_count = 0
#             for item in root.iter("item"):
#                 job_url = item.findtext("link", "")
#                 if job_url in seen_urls:
#                     continue
#                 seen_urls.add(job_url)
#                 rows.append({
#                     "title":       item.findtext("title", ""),
#                     "job_url":     job_url,
#                     "description": item.findtext("description", ""),
#                     "date_posted": item.findtext("pubDate", ""),
#                     "company":     "",
#                     "location":    "Germany",
#                     "site":        "stepstone",
#                 })
#                 new_count += 1
#                 if new_count >= limit_per_location:
#                     break
#             print(f"    → {new_count} new jobs from StepStone [{where}]")
#         except requests.exceptions.Timeout:
#             print(f"    ✗ StepStone timed out [{where}] — skipping")
#         except ET.ParseError:
#             print(f"    ✗ StepStone returned non-XML [{where}] — may be blocking, skipping")
#         except Exception as e:
#             print(f"    ✗ StepStone error [{where}]: {e}")

#     return pd.DataFrame(rows)


# def scrape_arbeitsagentur(keyword: str, size: int = 50) -> pd.DataFrame:
#     """Bundesagentur für Arbeit — single national endpoint, no location variant needed."""
#     url = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v3/jobs"
#     params = {
#         "was":         keyword,
#         "size":        size,
#         "page":        0,
#         "angebotsart": 1,  # full-time jobs
#     }
#     headers = {
#         "X-API-Key":  "jobboerse-jobsuche-ui",
#         "User-Agent": "Mozilla/5.0",
#         "Accept":     "application/json",
#     }
#     try:
#         r = requests.get(url, params=params, headers=headers, timeout=15)
#         if r.status_code != 200:
#             print(f"    ✗ Arbeitsagentur HTTP {r.status_code}: {r.text[:120]}")
#             return pd.DataFrame()
#         jobs = r.json().get("stellenangebote", [])
#         rows = [
#             {
#                 "title":       j.get("titel", ""),
#                 "company":     j.get("arbeitgeber", ""),
#                 "location":    j.get("arbeitsort", {}).get("ort", "Germany"),
#                 "job_url":     (
#                     "https://www.arbeitsagentur.de/jobsuche/jobdetail/"
#                     + j.get("hashId", "")
#                 ),
#                 "description": j.get("stellenbeschreibung", "") or "",
#                 "site":        "arbeitsagentur",
#             }
#             for j in jobs
#         ]
#         print(f"    → {len(rows)} jobs from Arbeitsagentur")
#         return pd.DataFrame(rows)
#     except Exception as e:
#         print(f"    ✗ Arbeitsagentur error: {e}")
#         return pd.DataFrame()


# def check_endpoints() -> None:
#     print("Checking Arbeitsagentur endpoint…")
#     try:
#         r = requests.get(
#             "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v3/jobs",
#             params={"was": "test", "size": 1, "angebotsart": 1},
#             headers={"X-API-Key": "jobboerse-jobsuche-ui", "Accept": "application/json"},
#             timeout=10,
#         )
#         status = "✓ OK" if r.status_code == 200 else f"✗ {r.status_code}"
#         print(f"  Arbeitsagentur: {status}")
#     except Exception as e:
#         print(f"  Arbeitsagentur: ✗ {e}")


# # ── Main ──────────────────────────────────────────────────────
# def main() -> None:
#     check_endpoints()

#     all_jobs: list = []
#     total = len(SEARCH_TERMS)

#     for idx, term in enumerate(SEARCH_TERMS, 1):
#         print(f"\n[{idx}/{total}] '{term}'")

#         # LinkedIn + Indeed — try both "Germany" and "Deutschland"
#         for location in LOCATIONS:
#             try:
#                 jobs = scrape_jobs(
#                     site_name=["linkedin", "indeed"],
#                     search_term=term,
#                     location=location,
#                     results_wanted=50,
#                     hours_old=24,
#                     country_indeed="Germany",
#                     linkedin_fetch_description=True,
#                     verbose=0,
#                 )
#                 all_jobs.append(jobs)
#                 print(f"    → {len(jobs)} jobs from LinkedIn/Indeed [{location}]")
#             except Exception as e:
#                 print(f"    ✗ jobspy error [{location}]: {e}")
#             time.sleep(2)  # brief pause between location variants

#         # StepStone — both location strings handled inside the function
#         all_jobs.append(scrape_stepstone(term))

#         # Bundesagentur für Arbeit — national API, no location variant needed
#         all_jobs.append(scrape_arbeitsagentur(term))

#         # Polite delay before next search term
#         time.sleep(3)

#     # ── Combine & deduplicate ─────────────────────────────────
#     combined = pd.concat(all_jobs, ignore_index=True)
#     combined.drop_duplicates(subset=["title", "company"], keep="first", inplace=True)
#     combined = combined[combined["title"].notna() & (combined["title"] != "")]
#     combined["scraped_date"] = str(date.today())

#     out_path = "jobs_today.csv"
#     combined.to_csv(out_path, index=False)

#     print(f"\n{'─'*50}")
#     print(f"✓ {len(combined)} unique jobs saved → {out_path}")
#     print("\nBreakdown by site:")
#     if "site" in combined.columns:
#         print(combined.groupby("site")["title"].count().to_string())
#     else:
#         print("  (no 'site' column found — jobspy may use different column names)")


# if __name__ == "__main__":
#     main()
