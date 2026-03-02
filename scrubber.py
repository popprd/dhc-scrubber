#!/usr/bin/env python3
"""
DHC Scrubber
============
Categorizes imaging centers from Imaging Centers.csv by:
  1. Scraping each center's website for service keywords
  2. Falling back to center name + Primary Taxonomy if website is unavailable

Categories assigned
-------------------
  Offers MRI/CT        – Website or name confirms MRI or CT services
  Mammography Only     – Breast / mammography focus; no MRI or CT detected
  Ultrasound Only      – Sonography / echo focus; no MRI or CT detected
  Cancer Center        – Radiation oncology / cancer-treatment facility
  Nuclear Medicine/PET – Nuclear medicine, PET, or SPECT imaging
  Ignore               – Mobile X-ray, plain radiography, urgent care/ER, pain clinic, non-imaging
  Closed               – Website explicitly closed or HTTP 404 + closed signals
  Needs Review         – Insufficient data to classify

Usage
-----
  python scrubber.py                   # process all unprocessed rows
  python scrubber.py --limit 100       # process only first 100 unprocessed
  python scrubber.py --workers 10      # change concurrency (default 8)
  python scrubber.py --reset           # wipe progress and start fresh
"""

import argparse
import json
import logging
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

# ── Suppress SSL warnings ──────────────────────────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "Imaging Centers.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "Imaging Centers - Categorized.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "progress.json")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Category keyword patterns ──────────────────────────────────────────────────

MRI_CT_RE = re.compile(
    r"\b("
    r"mri|magnetic\s+resonance|"
    r"ct\s+scan|computed\s+tomography|cat\s+scan|ct\s+imaging|"
    r"pet[\s/\-]?ct|pet\s+scan|pet\s+imaging|pet\/ct|"
    r"3[\.\s]?t\s+mri|1[\.\s]?5\s*t|open\s+mri|wide[\s\-]bore\s+mri|"
    r"body\s+mri|brain\s+mri|spine\s+mri|cardiac\s+mri|"
    r"ct\s+angiograph|cta\b|coronary\s+ct|cardiac\s+ct|"
    r"low[\s\-]dose\s+ct|lung\s+screening\s+ct|"
    r"cross[\s\-]sectional\s+imag"
    r")\b",
    re.IGNORECASE,
)

MAMMOGRAPHY_RE = re.compile(
    r"\b("
    r"mammograph|mammogram|tomosynthesis|3d\s+mamm|"
    r"breast\s+imag|breast\s+center|breast\s+health|"
    r"breast\s+screen|breast\s+cancer\s+screen|"
    r"digital\s+mammograph|breast\s+mri"       # breast MRI is mammography-adjacent
    r")\b",
    re.IGNORECASE,
)

NUCLEAR_RE = re.compile(
    r"\b("
    r"nuclear\s+medicine|nuclear\s+imaging|"
    r"pet\s+scan|pet\s+imaging|pet\/ct|pet[\s\-]ct|"
    r"positron\s+emission|spect\s+scan|spect\s+imaging|"
    r"gallium\s+scan|thyroid\s+scan|bone\s+scan(?!\s+mri)"
    r")\b",
    re.IGNORECASE,
)

CANCER_RE = re.compile(
    r"\b("
    r"cancer\s+center|cancer\s+care|cancer\s+treatment|"
    r"oncology\s+center|radiation\s+oncology|radiation\s+therapy|"
    r"chemotherapy|proton\s+therapy|cyberknife|gamma\s+knife|"
    r"tumor\s+board|linear\s+accelerator|\blinac\b|"
    r"stereotactic\s+radiosurgery|sbrt\b|imrt\b"
    r")\b",
    re.IGNORECASE,
)

ULTRASOUND_RE = re.compile(
    r"\b("
    r"ultrasound|sonograph|sonogram|"
    r"echocardiograph|echo\s+imag|vascular\s+ultrasound|"
    r"abdominal\s+ultrasound|pelvic\s+ultrasound|"
    r"obstetric\s+ultrasound|ob\s+ultrasound|duplex\s+scan"
    r")\b",
    re.IGNORECASE,
)

MOBILE_RE = re.compile(
    r"\b("
    r"mobile\s+x[\s\-]?ray|portable\s+x[\s\-]?ray|"
    r"mobile\s+imag|mobile\s+radiolog|portable\s+radiolog|"
    r"mobile\s+mammo|mobile\s+unit|roving\s+radiolog|"
    r"visiting\s+radiolog|bedside\s+x[\s\-]?ray"
    r")\b",
    re.IGNORECASE,
)

URGENT_CARE_RE = re.compile(
    r"\b("
    r"urgent\s+care|emergency\s+room|emergency\s+care|"
    r"er\s+service|\bfreestanding\s+er\b|freestanding\s+emergency|"
    r"walk[\s\-]in\s+clinic|walk[\s\-]in\s+urgent"
    r")\b",
    re.IGNORECASE,
)

NON_IMAGING_RE = re.compile(
    r"\b("
    r"pain\s+management|pain\s+clinic|pain\s+center|"
    r"interventional\s+pain|anesthesia\s+pain|"
    r"physical\s+therapy|chiropractic|chiropractor|"
    r"dermatolog|dental|optical|eye\s+care|vision\s+care|"
    r"hearing|pharmacy|weight\s+loss|sleep\s+clinic|"
    r"lab\s+only|laboratory\s+services\s+only"
    r")\b",
    re.IGNORECASE,
)

CLOSED_PAGE_RE = re.compile(
    r"\b("
    r"permanently\s+closed|we\s+have\s+closed|we\s+are\s+closed|"
    r"no\s+longer\s+in\s+business|no\s+longer\s+accepting|"
    r"out\s+of\s+business|this\s+location\s+is\s+closed|"
    r"this\s+office\s+is\s+closed|site\s+is\s+closed|"
    r"thank\s+you\s+for\s+your\s+years|we\s+have\s+served\s+our\s+last"
    r")\b",
    re.IGNORECASE,
)

# ── Name-based fallback patterns ───────────────────────────────────────────────

NAME_MRI_CT_RE = re.compile(
    r"\b("
    r"mri|\bct\b|magnetic\s+resonance|computed\s+tomography|"
    r"cross[\s\-]sectional|body\s+imag|diagnostic\s+imag|"
    r"radiology|imaging\s+centers?|medical\s+imag|advanced\s+imag"
    r")\b",
    re.IGNORECASE,
)

NAME_MAMMOGRAPHY_RE = re.compile(
    r"\b("
    r"mamm|breast\s+imag|breast\s+center|breast\s+health|"
    r"womens?\s+imag|women['\s]+s\s+health\s+imag"
    r")\b",
    re.IGNORECASE,
)

NAME_CANCER_RE = re.compile(
    r"\b(cancer|oncolog|radiation\s+therapy)\b",
    re.IGNORECASE,
)

NAME_MOBILE_RE = re.compile(
    r"\b(mobile|portable)\b",
    re.IGNORECASE,
)

NAME_ULTRASOUND_RE = re.compile(
    r"\b(ultrasound|sonograph|sonogram|echo)\b",
    re.IGNORECASE,
)

NAME_URGENT_RE = re.compile(
    r"\b(urgent\s+care|emergency\s+room|\bfreestanding\s+er\b)\b",
    re.IGNORECASE,
)

NAME_PAIN_RE = re.compile(
    r"\b(pain\s+management|pain\s+clinic|interventional\s+pain)\b",
    re.IGNORECASE,
)

# ── Taxonomy lookup sets ───────────────────────────────────────────────────────

TAXONOMY_MRI_CT = {
    "Radiology",
    "Diagnostic Radiology",
    "Magnetic Resonance Imaging (MRI)",
    "Magnetic Resonance Imaging",
    "Body Imaging",
}

TAXONOMY_NUCLEAR = {
    "Nuclear Radiology",
    "Nuclear Medicine",
}

TAXONOMY_IGNORE = {
    "Radiography",   # plain X-ray only
}

TAXONOMY_MOBILE = {
    "Radiology, Mobile",
    "Portable Xray Supplier",
    "Radiology, Mobile Mammography",   # mobile mammo → Ignore
}

TAXONOMY_MAMMOGRAPHY = {
    "Radiology, Mammography",
}

TAXONOMY_ULTRASOUND = {
    "Diagnostic Ultrasound",
    "Radiologic Technologist, Sonography",
    "Specialist/Technologist Cardiovascular, Sonography",
}

TAXONOMY_CANCER = {
    "Oncology, Radiation",
}

# ── HTTP fetch ─────────────────────────────────────────────────────────────────

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(url: str, timeout: int = 10):
    """Single GET attempt, returns requests.Response or raises."""
    return requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=timeout,
        allow_redirects=True,
        verify=False,
    )


def fetch_website_text(raw_url: str, timeout: int = 10):
    """
    Fetch a URL and return (text, error_message).
    Tries HTTPS first, then HTTP on SSL errors.
    text is capped at 15,000 characters of visible page content.
    """
    url = raw_url.strip()
    if not url:
        return None, "No URL provided"
    if not url.startswith("http"):
        url = "https://" + url

    try:
        resp = _get(url, timeout)
    except requests.exceptions.SSLError:
        try:
            url = url.replace("https://", "http://", 1)
            resp = _get(url, timeout)
        except Exception as e:
            return None, f"SSL error + HTTP fallback failed: {str(e)[:120]}"
    except requests.exceptions.ConnectionError:
        return None, "Connection error (site may be down or domain invalid)"
    except requests.exceptions.Timeout:
        return None, f"Request timed out after {timeout}s"
    except Exception as e:
        return None, f"Fetch error: {str(e)[:120]}"

    if resp.status_code == 404:
        return None, "HTTP 404 – page not found"
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:15_000], None


# ── Categorization logic ───────────────────────────────────────────────────────

def _first_match(pattern, text):
    """Return first match string or None."""
    m = pattern.search(text)
    return m.group().strip() if m else None


def _all_matches(pattern, text, limit=4):
    """Return up to `limit` unique matched strings (lowercased)."""
    return list({m.lower() for m in pattern.findall(text)})[:limit]


def categorize_from_website(text: str):
    """
    Categorize based solely on website text.
    Returns (category, rationale) or (None, None) if undetermined.
    """
    # 1. Closed signal
    m = _first_match(CLOSED_PAGE_RE, text)
    if m:
        return "Closed", f'Website content indicates closure ("{m}").'

    # 2. Cancer center (before MRI/CT – cancer centers may also have MRI)
    m = _first_match(CANCER_RE, text)
    if m:
        also_mri = bool(MRI_CT_RE.search(text))
        note = " Also lists MRI/CT services." if also_mri else ""
        return "Cancer Center", f'Website mentions cancer/oncology treatment ("{m}").{note}'

    # 3. Nuclear medicine / PET — but if MRI/CT is also present, promote to Offers MRI/CT
    m = _first_match(NUCLEAR_RE, text)
    if m:
        also_mri = bool(MRI_CT_RE.search(text))
        if also_mri:
            mri_terms = ", ".join(_all_matches(MRI_CT_RE, text))
            return "Offers MRI/CT", f'Website confirms MRI or CT services ({mri_terms}). Also mentions nuclear/PET imaging ("{m}").'
        return "Nuclear Medicine / PET", f'Website mentions nuclear medicine or PET imaging ("{m}"). No MRI or CT services detected.'

    # 4. Mobile / portable (check before MRI/CT — mobile MRI vans should be Ignore)
    m = _first_match(MOBILE_RE, text)
    if m:
        return "Ignore", f'Website indicates mobile/portable imaging services ("{m}"). Not a fixed imaging center.'

    # 5. Urgent care / ER (check before MRI/CT — ERs have CT scanners but are not imaging centers)
    m = _first_match(URGENT_CARE_RE, text)
    if m:
        return "Ignore", f'Website indicates urgent care or ER services ("{m}"). Imaging is incidental, not a dedicated imaging center.'

    # 6. MRI / CT
    matches = _all_matches(MRI_CT_RE, text)
    if matches:
        terms = ", ".join(matches)
        return "Offers MRI/CT", f"Website confirms MRI or CT services ({terms})."

    # 7. Non-imaging services (pain, PT, etc.)
    m = _first_match(NON_IMAGING_RE, text)
    if m:
        return "Ignore", f'Website primarily describes non-imaging services ("{m}").'

    # 8. Mammography only
    m = _first_match(MAMMOGRAPHY_RE, text)
    if m:
        return "Mammography Only", f'Website focuses on mammography/breast imaging ("{m}"). No MRI or CT services detected.'

    # 9. Ultrasound only
    m = _first_match(ULTRASOUND_RE, text)
    if m:
        return "Ultrasound Only", f'Website focuses on ultrasound/sonography ("{m}"). No MRI or CT services detected.'

    return None, None  # undetermined


def categorize_from_name_taxonomy(name: str, taxonomy: str, prefix: str = ""):
    """
    Fallback classification using center name and Primary Taxonomy.
    Returns (category, rationale).
    """
    p = prefix.strip() + " " if prefix.strip() else ""

    # Taxonomy takes priority over name
    if taxonomy in TAXONOMY_MOBILE:
        return "Ignore", f'{p}Taxonomy "{taxonomy}" indicates mobile/portable services.'
    if taxonomy in TAXONOMY_IGNORE:
        return "Ignore", f'{p}Taxonomy "{taxonomy}" indicates plain X-ray/radiography only — not MRI or CT.'
    if taxonomy in TAXONOMY_CANCER:
        return "Cancer Center", f'{p}Taxonomy "{taxonomy}" indicates radiation oncology.'
    if taxonomy in TAXONOMY_NUCLEAR:
        return "Nuclear Medicine / PET", f'{p}Taxonomy "{taxonomy}" indicates nuclear medicine or PET imaging.'
    if taxonomy in TAXONOMY_MRI_CT:
        return "Offers MRI/CT", f'{p}Taxonomy "{taxonomy}" indicates advanced diagnostic imaging.'
    if taxonomy in TAXONOMY_MAMMOGRAPHY:
        return "Mammography Only", f'{p}Taxonomy "{taxonomy}" indicates mammography services.'
    if taxonomy in TAXONOMY_ULTRASOUND:
        return "Ultrasound Only", f'{p}Taxonomy indicates ultrasound/sonography services.'

    # Name-based
    m = _first_match(NAME_CANCER_RE, name)
    if m:
        return "Cancer Center", f'{p}Center name suggests cancer/oncology services ("{m}").'

    m = _first_match(NAME_MOBILE_RE, name)
    if m:
        return "Ignore", f'{p}Center name contains "{m}" suggesting mobile/portable services.'

    m = _first_match(NAME_URGENT_RE, name)
    if m:
        return "Ignore", f'{p}Center name suggests urgent care or ER ("{m}"). Not a dedicated imaging center.'

    m = _first_match(NAME_PAIN_RE, name)
    if m:
        return "Ignore", f'{p}Center name suggests pain management services ("{m}").'

    m = _first_match(NAME_MRI_CT_RE, name)
    if m:
        return "Offers MRI/CT", f'{p}Center name contains "{m}" suggesting MRI/CT services.'

    m = _first_match(NAME_MAMMOGRAPHY_RE, name)
    if m:
        return "Mammography Only", f'{p}Center name suggests breast/mammography focus ("{m}"). No MRI or CT detected.'

    m = _first_match(NAME_ULTRASOUND_RE, name)
    if m:
        return "Ultrasound Only", f'{p}Center name suggests ultrasound/sonography focus ("{m}").'

    return "Needs Review", f'{p}Could not determine services from name or taxonomy. Manual review recommended.'


def classify_center(name: str, taxonomy: str, website_url: str):
    """
    Full classification pipeline for one center.
    Returns (category, rationale).
    """
    website_text = None
    website_error = None

    has_url = bool(website_url and str(website_url).strip() and str(website_url).strip().lower() != "nan")

    if has_url:
        website_text, website_error = fetch_website_text(website_url)
        time.sleep(0.15)  # polite crawl delay

    if website_text:
        category, rationale = categorize_from_website(website_text)
        if category:
            return category, f"[Website] {rationale}"
        # Website loaded but no decisive keywords → try name/taxonomy as tiebreaker
        prefix = "Website loaded but services could not be determined from content."
        category, rationale = categorize_from_name_taxonomy(name, taxonomy, prefix=prefix)
        return category, f"[Website + Name/Taxonomy] {rationale}"

    # No website or fetch failed
    if has_url and website_error:
        prefix = f"Website unavailable ({website_error})."
    else:
        prefix = "No website listed."

    category, rationale = categorize_from_name_taxonomy(name, taxonomy, prefix=prefix)
    return category, f"[Name/Taxonomy] {rationale}"


# ── Worker ─────────────────────────────────────────────────────────────────────

def process_row(args):
    """Called by thread pool. Returns (df_index, category, rationale)."""
    df_idx, row = args
    name     = str(row.get("Imaging Center Name", "")).strip()
    taxonomy = str(row.get("Primary Taxonomy ", "")).strip()
    website  = str(row.get("Website", "")).strip()

    try:
        category, rationale = classify_center(name, taxonomy, website)
    except Exception as exc:
        category  = "Needs Review"
        rationale = f"[Error] Unexpected error during processing: {str(exc)[:200]}"

    return df_idx, category, rationale.strip()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DHC Imaging Center Scrubber")
    parser.add_argument("--limit",   type=int, default=0,  help="Max centers to process (0 = all)")
    parser.add_argument("--workers", type=int, default=8,  help="Concurrent threads (default 8)")
    parser.add_argument("--reset",   action="store_true",  help="Clear progress and start fresh")
    args = parser.parse_args()

    # ── Load CSV ──
    logger.info("Loading %s …", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", dtype=str, low_memory=False)
    df["Category"]  = df["Category"].fillna("")
    df["Rationale"] = df["Rationale"].fillna("")

    # Active rows only (have a Definitive ID / name)
    active_mask = df["Imaging Center Name"].fillna("").str.strip().ne("")
    active_df   = df[active_mask]
    logger.info("Active centers in file: %s", f"{len(active_df):,}")

    # ── Progress ──
    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        logger.info("Progress reset.")

    progress: dict = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as fh:
            progress = json.load(fh)
        logger.info("Resuming — %s centers already done.", f"{len(progress):,}")

    # Apply previously saved results back to df
    for idx_str, result in progress.items():
        df.at[int(idx_str), "Category"]  = result["category"]
        df.at[int(idx_str), "Rationale"] = result["rationale"]

    # Build work queue (skip already done)
    to_process = [
        (idx, row)
        for idx, row in active_df.iterrows()
        if str(idx) not in progress
    ]

    if args.limit > 0:
        to_process = to_process[: args.limit]

    logger.info("Centers remaining to process: %s", f"{len(to_process):,}")
    if not to_process:
        logger.info("Nothing to do — writing output.")
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        return

    # ── Process ──
    SAVE_EVERY = 100
    done_count = 0

    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(to_process), desc="Categorizing", unit="center")
    except ImportError:
        progress_bar = None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_row, item): item[0] for item in to_process}

        for future in as_completed(futures):
            try:
                df_idx, category, rationale = future.result()
            except Exception as exc:
                logger.error("Future failed: %s", exc)
                if progress_bar:
                    progress_bar.update(1)
                continue

            df.at[df_idx, "Category"]  = category
            df.at[df_idx, "Rationale"] = rationale
            progress[str(df_idx)] = {"category": category, "rationale": rationale}
            done_count += 1

            if progress_bar:
                progress_bar.set_postfix(cat=category[:18])
                progress_bar.update(1)

            # Periodic checkpoint
            if done_count % SAVE_EVERY == 0:
                _save(df, progress)
                logger.info("Checkpoint: %s processed so far.", f"{done_count:,}")

    if progress_bar:
        progress_bar.close()

    _save(df, progress)
    logger.info("Finished. Output → %s", OUTPUT_CSV)

    # ── Summary ──
    counts = df[active_mask]["Category"].value_counts()
    logger.info("\n── Category Summary ───────────────────────────────")
    for cat, n in counts.items():
        bar = "█" * min(40, int(40 * n / len(active_df)))
        logger.info("  %-22s %5s  %s", cat, f"{n:,}", bar)
    logger.info("────────────────────────────────────────────────────")
    needs_review = counts.get("Needs Review", 0)
    logger.info("  %s centers need manual review.", f"{needs_review:,}")


def _save(df: pd.DataFrame, progress: dict):
    """Save progress JSON and output CSV."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as fh:
        json.dump(progress, fh)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
