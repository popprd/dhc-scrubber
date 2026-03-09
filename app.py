"""
DHC Scrubber – Streamlit App
Lumexa Imaging theme
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scrubber import classify_center

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DHC Scrubber | Lumexa Imaging",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lumexa theme ───────────────────────────────────────────────────────────────
NAVY    = "#192F6D"
BLUE    = "#245285"
CYAN    = "#4b9ed7"
GREEN   = "#669e49"
LOGO_URL = "https://www.lumexaimaging.com/wp-content/uploads/2025/09/lumexa-imaging-logo-lt-300x110.png"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

*, body, .stApp {{
    font-family: 'Inter', Arial, sans-serif;
}}

[data-testid="stSidebar"] {{
    background-color: {NAVY};
}}
[data-testid="stSidebar"] * {{
    color: #ffffff !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stMultiSelect label {{
    color: #ccd6f6 !important;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
[data-testid="stSidebar"] hr {{
    border-color: {BLUE} !important;
    opacity: 0.5;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: #ccd6f6 !important;
    font-size: 0.82rem;
}}

[data-testid="stButton"] > button[kind="primary"] {{
    background-color: {GREEN} !important;
    border-color: {GREEN} !important;
    color: #ffffff !important;
    font-weight: 600;
    border-radius: 6px;
}}
[data-testid="stButton"] > button[kind="primary"]:hover {{
    background-color: #5a8e40 !important;
}}
[data-testid="stButton"] > button[kind="secondary"] {{
    background-color: transparent !important;
    border: 1px solid {CYAN} !important;
    color: {CYAN} !important;
    font-weight: 600;
    border-radius: 6px;
}}

[data-testid="stMetric"] {{
    background-color: #f4f7fc;
    border-left: 4px solid {CYAN};
    border-radius: 6px;
    padding: 0.75rem 1rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    color: {NAVY} !important;
    letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {NAVY} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background-color: #f0f4fa;
    border-radius: 8px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: {NAVY};
    font-weight: 600;
    border-radius: 6px;
    padding: 0.4rem 1.2rem;
}}
.stTabs [aria-selected="true"] {{
    background-color: {NAVY} !important;
    color: #ffffff !important;
    border-bottom: none !important;
}}

.page-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 2px solid {CYAN};
    margin-bottom: 1.5rem;
}}
.page-header h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {NAVY};
    margin: 0;
}}
.page-header p {{
    font-size: 0.85rem;
    color: #555;
    margin: 0;
}}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORIES = [
    "HOPD",
    "ONO",
    "Offers MRI/CT",
    "Mammography Only",
    "Ultrasound Only",
    "Cancer Center",
    "Nuclear Medicine / PET",
    "Ignore",
    "Closed",
    "Needs Review",
]

CATEGORY_COLORS = {
    "HOPD":                   "#5c6bc0",
    "ONO":                    "#26a69a",
    "Offers MRI/CT":          CYAN,
    "Mammography Only":       "#f48fb1",
    "Ultrasound Only":        "#81c784",
    "Cancer Center":          "#ffb74d",
    "Nuclear Medicine / PET": "#ce93d8",
    "Ignore":                 "#bdbdbd",
    "Closed":                 "#ef9a9a",
    "Needs Review":           "#fff176",
}

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "Imaging Centers - Categorized.csv")

def _find_master_csv():
    """
    Search several candidate locations for the master imaging-center CSV.
    Returns the first path that exists, or None if none is found.
    Checks BASE_DIR and cwd so the app works both locally and on Streamlit Cloud.
    """
    filenames = ["All Imaging Centers.csv", "Imaging Centers.csv"]
    search_dirs = list(dict.fromkeys([BASE_DIR, os.getcwd()]))   # dedup, preserve order
    for d in search_dirs:
        for fn in filenames:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                return p
    return None

MASTER_CSV_PATH = _find_master_csv()

# ── Column detection ────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "name":      [
        "Imaging Center Name", "Center Name", "Facility Name", "Practice Name",
        "Name", "Organization Name", "Organization", "Facility", "Provider Name",
    ],
    "website":   [
        "Website", "Website URL", "URL", "Web", "Web Address", "Site", "Homepage",
        "Web Site", "Practice Website",
    ],
    "taxonomy":  [
        "Primary Taxonomy ",          # trailing space in original DHC export
        "Primary Taxonomy", "Taxonomy", "Provider Type", "Specialty",
        "Taxonomy Code Description", "NPI Taxonomy", "Taxonomy Description",
    ],
    "city":      ["City", "City Name", "Town", "Municipality", "Practice City"],
    "state":     ["State", "State Code", "ST", "State Abbrev", "State/Province", "Practice State"],
    "zip":       [
        "Zip Code (Cleansed)", "Zip Code", "ZIP", "Postal Code", "Zip", "ZipCode",
        "Postal", "PostalCode",
    ],
    "cbsa":      ["CBSA", "MSA", "Metro Area", "Market", "CBSA Name", "Metro", "Metropolitan Area",
                  "MSA Lookup"],
    "dhc_link":  ["DHC Profile Link", "DHC Link", "Profile URL", "DHC URL", "Profile Link", "DHC Profile"],
    "health_system": [
        "Health System Affiliated (0=Health System)",
        "Health System Affiliated",
        "Health System",
        "HS Affiliated",
    ],
    "physician_group": [
        "Physician Group Parent Name",
        "Physician Group Parent",
        "Physician Group Name",
        "Physician Group",
        "Physician Grp Parent Name",
    ],
    "address":   ["Address", "Address1", "Street Address", "Street"],
    "phone":     ["Phone", "Phone Number", "Telephone", "Tel"],
    "parent":    ["Parent (Master)", "Parent Master", "Parent", "Network Parent",
                  "Network Parent (or Network IF Network Parent is blank)"],
    "scrubber_category":  ["Scrubber Category"],
    "scrubber_rationale": ["Scrubber Rationale"],
}

FRIENDLY_NAMES = {
    "name":             "Center Name  *(required)*",
    "website":          "Website URL",
    "taxonomy":         "Primary Taxonomy",
    "city":             "City",
    "state":            "State",
    "zip":              "Zip Code",
    "cbsa":             "CBSA / Market",
    "dhc_link":         "DHC Profile Link",
    "health_system":    "Health System Affiliated",
    "physician_group":  "Physician Group Parent Name",
    "address":          "Address",
    "phone":            "Phone",
    "scrubber_category":  "Scrubber Category  *(output)*",
    "scrubber_rationale": "Scrubber Rationale  *(output)*",
}

INPUT_KEYS  = ["name", "website", "taxonomy", "city", "state", "zip", "cbsa",
               "dhc_link", "health_system", "physician_group", "address", "phone"]
OUTPUT_KEYS = ["scrubber_category", "scrubber_rationale"]


def detect_columns(df_columns: list) -> dict:
    cols_norm = {c.strip().lower(): c for c in df_columns}
    col_map = {}
    for key, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias.strip().lower() in cols_norm:
                found = cols_norm[alias.strip().lower()]
                break
        col_map[key] = found
    return col_map


# ── CSV loading ─────────────────────────────────────────────────────────────────
# Use cache_resource (no serialization) so Streamlit doesn't try to pickle
# a large DataFrame — that can exhaust memory or disk on Cloud deployments.

@st.cache_resource(show_spinner="Loading imaging center database…")
def _load_csv_shared(path: str):
    """Load and cache ONE shared copy of the CSV (no serialization overhead)."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    col_map = detect_columns(df.columns.tolist())

    if not col_map["scrubber_category"]:
        df["Scrubber Category"] = ""
        col_map["scrubber_category"] = "Scrubber Category"
    df[col_map["scrubber_category"]] = df[col_map["scrubber_category"]].fillna("")

    if not col_map["scrubber_rationale"]:
        df["Scrubber Rationale"] = ""
        col_map["scrubber_rationale"] = "Scrubber Rationale"
    df[col_map["scrubber_rationale"]] = df[col_map["scrubber_rationale"]].fillna("")

    return df, col_map


def load_master_csv(path: str):
    """Return a per-session copy so edits don't bleed across sessions."""
    df, col_map = _load_csv_shared(path)
    return df.copy(), dict(col_map)


# ── Session state initialization ───────────────────────────────────────────────
if "master_df" not in st.session_state:
    if MASTER_CSV_PATH is None:
        st.error(
            "**Imaging center database not found.**\n\n"
            "Expected `All Imaging Centers.csv` or `Imaging Centers.csv` "
            f"in the app directory.  Looked in: `{BASE_DIR}` and `{os.getcwd()}`."
        )
        st.stop()
    try:
        _df, _cmap = load_master_csv(MASTER_CSV_PATH)
    except Exception as _e:
        st.error(f"**Failed to load database:** {_e}")
        st.stop()
    st.session_state["master_df"]  = _df
    st.session_state["col_map"]    = _cmap
    st.session_state["unsaved"]    = False

if "discover_results" not in st.session_state:
    st.session_state["discover_results"] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def C(key: str, default: str = None):
    """Resolve a logical column key → actual DataFrame column name."""
    val = st.session_state.get("col_map", {}).get(key)
    return val if val else default


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    name_col = C("name")
    if name_col and name_col in df.columns:
        return df[df[name_col].fillna("").str.strip().ne("")]
    return df


def filter_by_location(df: pd.DataFrame, states: list, zips: list) -> pd.DataFrame:
    state_col = C("state")
    zip_col   = C("zip")
    result = df.copy()
    if states and state_col and state_col in result.columns:
        result = result[result[state_col].isin(states)]
    if zips and zip_col and zip_col in result.columns:
        result = result[result[zip_col].isin(zips)]
    return result


def to_download_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def extract_domain(url: str) -> str:
    if not url or str(url).strip().lower() in ("nan", "", "none"):
        return ""
    try:
        u = url.strip()
        if "://" not in u:
            u = "https://" + u
        return urlparse(u).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def run_scrubber(df: pd.DataFrame, indices, workers: int = 8):
    """
    Run classify_center on the given row indices.
    Writes results to the Scrubber Category/Rationale columns in-place.
    Yields (df, done, total) tuples.
    """
    col_map      = st.session_state["col_map"]
    name_col     = col_map.get("name", "Imaging Center Name")
    taxonomy_col = col_map.get("taxonomy")
    website_col  = col_map.get("website")
    hs_col       = col_map.get("health_system")
    pg_col       = col_map.get("physician_group")
    cat_col      = col_map.get("scrubber_category", "Scrubber Category")
    rat_col      = col_map.get("scrubber_rationale", "Scrubber Rationale")

    to_process = [(idx, df.loc[idx]) for idx in indices]
    total = len(to_process)
    if total == 0:
        yield df, 0, 0
        return

    done = 0

    def _process(args):
        idx, row = args
        name    = str(row.get(name_col, "")).strip()     if name_col     else ""
        tax     = str(row.get(taxonomy_col, "")).strip() if taxonomy_col else ""
        web     = str(row.get(website_col,  "")).strip() if website_col  else ""
        hs      = str(row.get(hs_col,  "")).strip()      if hs_col       else ""
        pg      = str(row.get(pg_col,  "")).strip()      if pg_col       else ""
        cat, rat = classify_center(name, tax, web, health_system_val=hs, physician_group_name=pg)
        return idx, cat, rat.strip()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, item): item[0] for item in to_process}
        for future in as_completed(futures):
            try:
                idx, cat, rat = future.result()
                df.at[idx, cat_col] = cat
                df.at[idx, rat_col] = rat
            except Exception:
                pass
            done += 1
            yield df, done, total


def build_display_cols(col_map: dict, available: list) -> list:
    order = ["name", "address", "city", "state", "zip", "cbsa", "parent",
             "scrubber_category", "scrubber_rationale",
             "website", "dhc_link"]
    return [col_map[k] for k in order if col_map.get(k) and col_map[k] in available]


def build_column_config(col_map: dict) -> dict:
    config = {}

    for key, label, width in [
        ("name",    "Center Name", "large"),
        ("address", "Address",     "medium"),
        ("city",    "City",        "small"),
        ("state",   "State",       "small"),
        ("zip",     "Zip Code",    "small"),
        ("cbsa",    "CBSA",        "medium"),
        ("parent",  "Parent",      "medium"),
    ]:
        col = col_map.get(key)
        if col:
            config[col] = st.column_config.TextColumn(label, width=width, disabled=True)

    cat_col = col_map.get("scrubber_category", "Scrubber Category")
    config[cat_col] = st.column_config.SelectboxColumn(
        "Scrubber Category", options=CATEGORIES, width="medium", required=True
    )

    rat_col = col_map.get("scrubber_rationale", "Scrubber Rationale")
    config[rat_col] = st.column_config.TextColumn("Scrubber Rationale", width="large", max_chars=500)

    web_col = col_map.get("website")
    if web_col:
        config[web_col] = st.column_config.LinkColumn("Website", width="medium", display_text="🔗 Visit")

    dhc_col = col_map.get("dhc_link")
    if dhc_col:
        config[dhc_col] = st.column_config.LinkColumn("DHC Profile", width="small", display_text="DHC ↗")

    return config


# ── Discover helpers ────────────────────────────────────────────────────────────

def search_and_classify_centers(zip_codes: list, state_name: str, existing_domains: set):
    """
    Use DuckDuckGo Maps to find imaging centers near each zip code.
    Filters out centers already in the master database (by domain).
    Classifies each new center using the scrubber engine.
    Returns a list of result dicts.
    """
    from duckduckgo_search import DDGS

    SEARCH_QUERIES = [
        "MRI imaging center",
        "radiology imaging center",
        "diagnostic imaging center",
        "CT scan center",
    ]

    results   = []
    seen_titles  = set()
    seen_domains = set(existing_domains)

    total_zips = len(zip_codes)
    prog_ph = st.empty()

    with DDGS() as ddgs:
        for zi, zc in enumerate(zip_codes, 1):
            prog_ph.caption(f"Searching zip {zc} ({zi}/{total_zips})…")
            for query in SEARCH_QUERIES:
                try:
                    maps_results = list(ddgs.maps(
                        f"{query} {zc} {state_name}",
                        place=f"{zc}",
                        max_results=8,
                    ))
                except Exception:
                    # maps() unavailable — fall back to text search
                    try:
                        text_results = list(ddgs.text(
                            f"{query} near {zc} {state_name}",
                            max_results=5,
                        ))
                        maps_results = [
                            {
                                "title":   r.get("title", ""),
                                "website": r.get("href", ""),
                                "address": "",
                                "city": "",
                                "state": state_name,
                                "postalCode": zc,
                                "phone": "",
                            }
                            for r in text_results
                        ]
                    except Exception:
                        maps_results = []

                for r in maps_results:
                    title   = (r.get("title") or "").strip()
                    website = (r.get("website") or r.get("href") or "").strip()
                    domain  = extract_domain(website)

                    if not title:
                        continue
                    title_key = title.lower()
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)
                    if domain and domain in seen_domains:
                        continue
                    if domain:
                        seen_domains.add(domain)

                    # Classify
                    try:
                        cat, rat = classify_center(title, "", website or "")
                    except Exception:
                        cat, rat = "Needs Review", "Classification error."

                    results.append({
                        "Name":               title,
                        "Address":            r.get("address", ""),
                        "City":               r.get("city", ""),
                        "State":              r.get("state", state_name),
                        "Zip":                r.get("postalCode", zc),
                        "Phone":              r.get("phone", ""),
                        "Website":            website,
                        "Scrubber Category":  cat,
                        "Scrubber Rationale": rat,
                        "Found Via Zip":      zc,
                    })

                time.sleep(0.3)  # polite delay between queries

    prog_ph.empty()
    return results


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:16px 8px 12px 8px;text-align:center;">'
        f'<img src="{LOGO_URL}" style="max-width:180px;width:100%;" />'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Location filter ────────────────────────────────────────────────────────
    st.markdown("**📍 LOCATION FILTER**")

    _master  = st.session_state["master_df"]
    _cmap    = st.session_state["col_map"]
    _state_c = _cmap.get("state")
    _zip_c   = _cmap.get("zip")

    _states_available = []
    if _state_c and _state_c in _master.columns:
        _states_available = sorted(_master[_state_c].fillna("").str.strip().replace("", pd.NA).dropna().unique().tolist())

    selected_states = st.multiselect(
        "State(s)",
        options=_states_available,
        default=st.session_state.get("selected_states", []),
        placeholder="Select one or more states…",
    )
    st.session_state["selected_states"] = selected_states

    _zips_available = []
    if selected_states and _zip_c and _zip_c in _master.columns:
        _state_mask    = _master[_state_c].isin(selected_states)
        _zips_available = sorted(
            _master[_state_mask][_zip_c].fillna("").str.strip().replace("", pd.NA).dropna().unique().tolist()
        )

    selected_zips = st.multiselect(
        "Zip Code(s)",
        options=_zips_available,
        default=[z for z in st.session_state.get("selected_zips", []) if z in _zips_available],
        placeholder="All zips in selected state(s)" if selected_states else "Select a state first",
        disabled=not selected_states,
    )
    st.session_state["selected_zips"] = selected_zips

    st.markdown("---")
    st.markdown("**⚙️ SCRUBBER SETTINGS**")
    workers = st.slider("Concurrent workers", 1, 16, 8)
    limit   = st.number_input("Row limit (0 = all)", min_value=0, value=0, step=100)

    st.markdown("---")
    st.markdown("**🏷️ RESULTS FILTER**")
    cat_filter  = st.multiselect("Categories", options=CATEGORIES, default=[], placeholder="All categories")
    search_term = st.text_input("Search by name", placeholder="e.g. Solis, RadNet…")

    st.markdown("---")
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
**Database:** Pre-loaded imaging center file.
Select **State(s)** then **Zip Code(s)** to filter the results.

**Categories:**
- 🏥 HOPD — Hospital Outpatient Dept (Health System = 0)
- 🩺 ONO — Specialty-group owned (Ortho, Neuro, etc.)
- 🔵 Offers MRI/CT — Independent center
- 🩷 Mammography Only
- 🟢 Ultrasound Only
- 🟠 Cancer Center
- 🟣 Nuclear Medicine / PET
- ⚪ Ignore
- 🔴 Closed
- 🟡 Needs Review

**Discover tab:** Searches DuckDuckGo Maps for imaging centers not already in the database for your selected zip codes.
        """)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="page-header">'
    f'<img src="{LOGO_URL}" style="height:48px;" />'
    f'<div>'
    f'<h1>DHC Imaging Center Scrubber</h1>'
    f'<p>Categorize imaging centers by services offered using website analysis and keyword matching</p>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── No state selected ──────────────────────────────────────────────────────────
if not selected_states:
    st.info(
        "👈 **Select one or more states** in the sidebar to get started. "
        "Optionally narrow down by zip code(s) after selecting a state.",
        icon=None,
    )
    st.stop()


# ── Resolve working dataframe ──────────────────────────────────────────────────
master_df = st.session_state["master_df"]
col_map   = st.session_state["col_map"]

active_all    = active_rows(master_df)
working_df    = filter_by_location(active_all, selected_states, selected_zips)

cat_col  = C("scrubber_category", "Scrubber Category")
rat_col  = C("scrubber_rationale", "Scrubber Rationale")
name_col = C("name")

categorized_mask   = working_df[cat_col].str.strip().ne("")
uncategorized_mask = working_df[cat_col].str.strip().eq("")
categorized        = working_df[categorized_mask]
uncategorized      = working_df[uncategorized_mask]

# ── Summary metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total (Filtered)",  f"{len(working_df):,}")
c2.metric("Categorized",       f"{len(categorized):,}")
c3.metric("Uncategorized",     f"{len(uncategorized):,}")
c4.metric("HOPD",              f"{(working_df[cat_col] == 'HOPD').sum():,}")
c5.metric("ONO",               f"{(working_df[cat_col] == 'ONO').sum():,}")
c6.metric("Offers MRI/CT",     f"{(working_df[cat_col] == 'Offers MRI/CT').sum():,}")

st.markdown("<br>", unsafe_allow_html=True)

if not name_col:
    st.error("⚠️ **Center Name column not detected.** Check the CSV column names.")
    st.stop()

# ── Run scrubber expander ──────────────────────────────────────────────────────
uncategorized_indices = uncategorized.index.tolist()
if int(limit) > 0:
    uncategorized_indices = uncategorized_indices[:int(limit)]

if len(uncategorized_indices) > 0:
    with st.expander(
        f"▶ **{len(uncategorized_indices):,} uncategorized centers** — click to run scrubber",
        expanded=len(categorized) == 0,
    ):
        _web_note = "" if C("website") else " *(no Website column — will use name/taxonomy only)*"
        est_lo = max(1, len(uncategorized_indices) // 60)
        est_hi = max(2, len(uncategorized_indices) // 40)
        st.caption(
            f"Will run HOPD check, ONO check, website analysis, and keyword matching.{_web_note} "
            f"Estimated time: ~{est_lo}–{est_hi} min at {workers} workers."
        )
        if st.button("🚀 Run Scrubber", type="primary"):
            prog_bar  = st.progress(0.0)
            status_ph = st.empty()
            last_df   = master_df.copy()

            for updated_df, done, total in run_scrubber(last_df, uncategorized_indices, workers=workers):
                if total > 0:
                    pct = done / total
                    prog_bar.progress(pct)
                    status_ph.caption(f"Processing… {done:,} / {total:,}  ({pct*100:.1f}%)")

            st.session_state["master_df"] = last_df
            st.session_state["unsaved"]   = True
            status_ph.success(f"✅ Done! {done:,} centers categorized.")
            save_csv(last_df, DEFAULT_OUTPUT)
            st.session_state["unsaved"] = False
            st.rerun()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_results, tab_discover, tab_summary = st.tabs(["📋 Known Centers", "🔍 Discover New Centers", "📊 Summary"])


# ════════════════════════════════════════════════════════════════════════════════
# KNOWN CENTERS TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_results:

    # Apply sidebar filters to working_df
    view = working_df.copy()
    if cat_filter:
        view = view[view[cat_col].isin(cat_filter)]
    if search_term and name_col and name_col in view.columns:
        view = view[view[name_col].fillna("").str.contains(search_term, case=False, na=False)]

    st.caption(f"Showing **{len(view):,}** of {len(working_df):,} centers")

    show_cols  = build_display_cols(col_map, view.columns.tolist())
    display_df = view[show_cols].copy().reset_index(drop=True)

    # Fix URLs for LinkColumn
    for link_col in [C("website"), C("dhc_link")]:
        if link_col and link_col in display_df.columns:
            def _fix(u):
                if pd.isna(u) or not str(u).strip() or str(u).strip().lower() in ("nan", ""):
                    return None
                u = str(u).strip()
                return u if u.startswith("http") else f"https://{u}"
            display_df[link_col] = display_df[link_col].apply(_fix)

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        height=550,
        column_config=build_column_config(col_map),
    )

    # Write edits back to master
    if not edited.equals(display_df):
        orig_indices = view.index.tolist()
        mdf = st.session_state["master_df"]
        for i, orig_idx in enumerate(orig_indices):
            if i < len(edited):
                if cat_col in edited.columns:
                    mdf.at[orig_idx, cat_col] = edited.at[i, cat_col]
                if rat_col in edited.columns:
                    mdf.at[orig_idx, rat_col] = edited.at[i, rat_col]
        st.session_state["master_df"] = mdf
        st.session_state["unsaved"]   = True

    col_save, col_dl, col_spacer = st.columns([1, 1, 4])

    with col_save:
        if st.session_state.get("unsaved"):
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                save_csv(st.session_state["master_df"], DEFAULT_OUTPUT)
                st.session_state["unsaved"] = False
                st.success("Saved.")
        else:
            st.button("💾 Saved ✓", disabled=True, use_container_width=True)

    with col_dl:
        # Download the filtered view only
        dl_df = filter_by_location(active_rows(st.session_state["master_df"]), selected_states, selected_zips)
        st.download_button(
            label="⬇ Download CSV",
            data=to_download_bytes(dl_df),
            file_name="Imaging Centers - Scrubbed.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.get("unsaved"):
        st.caption("⚠️ You have unsaved edits — click Save Changes or Download CSV.")


# ════════════════════════════════════════════════════════════════════════════════
# DISCOVER NEW CENTERS TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_discover:

    state_col = C("state")
    zip_col   = C("zip")
    web_col   = C("website")

    # Build set of domains already in the master database
    existing_domains: set = set()
    if web_col and web_col in master_df.columns:
        for url in master_df[web_col].dropna():
            d = extract_domain(str(url))
            if d:
                existing_domains.add(d)

    # Determine the state label to pass to search
    state_label = ", ".join(selected_states) if selected_states else ""

    # Which zips to search
    search_zips = selected_zips if selected_zips else _zips_available

    st.markdown("#### 🔍 Discover Imaging Centers Not in the Database")
    st.markdown(
        f"Searches DuckDuckGo Maps for imaging centers near each zip code "
        f"(**{len(search_zips)} zip codes**) in the selected area, then filters out centers "
        f"already in the database (matched by website domain)."
    )

    if not search_zips:
        st.info("Select at least one state (and optionally zip codes) in the sidebar to enable discovery search.")
    else:
        _col_btn, _col_info = st.columns([1, 3])
        with _col_btn:
            run_discover = st.button("🔍 Search for Unlisted Centers", type="primary", use_container_width=True)
        with _col_info:
            st.caption(
                f"Will search {len(search_zips)} zip code(s) × 4 query types = up to "
                f"{len(search_zips) * 4 * 8:,} candidate results. "
                f"Estimated time: {max(1, len(search_zips) * 20 // 60)}–{max(2, len(search_zips) * 40 // 60)} min."
            )

        if run_discover:
            with st.spinner("Searching…"):
                results = search_and_classify_centers(search_zips, state_label, existing_domains)
            st.session_state["discover_results"] = results
            if results:
                st.success(f"Found **{len(results)}** potential centers not in the database.")
            else:
                st.warning("No new centers found for the selected zip codes.")

        discover_results = st.session_state.get("discover_results")

        if discover_results:
            disc_df = pd.DataFrame(discover_results)

            # Category filter for discover results
            disc_cat_filter = st.multiselect(
                "Filter by Category",
                options=CATEGORIES,
                default=[],
                placeholder="All categories",
                key="disc_cat_filter",
            )
            if disc_cat_filter:
                disc_df = disc_df[disc_df["Scrubber Category"].isin(disc_cat_filter)]

            st.caption(f"Showing **{len(disc_df)}** discovered centers")

            # Fix website URLs
            if "Website" in disc_df.columns:
                disc_df["Website"] = disc_df["Website"].apply(
                    lambda u: (u if str(u).startswith("http") else f"https://{u}")
                    if u and str(u).strip() not in ("", "nan") else None
                )

            disc_config = {
                "Name":               st.column_config.TextColumn("Name", width="large"),
                "Address":            st.column_config.TextColumn("Address", width="medium"),
                "City":               st.column_config.TextColumn("City", width="small"),
                "State":              st.column_config.TextColumn("State", width="small"),
                "Zip":                st.column_config.TextColumn("Zip", width="small"),
                "Phone":              st.column_config.TextColumn("Phone", width="medium"),
                "Website":            st.column_config.LinkColumn("Website", width="medium", display_text="🔗 Visit"),
                "Scrubber Category":  st.column_config.SelectboxColumn(
                                          "Scrubber Category", options=CATEGORIES, width="medium"
                                      ),
                "Scrubber Rationale": st.column_config.TextColumn("Scrubber Rationale", width="large"),
                "Found Via Zip":      st.column_config.TextColumn("Found Via Zip", width="small"),
            }

            col_order = ["Name", "Address", "City", "State", "Zip", "Phone",
                         "Website", "Scrubber Category", "Scrubber Rationale", "Found Via Zip"]
            col_order = [c for c in col_order if c in disc_df.columns]

            edited_disc = st.data_editor(
                disc_df[col_order].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=520,
                column_config=disc_config,
            )

            st.download_button(
                label="⬇ Download Discovered Centers CSV",
                data=to_download_bytes(edited_disc),
                file_name="Discovered Imaging Centers.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_summary:

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.markdown("#### Category Breakdown (Filtered View)")
        cat_counts = (
            working_df[working_df[cat_col].str.strip().ne("")]
            [cat_col].value_counts().reset_index()
        )
        cat_counts.columns = ["Category", "Count"]

        colors = [CATEGORY_COLORS.get(c, "#90caf9") for c in cat_counts["Category"]]

        fig = go.Figure(go.Bar(
            x=cat_counts["Count"],
            y=cat_counts["Category"],
            orientation="h",
            marker_color=colors,
            text=cat_counts["Count"],
            textposition="outside",
            textfont=dict(size=12, color="#333"),
        ))
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Inter, Arial, sans-serif", color="#333"),
            xaxis=dict(showgrid=True, gridcolor="#e8edf2", title=""),
            yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
            margin=dict(l=10, r=60, t=20, b=20),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Counts by Category")
        total_cat = max(len(categorized), 1)
        for _, row in cat_counts.iterrows():
            pct = 100 * row["Count"] / total_cat
            color = CATEGORY_COLORS.get(row["Category"], "#90caf9")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 10px;margin:3px 0;border-radius:5px;'
                f'background:#f4f7fc;font-size:0.88rem;border-left:4px solid {color};">'
                f'<span style="color:{NAVY};font-weight:600;">{row["Category"]}</span>'
                f'<span style="color:#555;">{row["Count"]:,} &nbsp;'
                f'<span style="color:#999;font-size:0.78rem;">({pct:.1f}%)</span></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if len(uncategorized):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 10px;margin:3px 0;border-radius:5px;'
                f'background:#fff8e1;font-size:0.88rem;">'
                f'<span style="color:#e65100;font-weight:600;">⚠ Uncategorized</span>'
                f'<span style="color:#555;">{len(uncategorized):,}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # State breakdown
    state_col_k = C("state")
    if state_col_k and state_col_k in working_df.columns:
        st.markdown("#### Top States by MRI/CT Centers (Filtered View)")
        mri_by_state = (
            working_df[working_df[cat_col] == "Offers MRI/CT"]
            .groupby(state_col_k).size().reset_index(name="Count")
            .sort_values("Count", ascending=False).head(20)
        )
        if not mri_by_state.empty:
            fig2 = go.Figure(go.Bar(
                x=mri_by_state[state_col_k],
                y=mri_by_state["Count"],
                marker_color=CYAN,
                text=mri_by_state["Count"],
                textposition="outside",
            ))
            fig2.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Inter, Arial, sans-serif", color="#333"),
                xaxis=dict(title="", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="#e8edf2", title="Centers"),
                margin=dict(l=10, r=10, t=20, b=20),
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Needs Review spotlight
    needs_review_df = working_df[working_df[cat_col] == "Needs Review"]
    if not needs_review_df.empty:
        review_display = [
            c for c in [name_col, C("city"), C("state"), C("zip"), C("website"), rat_col]
            if c and c in needs_review_df.columns
        ]
        with st.expander(f"🔍 {len(needs_review_df):,} centers need manual review"):
            st.dataframe(
                needs_review_df[review_display].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
