"""
DHC Scrubber – Streamlit App
Lumexa Imaging theme
"""

import os
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Import classification engine from scrubber.py
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

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {NAVY};
}}
[data-testid="stSidebar"] * {{
    color: #ffffff !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stFileUploader label,
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

/* ── Primary buttons ── */
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

/* ── Secondary buttons ── */
[data-testid="stButton"] > button[kind="secondary"] {{
    background-color: transparent !important;
    border: 1px solid {CYAN} !important;
    color: {CYAN} !important;
    font-weight: 600;
    border-radius: 6px;
}}

/* ── Metric cards ── */
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

/* ── Tabs ── */
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

/* ── File uploader in sidebar ── */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background-color: rgba(255,255,255,0.07) !important;
    border: 2px dashed {CYAN} !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] p {{
    color: #ccd6f6 !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
    background-color: {CYAN} !important;
    color: {NAVY} !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 5px !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {{
    color: #8ba8cc !important;
}}

/* ── Header ── */
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

/* ── Overview cards ── */
.overview-card {{
    background: #f8faff;
    border: 1px solid #dce8f5;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}}
.overview-card h4 {{
    margin: 0 0 0.5rem 0;
    color: {NAVY};
    font-size: 0.95rem;
}}
.field-badge {{
    display: inline-block;
    background: {NAVY};
    color: #fff !important;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 4px;
}}
.field-badge.opt {{
    background: {CYAN};
}}
.field-badge.out {{
    background: {GREEN};
}}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORIES = [
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
DEFAULT_INPUT  = os.path.join(BASE_DIR, "Imaging Centers.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "Imaging Centers - Categorized.csv")

# ── Column detection ────────────────────────────────────────────────────────────
# Maps each logical field → recognized column name variants (checked in order,
# case-insensitive, leading/trailing whitespace ignored).
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
        "Primary Taxonomy ",          # note: trailing space in original DHC export
        "Primary Taxonomy", "Taxonomy", "Provider Type", "Specialty",
        "Taxonomy Code Description", "NPI Taxonomy", "Taxonomy Description",
    ],
    "city":      ["City", "City Name", "Town", "Municipality", "Practice City"],
    "state":     ["State", "State Code", "ST", "State Abbrev", "State/Province", "Practice State"],
    "cbsa":      ["CBSA", "MSA", "Metro Area", "Market", "CBSA Name", "Metro", "Metropolitan Area"],
    "dhc_link":  ["DHC Profile Link", "DHC Link", "Profile URL", "DHC URL", "Profile Link", "DHC Profile"],
    "category":  ["Category", "Classification", "Type", "Service Type", "Center Type"],
    "rationale": ["Rationale", "Notes", "Reason", "Comments", "Description", "Explanation"],
}

FRIENDLY_NAMES = {
    "name":      "Center Name  *(required)*",
    "website":   "Website URL",
    "taxonomy":  "Primary Taxonomy",
    "city":      "City",
    "state":     "State",
    "cbsa":      "CBSA / Market",
    "dhc_link":  "DHC Profile Link",
    "category":  "Category  *(output)*",
    "rationale": "Rationale  *(output)*",
}

INPUT_KEYS  = ["name", "website", "taxonomy", "city", "state", "cbsa", "dhc_link"]
OUTPUT_KEYS = ["category", "rationale"]


def detect_columns(df_columns: list) -> dict:
    """
    Auto-detect which actual column maps to each logical field.
    Returns {logical_key: actual_column_name or None}.
    """
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


# ── Session state defaults ─────────────────────────────────────────────────────
for _k, _v in [("df", None), ("col_map", {}), ("unsaved", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ────────────────────────────────────────────────────────────────────

def C(key: str, default: str = None):
    """Resolve a logical column key → actual DataFrame column name."""
    val = st.session_state.get("col_map", {}).get(key)
    return val if val else default


def load_and_init(path_or_buffer):
    """
    Load CSV, detect columns, ensure output columns exist.
    Returns (df, col_map).
    """
    df = pd.read_csv(path_or_buffer, encoding="utf-8-sig", dtype=str, low_memory=False)
    col_map = detect_columns(df.columns.tolist())

    # Ensure Category output column exists
    if not col_map["category"]:
        df["Category"] = ""
        col_map["category"] = "Category"
    df[col_map["category"]] = df[col_map["category"]].fillna("")

    # Ensure Rationale output column exists
    if not col_map["rationale"]:
        df["Rationale"] = ""
        col_map["rationale"] = "Rationale"
    df[col_map["rationale"]] = df[col_map["rationale"]].fillna("")

    return df, col_map


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    name_col = C("name")
    if name_col and name_col in df.columns:
        return df[df[name_col].fillna("").str.strip().ne("")]
    return df


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def to_download_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def run_scrubber(df: pd.DataFrame, workers: int = 8, limit: int = 0):
    """
    Run classify_center on all rows with an empty Category.
    Yields (df, done, total) tuples so the caller can update progress.
    """
    name_col     = C("name", "Imaging Center Name")
    taxonomy_col = C("taxonomy")
    website_col  = C("website")
    cat_col      = C("category", "Category")
    rat_col      = C("rationale", "Rationale")

    active = active_rows(df)
    to_process = [
        (idx, row) for idx, row in active.iterrows()
        if not str(row.get(cat_col, "")).strip()
    ]
    if limit > 0:
        to_process = to_process[:limit]

    total = len(to_process)
    if total == 0:
        yield df, 0, 0
        return

    done = 0

    def _process(args):
        idx, row = args
        name     = str(row.get(name_col, "")).strip()     if name_col     else ""
        taxonomy = str(row.get(taxonomy_col, "")).strip() if taxonomy_col else ""
        website  = str(row.get(website_col, "")).strip()  if website_col  else ""
        cat, rat = classify_center(name, taxonomy, website)
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
    order = ["name", "city", "state", "cbsa", "category", "rationale", "website", "dhc_link"]
    return [col_map[k] for k in order if col_map.get(k) and col_map[k] in available]


def build_column_config(col_map: dict) -> dict:
    config = {}

    name_col = col_map.get("name")
    if name_col:
        config[name_col] = st.column_config.TextColumn("Center Name", width="large", disabled=True)

    city_col = col_map.get("city")
    if city_col:
        config[city_col] = st.column_config.TextColumn("City", width="small", disabled=True)

    state_col = col_map.get("state")
    if state_col:
        config[state_col] = st.column_config.TextColumn("State", width="small", disabled=True)

    cbsa_col = col_map.get("cbsa")
    if cbsa_col:
        config[cbsa_col] = st.column_config.TextColumn("CBSA", width="medium", disabled=True)

    cat_col = col_map.get("category", "Category")
    config[cat_col] = st.column_config.SelectboxColumn(
        "Category", options=CATEGORIES, width="medium", required=True
    )

    rat_col = col_map.get("rationale", "Rationale")
    config[rat_col] = st.column_config.TextColumn("Rationale", width="large", max_chars=500)

    web_col = col_map.get("website")
    if web_col:
        config[web_col] = st.column_config.LinkColumn("Website", width="medium", display_text="🔗 Visit")

    dhc_col = col_map.get("dhc_link")
    if dhc_col:
        config[dhc_col] = st.column_config.LinkColumn("DHC Profile", width="small", display_text="DHC ↗")

    return config


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:16px 8px 12px 8px;text-align:center;">'
        f'<img src="{LOGO_URL}" style="max-width:180px;width:100%;" />'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**📂 UPLOAD FILE**")
    uploaded = st.file_uploader(
        "Drop or browse a CSV file",
        type=["csv"],
        help="Upload any CSV with imaging center data. Column names are auto-detected.",
    )
    if uploaded:
        _df, _cmap = load_and_init(uploaded)
        st.session_state["df"]      = _df
        st.session_state["col_map"] = _cmap
        st.session_state["unsaved"] = False

    # ── Column mapping (shown after file loads) ────────────────────────────────
    if st.session_state["df"] is not None:
        _cm  = st.session_state["col_map"]
        _df  = st.session_state["df"]
        _available = ["(not in file)"] + _df.columns.tolist()

        _name_missing     = not _cm.get("name")
        _important_missing = [k for k in ["website", "taxonomy"] if not _cm.get(k)]
        _expand_mapping   = _name_missing or bool(_important_missing)

        st.markdown("---")
        with st.expander("📋 COLUMN MAPPING", expanded=_expand_mapping):
            if _name_missing:
                st.error("⚠️ **Center Name** column not detected — required for the tool to work.")
            elif _important_missing:
                st.caption(
                    f"⚠️ Optional field(s) not detected: "
                    + ", ".join(f"*{FRIENDLY_NAMES[k]}*" for k in _important_missing)
                    + ". Accuracy may be reduced."
                )
            else:
                st.caption("✅ All key columns detected. Adjust if needed.")

            _new_map = dict(_cm)
            for _key in INPUT_KEYS:
                _cur = _cm.get(_key)
                _idx = _available.index(_cur) if _cur in _available else 0
                _sel = st.selectbox(
                    FRIENDLY_NAMES[_key],
                    options=_available,
                    index=_idx,
                )
                _new_map[_key] = None if _sel == "(not in file)" else _sel

            # Output cols: never change via this UI — just show them
            st.caption(
                f"Output → **{_cm.get('category', 'Category')}** "
                f"& **{_cm.get('rationale', 'Rationale')}**"
            )
            st.session_state["col_map"] = _new_map

    st.markdown("---")
    st.markdown("**⚙️ SCRUBBER SETTINGS**")
    workers = st.slider("Concurrent workers", 1, 16, 8)
    limit   = st.number_input("Row limit (0 = all)", min_value=0, value=0, step=100)

    st.markdown("---")
    st.markdown("**🏷️ FILTER RESULTS**")

    _state_col     = st.session_state.get("col_map", {}).get("state")
    _state_options = []
    if st.session_state["df"] is not None and _state_col and _state_col in st.session_state["df"].columns:
        _state_options = sorted(
            active_rows(st.session_state["df"])[_state_col].dropna().unique().tolist()
        )

    cat_filter = st.multiselect(
        "Categories",
        options=CATEGORIES,
        default=[],
        placeholder="All categories",
    )
    state_filter = st.multiselect(
        "State(s)",
        options=_state_options,
        default=[],
        placeholder="All states",
    )
    search_term = st.text_input("Search by name", placeholder="e.g. Solis, RadNet…")

    # ── About (always accessible) ──────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
**Required:** Center Name column
**Recommended:** Website URL, Primary Taxonomy
**Optional:** City, State, CBSA, DHC Profile Link

The tool auto-detects column names from your CSV. Use the Column Mapping section above to confirm or correct any field.

**Categories assigned:**
- 🔵 Offers MRI/CT
- 🩷 Mammography Only
- 🟢 Ultrasound Only
- 🟠 Cancer Center
- 🟣 Nuclear Medicine / PET
- ⚪ Ignore
- 🔴 Closed
- 🟡 Needs Review
        """)


# ── Main header ────────────────────────────────────────────────────────────────
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

# ── No data state → overview panel ────────────────────────────────────────────
if st.session_state["df"] is None:

    st.markdown("### 👋 Getting Started")
    st.markdown(
        "Upload any CSV containing imaging center data using the sidebar. "
        "The tool will auto-detect your column names and can work with a variety of export formats."
    )

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown(
            f"""
<div class="overview-card">
<h4>📋 Required Field</h4>
<table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
<tr style="border-bottom:1px solid #dce8f5;">
  <td style="padding:6px 8px;font-weight:700;color:{NAVY};">Center Name</td>
  <td style="padding:6px 8px;color:#444;">Identifies each imaging center.<br>
    Recognized as: <em>Imaging Center Name, Facility Name, Name, Organization, Provider Name…</em>
  </td>
</tr>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="overview-card">
<h4>🔎 Recommended Fields <span style="font-weight:400;font-size:0.82rem;color:#666;">(improve accuracy)</span></h4>
<table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
<tr style="border-bottom:1px solid #dce8f5;">
  <td style="padding:6px 8px;font-weight:700;color:{NAVY};white-space:nowrap;">Website URL</td>
  <td style="padding:6px 8px;color:#444;">The tool visits each center's website to detect MRI, CT, and other services.<br>
    Recognized as: <em>Website, URL, Web Address, Homepage…</em>
  </td>
</tr>
<tr>
  <td style="padding:6px 8px;font-weight:700;color:{NAVY};white-space:nowrap;">Primary Taxonomy</td>
  <td style="padding:6px 8px;color:#444;">Used as a fallback when the website is unavailable or uninformative.<br>
    Recognized as: <em>Primary Taxonomy, Taxonomy, Provider Type, Specialty…</em>
  </td>
</tr>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
<div class="overview-card">
<h4>📦 Optional Display Fields</h4>
<table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
<tr style="border-bottom:1px solid #dce8f5;">
  <td style="padding:5px 8px;font-weight:700;color:{NAVY};">City</td>
  <td style="padding:5px 8px;color:#444;">Shown in results table and used for filtering</td>
</tr>
<tr style="border-bottom:1px solid #dce8f5;">
  <td style="padding:5px 8px;font-weight:700;color:{NAVY};">State</td>
  <td style="padding:5px 8px;color:#444;">Filtering + state-level MRI/CT summary chart</td>
</tr>
<tr style="border-bottom:1px solid #dce8f5;">
  <td style="padding:5px 8px;font-weight:700;color:{NAVY};">CBSA / Market</td>
  <td style="padding:5px 8px;color:#444;">Shown in results table</td>
</tr>
<tr>
  <td style="padding:5px 8px;font-weight:700;color:{NAVY};white-space:nowrap;">DHC Profile Link</td>
  <td style="padding:5px 8px;color:#444;">Rendered as a clickable link in the results table</td>
</tr>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="overview-card">
<h4>✅ Output Fields <span style="font-weight:400;font-size:0.82rem;color:#666;">(auto-created if missing)</span></h4>
<table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
<tr style="border-bottom:1px solid #dce8f5;">
  <td style="padding:5px 8px;font-weight:700;color:{NAVY};">Category</td>
  <td style="padding:5px 8px;color:#444;">
    Offers MRI/CT · Mammography Only · Ultrasound Only ·
    Cancer Center · Nuclear Medicine/PET · Ignore · Closed · Needs Review
  </td>
</tr>
<tr>
  <td style="padding:5px 8px;font-weight:700;color:{NAVY};">Rationale</td>
  <td style="padding:5px 8px;color:#444;">Explanation of how the category was determined (editable)</td>
</tr>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

    st.info(
        "💡 **Column names don't need to match exactly.** The tool auto-detects common variants "
        "(e.g. *Facility Name* works just as well as *Imaging Center Name*). After uploading, "
        "check the **📋 Column Mapping** section in the sidebar to confirm or adjust the detected fields.",
        icon=None,
    )

    st.markdown("---")
    st.markdown("#### How categorization works")

    steps_col, logic_col = st.columns([1, 1], gap="large")

    with steps_col:
        st.markdown(
            f"""
**Step 1 — Website scraping** *(if Website URL is available)*
The tool fetches each center's website and scans for service keywords. This is the most accurate signal.

**Step 2 — Keyword fallback** *(if website is unavailable)*
Falls back to the center name and Primary Taxonomy to infer services.

**Step 3 — Human review**
Centers where neither step yields a confident result are flagged as **Needs Review** for manual classification.
""",
        )

    with logic_col:
        st.markdown(
            f"""
**Detection priority (website):**
Closed → Cancer Center → Nuclear/PET → Mobile → Urgent Care/ER → **MRI/CT** → Non-imaging → Mammography → Ultrasound

**Key rules:**
- Emergency rooms / urgent care → **Ignore** (even if they have a CT scanner)
- Nuclear/PET sites that *also* mention MRI or CT → promoted to **Offers MRI/CT**
- Mobile imaging units → **Ignore**
- Plain X-ray / Radiography taxonomy → **Ignore**
""",
        )

    st.stop()


# ── Data loaded — resolve columns ──────────────────────────────────────────────
df      = st.session_state["df"]
col_map = st.session_state["col_map"]

cat_col = C("category", "Category")
rat_col = C("rationale", "Rationale")
name_col  = C("name")
state_col = C("state")

active       = active_rows(df)
categorized  = active[active[cat_col].str.strip().ne("")]
uncategorized = active[active[cat_col].str.strip().eq("")]

# ── Summary metrics row ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Centers",  f"{len(active):,}")
c2.metric("Categorized",    f"{len(categorized):,}")
c3.metric("Uncategorized",  f"{len(uncategorized):,}")
c4.metric("Offers MRI/CT",  f"{(active[cat_col] == 'Offers MRI/CT').sum():,}")
c5.metric("Needs Review",   f"{(active[cat_col] == 'Needs Review').sum():,}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Run scrubber section ───────────────────────────────────────────────────────
if not name_col:
    st.error(
        "⚠️ **Center Name column not detected.** "
        "Please use the **📋 Column Mapping** section in the sidebar to map the correct field."
    )
    st.stop()

if len(uncategorized) > 0:
    with st.expander(
        f"▶ **{len(uncategorized):,} uncategorized centers** — click to run scrubber",
        expanded=len(categorized) == 0,
    ):
        _web_note = "" if C("website") else " *(no Website column detected — will use name/taxonomy only)*"
        _tax_note = "" if C("taxonomy") else " *(no Taxonomy column detected)*"
        st.caption(
            f"The scrubber will visit each center's website and apply keyword matching.{_web_note}{_tax_note} "
            f"Estimated time: ~{max(1, len(uncategorized) // 60)} – "
            f"{max(2, len(uncategorized) // 40)} minutes at {workers} workers."
        )
        if st.button("🚀 Run Scrubber", type="primary"):
            prog_bar  = st.progress(0.0)
            status_ph = st.empty()
            last_df   = df.copy()

            for updated_df, done, total in run_scrubber(df, workers=workers, limit=int(limit)):
                last_df = updated_df
                if total > 0:
                    pct = done / total
                    prog_bar.progress(pct)
                    status_ph.caption(f"Processing… {done:,} / {total:,}  ({pct*100:.1f}%)")

            st.session_state["df"] = last_df
            st.session_state["unsaved"] = True
            status_ph.success(f"✅ Done! {done:,} centers categorized.")
            save_csv(last_df, DEFAULT_OUTPUT)
            st.session_state["unsaved"] = False
            st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_results, tab_summary = st.tabs(["📋 Results", "📊 Summary"])

# ════════════════════════════════════════════════════════════════════════════════
# RESULTS TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_results:

    # Apply filters
    view = active.copy()
    if cat_filter:
        view = view[view[cat_col].isin(cat_filter)]
    if state_filter and state_col and state_col in view.columns:
        view = view[view[state_col].isin(state_filter)]
    if search_term and name_col and name_col in view.columns:
        view = view[view[name_col].fillna("").str.contains(search_term, case=False, na=False)]

    st.caption(f"Showing **{len(view):,}** of {len(active):,} centers")

    # Build display dataframe
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

    # Detect edits and write back to main df
    if not edited.equals(display_df):
        orig_indices = view.index.tolist()
        for i, orig_idx in enumerate(orig_indices):
            if i < len(edited):
                if cat_col in edited.columns:
                    df.at[orig_idx, cat_col] = edited.at[i, cat_col]
                if rat_col in edited.columns:
                    df.at[orig_idx, rat_col] = edited.at[i, rat_col]
        st.session_state["df"] = df
        st.session_state["unsaved"] = True

    # Action row
    col_save, col_dl, col_spacer = st.columns([1, 1, 4])

    with col_save:
        if st.session_state.get("unsaved"):
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                save_csv(df, DEFAULT_OUTPUT)
                st.session_state["unsaved"] = False
                st.success("Saved to Imaging Centers - Categorized.csv")
        else:
            st.button("💾 Saved ✓", disabled=True, use_container_width=True)

    with col_dl:
        st.download_button(
            label="⬇ Download CSV",
            data=to_download_bytes(df),
            file_name="Imaging Centers - Categorized.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.get("unsaved"):
        st.caption("⚠️ You have unsaved edits — click Save Changes or Download CSV.")


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_summary:

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.markdown("#### Category Breakdown")
        cat_counts = (
            active[active[cat_col].str.strip().ne("")]
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
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Counts by Category")
        for _, row in cat_counts.iterrows():
            pct = 100 * row["Count"] / max(len(categorized), 1)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 10px;margin:3px 0;border-radius:5px;'
                f'background:#f4f7fc;font-size:0.88rem;">'
                f'<span style="color:{NAVY};font-weight:600;">{row["Category"]}</span>'
                f'<span style="color:#555;">{row["Count"]:,} &nbsp;'
                f'<span style="color:#999;font-size:0.78rem;">({pct:.1f}%)</span></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        uncategorized_n = len(uncategorized)
        if uncategorized_n:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 10px;margin:3px 0;border-radius:5px;'
                f'background:#fff8e1;font-size:0.88rem;">'
                f'<span style="color:#e65100;font-weight:600;">⚠ Uncategorized</span>'
                f'<span style="color:#555;">{uncategorized_n:,}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # State breakdown (only if state column detected)
    if state_col and state_col in active.columns:
        st.markdown("#### Top States by MRI/CT Centers")
        mri_by_state = (
            active[active[cat_col] == "Offers MRI/CT"]
            .groupby(state_col).size().reset_index(name="Count")
            .sort_values("Count", ascending=False).head(20)
        )
        if not mri_by_state.empty:
            fig2 = go.Figure(go.Bar(
                x=mri_by_state[state_col],
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
    else:
        st.info("State column not detected — state-level breakdown unavailable.", icon="ℹ️")

    # Needs Review spotlight
    needs_review = active[active[cat_col] == "Needs Review"]
    if not needs_review.empty:
        review_display = [
            c for c in [name_col, C("city"), C("state"), C("website"), rat_col]
            if c and c in needs_review.columns
        ]
        with st.expander(f"🔍 {len(needs_review):,} centers need manual review"):
            st.dataframe(
                needs_review[review_display].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
