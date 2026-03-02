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

/* ── Category badge colors ── */
.cat-mri      {{ background:#e8f4ff; color:{NAVY};   border:1px solid {CYAN}; }}
.cat-mammo    {{ background:#fff0f6; color:#a0106a;  border:1px solid #f48fb1; }}
.cat-us       {{ background:#e8fff0; color:#1b5e20;  border:1px solid #81c784; }}
.cat-cancer   {{ background:#fff3e0; color:#e65100;  border:1px solid #ffb74d; }}
.cat-nuclear  {{ background:#f3e5f5; color:#6a1b9a;  border:1px solid #ce93d8; }}
.cat-ignore   {{ background:#fafafa; color:#616161;  border:1px solid #bdbdbd; }}
.cat-closed   {{ background:#ffebee; color:#b71c1c;  border:1px solid #ef9a9a; }}
.cat-review   {{ background:#fffde7; color:#f57f17;  border:1px solid #fff176; }}

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

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "Imaging Centers.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "Imaging Centers - Categorized.csv")

DISPLAY_COLS = [
    "Imaging Center Name", "City", "State", "CBSA",
    "Category", "Rationale", "Website", "DHC Profile Link",
]

# ── Session state defaults ─────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
if "unsaved" not in st.session_state:
    st.session_state["unsaved"] = False

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer, encoding="utf-8-sig", dtype=str, low_memory=False)
    df["Category"]  = df["Category"].fillna("")
    df["Rationale"] = df["Rationale"].fillna("")
    return df


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Imaging Center Name"].fillna("").str.strip().ne("")]


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
    active = active_rows(df)
    to_process = [
        (idx, row) for idx, row in active.iterrows()
        if not str(row.get("Category", "")).strip()
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
        name     = str(row.get("Imaging Center Name", "")).strip()
        taxonomy = str(row.get("Primary Taxonomy ", "")).strip()
        website  = str(row.get("Website", "")).strip()
        cat, rat = classify_center(name, taxonomy, website)
        return idx, cat, rat.strip()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, item): item[0] for item in to_process}
        for future in as_completed(futures):
            try:
                idx, cat, rat = future.result()
                df.at[idx, "Category"]  = cat
                df.at[idx, "Rationale"] = rat
            except Exception as exc:
                pass
            done += 1
            yield df, done, total


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:16px 8px 12px 8px;text-align:center;">'
        f'<img src="{LOGO_URL}" style="max-width:180px;width:100%;" />'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**📂 DATA SOURCE**")
    use_default = st.checkbox(
        "Use default file (Imaging Centers.csv)",
        value=os.path.exists(DEFAULT_OUTPUT) or os.path.exists(DEFAULT_INPUT),
    )

    if use_default:
        load_path = DEFAULT_OUTPUT if os.path.exists(DEFAULT_OUTPUT) else DEFAULT_INPUT
        if st.button("Load / Refresh File", type="primary", use_container_width=True):
            st.session_state["df"] = load_csv(load_path)
            st.session_state["unsaved"] = False
            st.rerun()
        if st.session_state["df"] is None and os.path.exists(load_path):
            st.session_state["df"] = load_csv(load_path)
    else:
        uploaded = st.file_uploader("Upload DHC CSV", type=["csv"])
        if uploaded:
            st.session_state["df"] = load_csv(uploaded)
            st.session_state["unsaved"] = False

    st.markdown("---")
    st.markdown("**⚙️ SCRUBBER SETTINGS**")
    workers = st.slider("Concurrent workers", 1, 16, 8)
    limit   = st.number_input("Row limit (0 = all)", min_value=0, value=0, step=100)

    st.markdown("---")
    st.markdown("**🏷️ FILTER RESULTS**")
    cat_filter = st.multiselect(
        "Categories",
        options=CATEGORIES,
        default=[],
        placeholder="All categories",
    )
    state_filter = st.multiselect(
        "State(s)",
        options=sorted(
            active_rows(st.session_state["df"])["State"].dropna().unique().tolist()
        ) if st.session_state["df"] is not None else [],
        default=[],
        placeholder="All states",
    )
    search_term = st.text_input("Search by name", placeholder="e.g. Solis, RadNet…")

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

# ── No data state ──────────────────────────────────────────────────────────────
if st.session_state["df"] is None:
    st.info("👈 Load a file using the sidebar to get started.")
    st.stop()

df = st.session_state["df"]
active = active_rows(df)
categorized = active[active["Category"].str.strip().ne("")]
uncategorized = active[active["Category"].str.strip().eq("")]

# ── Summary metrics row ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Centers",    f"{len(active):,}")
c2.metric("Categorized",      f"{len(categorized):,}")
c3.metric("Uncategorized",    f"{len(uncategorized):,}")
c4.metric("Offers MRI/CT",    f"{(active['Category'] == 'Offers MRI/CT').sum():,}")
c5.metric("Needs Review",     f"{(active['Category'] == 'Needs Review').sum():,}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Run scrubber section ───────────────────────────────────────────────────────
if len(uncategorized) > 0:
    with st.expander(
        f"▶ **{len(uncategorized):,} uncategorized centers** — click to run scrubber",
        expanded=len(categorized) == 0,
    ):
        st.caption(
            f"The scrubber will visit each center's website and apply keyword matching. "
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
        view = view[view["Category"].isin(cat_filter)]
    if state_filter:
        view = view[view["State"].isin(state_filter)]
    if search_term:
        view = view[
            view["Imaging Center Name"].fillna("").str.contains(search_term, case=False, na=False)
        ]

    st.caption(f"Showing **{len(view):,}** of {len(active):,} centers")

    # Build display dataframe
    show_cols = [c for c in DISPLAY_COLS if c in view.columns]
    display_df = view[show_cols].copy().reset_index(drop=True)

    # Fix URLs for LinkColumn
    for link_col in ["Website", "DHC Profile Link"]:
        if link_col in display_df.columns:
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
        column_config={
            "Imaging Center Name": st.column_config.TextColumn(
                "Center Name", width="large", disabled=True
            ),
            "City":  st.column_config.TextColumn("City",  width="small",  disabled=True),
            "State": st.column_config.TextColumn("State", width="small",  disabled=True),
            "CBSA":  st.column_config.TextColumn("CBSA",  width="medium", disabled=True),
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=CATEGORIES,
                width="medium",
                required=True,
            ),
            "Rationale": st.column_config.TextColumn(
                "Rationale",
                width="large",
                max_chars=500,
            ),
            "Website": st.column_config.LinkColumn(
                "Website", width="medium", display_text="🔗 Visit"
            ),
            "DHC Profile Link": st.column_config.LinkColumn(
                "DHC Profile", width="small", display_text="DHC ↗"
            ),
        },
    )

    # Detect edits and write back to main df
    if not edited.equals(display_df):
        # Map edits back by position (display_df was reset_index)
        orig_indices = view.index.tolist()
        for i, orig_idx in enumerate(orig_indices):
            if i < len(edited):
                df.at[orig_idx, "Category"]  = edited.at[i, "Category"]  if "Category"  in edited.columns else df.at[orig_idx, "Category"]
                df.at[orig_idx, "Rationale"] = edited.at[i, "Rationale"] if "Rationale" in edited.columns else df.at[orig_idx, "Rationale"]
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
            active[active["Category"].str.strip().ne("")]
            ["Category"].value_counts().reset_index()
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

    # State breakdown
    st.markdown("#### Top States by MRI/CT Centers")
    mri_by_state = (
        active[active["Category"] == "Offers MRI/CT"]
        .groupby("State").size().reset_index(name="Count")
        .sort_values("Count", ascending=False).head(20)
    )
    if not mri_by_state.empty:
        fig2 = go.Figure(go.Bar(
            x=mri_by_state["State"],
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
    needs_review = active[active["Category"] == "Needs Review"]
    if not needs_review.empty:
        with st.expander(f"🔍 {len(needs_review):,} centers need manual review"):
            review_cols = [c for c in ["Imaging Center Name", "City", "State", "Website", "Rationale"] if c in needs_review.columns]
            st.dataframe(
                needs_review[review_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
