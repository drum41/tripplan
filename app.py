import streamlit as st
from streamlit_gsheets import GSheetsConnection
import ssl
import certifi
import urllib.request
import pandas as pd
import altair as alt
import os
import json
try:
    from streamlit_timeline import timeline as st_timeline
except Exception:
    st_timeline = None
from streamlit_extras.stylable_container import stylable_container
import datetime as _dt
base_dir = os.path.dirname(os.path.abspath(__file__))

ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
)

def vmk_data_container(key_suffix=""):
    """
    Creates a VMK-styled data container with a light background and border
    
    Args:
        key_suffix (str): Suffix for unique container key
    """
    return st.container(border=True)

# Use the standard spreadsheet URL (not the CSV export URL) for CRUD
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1T2RXkiXKj6ZL9mZTKpV3HF_IyPfyn4ZiQ7cHFNAfzWg"

conn = st.connection("gsheets", type=GSheetsConnection)

st.set_page_config(page_title="Trip Planner", layout="wide", initial_sidebar_state="collapsed")

@st.cache_data(show_spinner=False)
def get_sheet_data() -> pd.DataFrame:
    # Omitting worksheet selects the first sheet
    spreadsheet_id = extract_spreadsheet_id(SPREADSHEET_URL)
    return conn.read(spreadsheet=spreadsheet_id)

def extract_spreadsheet_id(spreadsheet_url: str) -> str:
    try:
        # Typical URL: https://docs.google.com/spreadsheets/d/{ID}/edit
        parts = spreadsheet_url.split("/d/")
        if len(parts) > 1:
            tail = parts[1]
            return tail.split("/")[0]
        # If already an ID, return as-is
        return spreadsheet_url
    except Exception:
        return spreadsheet_url

# ---------- Timeline helpers ----------
def _parse_date_to_parts(date_value) -> dict:
    """Return {year, month, day} from a Date cell.
    Accepts formats like '3/9' (m/d), '2025-09-03', pandas Timestamp, or date.
    Missing year defaults to the current year.
    """
    try:
        # Handle NaN
        if pd.isna(date_value):
            today = _dt.date.today()
            return {"year": today.year, "month": today.month, "day": today.day}

        # Already a date-like object
        if hasattr(date_value, "year") and hasattr(date_value, "month") and hasattr(date_value, "day"):
            return {"year": int(date_value.year),
                    "month": int(date_value.month),
                    "day": int(date_value.day)}

        s = str(date_value).strip()

        # m/d or mm/dd (US style month/day)
        if "/" in s and s.count("/") == 1:
            m_str, d_str = s.split("/")
            today = _dt.date.today()
            return {"year": today.year,
                    "month": int(m_str),
                    "day": int(d_str)}

        # Fallback: use pandas to_datetime
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return {"year": int(ts.year),
                    "month": int(ts.month),
                    "day": int(ts.day)}

    except Exception:
        pass

    # Final fallback: today
    today = _dt.date.today()
    return {"year": today.year, "month": today.month, "day": today.day}

def build_timeline_json(
    df_source: pd.DataFrame,
    date_col: str = "Date",
    hour_col: str = "Hour_num",
    activity_col: str = "Activity",
    place_col: str = "Place",
    proposer_col: str = "Propose person",
    image_col: str = "Image",
    description_col: str = "Description",
    end_date_col: str = "End Date",
    end_hour_col: str = "End Hour",
) -> dict:
    """Convert rows into TimelineJS-compatible JSON.
    Each row becomes an event. Requires at least a date; optionally uses hour for sorting.
    Uses Image and Description columns when available.
    """
    rows = df_source.copy()
    # Sort for consistency
    sort_keys = []
    if date_col in rows.columns:
        sort_keys.append(date_col)
    if hour_col in rows.columns:
        sort_keys.append(hour_col)
    elif hour_col != "Hour_num" and hour_col in rows.columns:
        sort_keys.append(hour_col)
    if sort_keys:
        rows = rows.sort_values(sort_keys)

    events = []
    for _, r in rows.iterrows():
        date_parts = _parse_date_to_parts(r.get(date_col))
        # Determine start hour
        start_hour_val = None
        try:
            if hour_col in rows.columns and pd.notna(r.get(hour_col)):
                start_hour_val = int(float(r.get(hour_col)))
        except Exception:
            start_hour_val = None
        # Derive end date
        end_parts = None
        end_hour_val = None
        if end_date_col in rows.columns and pd.notna(r.get(end_date_col)):
            end_parts = _parse_date_to_parts(r.get(end_date_col))
            # Optional explicit end hour
            if end_hour_col in rows.columns and pd.notna(r.get(end_hour_col)):
                try:
                    end_hour_val = int(float(r.get(end_hour_col)))
                except Exception:
                    end_hour_val = None
        else:
            try:
                # If we have hour and duration, estimate same-day vs next-day
                if (hour_col in rows.columns and "Time_h" in rows.columns and
                    pd.notna(r.get(hour_col)) and pd.notna(r.get("Time_h"))):
                    start_num = float(r.get(hour_col))
                    dur = float(r.get("Time_h"))
                    carry_next_day = (start_num + dur) >= 24.0
                    import datetime as _dt
                    start_dt = _dt.date(year=int(date_parts["year"]), month=int(date_parts["month"]), day=int(date_parts["day"]))
                    end_dt = start_dt + _dt.timedelta(days=1) if carry_next_day else start_dt
                    end_parts = {"year": end_dt.year, "month": end_dt.month, "day": end_dt.day}
                    # Compute end hour modulo 24
                    try:
                        end_hour_val = int((start_num + dur) % 24)
                    except Exception:
                        end_hour_val = None
            except Exception:
                end_parts = None
        if end_parts is None:
            end_parts = dict(date_parts)
        activity = str(r.get(activity_col)) if activity_col in rows.columns else ""
        place = str(r.get(place_col)) if place_col in rows.columns else ""
        proposer = str(r.get(proposer_col)) if proposer_col in rows.columns else ""
        img_url = r.get(image_col) if image_col in rows.columns else None
        desc = r.get(description_col) if description_col in rows.columns else None

        # Headline and text
        headline = " ".join([x for x in [activity, "—", place] if x]) if place or activity else "Trip event"
        text_body = str(desc) if pd.notna(desc) else ""

        media = {}
        if pd.notna(img_url):
            media = {
                "url": str(img_url),
                "credit": proposer if proposer else "",
                "caption": place if place else activity,
            }

        event = {
            "start_date": {
                "year": int(date_parts["year"]),
                "month": int(date_parts["month"]),
                "day": int(date_parts["day"]),
            },
            "end_date": {
                "year": int(end_parts["year"]),
                "month": int(end_parts["month"]),
                "day": int(end_parts["day"]),
            },
            "text": {"headline": headline, "text": text_body},
        }
        # Attach hours if available (TimelineJS supports hour/minute/second)
        if start_hour_val is not None:
            event["start_date"]["hour"] = int(start_hour_val)
            event["start_date"]["minute"] = 0
            event["start_date"]["second"] = 0
        if end_hour_val is not None:
            event["end_date"]["hour"] = int(end_hour_val)
            event["end_date"]["minute"] = 0
            event["end_date"]["second"] = 0
        if media:
            event["media"] = media
        events.append(event)

    return {"events": events}

def save_timeline_json(timeline_obj: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(timeline_obj, f, ensure_ascii=False, indent=2)

# ---------- Alternate builder (CSV/Excel-like semantics) ----------
def sheet_to_timeline_df(df_input: pd.DataFrame, default_duration_minutes: int = 60, default_year: int | None = None) -> dict:
    """Convert a DataFrame with Date (e.g., '3/9'), Hour (e.g., '12h'), Activity, Place,
    optional Image, Description, Propose person, Plan cost, Actual cost into TimelineJS JSON.

    - End time is next row's start or default_duration_minutes for the last row.
    - If Date lacks a year, uses default_year or the current year.
    """
    import datetime as _dt

    def _parse_day_month(value: str) -> tuple[int, int, int]:
        s = str(value).strip()
        # Try d/m formats first
        if "/" in s and s.count("/") == 1:
            day_str, month_str = s.split("/")
            year_val = default_year or _dt.date.today().year
            return int(year_val), int(month_str), int(day_str)
        # Try pandas to_datetime
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return int(ts.year), int(ts.month), int(ts.day)
        # Fallback to today
        today = _dt.date.today()
        return today.year, today.month, today.day

    def _parse_hour(value: str) -> int:
        try:
            return int(str(value).replace("h", "").strip())
        except Exception:
            return 0

    events: list[dict] = []
    n = len(df_input)
    for i in range(n):
        row = df_input.iloc[i]
        y, m, d = _parse_day_month(row.get("Date"))
        h = _parse_hour(row.get("Hour"))
        start = _dt.datetime(y, m, d, h, 0, 0)

        if i + 1 < n:
            next_row = df_input.iloc[i + 1]
            y2, m2, d2 = _parse_day_month(next_row.get("Date"))
            h2 = _parse_hour(next_row.get("Hour"))
            end = _dt.datetime(y2, m2, d2, h2, 0, 0)
        else:
            end = start + _dt.timedelta(minutes=int(default_duration_minutes))

        event = {
            "start_date": {
                "year": start.year,
                "month": start.month,
                "day": start.day,
                "hour": start.hour,
                "minute": start.minute,
                "second": start.second,
            },
            "end_date": {
                "year": end.year,
                "month": end.month,
                "day": end.day,
                "hour": end.hour,
                "minute": end.minute,
                "second": end.second,
            },
            "text": {
                "headline": f"{row.get('Activity', '')} — {row.get('Place', '')}",
                "text": row.get("Description", "") or "",
            },
            "media": {
                "url": row.get("Image", "") or "",
            },
            "extra": {
                "Propose person": row.get("Propose person", "") or "",
                "Plan cost": row.get("Plan cost", "") or "",
                "Actual cost": row.get("Actual cost", "") or "",
            },
        }
        events.append(event)

    return {"events": events}

# Read full sheet (cached) with error handling
try:
    data = get_sheet_data()
except Exception as e:
    st.error(f"Failed to load sheet. Check sharing and URL. Details: {e}")
    st.stop()

# Basic cleanup / typing
df = data.copy()
# df = df.dropna(how="all")

if "Hour" in df.columns:
    df["Hour_num"] = pd.to_numeric(
        df["Hour"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    # Compute time as next - current within the same date
    if {"Date"}.issubset(df.columns):
        df_sorted = df.sort_values(["Date", "Hour_num"]).copy()
        df_sorted["_next_hour"] = df_sorted.groupby("Date")["Hour_num"].shift(-1)
        df_sorted["Time_h"] = (df_sorted["_next_hour"] - df_sorted["Hour_num"]).fillna(0).clip(lower=0)
        df["Time_h"] = df_sorted["Time_h"].reindex(df.index)
    else:
        next_hour = df["Hour_num"].shift(-1)
        df["Time_h"] = (next_hour - df["Hour_num"]).fillna(0).clip(lower=0)

for col in ["Plan cost", "Actual cost"]:
    if col in df.columns:
        # Normalize currency-formatted strings to numeric
        s = df[col].astype(str)
        # Remove currency symbols and letters, keep digits, separators, and minus
        s = s.str.replace(r"[^\d\.,\-]", "", regex=True)
        # Treat comma as thousands separator by default; strip it
        s = s.str.replace(",", "")
        # Now parse to numeric
        df[col] = pd.to_numeric(s, errors="coerce")


# Sidebar filters
st.sidebar.header("Filters")
if "Date" in df.columns:
    unique_dates = sorted([d for d in df["Date"].dropna().unique().tolist()])
    if "__selected_dates__" not in st.session_state:
        st.session_state["__selected_dates__"] = unique_dates
    with st.sidebar.form("filters_form"):
        sel = st.multiselect("Date", unique_dates, default=st.session_state["__selected_dates__"])
        apply_filters = st.form_submit_button("Apply")
    if apply_filters:
        st.session_state["__selected_dates__"] = sel if sel else unique_dates
    selected_dates = st.session_state["__selected_dates__"]
    df_view = df[df["Date"].isin(selected_dates)] if selected_dates else df
else:
    df_view = df

with stylable_container(key="vmk_timeline_container",
        css_styles="""
            {
                background: white;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                padding: 1.25rem;
                margin: 0;
            }
        """,):
    st.subheader("Trip Story")
    if st_timeline is None:
        st.info("Install streamlit-timeline to enable this feature: pip install streamlit-timeline")
    else:
        try:
            source_df = df_view if not df_view.empty else df
            tj = build_timeline_json(source_df)
            data_str = json.dumps(tj, ensure_ascii=False, indent=2)
            st_timeline(data_str, height=800)
        except Exception as e:
            st.error(f"Failed to render timeline: {e}")

# Sidebar: Trip photos uploader
with st.sidebar.expander("Trip photos", expanded=False):
    uploaded = st.file_uploader(
        "Upload images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
    )
    if uploaded:
        st.session_state["trip_photos"] = [
            {"name": f.name, "data": f.read(), "source": "upload"} for f in uploaded
        ]
    add_url = st.text_input("Add image URL")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Add URL") and add_url:
            photos = st.session_state.get("trip_photos", [])
            photos.append({"name": add_url, "url": add_url, "source": "url"})
            st.session_state["trip_photos"] = photos
    with c2:
        if st.button("Clear photos"):
            st.session_state["trip_photos"] = []
    # Tiny preview count
    count = len(st.session_state.get("trip_photos", []))
    st.caption(f"{count} photo(s) stored")

# Decide which cost column to use (prefer Actual)
cost_col = None
if "Actual cost" in df_view.columns and df_view["Actual cost"].notna().any():
    cost_col = "Actual cost"
elif "Plan cost" in df_view.columns:
    cost_col = "Plan cost"


# Precompute aggregates
act = (
    df_view.groupby("Activity")[ ["Time_h"] + ([cost_col] if cost_col else []) ].sum().reset_index()
    if "Activity" in df_view.columns and "Time_h" in df_view.columns else pd.DataFrame()
)
# involvement score = normalized hours + normalized cost (when cost available)
if not act.empty:
    def _norm(s: pd.Series) -> pd.Series:
        s = s.fillna(0)
        span = (s.max() - s.min())
        return (s - s.min()) / span if span and span != 0 else s * 0
    act["_hours_norm"] = _norm(act["Time_h"]) if "Time_h" in act.columns else 0
    if cost_col and cost_col in act.columns:
        act["_cost_norm"] = _norm(act[cost_col])
    else:
        act["_cost_norm"] = 0
    act["involvement"] = act["_hours_norm"] + act["_cost_norm"]
    act = act.sort_values("involvement", ascending=False)

# Auto-generate tl.json when data has Image/Description columns
try:
    required_cols = {"Date", "Activity", "Place", "Propose person"}
    has_required = required_cols.issubset(df.columns)
    has_media = ("Image" in df.columns) or ("Description" in df.columns)
    if has_required and has_media:
        timeline_obj = build_timeline_json(df)
        save_timeline_json(timeline_obj, os.path.join(base_dir, "tl.json"))
except Exception:
    pass
proposer = (
    df_view["Propose person"].fillna("Unknown").value_counts().reset_index()
    if "Propose person" in df_view.columns else pd.DataFrame()
)
if not proposer.empty:
    proposer.columns = ["Propose person", "Count"]
place = (
    df_view.groupby("Place")[ ["Time_h"] + ([cost_col] if cost_col else []) ].sum().reset_index()
    if "Place" in df_view.columns and "Time_h" in df_view.columns else pd.DataFrame()
)
if not place.empty and cost_col:
    place["Cost per hour"] = place.apply(
        lambda r: (r[cost_col] / r["Time_h"]) if r["Time_h"] and r["Time_h"] > 0 else None,
        axis=1,
    )
    # correlation between hours and cost
    try:
        corr = float(place[["Time_h", cost_col]].corr().iloc[0, 1])
    except Exception:
        corr = None

# Helper: custom metric card
def render_metric_card(title: str, value: str, delta: str | None = None, icon: str | None = None, color: str = "#3b82f6") -> None:
    icon_span = f"<span style='font-size:20px;margin-right:8px'>{icon}</span>" if icon else ""
    delta_html = f"<div style='color:#059669;font-weight:600;margin-top:2px'>{delta}</div>" if delta else ""
    html = f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,.04);">
      <div style="display:flex;align-items:center;gap:8px;color:#6b7280;font-size:12px;font-weight:600;letter-spacing:.02em;text-transform:uppercase;">
        <div style="width:8px;height:8px;border-radius:50%;background:{color}"></div>
        <div>{title}</div>
      </div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-top:6px;">
        {icon_span}
        <div style="font-size:20px;font-weight:700;color:#111827;line-height:1.1">{value}</div>
      </div>
      {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Summary container
with stylable_container(key="vmk_summary_container",
        css_styles="""
            {
                background: white;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                padding: 1.25rem;
                margin: 0;
            }
        """,):
    # Summary cards under timeline
    cards = st.columns(5)
    with cards[0]:
        if not act.empty and "Time_h" in act.columns:
            top_time_row = act.sort_values("Time_h", ascending=False).iloc[0]
            total_time = float(act["Time_h"].fillna(0).sum()) if "Time_h" in act.columns else 0
            share_time = (float(top_time_row["Time_h"]) / total_time) if total_time else 0
            render_metric_card(
                title="Most time-consuming activity",
                value=f"{top_time_row['Activity']} — {int(top_time_row['Time_h'])}h",
                delta=f"{share_time:.0%} of total",
                icon="⏱️",
                color="#3b82f6",
            )
            # tiny bar of top 5 hours
            small = act.sort_values(cost_col, ascending=False).head(5)
            ch = (
                alt.Chart(small)
                .mark_bar(color="#60a5fa")
                .encode(
                    x=alt.X("Time_h:Q", title=None),
                    y=alt.Y("Activity:N", sort=small.sort_values("Time_h")["Activity"].tolist(), title=None),
                    tooltip=["Activity", alt.Tooltip("Time_h:Q", format=",.0f")],
                )
                .properties(height=120)
            )
            labels = (
                alt.Chart(small)
                .transform_calculate(label="datum.Activity + ' — ' + toString(round(datum.Time_h))")
                .mark_text(align="left", dx=3, color="#111827")
                .encode(
                    x=alt.X("Time_h:Q"),
                    y=alt.Y("Activity:N", sort=small.sort_values("Time_h")["Activity"].tolist()),
                    text=alt.Text("label:N"),
                )
            )
            st.altair_chart((ch + labels).configure_axis(labels=False, ticks=False, domain=False), use_container_width=True)
        else:
            render_metric_card("Most time-consuming activity", "–", icon="⏱️")
    with cards[1]:
        if not act.empty and cost_col and cost_col in act.columns and act[cost_col].fillna(0).sum() > 0:
            top_cost_row = act.sort_values(cost_col, ascending=False).iloc[0]
            total_cost = float(act[cost_col].fillna(0).sum())
            share_cost = float(top_cost_row[cost_col]) / total_cost if total_cost else 0
            render_metric_card(
                title="Most costly activity",
                value=f"{top_cost_row['Activity']} — {top_cost_row[cost_col]:,.0f}",
                delta=f"{share_cost:.0%} of total",
                icon="💸",
                color="#10b981",
            )

            # tiny bar of top 5 costs
            small_cost = act.sort_values(cost_col, ascending=False).head(5)

            ch_cost = (
                alt.Chart(small_cost)
                .mark_bar(color="#34d399")  # softer green
                .encode(
                    x=alt.X(f"{cost_col}:Q", title=None),
                    y=alt.Y("Activity:N",
                            sort=small_cost.sort_values(cost_col)["Activity"].tolist(),
                            title=None),
                    tooltip=["Activity", alt.Tooltip(f"{cost_col}:Q", format=",.0f")],
                )
                .properties(height=120)
            )

            labels_cost = (
                alt.Chart(small_cost)
                .transform_calculate(
                    label="datum.Activity + ' — ' + format(datum['{}'], ',.0f')".format(cost_col)
                )
                .mark_text(align="left", dx=3, color="#111827")
                .encode(
                    x=alt.X(f"{cost_col}:Q"),
                    y=alt.Y("Activity:N",
                            sort=small_cost.sort_values(cost_col)["Activity"].tolist()),
                    text=alt.Text("label:N"),
                )
            )

            st.altair_chart(
                (ch_cost + labels_cost)
                .configure_axis(labels=False, ticks=False, domain=False),
                use_container_width=True
            )
        else:
            render_metric_card("Most costly activity", "–", icon="💸")
    with cards[2]:
        if not proposer.empty:
            top_prop = proposer.iloc[0]
            total_props = int(proposer["Count"].sum()) if "Count" in proposer.columns else None
            share_prop = (int(top_prop["Count"]) / total_props) if total_props else 0
            render_metric_card(
                title="Top proposer",
                value=f"{top_prop['Propose person']} — {int(top_prop['Count'])}",
                delta=(f"{share_prop:.0%} of proposals" if total_props else None),
                icon="🏷️",
                color="#f59e0b",
            )
            sp = proposer.head(5)

            ch3 = (
                alt.Chart(sp)
                .mark_bar(color="#fbbf24")
                .encode(
                    x=alt.X("Count:Q", title=None),
                    y=alt.Y("Propose person:N",
                            sort=sp.sort_values("Count")["Propose person"].tolist(),
                            title=None),
                    tooltip=["Propose person", "Count"],
                )
                .properties(height=110)
            )

            labels3 = (
                alt.Chart(sp)
                .transform_calculate(
                    label="datum['Propose person'] + ' — ' + toString(datum.Count)"
                )
                .mark_text(align="left", dx=3, color="#111827")
                .encode(
                    x=alt.X("Count:Q"),
                    y=alt.Y("Propose person:N",
                            sort=sp.sort_values("Count")["Propose person"].tolist()),
                    text=alt.Text("label:N"),
                )
            )

            st.altair_chart((ch3 + labels3).configure_axis(labels=False, ticks=False, domain=False),
                            use_container_width=True)
        else:
            render_metric_card("Top proposer", "–", icon="🏷️")
    with cards[3]:
        if not place.empty and cost_col:
            top_cost_place = place.sort_values(cost_col, ascending=False).iloc[0]
            total_place_cost = float(place[cost_col].fillna(0).sum())
            share_place_cost = float(top_cost_place[cost_col]) / total_place_cost if total_place_cost else 0
            render_metric_card(
                title="Highest total cost (place)",
                value=f"{top_cost_place['Place']} — {top_cost_place[cost_col]:,.0f}",
                delta=f"{share_place_cost:.0%} of total",
                icon="📍",
                color="#6366f1",
            )
            sp = place.sort_values(cost_col, ascending=False).head(5)

            chp1 = (
                alt.Chart(sp)
                .mark_bar(color="#93c5fd")
                .encode(
                    x=alt.X(f"{cost_col}:Q", title=None),
                    y=alt.Y("Place:N",
                            sort=sp.sort_values(cost_col)["Place"].tolist(),
                            title=None),
                    tooltip=["Place", alt.Tooltip(f"{cost_col}:Q", format=",.0f")],
                )
                .properties(height=110)
            )

            labels_p1 = (
                alt.Chart(sp)
                .transform_calculate(
                    label="datum.Place + ' — ' + format(datum['{}'], ',.0f')".format(cost_col)
                )
                .mark_text(align="left", dx=3, color="#111827")
                .encode(
                    x=alt.X(f"{cost_col}:Q"),
                    y=alt.Y("Place:N",
                            sort=sp.sort_values(cost_col)["Place"].tolist()),
                    text=alt.Text("label:N"),
                )
            )

            st.altair_chart((chp1 + labels_p1).configure_axis(labels=False, ticks=False, domain=False),
                            use_container_width=True)
        else:
            render_metric_card("Highest total cost (place)", "–", icon="📍", color="#6366f1")
    with cards[4]:
        if not place.empty and "Cost per hour" in place.columns:
            cph_df = place.dropna(subset=["Cost per hour"]).copy()
            if not cph_df.empty:
                top_cph_place = cph_df.sort_values("Cost per hour", ascending=False).iloc[0]
                render_metric_card(
                    title="Highest cost/hour (place)",
                    value=f"{top_cph_place['Place']} — {top_cph_place['Cost per hour']:,.0f}/h",
                    delta=None,
                    icon="⚡",
                    color="#22c55e",
                )
                spc = cph_df.sort_values("Cost per hour", ascending=False).head(5)

                chp2 = (
                    alt.Chart(spc)
                    .mark_bar(color="#86efac")
                    .encode(
                        x=alt.X("Cost per hour:Q", title=None),
                        y=alt.Y("Place:N",
                                sort=spc.sort_values("Cost per hour")["Place"].tolist(),
                                title=None),
                        tooltip=["Place", alt.Tooltip("Cost per hour:Q", format=",.0f")],
                    )
                    .properties(height=110)
                )

                labels_p2 = (
                    alt.Chart(spc)
                    .transform_calculate(
                        label="datum.Place + ' — ' + format(datum['Cost per hour'], ',.0f') + '/h'"
                    )
                    .mark_text(align="left", dx=3, color="#111827")
                    .encode(
                        x=alt.X("Cost per hour:Q"),
                        y=alt.Y("Place:N",
                                sort=spc.sort_values("Cost per hour")["Place"].tolist()),
                        text=alt.Text("label:N"),
                    )
                )

                st.altair_chart((chp2 + labels_p2).configure_axis(labels=False, ticks=False, domain=False),
                                use_container_width=True)
            else:
                render_metric_card("Highest cost/hour (place)", "–", icon="⚡", color="#22c55e")
        else:
            render_metric_card("Highest cost/hour (place)", "–", icon="⚡", color="#22c55e")
with st.expander("raw data and photos", expanded=False):       
    with stylable_container(key="vmk_more_container",
            css_styles="""
                {
                    background: white;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                    padding: 1.25rem;
                    margin: 0;
                }
            """,):
        st.caption("Edit data and save back to Google Sheets")
        scope = st.radio("Edit scope", ["Filtered view", "Full dataset"], horizontal=True)

        base_df = df_view if scope == "Filtered view" else df.copy()

        # Columns that should NOT be persisted back (derived or helper columns)
        computed_cols = [
            "Hour_num",
            "Time_h",
            "_hours_norm",
            "_cost_norm",
            "involvement",
            "Cost per hour",
        ]

        editable_cols = [c for c in base_df.columns if c not in computed_cols]
        editable_df = base_df[editable_cols].copy()

        st.info("Note: Saving will overwrite the corresponding sheet range with the edited rows.")
        edited = st.data_editor(
            editable_df,
            use_container_width=True,
            num_rows="dynamic",
        )

        col_l, col_r = st.columns([1, 2])
        with col_l:
            confirm = st.checkbox("I understand this will overwrite the sheet")
        with col_r:
            save_clicked = st.button("Save to Google Sheets", type="primary", disabled=not confirm)

        if save_clicked and confirm:
            try:
                # Ensure basic dtype cleanup before saving
                to_save = edited.copy()
                # Optional: coerce known numeric columns
                for col in [c for c in ["Plan cost", "Actual cost"] if c in to_save.columns]:
                    to_save[col] = pd.to_numeric(to_save[col], errors="ignore")
                if "Hour" in to_save.columns:
                    to_save["Hour"] = to_save["Hour"].astype(str)

                # Omitting worksheet writes to the first sheet; pass spreadsheet ID
                conn.update(spreadsheet=extract_spreadsheet_id(SPREADSHEET_URL), data=to_save)
                # Invalidate cache so subsequent load picks up changes
                get_sheet_data.clear()
                # Regenerate timeline after save if possible
                try:
                    fresh = get_sheet_data()
                    timeline_obj = build_timeline_json(fresh)
                    save_timeline_json(timeline_obj, os.path.join(base_dir, "tl.json"))
                except Exception:
                    pass
                st.success("Saved to Google Sheets.")
            except Exception as e:
                st.error(f"Failed to save: {e}")

        st.caption("Uploaded and linked photos")
        photos = st.session_state.get("trip_photos", [])
        if not photos:
            st.info("Add photos from the sidebar → Trip photos")
        else:
            cols = st.columns(3)
            for idx, p in enumerate(photos):
                with cols[idx % 3]:
                    if p.get("source") == "upload" and p.get("data"):
                        st.image(p["data"], caption=p.get("name"), use_container_width=True)
                    elif p.get("source") == "url" and p.get("url"):
                        st.image(p["url"], caption=p.get("name"), use_container_width=True)

        ## Story tab removed (timeline moved to top)

