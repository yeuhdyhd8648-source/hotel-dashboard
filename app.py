"""
Hotel Bookings Analytics Dashboard
------------------------------------------------
A professional, interactive Streamlit dashboard for exploring the
Hotel Booking Demand dataset (hotel_bookings.csv).

Run with:
    streamlit run app.py

Place hotel_bookings.csv in the same folder as this script,
or upload it via the sidebar file uploader.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Hotel Bookings Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# CUSTOM CSS — modern styling
# ==================================================================
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        /* KPI cards */
        .kpi-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border-radius: 14px;
            padding: 18px 20px;
            color: white;
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
            text-align: center;
        }
        .kpi-value { font-size: 26px; font-weight: 700; margin: 4px 0 0 0; }
        .kpi-label { font-size: 13px; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; }

        /* Header */
        .dashboard-title {
            font-size: 38px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0px;
        }
        .dashboard-subtitle {
            font-size: 16px;
            color: #4b5563;
            margin-top: 4px;
            margin-bottom: 20px;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
        }
        section[data-testid="stSidebar"] * {
            color: #f3f4f6 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #1f2937;
        }
    </style>
""", unsafe_allow_html=True)


# ==================================================================
# DATA LOADING
# ==================================================================
@st.cache_data(show_spinner="Loading dataset...")
def load_data(file) -> pd.DataFrame:
    """Load and lightly clean the hotel bookings dataset."""
    df = pd.read_csv(file)

    # --- Data type handling / cleaning -----------------------------------
    # Numeric columns that may contain NaNs -> fill sensibly
    for col in ["children", "agent", "company"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "children" in df.columns:
        df["children"] = df["children"].fillna(0).astype(int)

    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    # Build a proper arrival date column from year/month/day if available
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    if {"arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"}.issubset(df.columns):
        df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
        df["arrival_date"] = pd.to_datetime(
            dict(
                year=df["arrival_date_year"],
                month=df["arrival_month_num"],
                day=df["arrival_date_day_of_month"],
            ),
            errors="coerce",
        )

    # reservation_status_date -> datetime
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = pd.to_datetime(
            df["reservation_status_date"], errors="coerce"
        )

    # Total nights & total guests (handy derived metrics)
    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    if {"adults", "children", "babies"}.issubset(df.columns):
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # adr (avg daily rate) sanity: clip negative/extreme outliers
    if "adr" in df.columns:
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
        df["adr"] = df["adr"].clip(lower=0)

    # is_canceled as readable label
    if "is_canceled" in df.columns:
        df["booking_status"] = df["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})

    return df


# --- File source: default path or uploader ---
st.sidebar.markdown("## 🏨 Hotel Bookings")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload a different CSV (optional)", type=["csv"])
DEFAULT_PATH = "hotel_bookings.csv"

try:
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)
    else:
        raw_df = load_data(DEFAULT_PATH)
except FileNotFoundError:
    st.error(
        f"⚠️ Could not find `{DEFAULT_PATH}`. Please place the file next to this "
        "script or upload a CSV using the sidebar."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

if raw_df.empty:
    st.warning("The dataset is empty. Nothing to display.")
    st.stop()

# ==================================================================
# HEADER
# ==================================================================
st.markdown('<p class="dashboard-title">🏨 Hotel Bookings Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="dashboard-subtitle">Exploring booking demand, cancellations, revenue and guest '
    'behavior across City &amp; Resort hotels — filter on the left to drill in.</p>',
    unsafe_allow_html=True,
)

# ==================================================================
# SIDEBAR FILTERS
# ==================================================================
st.sidebar.markdown("### 🔎 Filters")

df = raw_df.copy()

# Hotel type filter
if "hotel" in df.columns:
    hotel_options = sorted(df["hotel"].dropna().unique().tolist())
    selected_hotels = st.sidebar.multiselect("Hotel Type", hotel_options, default=hotel_options)
    if selected_hotels:
        df = df[df["hotel"].isin(selected_hotels)]

# Booking status filter
if "booking_status" in df.columns:
    status_options = sorted(df["booking_status"].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect("Booking Status", status_options, default=status_options)
    if selected_status:
        df = df[df["booking_status"].isin(selected_status)]

# Year filter
if "arrival_date_year" in df.columns:
    year_options = sorted(df["arrival_date_year"].dropna().unique().tolist())
    selected_years = st.sidebar.multiselect("Arrival Year", year_options, default=year_options)
    if selected_years:
        df = df[df["arrival_date_year"].isin(selected_years)]

# Market segment filter
if "market_segment" in df.columns:
    segment_options = sorted(df["market_segment"].dropna().unique().tolist())
    selected_segments = st.sidebar.multiselect("Market Segment", segment_options, default=segment_options)
    if selected_segments:
        df = df[df["market_segment"].isin(selected_segments)]

# Customer type filter
if "customer_type" in df.columns:
    cust_options = sorted(df["customer_type"].dropna().unique().tolist())
    selected_cust = st.sidebar.multiselect("Customer Type", cust_options, default=cust_options)
    if selected_cust:
        df = df[df["customer_type"].isin(selected_cust)]

# Arrival date range filter
if "arrival_date" in df.columns and df["arrival_date"].notna().any():
    min_date = df["arrival_date"].min()
    max_date = df["arrival_date"].max()
    date_range = st.sidebar.date_input(
        "Arrival Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df["arrival_date"] >= pd.to_datetime(start_date))
            & (df["arrival_date"] <= pd.to_datetime(end_date))
        ]

# Lead time slider
if "lead_time" in df.columns:
    lt_min, lt_max = int(df["lead_time"].min()), int(df["lead_time"].max())
    if lt_min < lt_max:
        lead_range = st.sidebar.slider("Lead Time (days)", lt_min, lt_max, (lt_min, lt_max))
        df = df[(df["lead_time"] >= lead_range[0]) & (df["lead_time"] <= lead_range[1])]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** of **{len(raw_df):,}** total bookings")

if df.empty:
    st.warning("No data matches the selected filters. Please broaden your filter selection.")
    st.stop()

# ==================================================================
# KPI CARDS
# ==================================================================
total_bookings = len(df)
cancellations = int(df["is_canceled"].sum()) if "is_canceled" in df.columns else 0
cancel_rate = (cancellations / total_bookings * 100) if total_bookings else 0
avg_adr = df["adr"].mean() if "adr" in df.columns else np.nan
avg_lead_time = df["lead_time"].mean() if "lead_time" in df.columns else np.nan
avg_nights = df["total_nights"].mean() if "total_nights" in df.columns else np.nan
total_guests = int(df["total_guests"].sum()) if "total_guests" in df.columns else 0

kpi_cols = st.columns(6)
kpi_data = [
    ("Total Bookings", f"{total_bookings:,}"),
    ("Cancellation Rate", f"{cancel_rate:.1f}%"),
    ("Avg Daily Rate", f"${avg_adr:,.2f}" if pd.notna(avg_adr) else "N/A"),
    ("Avg Lead Time", f"{avg_lead_time:,.0f} days" if pd.notna(avg_lead_time) else "N/A"),
    ("Avg Stay Length", f"{avg_nights:,.1f} nights" if pd.notna(avg_nights) else "N/A"),
    ("Total Guests", f"{total_guests:,}"),
]

for col, (label, value) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ==================================================================
# CHARTS — ROW 1: Bookings over time & Hotel comparison
# ==================================================================
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader("📈 Booking Trends Over Time")
    if "arrival_date" in df.columns and df["arrival_date"].notna().any():
        monthly = (
            df.dropna(subset=["arrival_date"])
            .groupby([pd.Grouper(key="arrival_date", freq="MS"), "booking_status"])
            .size()
            .reset_index(name="bookings")
        )
        fig_trend = px.line(
            monthly,
            x="arrival_date",
            y="bookings",
            color="booking_status" if "booking_status" in df.columns else None,
            markers=True,
            template="plotly_white",
            color_discrete_map={"Not Canceled": "#2563eb", "Canceled": "#ef4444"},
        )
        fig_trend.update_layout(
            xaxis_title="Arrival Month",
            yaxis_title="Number of Bookings",
            legend_title="Status",
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No valid arrival date data available for trend chart.")

with row1_col2:
    st.subheader("🏨 Bookings by Hotel")
    if "hotel" in df.columns:
        hotel_counts = df["hotel"].value_counts().reset_index()
        hotel_counts.columns = ["hotel", "count"]
        fig_hotel = px.pie(
            hotel_counts,
            names="hotel",
            values="count",
            hole=0.55,
            template="plotly_white",
            color_discrete_sequence=["#2563eb", "#f97316"],
        )
        fig_hotel.update_traces(textinfo="percent+label")
        fig_hotel.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, showlegend=False)
        st.plotly_chart(fig_hotel, use_container_width=True)
    else:
        st.info("Column 'hotel' not found.")

# ==================================================================
# CHARTS — ROW 2: Cancellations by segment & ADR by hotel/month
# ==================================================================
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("❌ Cancellation Rate by Market Segment")
    if {"market_segment", "is_canceled"}.issubset(df.columns):
        seg = (
            df.groupby("market_segment")["is_canceled"]
            .mean()
            .mul(100)
            .reset_index(name="cancel_rate")
            .sort_values("cancel_rate", ascending=False)
        )
        fig_seg = px.bar(
            seg,
            x="market_segment",
            y="cancel_rate",
            template="plotly_white",
            color="cancel_rate",
            color_continuous_scale="Reds",
            text=seg["cancel_rate"].round(1).astype(str) + "%",
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(
            xaxis_title="Market Segment",
            yaxis_title="Cancellation Rate (%)",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.info("Required columns for this chart are missing.")

with row2_col2:
    st.subheader("💰 Average Daily Rate by Hotel & Month")
    if {"hotel", "arrival_date_month", "adr"}.issubset(df.columns):
        month_order = list(month_map.keys()) if "month_map" in dir() else [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"
        ]
        adr_month = (
            df.groupby(["arrival_date_month", "hotel"])["adr"]
            .mean()
            .reset_index()
        )
        adr_month["arrival_date_month"] = pd.Categorical(
            adr_month["arrival_date_month"], categories=month_order, ordered=True
        )
        adr_month = adr_month.sort_values("arrival_date_month")
        fig_adr = px.line(
            adr_month,
            x="arrival_date_month",
            y="adr",
            color="hotel",
            markers=True,
            template="plotly_white",
            color_discrete_sequence=["#2563eb", "#f97316"],
        )
        fig_adr.update_layout(
            xaxis_title="Month",
            yaxis_title="Avg Daily Rate ($)",
            legend_title="Hotel",
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig_adr, use_container_width=True)
    else:
        st.info("Required columns for this chart are missing.")

# ==================================================================
# CHARTS — ROW 3: Top countries & Lead time distribution
# ==================================================================
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("🌍 Top 10 Guest Countries")
    if "country" in df.columns:
        top_countries = (
            df[df["country"] != "Unknown"]["country"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_countries.columns = ["country", "bookings"]
        fig_country = px.bar(
            top_countries.sort_values("bookings"),
            x="bookings",
            y="country",
            orientation="h",
            template="plotly_white",
            color="bookings",
            color_continuous_scale="Blues",
        )
        fig_country.update_layout(
            xaxis_title="Bookings",
            yaxis_title="Country",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("Column 'country' not found.")

with row3_col2:
    st.subheader("⏳ Lead Time Distribution")
    if "lead_time" in df.columns:
        fig_lead = px.histogram(
            df,
            x="lead_time",
            nbins=40,
            template="plotly_white",
            color_discrete_sequence=["#2563eb"],
        )
        fig_lead.update_layout(
            xaxis_title="Lead Time (days)",
            yaxis_title="Number of Bookings",
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
            bargap=0.05,
        )
        st.plotly_chart(fig_lead, use_container_width=True)
    else:
        st.info("Column 'lead_time' not found.")

# ==================================================================
# CHARTS — ROW 4: Room type & Special requests
# ==================================================================
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    st.subheader("🛏️ Reserved vs Assigned Room Type")
    if {"reserved_room_type", "assigned_room_type"}.issubset(df.columns):
        room_match = np.where(
            df["reserved_room_type"] == df["assigned_room_type"], "Same Room", "Room Changed"
        )
        room_df = pd.Series(room_match, name="match").value_counts().reset_index()
        room_df.columns = ["status", "count"]
        fig_room = px.bar(
            room_df,
            x="status",
            y="count",
            template="plotly_white",
            color="status",
            color_discrete_map={"Same Room": "#22c55e", "Room Changed": "#f59e0b"},
            text="count",
        )
        fig_room.update_traces(textposition="outside")
        fig_room.update_layout(
            xaxis_title="",
            yaxis_title="Number of Bookings",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
        )
        st.plotly_chart(fig_room, use_container_width=True)
    else:
        st.info("Required columns for this chart are missing.")

with row4_col2:
    st.subheader("⭐ Special Requests Distribution")
    if "total_of_special_requests" in df.columns:
        req_counts = df["total_of_special_requests"].value_counts().sort_index().reset_index()
        req_counts.columns = ["requests", "bookings"]
        fig_req = px.bar(
            req_counts,
            x="requests",
            y="bookings",
            template="plotly_white",
            color_discrete_sequence=["#8b5cf6"],
            text="bookings",
        )
        fig_req.update_traces(textposition="outside")
        fig_req.update_layout(
            xaxis_title="Number of Special Requests",
            yaxis_title="Number of Bookings",
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
        )
        st.plotly_chart(fig_req, use_container_width=True)
    else:
        st.info("Column 'total_of_special_requests' not found.")

# ==================================================================
# DATA TABLE + DOWNLOAD
# ==================================================================
st.markdown("---")
st.subheader("📋 Filtered Data Preview")

preview_rows = st.slider("Rows to preview", min_value=5, max_value=100, value=15, step=5)
st.dataframe(df.head(preview_rows), use_container_width=True, height=380)


@st.cache_data
def convert_df_to_csv(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


csv_data = convert_df_to_csv(df)
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=csv_data,
    file_name=f"hotel_bookings_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

# ==================================================================
# FOOTER
# ==================================================================
st.markdown("---")
st.caption(
    "Dashboard built with Streamlit & Plotly · Data: Hotel Booking Demand dataset "
    f"· Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)