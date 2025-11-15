# app.py
# Mukherjee Screener – Offline Ultra UI (Forex + US500 + Nifty500 + Indices)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests

# Optional: auto-refresh (if installed)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# =========================
# 🔐 TELEGRAM CONFIG (OPTIONAL)
# =========================
TELEGRAM_BOT_TOKEN = ""  # yahan apna bot token daalo (optional)
TELEGRAM_CHAT_ID = ""    # yahan apna chat id daalo (optional)


def send_telegram_message(text: str):
    """Simple Telegram push (safe if token/chat_id empty)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Telegram error:", e)


# =========================
# 🔗 OFFLINE SYMBOL LISTS
# =========================

FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    "GBPJPY=X", "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",
    "AUDJPY=X", "AUDNZD=X", "AUDCAD=X", "AUDCHF=X",
    "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
    "CADJPY=X", "CADCHF=X",
    "CHFJPY=X"
]

# ✅ US TOP 500 (sample list – baaki tickers yahan add kar sakte ho)
US500_LIST = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","UNH","XOM",
    "JNJ","JPM","V","PG","AVGO","HD","LLY","CVX","MRK","PEP",
    "ABBV","KO","COST","ADBE","MCD","TMO","CSCO","PFE","ACN","DHR",
    "LIN","ABT","WMT","CRM","INTC","AMD","NFLX","BAC","DIS","TXN",
    "QCOM","HON","PM","AMGN","IBM","LOW","ORCL","CAT","GE","SPGI",
    "NOW","INTU","RTX","GS","MS","BKNG","ADI","MU","LRCX","AMAT",
    # 👇 yahan se aage apne hisaab se baaki S&P 500 tickers add karo
]

# ✅ NIFTY 500 (sample list – baaki tickers yahan add kar sakte ho)
NIFTY500_LIST = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","SBIN.NS","BHARTIARTL.NS",
    "KOTAKBANK.NS","LT.NS","ITC.NS","AXISBANK.NS","ASIANPAINT.NS","HINDUNILVR.NS","MARUTI.NS",
    "BAJFINANCE.NS","SUNPHARMA.NS","HCLTECH.NS","WIPRO.NS","ULTRACEMCO.NS","POWERGRID.NS",
    "NTPC.NS","ONGC.NS","ADANIENT.NS","ADANIPORTS.NS","TECHM.NS","TITAN.NS","NESTLEIND.NS",
    "JSWSTEEL.NS","GRASIM.NS","HDFCLIFE.NS","BAJAJFINSV.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "DIVISLAB.NS","COALINDIA.NS","DRREDDY.NS","BRITANNIA.NS","HEROMOTOCO.NS","EICHERMOT.NS",
    "CIPLA.NS","BPCL.NS","IOC.NS","SHREECEM.NS","ICICIPRULI.NS","HAVELLS.NS","PIDILITIND.NS",
    "DABUR.NS","INDUSINDBK.NS","SBICARD.NS",
    # 👇 yahan se aage apne hisaab se baaki Nifty 500 tickers add karo
]

INDICES_MAP = {
    "US30": "^DJI",
    "NAS100": "^NDX",
    "SPX500": "^GSPC",
    "GER40": "^GDAXI",
    "UK100": "^FTSE",
    "JPN225": "^N225",
    "AUS200": "^AXJO",
    "HK50": "^HSI",
}


def resolve_symbol(sym: str) -> str:
    """Indices ke display naam ko Yahoo symbol me map karo."""
    return INDICES_MAP.get(sym, sym)


# =========================
# 🎨 THEME / CSS
# =========================
def inject_css():
    st.markdown(
        """
    <style>
    .stApp {
        background-color: #0d1117 !important;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1300px;
    }
    .option-box {
        background: rgba(255,255,255,0.05);
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 0 15px rgba(0,0,0,0.4);
    }
    h1, h2, h3, h4, h5 {
        color: #ffffff !important;
    }
    label, .stRadio label, .stCheckbox label {
        color: #dddddd !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #005cf0 0%, #0084ff 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 25px;
        border: none;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 0 18px rgba(0,120,250,0.55);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px rgba(0,150,255,0.9);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# =========================
# 📥 DATA LOADERS
# =========================
@st.cache_data(show_spinner=False)
def load_ohlc_screener(symbol: str, timeframe: str, lookback_days: int = 90) -> pd.DataFrame:
    """
    Screener ke liye TF: 4h, 6h, 8h, 10h, 12h, 1d
    Multi-hour TF = 60m se resample.
    """
    real_symbol = resolve_symbol(symbol)
    custom_tfs = ["4h", "6h", "8h", "10h", "12h"]

    if timeframe in custom_tfs:
        df = yf.download(real_symbol, interval="60m", period=f"{lookback_days}d", auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(
            columns={
                "Datetime": "time",
                "Date": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df.set_index("time")
        rule = timeframe.upper()  # "4h" -> "4H"
        df = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
        return df[["time", "open", "high", "low", "close", "volume"]]

    if timeframe == "1d":
        df = yf.download(real_symbol, interval="1d", period=f"{lookback_days}d", auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(
            columns={
                "Datetime": "time",
                "Date": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return df[["time", "open", "high", "low", "close", "volume"]]

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_chart_data(symbol: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Chart ke liye: 1m,5m,15m,30m,1h,2h,4h,6h,8h,10h,12h,1d,1w
    """
    real_symbol = resolve_symbol(symbol)
    tf_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "2h": "60m",
        "4h": "60m",
        "6h": "60m",
        "8h": "60m",
        "10h": "60m",
        "12h": "60m",
        "1d": "1d",
        "1w": "1wk",
    }
    base_tf = tf_map[timeframe]
    df = yf.download(real_symbol, interval=base_tf, period=f"{lookback_days}d", auto_adjust=False)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(
        columns={
            "Datetime": "time",
            "Date": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    if timeframe in ["2h", "4h", "6h", "8h", "10h", "12h"]:
        df = df.set_index("time")
        rule = timeframe.upper()
        df = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )

    return df[["time", "open", "high", "low", "close", "volume"]]


# =========================
# 📐 ZONE ENGINE
# =========================
def detect_trend(df: pd.DataFrame, ema_len: int = 50) -> str:
    if len(df) < ema_len + 5:
        return "Sideways"
    ema = df["close"].ewm(span=ema_len, adjust=False).mean()
    if df["close"].iloc[-1] > ema.iloc[-1]:
        return "Uptrend"
    if df["close"].iloc[-1] < ema.iloc[-1]:
        return "Downtrend"
    return "Sideways"


def detect_base_type(df: pd.DataFrame, i: int, base_count: int = 1, base_type: str = "Normal Base") -> bool:
    window = df.iloc[i - (base_count - 1) : i + 1]

    if base_type == "Normal Base":
        for _, row in window.iterrows():
            o, c, h, l = row[["open", "close", "high", "low"]]
            body = abs(c - o)
            rng = h - l
            if rng == 0 or body / rng > 0.40:
                return False
        return True

    if base_type == "Doji Base":
        for _, row in window.iterrows():
            o, c, h, l = row[["open", "close", "high", "low"]]
            body = abs(c - o)
            rng = h - l
            if rng == 0 or body / rng > 0.15:
                return False
        return True

    if base_type == "Inside Bar":
        if base_count == 1:
            return True
        high_prev = window.iloc[0]["high"]
        low_prev = window.iloc[0]["low"]
        for _, row in window.iterrows():
            if row["high"] > high_prev or row["low"] < low_prev:
                return False
        return True

    if base_type == "Compression Base":
        base_range = window["high"].max() - window["low"].min()
        if len(df) < 30:
            return False
        avg_range = (df["high"] - df["low"]).rolling(30).mean().iloc[i]
        if pd.isna(avg_range) or avg_range == 0:
            return False
        return base_range <= avg_range * 0.6

    return False


def find_bases(df: pd.DataFrame, base_count: int = 1, base_type: str = "Normal Base", reject_wide: bool = True):
    bases = []
    for i in range(base_count, len(df) - base_count):
        if not detect_base_type(df, i, base_count, base_type):
            continue
        if reject_wide and i >= 25:
            zone_range = df["high"].iloc[i] - df["low"].iloc[i]
            avg_range = (df["high"] - df["low"]).rolling(25).mean().iloc[i]
            if not pd.isna(avg_range) and zone_range > avg_range * 2.0:
                continue
        bases.append(i)
    return bases


def classify_pattern(df: pd.DataFrame, idx: int) -> str:
    if idx < 1 or idx >= len(df) - 1:
        return "UNKNOWN"
    o_prev, c_prev = df.loc[idx - 1, ["open", "close"]]
    o_next, c_next = df.loc[idx + 1, ["open", "close"]]
    leg_in_up = c_prev > o_prev
    leg_in_down = c_prev < o_prev
    leg_out_up = c_next > o_next
    leg_out_down = c_next < o_next
    if leg_in_up and leg_out_up:
        return "RBR"
    if leg_in_up and leg_out_down:
        return "RBD"
    if leg_in_down and leg_out_up:
        return "DBR"
    if leg_in_down and leg_out_down:
        return "DBD"
    return "UNKNOWN"


def check_departure_strength(
    df: pd.DataFrame,
    base_index: int,
    required_impulse: int = 2,
    multiplier: float = 2.0,
) -> bool:
    if base_index >= len(df) - required_impulse - 1:
        return False

    base_low = df["low"].iloc[base_index]
    base_high = df["high"].iloc[base_index]
    base_range = base_high - base_low
    if base_range <= 0:
        return False

    future = df.iloc[base_index + 1 : base_index + 1 + required_impulse]

    for _, row in future.iterrows():
        body = abs(row["close"] - row["open"])
        if body < base_range * multiplier:
            return False

    final_close = future["close"].iloc[-1]
    if final_close > base_high + base_range * 0.5:
        return True
    if final_close < base_low - base_range * 0.5:
        return True
    return False


def mark_fresh(df: pd.DataFrame, zone: dict, start_idx: int) -> bool:
    y0 = min(zone["proximal"], zone["distal"])
    y1 = max(zone["proximal"], zone["distal"])
    for i in range(start_idx + 1, len(df)):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        if zone["type"] == "Demand":
            if l <= y1 and h >= y0:
                return False
        else:
            if h >= y0 and l <= y1:
                return False
    return True


def zones_to_df(zones: list, symbol: str, timeframe: str) -> pd.DataFrame:
    if not zones:
        return pd.DataFrame()
    df = pd.DataFrame(zones)
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    return df[
        [
            "symbol",
            "timeframe",
            "time",
            "type",
            "pattern",
            "proximal",
            "distal",
            "fresh",
            "trend",
            "rrr_est",
            "score",
        ]
    ]


def find_zones(
    df: pd.DataFrame,
    base_count: int = 1,
    base_type: str = "Normal Base",
    reject_wide: bool = True,
    departure_strength: str = "Strong",
    required_impulse: int = 2,
    strength_multiplier: float = 2.0,
):
    zones = []
    if df.empty or len(df) < 30:
        return zones

    trend = detect_trend(df)
    bases = find_bases(df, base_count=base_count, base_type=base_type, reject_wide=reject_wide)

    for b in bases:
        pattern = classify_pattern(df, b)
        if pattern == "UNKNOWN":
            continue

        h = df["high"].iloc[b]
        l = df["low"].iloc[b]

        if pattern in ["DBR", "RBR"]:
            ztype = "Demand"
            distal = l
            proximal = h
        elif pattern in ["RBD", "DBD"]:
            ztype = "Supply"
            distal = h
            proximal = l
        else:
            continue

        strong_dep = check_departure_strength(
            df, b, required_impulse=required_impulse, multiplier=strength_multiplier
        )

        if departure_strength == "Strong" and not strong_dep:
            continue
        if departure_strength == "Very Strong" and not strong_dep:
            continue

        zone = {
            "index": b,
            "time": df["time"].iloc[b],
            "type": ztype,
            "pattern": pattern,
            "proximal": proximal,
            "distal": distal,
        }

        zone["fresh"] = mark_fresh(df, zone, b)
        zone["trend"] = trend

        future = df.iloc[b + 1 : b + 11]
        zh = abs(proximal - distal)
        if future.empty or zh == 0:
            rrr = np.nan
        else:
            if ztype == "Demand":
                max_move = future["high"].max() - proximal
            else:
                max_move = proximal - future["low"].min()
            rrr = max_move / zh if zh != 0 else np.nan
        zone["rrr_est"] = rrr

        score = 50
        if trend == "Uptrend" and ztype == "Demand":
            score += 15
        if trend == "Downtrend" and ztype == "Supply":
            score += 15
        if zone["fresh"]:
            score += 15
        if not np.isnan(rrr) and rrr >= 3:
            score += 10
        if strong_dep:
            score += 10

        zone["score"] = max(0, min(100, score))
        zones.append(zone)

    return zones


# =========================
# 🚀 MAIN APP
# =========================
def main():
    st.set_page_config(page_title="Mukherjee Screener PRO Offline", layout="wide")
    inject_css()

    st.markdown(
        "<h1 style='text-align:center;'>📊 Mukherjee Screener – Offline Ultra</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 🔄 Auto-refresh (optional)
    st.markdown("### 🔄 Live Auto-Refresh (Optional)")
    crr1, crr2 = st.columns([1, 3])
    with crr1:
        refresh_rate = st.selectbox("Refresh (sec)", [0, 10, 30, 60], index=0)
    with crr2:
        if refresh_rate > 0:
            if st_autorefresh:
                st.info(f"Auto-refresh every {refresh_rate} sec enabled.")
                st_autorefresh(interval=refresh_rate * 1000, key="auto_refresh")
            else:
                st.warning("Install: pip install streamlit-autorefresh")
    st.markdown("---")

    # ==============================
    #       CONTROL PANEL UI
    # ==============================
    st.markdown("### 🔧 Screener Controls")
    col1, col2, col3 = st.columns(3)

    # COLUMN 1 – MARKET + SYMBOLS
    with col1:
        st.markdown("<div class='option-box'>", unsafe_allow_html=True)
        market_choice = st.selectbox(
            "Select Market",
            ["Forex All Pairs", "US Top 500", "Nifty 500", "Indices"],
        )
        if market_choice == "Forex All Pairs":
            universe = FOREX_PAIRS
        elif market_choice == "US Top 500":
            universe = US500_LIST
        elif market_choice == "Nifty 500":
            universe = NIFTY500_LIST
        else:
            universe = list(INDICES_MAP.keys())
        selected_symbols = st.multiselect(
            "Select Symbols",
            universe,
            default=universe[:5] if len(universe) >= 5 else universe,
        )
        fast_mode = st.checkbox("⚡ Fast Scan (limit symbols)", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # COLUMN 2 – TIMEFRAMES
    with col2:
        st.markdown("<div class='option-box'>", unsafe_allow_html=True)
        timeframe_choice = st.multiselect(
            "Screener Timeframes",
            ["4h", "6h", "8h", "10h", "12h", "1d"],
            default=["4h", "1d"],
        )
        lookback_days = st.slider("Lookback Days", 30, 200, 90)
        st.markdown("</div>", unsafe_allow_html=True)

    # COLUMN 3 – ZONE SETTINGS
    with col3:
        st.markdown("<div class='option-box'>", unsafe_allow_html=True)
        zone_type_choice = st.radio("Zone Type", ["Both", "Supply Only", "Demand Only"])
        fresh_only = st.checkbox("Fresh Zones Only", value=True)
        valid_only = st.checkbox("Valid Zones Only (Score ≥ 80)", value=False)
        min_score = st.slider("Min Score", 0, 100, 70)

        base_candle_count = st.selectbox(
            "Base Candle Count",
            [1, 2, 3, 4, 5, 6],
            index=1,
        )
        base_type = st.selectbox(
            "Base Type",
            ["Normal Base", "Doji Base", "Inside Bar", "Compression Base"],
            index=0,
        )
        reject_wide_base = st.checkbox("Reject Wide Base", value=True)

        departure_strength = st.selectbox(
            "Departure Strength",
            ["Normal", "Strong", "Very Strong"],
            index=1,
        )
        required_impulse = st.selectbox("Impulse Candles", [1, 2, 3], index=1)
        strength_multiplier = st.selectbox(
            "Strength Multiplier (x)",
            [1.0, 1.5, 2.0, 3.0, 4.0],
            index=2,
        )

        telegram_alert = st.checkbox("📲 Telegram Alert on Fresh A+ Zones", value=False)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    scan_button = st.button("🚀 START SCAN", use_container_width=True)
    st.markdown("---")

    filtered = pd.DataFrame()
    result_df = pd.DataFrame()

    # ==============================
    #   SCREENER EXECUTION
    # ==============================
    if scan_button:
        if not selected_symbols or not timeframe_choice:
            st.error("Please select at least 1 symbol & 1 timeframe.")
        else:
            if fast_mode:
                symbols_to_scan = selected_symbols[: min(len(selected_symbols), 10)]
            else:
                symbols_to_scan = selected_symbols

            all_results = []
            total = len(symbols_to_scan) * len(timeframe_choice)
            done = 0
            progress = st.progress(0.0, text="Scanning...")

            for sym in symbols_to_scan:
                for tf in timeframe_choice:
                    df = load_ohlc_screener(sym, tf, lookback_days)
                    if df.empty or len(df) < 30:
                        done += 1
                        progress.progress(done / total, text=f"{sym} {tf}: no data")
                        continue

                    zones = find_zones(
                        df,
                        base_count=base_candle_count,
                        base_type=base_type,
                        reject_wide=reject_wide_base,
                        departure_strength=departure_strength,
                        required_impulse=required_impulse,
                        strength_multiplier=strength_multiplier,
                    )
                    zdf = zones_to_df(zones, sym, tf)
                    if not zdf.empty:
                        all_results.append(zdf)

                    done += 1
                    progress.progress(done / total, text=f"Processed {sym} {tf}")

            if not all_results:
                st.warning("No zones detected for current settings.")
            else:
                result_df = pd.concat(all_results, ignore_index=True)
                filtered = result_df.copy()

                if zone_type_choice == "Supply Only":
                    filtered = filtered[filtered["type"] == "Supply"]
                elif zone_type_choice == "Demand Only":
                    filtered = filtered[filtered["type"] == "Demand"]

                if fresh_only:
                    filtered = filtered[filtered["fresh"] == True]

                if valid_only:
                    filtered = filtered[filtered["score"] >= 80]

                filtered = filtered[filtered["score"] >= min_score]
                filtered = filtered.sort_values("score", ascending=False)

                st.subheader("📋 Screener Output")
                st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

                csv_data = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Zones CSV",
                    data=csv_data,
                    file_name="mukherjee_zones.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # Telegram alerts
                if telegram_alert and not filtered.empty:
                    hiq = filtered[
                        (filtered["fresh"] == True) & (filtered["score"] >= 90)
                    ].head(5)
                    if not hiq.empty:
                        lines = ["Mukherjee Screener Alert 🚨", ""]
                        for _, r in hiq.iterrows():
                            lines.append(
                                f"{r['symbol']} ({r['timeframe']}) - {r['type']} | Score: {r['score']} | Trend: {r['trend']}"
                            )
                        send_telegram_message("\n".join(lines))

    # ==============================
    # 📈 CHART VIEWER + ZONES
    # ==============================
    st.markdown("### 📈 Chart Viewer (Selected Symbol + Zones)")

    chart_timeframes = [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "10h",
        "12h",
        "1d",
        "1w",
    ]

    if not selected_symbols:
        st.info("Select some symbols above to view chart.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        chart_symbol = st.selectbox("Chart Symbol", selected_symbols, index=0)
    with c2:
        chart_tf = st.selectbox("Chart Timeframe", chart_timeframes, index=5)
    with c3:
        lookback_chart = st.slider("Chart Lookback (days)", 5, 120, 30)

    cdf = load_chart_data(chart_symbol, chart_tf, lookback_chart)
    if cdf.empty:
        st.error("No chart data for this symbol/timeframe.")
        return

    fig = go.Figure()
    fig.add_candlestick(
        x=cdf["time"],
        open=cdf["open"],
        high=cdf["high"],
        low=cdf["low"],
        close=cdf["close"],
        name="Price",
    )

    if not filtered.empty:
        zones_for_chart = filtered[filtered["symbol"] == chart_symbol]
        if not zones_for_chart.empty:
            x0 = cdf["time"].min()
            x1 = cdf["time"].max()
            for _, z in zones_for_chart.iterrows():
                y0 = min(z["proximal"], z["distal"])
                y1 = max(z["proximal"], z["distal"])
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor="rgba(0,255,0,0.15)"
                    if z["type"] == "Demand"
                    else "rgba(255,0,0,0.15)",
                    line=dict(width=0),
                    layer="below",
                )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
