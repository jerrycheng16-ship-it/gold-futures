import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面配置
st.set_page_config(
    page_title="黃金價格趨勢與動量監控系統",
    page_icon="🪙",
    layout="wide"
)

# 標題與說明
st.title("🪙 紐約黃金期貨 (GC=F) 價格趨勢與動量監控")
st.caption("即時抓取 Yahoo Finance 最新數據，結合均線趨勢與 RSI/ATR 動量指標進行市場狀態判讀。")

# 側邊欄：控制參數
st.sidebar.header("⚙️ 策略與顯示參數設定")
ticker_symbol = st.sidebar.selectbox("選擇黃金標的", ["GC=F", "XAUUSD=X"], index=0, help="GC=F: 紐約黃金期貨；XAUUSD=X: 現貨黃金")
time_period = st.sidebar.selectbox("歷史數據區間", ["3m", "6m", "1y", "2y", "5y"], index=2)
ema_fast_len = st.sidebar.slider("快速均線 (EMA 1)", 5, 30, 20)
ema_slow_len = st.sidebar.slider("慢速均線 (EMA 2)", 20, 100, 50)
rsi_period = st.sidebar.slider("RSI 週期", 7, 21, 14)

# 資料抓取與計算函數
@st.cache_data(ttl=300)  # 快取 5 分鐘，避免頻繁請求
def load_and_process_data(ticker, period, fast_n, slow_n, rsi_n):
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        return None
    
    # 處理 yfinance 可能產生的 MultiIndex 欄位
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 計算均線
    df['EMA_Fast'] = df['Close'].ewm(span=fast_n, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_n, adjust=False).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 計算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_n).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 計算真實波動區間 (ATR)
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = np.maximum(high_low, np.maximum(high_cp, low_cp))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    return df

# 載入資料
df = load_and_process_data(ticker_symbol, time_period, ema_fast_len, ema_slow_len, rsi_period)

if df is None or df.empty:
    st.error("無法取得黃金市場數據，請檢查網路連接或稍後再試。")
else:
    # 取得最新一筆與前一筆數據
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    latest_close = float(latest['Close'])
    prev_close = float(prev['Close'])
    change_dollar = latest_close - prev_close
    change_pct = (change_dollar / prev_close) * 100

    fast_ema = float(latest['EMA_Fast'])
    slow_ema = float(latest['EMA_Slow'])
    current_rsi = float(latest['RSI'])
    current_atr = float(latest['ATR'])

    # 判斷趨勢狀態與信號
    if latest_close > fast_ema and fast_ema > slow_ema:
        trend_status = "多頭排列 (Bullish)"
        status_color = "🟢"
    elif latest_close < fast_ema and fast_ema < slow_ema:
        trend_status = "空頭排列 (Bearish)"
        status_color = "🔴"
    else:
        trend_status = "震盪盤整 (Consolidation)"
        status_color = "🟡"

    # 第一區塊：核心指標卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("當前黃金價格", f"${latest_close:,.2f}", f"{change_dollar:+.2f} ({change_pct:+.2f}%)")
    col2.metric("趨勢狀態", f"{status_color} {trend_status}")
    col3.metric(f"快速 EMA ({ema_fast_len})", f"${fast_ema:,.2f}")
    col4.metric(f"RSI ({rsi_period})", f"{current_rsi:.1f}", "超買 (>70)" if current_rsi > 70 else ("超賣 (<30)" if current_rsi < 30 else "常態"))
    col5.metric("ATR (14日波動度)", f"${current_atr:.2f}")

    st.markdown("---")

    # 第二區塊：Plotly 互動 K 線圖與動量子圖
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("黃金日 K 線與均線系統", "RSI 動量指標")
    )

    # 繪製 K 線圖
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="K線"
    ), row=1, col=1)

    # 繪製均線
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name=f'EMA {ema_fast_len}', line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name=f'EMA {ema_slow_len}', line=dict(color='blue', width=1.5)), row=1, col=1)
    if not df['SMA_200'].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200', line=dict(color='purple', width=1.5, dash='dash')), row=1, col=1)

    # 繪製 RSI 子圖
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='green', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 圖表版面樣式設定
    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 第三區塊：最新歷史數據與策略建議
    with st.expander("查看近 10 個交易日數據明細"):
        show_cols = ['Open', 'High', 'Low', 'Close', 'EMA_Fast', 'EMA_Slow', 'RSI', 'ATR']
        st.dataframe(df[show_cols].tail(10).style.format("{:.2f}"))
