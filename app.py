import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面設定
st.set_page_config(
    page_title="黃金期貨趨勢動量監控與歷史回測",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 黃金期貨 (GC=F) 趨勢動量監控與歷史回測系統")
st.caption("結合雙均線趨勢濾網、RSI 動量與歷史單筆進出場模擬的回測分析平台。")

# ---------------------------------------------------------
# 1. 策略說明頁籤 / 區塊
# ---------------------------------------------------------
with st.expander("📖 點擊展開：多空判斷邏輯與回測機制說明", expanded=False):
    st.markdown("""
    ### 策略架構說明
    本系統採用 **Trend + Momentum** 的組合邏輯，對黃金進行趨勢識別與歷史回測：

    * **1. 多空趨勢判讀 (Trend Rules)**：
        * **🟢 多頭訊號 (Long)**：當 `收盤價 > 快速 EMA` 且 `快速 EMA > 慢速 EMA` 時，判定為多頭強勢區，策略持有多單。
        * **🔴 空頭訊號 (Short)**：當 `收盤價 < 快速 EMA` 且 `快速 EMA < 慢速 EMA` 時，判定為空頭強勢區，策略持有空單（或空倉）。
        * **🟡 震盪觀望 (Neutral)**：當均線發生交叉糾纏或價格夾於雙均線之間時，判定為盤整，離場觀望。
    * **2. 輔助動量指標 (Momentum & Volatility)**：
        * **RSI (14日)**：衡量價格超買（>70）或超賣（<30）狀態。
        * **ATR (14日)**：真實平均波幅，反映當前市場波動度，可用於設定動態停損或部位規模。
    * **3. 回測計算方式**：
        * **訊號觸發與執行**：依據前一交易日收盤確認的訊號，於次日以當日收盤價執行換倉。
        * **績效對比**：同時計算「策略累積報酬」與「買入持有（Buy & Hold）報酬」。
    """)

# ---------------------------------------------------------
# 側邊欄設定
# ---------------------------------------------------------
st.sidebar.header("⚙️ 參數與回測設定")
ticker_symbol = st.sidebar.selectbox("標的選擇", ["GC=F", "XAUUSD=X"], index=0)
time_period = st.sidebar.selectbox("歷史數據區間", ["1y", "2y", "3y", "5y", "max"], index=2)
ema_fast_len = st.sidebar.slider("快速均線 (EMA 1)", 5, 30, 20)
ema_slow_len = st.sidebar.slider("慢速均線 (EMA 2)", 20, 100, 50)
rsi_period = st.sidebar.slider("RSI 週期", 7, 21, 14)
allow_short = st.sidebar.checkbox("開啟做空機制（若關閉則盤整/空頭時皆空倉觀望）", value=True)

# ---------------------------------------------------------
# 資料抓取與計算指標 / 回測邏輯
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def get_data_and_backtest(ticker, period, fast_n, slow_n, rsi_n, enable_short):
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=['Close']).copy()

    # 指標計算
    df['EMA_Fast'] = df['Close'].ewm(span=fast_n, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_n, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_n).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = np.maximum(high_low, np.maximum(high_cp, low_cp))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # 多空目標部位 Signal (1: 多頭, -1: 空頭, 0: 觀望)
    df['Signal'] = 0
    long_cond = (df['Close'] > df['EMA_Fast']) & (df['EMA_Fast'] > df['EMA_Slow'])
    short_cond = (df['Close'] < df['EMA_Fast']) & (df['EMA_Fast'] < df['EMA_Slow'])

    df.loc[long_cond, 'Signal'] = 1
    if enable_short:
        df.loc[short_cond, 'Signal'] = -1

    # 實際持倉與損益計算 (次日生效，避免Look-ahead Bias)
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Return'] = df['Position'] * df['Market_Return']

    # 累積報酬率與權益曲線
    df['Cum_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()

    # 紀錄換倉點 (進出場訊號)
    df['Trades'] = df['Position'].diff()

    return df

df = get_data_and_backtest(ticker_symbol, time_period, ema_fast_len, ema_slow_len, rsi_period, allow_short)

if df is None or df.empty:
    st.error("無法讀取市場資料，請確認網路連線或調整選取區間。")
else:
    # ---------------------------------------------------------
    # 2. 當前最新市場狀態卡片
    # ---------------------------------------------------------
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    latest_close = float(latest['Close'])
    prev_close = float(prev['Close'])
    change_pct = ((latest_close - prev_close) / prev_close) * 100

    fast_ema = float(latest['EMA_Fast'])
    slow_ema = float(latest['EMA_Slow'])
    current_signal = int(latest['Signal'])

    if current_signal == 1:
        status_text = "🟢 多頭排列 (Long)"
    elif current_signal == -1:
        status_text = "🔴 空頭排列 (Short)"
    else:
        status_text = "🟡 震盪觀望 (Neutral)"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新黃金價格", f"${latest_close:,.2f}", f"{change_pct:+.2f}%")
    col2.metric("當前市場訊號", status_text)
    col3.metric("RSI (14日)", f"{latest['RSI']:.1f}")
    col4.metric("ATR (14日波動度)", f"${latest['ATR']:.2f}")

    st.markdown("---")

    # ---------------------------------------------------------
    # 3. 回測績效統計 Summary
    # ---------------------------------------------------------
    st.subheader("📊 策略歷史回測績效表現")

    total_strat_ret = (df['Cum_Strategy_Return'].iloc[-1] - 1) * 100
    total_mkt_ret = (df['Cum_Market_Return'].iloc[-1] - 1) * 100

    # 計算勝率與交易次數
    trade_returns = df[df['Trades'] != 0]['Strategy_Return']
    num_trades = len(df[df['Trades'] != 0])
    
    # 計算最大回撤 (MDD)
    cum_ret = df['Cum_Strategy_Return']
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_drawdown = drawdown.min() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("策略總累積報酬率", f"{total_strat_ret:+.2f}%", delta_color="normal")
    m2.metric("買入持有報酬率", f"{total_mkt_ret:+.2f}%")
    m3.metric("總調倉次數", f"{num_trades} 次")
    m4.metric("策略最大回撤 (MDD)", f"{max_drawdown:.2f}%")

    # ---------------------------------------------------------
    # 4. 圖表繪製 (K線訊號圖 + 權益曲線圖 + RSI)
    # ---------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=("黃金 K 線與歷史進出場多空訊號", "策略歷史權益曲線 (Equity Curve)", "RSI 動量")
    )

    # 1. K線圖與均線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name=f'EMA {ema_fast_len}', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name=f'EMA {ema_slow_len}', line=dict(color='blue', width=1)), row=1, col=1)

    # 標示買進/賣出/平倉點位
    buy_signals = df[df['Trades'] > 0]
    sell_signals = df[df['Trades'] < 0]

    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Low'] * 0.99,
        mode='markers', name='多單建倉/空單平倉',
        marker=dict(symbol='triangle-up', size=11, color='green')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['High'] * 1.01,
        mode='markers', name='空單建倉/多單平倉',
        marker=dict(symbol='triangle-down', size=11, color='red')
    ), row=1, col=1)

    # 2. 權益曲線
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Cum_Strategy_Return'], mode='lines', name='策略權益', line=dict(color='green', width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Cum_Market_Return'], mode='lines', name='買入持有', line=dict(color='gray', width=1.5, dash='dash')
    ), row=2, col=1)

    # 3. RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=1)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=850,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 5. 歷史多空訊號明細表
    # ---------------------------------------------------------
    st.subheader("📋 歷史多空換倉紀錄與明細")
    
    trades_df = df[df['Trades'] != 0].copy()
    if not trades_df.empty:
        trades_df['動作'] = trades_df['Position'].map({1: '買進持有 (Long)', -1: '賣出做空 (Short)', 0: '平倉觀望 (Flat)'})
        show_trades = trades_df[['Close', 'EMA_Fast', 'EMA_Slow', 'RSI', '動作']].sort_index(ascending=False)
        st.dataframe(show_trades.style.format({
            'Close': '${:,.2f}',
            'EMA_Fast': '${:,.2f}',             'EMA_Slow': '${:,.2f}',
            'RSI': '{:.1f}'
        }), height=300)
    else:
        st.info("選定區間內未發生部位切換。")
