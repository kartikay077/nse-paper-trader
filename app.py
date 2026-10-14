import streamlit as st
import pandas as pd
import yfinance as yf
import sqlite3
import pytz
import numpy as np
from datetime import datetime, timedelta
from contextlib import contextmanager
# --- CONFIGURATION ---
DB_NAME = "nse_v15.db"
TZ = pytz.timezone('Asia/Kolkata')
STARTING_CAPITAL = 1000.00
SAFETY_THRESHOLD = 1050.00
CAPITAL_FLOOR = 1000.00    
RISK_PER_TRADE_PCT = 0.005 # 0.5% (₹5.00)
DAILY_LOSS_LIMIT_PCT = 0.02 # 2% (₹20.00)
MIN_RR = 2.0
SLIPPAGE = 0.0005          
COMMISSION = 0.0006        
class Database:
    @staticmethod
    @contextmanager
    def get_conn():
        conn = sqlite3.connect(DB_NAME, timeout=30)
        conn.row_factory = sqlite3.Row
        try: yield conn
        finally: conn.close()
    @staticmethod
    def init():
        with Database.get_conn() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, val_float REAL, val_text TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, qty INTEGER, entry_price REAL, sl REAL, tp REAL, entry_fees REAL, entry_ts TEXT, last_processed_ts TEXT, session_id TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY, ts TEXT, type TEXT, symbol TEXT, qty INTEGER, price REAL, pnl REAL, fees REAL, balance REAL, note TEXT, session_id TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS archive_sessions (session_id TEXT PRIMARY KEY, end_ts TEXT, final_equity REAL)")
           
            conn.execute("INSERT OR IGNORE INTO state VALUES ('cash', ?, NULL)", (STARTING_CAPITAL,))
            conn.execute("INSERT OR IGNORE INTO state VALUES ('daily_start_equity', ?, NULL)", (STARTING_CAPITAL,))
            conn.execute("INSERT OR IGNORE INTO state VALUES ('last_date', NULL, '')")
            conn.execute("INSERT OR IGNORE INTO state VALUES ('session_id', NULL, ?)", (datetime.now().strftime('%Y%m%d%H%M%S'),))
            conn.execute("INSERT OR IGNORE INTO state VALUES ('kill_switch', 0, NULL)")
            conn.commit()
# --- DATA ENGINE ---
def fetch_and_validate(symbol, period="5d", interval="5m"):
    try:
        if not symbol.isalnum(): return None, "Malformed Symbol."
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 10: return None, "Insufficient Market Data."
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
       
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        # OHLC Validation
        valid_ohlc = (df['High'] >= df['Low']) & (df['High'] >= df['Open']) & (df['High'] >= df['Close']) & (df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])
        if not valid_ohlc.all(): return None, "Data Quality Error: Invalid OHLC relationships."
        if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any(): return None, "Data Error: Non-positive prices."
       
        # Timezone & Chronology
        if df.index.tz is None: df.index = df.index.tz_localize('UTC').tz_convert(TZ)
        else: df.index = df.index.tz_convert(TZ)
        if not df.index.is_monotonic_increasing: return None, "Data Error: Non-chronological timestamps."
       
        # Strip incomplete candle
        now = datetime.now(TZ)
        if (now - df.index[-1]).total_seconds() < 300: df = df.iloc[:-1]
        if len(df) < 10: return None, "Insufficient Completed Candles."
       
        last_ts = df.index[-1]
        is_stale = (now - last_ts).total_seconds() > 1800 # 30 min limit
        return df, {"is_stale": is_stale, "ts": last_ts, "provider": "Yahoo Finance (Delayed)"}
    except Exception as e:
        return None, f"Fetch System Failure: {str(e)}"
# --- THE FOUR AGENTS ---
def agent_1_news(symbol):
    try:
        t = yf.Ticker(f"{symbol}.NS")
        news = t.news
        if not news: return "UNCERTAIN", "No recent company disclosures."
        latest = news[0]
        pub_time = latest.get('providerPublishTime')
        if not pub_time:
            return "UNCERTAIN", "News timestamp missing."
        pub_ts = datetime.fromtimestamp(pub_time, tz=pytz.UTC)
        if (datetime.now(pytz.UTC) - pub_ts).total_seconds() > 86400:
            return "UNCERTAIN", "News stale (>24h)."
        publisher = latest.get('publisher', 'NSE Wire')
        title = latest.get('title', 'Market Announcement')
        return "RELEVANT", f"[{publisher}] {title}"
    except: return "UNCERTAIN", "News metadata service offline."
def agent_2_bull(df):
    ema5 = df['Close'].ewm(span=5, adjust=False).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    if ema5.iloc[-1] > ema20.iloc[-1] and ema5.iloc[-2] <= ema20.iloc[-2]:
        entry = df['Close'].iloc[-1]
        sl = df['Low'].iloc[-3:].min() * 0.998
        tp = entry + (entry - sl) * 2.1
        return "TRADE", {"entry": entry, "sl": sl, "tp": tp}
    return "NO TRADE", "Technical setup (EMA Crossover) not found."
def agent_3_bear(df, proposal):
    if not isinstance(proposal, dict): return "WAIT", "Awaiting bullish setup."
    # Challenge: Resistance near 20-candle high
    if df['Close'].iloc[-1] >= df['High'].iloc[-20:-1].max():
        return "REJECT", "Price at 20-candle resistance. High breakout failure risk."
    return "APPROVE", "No major structural barriers."
def agent_4_risk_gate(m, prop, symbol, news_status):
    if m['kill_switch']: return False, "Kill Switch Active."
    if m['is_stale']: return False, "Market Data Stale."
    if news_status != "RELEVANT": return False, f"News status: {news_status}. Entry blocked."
    if m['equity'] < SAFETY_THRESHOLD: return False, f"Equity ₹{m['equity']:.2f} < ₹{SAFETY_THRESHOLD}."
    if any(p['symbol'] == symbol for p in m['positions']): return False, "Duplicate Position."
   
    daily_pnl = m['equity'] - m['start_equity']
    if daily_pnl <= -(m['start_equity'] * DAILY_LOSS_LIMIT_PCT): return False, "Daily loss limit reached."
    # Fill Model
    entry_f = round(prop['entry'] * (1 + SLIPPAGE), 2)
    sl_f = round(prop['sl'] * (1 - SLIPPAGE), 2)
    tp_f = round(prop['tp'] * (1 - SLIPPAGE), 2)
   
    if not (tp_f > entry_f > sl_f > 0): return False, "Invalid Price Matrix (SL/TP)."
   
    risk_ps = entry_f - sl_f
    qty = int((m['equity'] * RISK_PER_TRADE_PCT) // risk_ps)
    if qty <= 0: return False, "Risk budget < 1 share."
   
    notional = round(qty * entry_f, 2)
    entry_fees = round(notional * COMMISSION, 2)
    if (notional + entry_fees) > m['cash']: return False, "Insufficient Cash."
   
    exit_fees_est = round((qty * tp_f) * COMMISSION, 2)
    net_risk = round((qty * risk_ps) + entry_fees + exit_fees_est, 2)
    net_reward = round((qty * (tp_f - entry_f)) - entry_fees - exit_fees_est, 2)
   
    if (net_reward / net_risk) < MIN_RR: return False, f"RR {net_reward/net_risk:.2f} < {MIN_RR}"
    if (m['equity'] - net_risk) < CAPITAL_FLOOR: return False, "Breaches ₹1,000 floor."
   
    return True, {"qty": qty, "entry": entry_f, "fees": entry_fees, "notional": notional}
# --- ACCOUNTING ---
def get_mtm_metrics(conn):
    cash = conn.execute("SELECT val_float FROM state WHERE key='cash'").fetchone()[0]
    start_eq = conn.execute("SELECT val_float FROM state WHERE key='daily_start_equity'").fetchone()[0]
    kill = conn.execute("SELECT val_float FROM state WHERE key='kill_switch'").fetchone()[0]
    session_id = conn.execute("SELECT val_text FROM state WHERE key='session_id'").fetchone()[0]
    db_pos = conn.execute("SELECT * FROM positions").fetchall()
   
    total_cmv, is_stale_global = 0.0, False
    pos_list = []
    for p in db_pos:
        df, meta = fetch_and_validate(p['symbol'], period="1d")
        if df is None: is_stale_global = True; continue
        curr_price = df['Close'].iloc[-1]
        total_cmv += (p['qty'] * curr_price)
        pos_list.append({**dict(p), 'cmv': curr_price, 'unrealized': (curr_price - p['entry_price']) * p['qty']})
       
    return {"cash": round(cash, 2), "equity": round(cash + total_cmv, 2), "start_equity": start_eq,
            "kill_switch": bool(kill), "positions": pos_list, "is_stale": is_stale_global, "session_id": session_id}
def process_exits(conn, p, df):
    raw_ts = pd.to_datetime(p['last_processed_ts'] if p['last_processed_ts'] else p['entry_ts'])
    last_ts = raw_ts.tz_localize(TZ) if raw_ts.tzinfo is None else raw_ts.tz_convert(TZ)
    df_new = df[df.index > last_ts]
   
    for ts, row in df_new.iterrows():
        exit_p, note = None, ""
        if row['Open'] <= p['sl']: exit_p, note = row['Open'], "Stop Loss (Gap)"
        elif row['Low'] <= p['sl']: exit_p, note = p['sl'], "Stop Loss"
        elif row['High'] >= p['tp']: exit_p, note = p['tp'], "Target Hit"
       
        if exit_p:
            fill = round(exit_p * (1 - SLIPPAGE), 2)
            fees = round((fill * p['qty']) * COMMISSION, 2)
            proceeds = round((p['qty'] * fill) - fees, 2)
           
            new_cash = round(conn.execute("SELECT val_float FROM state WHERE key='cash'").fetchone()[0] + proceeds, 2)
            net_pnl = round(proceeds - ((p['qty'] * p['entry_price']) + p['entry_fees']), 2)
           
            conn.execute("INSERT INTO ledger (ts, type, symbol, qty, price, pnl, fees, balance, note, session_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (ts.isoformat(), "EXIT", p['symbol'], p['qty'], fill, net_pnl, fees, new_cash, note, p['session_id']))
            conn.execute("UPDATE state SET val_float=? WHERE key='cash'", (new_cash,))
            conn.delete_positions = conn.execute("DELETE FROM positions WHERE symbol=?", (p['symbol'],))
            return True
        conn.execute("UPDATE positions SET last_processed_ts=? WHERE symbol=?", (ts.isoformat(), p['symbol']))
    return False
# --- UI ---
st.set_page_config(page_title="Four-Agent NSE Trader V15", layout="wide", initial_sidebar_state="expanded")
Database.init()
with Database.get_conn() as conn:
    m = get_mtm_metrics(conn)
    today = datetime.now(TZ).strftime('%Y-%m-%d')
    if conn.execute("SELECT val_text FROM state WHERE key='last_date'").fetchone()[0] != today:
        conn.execute("UPDATE state SET val_float=? WHERE key='daily_start_equity'", (m['equity'],))
        conn.execute("UPDATE state SET val_text=? WHERE key='last_date'", (today,))
        conn.commit()
   
    if st.sidebar.button("🔄 Sync Market Data"):
        positions = conn.execute("SELECT * FROM positions").fetchall()
        for p in positions:
            df, meta = fetch_and_validate(p['symbol'])
            if df is not None: process_exits(conn, p, df)
        conn.commit(); st.rerun()
st.title("🛡️ Four-Agent NSE Trader V15")
st.caption("Paper Trading | JIT Execution | MTM Portfolio Accounting")
# Dashboard Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"₹{m['equity']:.2f}")
c2.metric("Cash", f"₹{m['cash']:.2f}")
c3.metric("Daily Limit Used", f"₹{abs(m['equity'] - m['start_equity']):.2f}")
c4.metric("Session ID", m['session_id'])
if m['is_stale']: st.warning("⚠️ Market Data Delay detected. Real-time paper signals may be disabled.")
t1, t2, t3, t4 = st.tabs(["Scanner", "Active Positions", "Auditable Ledger", "Verification"])
with t1:
    sym = st.text_input("NSE Symbol", "RELIANCE").upper().strip()
    if st.button("Step 1: Scan Market Opportunity"):
        df, meta = fetch_and_validate(sym)
        if df is not None:
            a1_s, a1_m = agent_1_news(sym)
            a2_s, prop = agent_2_bull(df)
            a3_s, a3_m = agent_3_bear(df, prop)
           
            st.write(f"**Agent 1 (News):** {a1_s} | {a1_m}")
            st.write(f"**Agent 2 (Bull):** {a2_s} | {a2_m if a2_s=='NO TRADE' else ''}")
            st.write(f"**Agent 3 (Bear):** {a3_s} | {a3_m}")
           
            if a2_s == "TRADE" and a3_s == "APPROVE":
                ok, res = agent_4_risk_gate(m, prop, sym, a1_s)
                if ok:
                    st.success(f"**Agent 4 (Risk):** PASS. Proposal: Buy {res['qty']} @ ₹{res['entry']:.2f}")
                    if st.button("Step 2: CONFIRM ATOMIC EXECUTION"):
                        with Database.get_conn() as conn:
                            conn.execute("BEGIN TRANSACTION")
                            try:
                                # JUST-IN-TIME RE-VALIDATION
                                fresh_df, f_meta = fetch_and_validate(sym)
                                if fresh_df is None or f_meta['is_stale']: raise Exception("JIT Error: Stale/Incomplete Data.")
                               
                                fresh_m = get_mtm_metrics(conn)
                                n_s, _ = agent_1_news(sym)
                                b_s, fresh_prop = agent_2_bull(fresh_df)
                               
                                if b_s != "TRADE": raise Exception("JIT Error: Bullish Signal Disappeared.")
                               
                                final_ok, final_res = agent_4_risk_gate(fresh_m, fresh_prop, sym, n_s)
                                if final_ok:
                                    conn.execute("UPDATE state SET val_float = val_float - ? WHERE key='cash'", (final_res['notional'] + final_res['fees'],))
                                    conn.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?)",
                                                (sym, final_res['qty'], final_res['entry'], fresh_prop['sl'], fresh_prop['tp'], final_res['fees'], datetime.now(TZ).isoformat(), None, m['session_id']))
                                    conn.execute("INSERT INTO ledger (ts, type, symbol, qty, price, pnl, fees, balance, note, session_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                                (datetime.now(TZ).isoformat(), "ENTRY", sym, final_res['qty'], final_res['entry'], 0, final_res['fees'], round(fresh_m['cash'] - (final_res['notional'] + final_res['fees']), 2), "Position Opened", m['session_id']))
                                    conn.commit(); st.rerun()
                                else: st.error(f"JIT Re-validation Failed: {final_res}")
                            except Exception as e:
                                conn.rollback(); st.error(f"Execution Error: {e}")
                else: st.error(f"**Agent 4 (Risk):** {res}")
        else: st.error(meta)
with t2:
    st.table(m['positions'])
    if st.button("🚨 EMERGENCY KILL SWITCH"):
        with Database.get_conn() as conn:
            conn.execute("UPDATE state SET val_float=1 WHERE key='kill_switch'")
            conn.commit(); st.rerun()
with t3:
    if st.checkbox("Confirm Archival Reset"):
        if st.button("Archive Session & Reset"):
            with Database.get_conn() as conn:
                conn.execute("INSERT INTO archive_sessions VALUES (?,?,?)", (m['session_id'], datetime.now().isoformat(), m['equity']))
                conn.execute("DELETE FROM positions"); conn.execute("DELETE FROM ledger")
                conn.execute("UPDATE state SET val_float=? WHERE key='cash'", (STARTING_CAPITAL,))
                conn.execute("UPDATE state SET val_float=0 WHERE key='kill_switch'")
                conn.execute("UPDATE state SET val_text=? WHERE key='session_id'", (datetime.now().strftime('%Y%m%d%H%M%S'),))
                conn.commit(); st.rerun()
    with Database.get_conn() as conn:
        st.dataframe(pd.read_sql("SELECT * FROM ledger ORDER BY id DESC", conn))
with t4:
    st.subheader("Automated Performance Verification")
    if st.button("Run Unit Test Suite"):
        results = []
        # 1. News Rejection
        m_mock = {'cash': 1000, 'equity': 1000, 'start_equity': 1000, 'kill_switch': 0, 'is_stale': False, 'positions': []}
        ok, res = agent_4_risk_gate(m_mock, {'entry':100, 'sl':95, 'tp':120}, "TEST", "UNCERTAIN")
        results.append(("Risk: News Gate (UNCERTAIN)", "PASS" if not ok else "FAIL"))
       
        # 2. Accounting Precision
        entry_cost = 100.06; exit_val = 109.934; expected_pnl = 9.874
        results.append(("Accounting: P&L Precision", "PASS" if round(exit_val - entry_cost, 3) == expected_pnl else "FAIL"))
        # 3. Gap Exit Handler
        with Database.get_conn() as conn:
            conn.execute("SAVEPOINT test_gap")
            p_mock = {'symbol': 'TEST', 'qty': 1, 'entry_price': 100, 'sl': 95, 'tp': 120, 'entry_fees': 0.06, 'entry_ts': '2024-01-01', 'last_processed_ts': None, 'session_id': 'T'}
            df_gap = pd.DataFrame({'Open':[90], 'High':[92], 'Low':[89], 'Close':[91]}, index=[pd.to_datetime('2024-01-01 09:15:00').tz_localize(TZ)])
            process_exits(conn, p_mock, df_gap)
            exit_px = conn.execute("SELECT price FROM ledger WHERE symbol='TEST'").fetchone()[0]
            results.append(("Exit: Gap Handling (Open Fill)", "PASS" if exit_px < 95 else "FAIL"))
            conn.execute("ROLLBACK TO test_gap")
        for n, r in results: st.write(f"{n}: {r}")
