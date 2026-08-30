"""
CoinDCX XAGUSDT (Silver) Futures - PAPER TRADING Bot (GitHub Actions version)
------------------------------------------------------------------------------
Runs ONCE per invocation - GitHub Actions triggers it every 15 minutes
(Mon-Fri only) via cron schedule (see .github/workflows/run-bot.yml).

State is persisted to state.json, committed back to the repo after each run.

Strategy: CCI(20, HLC/3) vs its own 14 EMA, with 20 EMA price trend filter
  Entry Long  : CCI(20) crosses ABOVE its 14 EMA, AND price > 20 EMA
  Entry Short : CCI(20) crosses BELOW its 14 EMA, AND price < 20 EMA
  Exit        : CCI(20) crosses its 14 EMA the OPPOSITE way to the open
                position (price filter does NOT apply to exits — a cross
                against the position always closes it, even if a new
                entry signal fires on the very same candle)
  Stop Loss   : none (only the signal-cross exit / take-profit / liquidation)
  Take Profit : 2% away from entry price
  Timeframe   : 15 minute candles
  Pair        : B-XAG_USDT
  Active days : Monday-Friday only (matches real silver market hours)

Capital & Risk (paper/simulated):
  Starting capital : 1000 (unit = quote currency, USDT)
  Capital per trade: 25% of current capital used as margin
  Leverage         : 10x

100% PAPER TRADING - only public endpoints, no API key needed, no real orders.

------------------------------------------------------------------------------
FIX (this version) — vs the original:
------------------------------------------------------------------------------
The original check_signals() conflated entry and exit into ONE signal per
candle:
    if bullish_cross and price_above: return "enter_long"
    if bearish_cross and price_below: return "enter_short"
    if bullish_cross or bearish_cross: return "exit"

Problem: during a strong trending move, a CCI cross often lines up WITH the
price filter (e.g. bullish_cross + price already above the 20 EMA). In that
case the function returned "enter_long"/"enter_short" instead of "exit" — so
an open SHORT position was never closed even though the trend had clearly
reversed against it. The position only closed later, once a cross happened
that did NOT match the price filter, by which point the loss had grown a lot
(this is exactly what happened on 2026-08-28: short opened 05:30 @ 68.77,
should have exited ~06:00 on the bullish cross, but the price filter lined up
with that cross so it was read as a long entry signal instead, and the
position wasn't actually closed until 08:00 @ 70.77, a much bigger loss).

Fix: exits and entries are now evaluated separately, in this order:
  1. If a position is open, check for a cross AGAINST that position's side.
     If found, close it immediately — no price filter involved.
  2. If flat (either already flat, or just closed above), check for a cross
     WITH the price filter to open a new position, same candle allowed.
"""

import time
import csv
import os
import json
from datetime import datetime, timezone

import requests

# ---------------- CONFIG ----------------
PAIR = "B-XAG_USDT"
RESOLUTION = "15"
CANDLE_DURATION_MS = 15 * 60 * 1000

STARTING_CAPITAL = 1000.0
CAPITAL_PER_TRADE_PCT = 0.25
LEVERAGE = 10
FEE_RATE = 0.0001          # CoinDCX commodity pairs promo fee: 0.01% maker/taker
TAKE_PROFIT_PCT = 0.02     # 2% target, no stop loss

HISTORY_HOURS = 240  # 10 days lookback - warms up CCI(20)+EMA(14) and price EMA(20)

CCI_PERIOD = 20
CCI_EMA_PERIOD = 14
PRICE_EMA_PERIOD = 20

STATE_FILE = "state.json"
TRADE_LOG_FILE = "xagusdt_paper_trades.csv"
CANDLES_URL = "https://public.coindcx.com/market_data/candlesticks"


# ---------------- DATA FETCH ----------------
def fetch_candles(lookback_hours):
    now = int(time.time())
    frm = now - lookback_hours * 3600
    params = {
        "pair": PAIR,
        "from": frm,
        "to": now,
        "resolution": RESOLUTION,
        "pcode": "f",
    }
    resp = requests.get(CANDLES_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    dedup = {c["time"]: c for c in data}
    return [dedup[t] for t in sorted(dedup)]


def only_closed_candles(candles):
    now_ms = int(time.time() * 1000)
    return [c for c in candles if c["time"] + CANDLE_DURATION_MS <= now_ms]


# ---------------- INDICATORS ----------------
def compute_ema(values, period):
    """EMA that tolerates leading None values (e.g. CCI isn't defined until
    enough candles exist). Starts once the first non-None value appears."""
    out = [None] * len(values)
    k = 2 / (period + 1)
    ema = None
    for i, v in enumerate(values):
        if v is None:
            continue
        ema = v if ema is None else v * k + ema * (1 - k)
        out[i] = ema
    return out


def compute_cci(candles, period):
    tp = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles]
    cci = [None] * len(tp)
    for i in range(period - 1, len(tp)):
        window = tp[i - period + 1:i + 1]
        sma = sum(window) / period
        mean_dev = sum(abs(x - sma) for x in window) / period
        cci[i] = 0.0 if mean_dev == 0 else (tp[i] - sma) / (0.015 * mean_dev)
    return cci


def add_indicators(candles):
    cci = compute_cci(candles, CCI_PERIOD)
    cci_ema = compute_ema(cci, CCI_EMA_PERIOD)
    price_ema = compute_ema([c["close"] for c in candles], PRICE_EMA_PERIOD)
    for i, c in enumerate(candles):
        c["cci"] = cci[i]
        c["cci_ema"] = cci_ema[i]
        c["price_ema"] = price_ema[i]
    return candles


# ---------------- STATE (persisted across runs) ----------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"capital": STARTING_CAPITAL, "position": None, "last_processed_time": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def init_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_time", "exit_time", "side", "entry_price", "exit_price",
                "quantity", "pnl", "reason", "capital_after"
            ])


def log_trade(row):
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ---------------- TRADE LOGIC ----------------
def open_position(state, side, price, ts):
    margin = state["capital"] * CAPITAL_PER_TRADE_PCT
    notional = margin * LEVERAGE
    quantity = notional / price
    liq_price = price * (1 - 1 / LEVERAGE) if side == "long" else price * (1 + 1 / LEVERAGE)
    tp_price = price * (1 + TAKE_PROFIT_PCT) if side == "long" else price * (1 - TAKE_PROFIT_PCT)

    state["position"] = {
        "side": side,
        "entry_price": price,
        "entry_time": ts,
        "quantity": quantity,
        "margin": margin,
        "liq_price": liq_price,
        "tp_price": tp_price,
    }
    print(f"[{ts}] OPEN {side.upper()} @ {price:.4f} | qty={quantity:.4f} "
          f"| margin={margin:.2f} | liq~{liq_price:.4f} | tp~{tp_price:.4f}")


def close_position(state, price, ts, reason):
    pos = state["position"]
    direction = 1 if pos["side"] == "long" else -1
    gross_pnl = (price - pos["entry_price"]) * pos["quantity"] * direction

    entry_notional = pos["entry_price"] * pos["quantity"]
    exit_notional = price * pos["quantity"]
    fees = (entry_notional + exit_notional) * FEE_RATE

    pnl = gross_pnl - fees
    if reason == "liquidated":
        pnl = -pos["margin"]

    state["capital"] += pnl
    print(f"[{ts}] CLOSE {pos['side'].upper()} @ {price:.4f} | pnl={pnl:.2f} "
          f"| reason={reason} | capital={state['capital']:.2f}")

    log_trade([
        pos["entry_time"], ts, pos["side"], pos["entry_price"], price,
        pos["quantity"], round(pnl, 4), reason, round(state["capital"], 4)
    ])
    state["position"] = None


def check_liquidation(state, current_price, ts):
    pos = state.get("position")
    if not pos:
        return
    if pos["side"] == "long" and current_price <= pos["liq_price"]:
        close_position(state, pos["liq_price"], ts, "liquidated")
    elif pos["side"] == "short" and current_price >= pos["liq_price"]:
        close_position(state, pos["liq_price"], ts, "liquidated")


def check_take_profit(state, current_price, ts):
    pos = state.get("position")
    if not pos:
        return
    if pos["side"] == "long" and current_price >= pos["tp_price"]:
        close_position(state, pos["tp_price"], ts, "take_profit")
    elif pos["side"] == "short" and current_price <= pos["tp_price"]:
        close_position(state, pos["tp_price"], ts, "take_profit")


def compute_crosses(prev_c, curr_c):
    """Returns (bullish_cross, bearish_cross), or None if indicators aren't
    ready yet (still warming up)."""
    if prev_c["cci"] is None or prev_c["cci_ema"] is None:
        return None
    if curr_c["cci"] is None or curr_c["cci_ema"] is None:
        return None

    bullish_cross = prev_c["cci"] <= prev_c["cci_ema"] and curr_c["cci"] > curr_c["cci_ema"]
    bearish_cross = prev_c["cci"] >= prev_c["cci_ema"] and curr_c["cci"] < curr_c["cci_ema"]
    return bullish_cross, bearish_cross


def check_exit(position_side, bullish_cross, bearish_cross):
    """Exit is based ONLY on the CCI cross going against the open position.
    The price-vs-20EMA filter is NOT applied here — that filter is for
    entries only. This is the actual fix: previously a cross that also
    happened to satisfy the price filter got misread as a fresh entry
    signal instead of an exit, so the position was never closed."""
    if position_side == "long" and bearish_cross:
        return True
    if position_side == "short" and bullish_cross:
        return True
    return False


def check_entry(curr_c, bullish_cross, bearish_cross):
    if curr_c["price_ema"] is None:
        return None
    price_above = curr_c["close"] > curr_c["price_ema"]
    price_below = curr_c["close"] < curr_c["price_ema"]

    if bullish_cross and price_above:
        return "enter_long"
    if bearish_cross and price_below:
        return "enter_short"
    return None


# ---------------- MAIN (one run) ----------------
def main():
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:  # 5=Saturday, 6=Sunday
        print(f"Weekend ({now_utc.strftime('%A')}) - bot inactive Mon-Fri only, skipping run.")
        return

    init_trade_log()
    state = load_state()
    is_first_run = state["last_processed_time"] == 0

    print(f"Loaded state: capital={state['capital']:.2f}, "
          f"position={'flat' if not state['position'] else state['position']['side']}")

    raw = fetch_candles(HISTORY_HOURS)
    if not raw:
        print("No candle data returned this run, skipping.")
        save_state(state)
        return

    current_price = float(raw[-1]["close"])
    ts_now = now_utc.isoformat()
    check_liquidation(state, current_price, ts_now)
    check_take_profit(state, current_price, ts_now)

    closed = only_closed_candles(raw)
    if not closed:
        print("No closed candles yet, skipping.")
        save_state(state)
        return

    closed = add_indicators(closed)

    if is_first_run:
        state["last_processed_time"] = closed[-1]["time"]
        print("First run: indicators warmed up, no historical signals executed. "
              "Bot will start reacting to the next new candle.")
    else:
        new_candles = [c for c in closed if c["time"] > state["last_processed_time"]]
        start_idx = len(closed) - len(new_candles)

        for i in range(max(start_idx, 1), len(closed)):
            prev_c = closed[i - 1]
            curr_c = closed[i]
            ts = datetime.fromtimestamp(curr_c["time"] / 1000, tz=timezone.utc).isoformat()
            price = float(curr_c["close"])

            crosses = compute_crosses(prev_c, curr_c)
            if crosses is None:
                continue
            bullish_cross, bearish_cross = crosses

            # 1. Exit check first — a cross against an open position closes
            #    it regardless of the price filter.
            if state["position"] and check_exit(state["position"]["side"], bullish_cross, bearish_cross):
                close_position(state, price, ts, "signal_exit")

            # 2. Entry check — only when flat (either already flat, or just
            #    closed above), same-candle re-entry allowed.
            if not state["position"]:
                signal = check_entry(curr_c, bullish_cross, bearish_cross)
                if signal == "enter_long":
                    open_position(state, "long", price, ts)
                elif signal == "enter_short":
                    open_position(state, "short", price, ts)

        state["last_processed_time"] = closed[-1]["time"]

    save_state(state)
    print(f"Run complete. Capital: {state['capital']:.2f} | "
          f"Position: {'flat' if not state['position'] else state['position']['side']}")


if __name__ == "__main__":
    main()
