#!/usr/bin/env python3
"""
backtest_nnfx.py - DaviddTech 5-indicator NNFX trend-following backtester (1H / 4H bars)
========================================================================================
Rules (evaluated on each bar's CLOSE, order filled at the NEXT bar's OPEN):
  1. Baseline:      EMA(20). Longs only if close > EMA, shorts only if close < EMA.
  2. Confirmation:  MACD(12,26,9) line crosses its signal line in the trade direction.
  3. Confirmation:  RSI(14) crosses above 50 (long) / below 50 (short).
  4. Volume guard:  RVOL = volume / SMA(volume, 20) >= 1.3.
  5. Exits:         ATR(14) bracket set from the fill price: stop 1.5x ATR, target 3.0x ATR (1:2 R:R).
                    If a bar touches both, the stop is assumed hit first (conservative). A bar that
                    opens past the stop/target fills at the open (gap).
  Sizing:           units = equity * 2% / (1.5 * ATR) on current equity (compounding; --no-compound
                    sizes on starting equity), capped so notional <= equity (no leverage).
                    Fractional units allowed (crypto, Alpaca fractional shares).
  Circuit breaker:  after 3 consecutive losing trades in one session (ET date for stocks, UTC date
                    for crypto), no new entries until the next session.
  One position per symbol at a time; each symbol runs on its own $1,000 account.

--window N relaxes rule 2/3 so the MACD and RSI crosses may occur within the last N bars (both
still have to be on the right side on the signal bar). N=1 (default) = same-bar crosses, as specified.

Data: Yahoo 1-hour bars (last ~730 days). 4H bars are built from the 1H bars.
Outputs go to results/nnfx/ only.

Usage:
  python src/backtest_nnfx.py                              # BTC-USD ETH-USD NVDA AMD, 1H and 4H
  python src/backtest_nnfx.py --symbols NVDA --tf 4h --window 3 --fee-bps 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "nnfx"


# --------------------------------------------------------------------------- data
def load_1h(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="730d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["Volume"] > 0]
    is_crypto = symbol.endswith("-USD")
    df.index = df.index.tz_convert("UTC" if is_crypto else "America/New_York")
    return df


def to_4h(df: pd.DataFrame, is_crypto: bool) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if is_crypto:
        return df.resample("4h").agg(agg).dropna()
    # Stocks: 4 consecutive session bars per block inside each day (09:30-13:30, 13:30-16:00).
    day = df.index.date
    block = df.groupby(day).cumcount() // 4
    out = df.groupby([day, block]).agg(agg)
    out.index = df.index.to_series().groupby([day, block]).first().values
    return out


# --------------------------------------------------------------------------- indicators
def add_indicators(df: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()   # Wilder smoothing
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss)

    d["rvol"] = d["Volume"] / d["Volume"].rolling(20).mean()

    prev_c = c.shift()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_c).abs(), (d["Low"] - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    macd_above = d["macd"] > d["macd_sig"]
    rsi_above = d["rsi"] > 50
    macd_x_up = macd_above & ~macd_above.shift(fill_value=False)
    macd_x_dn = ~macd_above & macd_above.shift(fill_value=False)
    rsi_x_up = rsi_above & ~rsi_above.shift(fill_value=False)
    rsi_x_dn = ~rsi_above & rsi_above.shift(fill_value=False)
    recent = lambda s: s.astype(int).rolling(window, min_periods=1).max().astype(bool)

    # LONG: baseline up + MACD crossed up + RSI crossed above 50 + volume surge
    d["long_sig"] = (
        (c > d["ema20"])
        & recent(macd_x_up) & macd_above
        & recent(rsi_x_up) & rsi_above
        & (d["rvol"] >= 1.3)
    )
    # SHORT: mirror image
    d["short_sig"] = (
        (c < d["ema20"])
        & recent(macd_x_dn) & ~macd_above
        & recent(rsi_x_dn) & ~rsi_above
        & (d["rvol"] >= 1.3)
    )
    warmup = 50
    d.iloc[:warmup, d.columns.get_indexer(["long_sig", "short_sig"])] = False
    return d


# --------------------------------------------------------------------------- simulation
def backtest(d: pd.DataFrame, symbol: str, tf: str, equity0: float, risk: float,
             fee_bps: float, allow_short: bool, compound: bool = True) -> tuple[list[dict], pd.Series]:
    fee = fee_bps / 1e4
    equity = equity0
    pos = None                      # open position dict
    pending = None                  # signal from previous bar -> fill at this bar's open
    session, loss_streak, halted = None, 0, False
    curve = []

    O, H, L, C = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr, ls, ss = d["atr"].to_numpy(), d["long_sig"].to_numpy(), d["short_sig"].to_numpy()
    idx = d.index
    trades = []

    def close_pos(i, price, reason):
        nonlocal equity, pos, loss_streak, halted
        side = pos["side"]
        gross = (price - pos["entry"]) * pos["units"] * side
        cost = (pos["entry"] + price) * pos["units"] * fee
        pnl = gross - cost
        equity += pnl
        trades.append({
            "symbol": symbol, "tf": tf, "side": "LONG" if side == 1 else "SHORT",
            "entry_time": pos["time"], "entry": pos["entry"], "stop": pos["sl"], "target": pos["tp"],
            "units": pos["units"], "exit_time": idx[i], "exit": price, "reason": reason,
            "pnl": pnl, "r_multiple": pnl / pos["risk_usd"], "equity": equity,
        })
        loss_streak = loss_streak + 1 if pnl < 0 else 0
        if loss_streak >= 3:
            halted = True           # daily circuit breaker
        pos = None

    for i in range(len(d)):
        # New session resets the circuit breaker.
        sess = idx[i].date()
        if sess != session:
            session, loss_streak, halted = sess, 0, False

        # 1) Fill a pending entry at this bar's open.
        if pending is not None and pos is None and not halted:
            side, a = pending
            entry = O[i]
            base = equity if compound else equity0      # compound: size on current equity
            units = base * risk / (1.5 * a)
            units = min(units, equity / entry)          # no leverage
            if units > 0:
                pos = {"side": side, "entry": entry, "units": units, "time": idx[i],
                       "sl": entry - side * 1.5 * a, "tp": entry + side * 3.0 * a,
                       "risk_usd": units * 1.5 * a}
        pending = None

        # 2) Manage the open position against this bar's range.
        if pos is not None:
            s, sl, tp = pos["side"], pos["sl"], pos["tp"]
            if s == 1:
                if O[i] <= sl:   close_pos(i, O[i], "SL gap")
                elif O[i] >= tp: close_pos(i, O[i], "TP gap")
                elif L[i] <= sl: close_pos(i, sl, "SL")
                elif H[i] >= tp: close_pos(i, tp, "TP")
            else:
                if O[i] >= sl:   close_pos(i, O[i], "SL gap")
                elif O[i] <= tp: close_pos(i, O[i], "TP gap")
                elif H[i] >= sl: close_pos(i, sl, "SL")
                elif L[i] <= tp: close_pos(i, tp, "TP")

        # 3) Signal on this bar's close -> queue for next open.
        if pos is None and not halted and np.isfinite(atr[i]) and atr[i] > 0:
            if ls[i]:
                pending = (1, atr[i])
            elif ss[i] and allow_short:
                pending = (-1, atr[i])

        mark = equity
        if pos is not None:
            mark += (C[i] - pos["entry"]) * pos["units"] * pos["side"]
        curve.append(mark)

    if pos is not None:
        close_pos(len(d) - 1, C[-1], "end of data")
        curve[-1] = equity
    return trades, pd.Series(curve, index=idx)


def summarize(trades: list[dict], curve: pd.Series, equity0: float, d: pd.DataFrame, first: int = 50) -> dict:
    t = pd.DataFrame(trades)
    n = len(t)
    wins = t[t.pnl > 0].pnl.sum() if n else 0.0
    losses = -t[t.pnl < 0].pnl.sum() if n else 0.0
    dd = (curve / curve.cummax() - 1).min() * 100 if len(curve) else 0.0
    final = curve.iloc[-1] if len(curve) else equity0
    return {
        "trades": n,
        "long/short": f"{(t.side == 'LONG').sum() if n else 0}/{(t.side == 'SHORT').sum() if n else 0}",
        "win_rate_%": round((t.pnl > 0).mean() * 100, 1) if n else 0.0,
        "profit_factor": round(wins / losses, 2) if losses > 0 else (np.inf if wins > 0 else 0.0),
        "avg_R": round(t.r_multiple.mean(), 2) if n else 0.0,
        "max_dd_%": round(dd, 1),
        "final_equity_$": round(final, 2),
        "return_%": round((final / equity0 - 1) * 100, 1),
        "buy_hold_%": round((d.Close.iloc[-1] / d.Close.iloc[first] - 1) * 100, 1),
        "start": d.index[first].strftime("%Y-%m-%d"),
        "end": d.index[-1].strftime("%Y-%m-%d"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD", "NVDA", "AMD"])
    ap.add_argument("--tf", nargs="+", default=["1h", "4h"], choices=["1h", "4h"])
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--risk", type=float, default=0.02, help="fraction of equity risked per trade")
    ap.add_argument("--window", type=int, default=1, help="bars within which MACD/RSI crosses may occur")
    ap.add_argument("--fee-bps", type=float, default=5.0, help="fees+slippage per side, basis points")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--no-compound", action="store_true", help="size every trade on starting equity")
    ap.add_argument("--start", help="first trading date YYYY-MM-DD (indicators still warm up on earlier bars)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, all_trades = [], []
    for sym in args.symbols:
        raw = load_1h(sym)
        for tf in args.tf:
            bars = raw if tf == "1h" else to_4h(raw, sym.endswith("-USD"))
            d = add_indicators(bars, args.window)
            first = 50
            if args.start:
                d = d[d.index >= pd.Timestamp(args.start, tz=d.index.tz)]
                first = 0
            trades, curve = backtest(d, sym, tf, args.equity, args.risk, args.fee_bps, not args.long_only,
                                    not args.no_compound)
            rows.append({"symbol": sym, "tf": tf, **summarize(trades, curve, args.equity, d, first)})
            all_trades += trades

    summary = pd.DataFrame(rows)
    tag = f"_{args.tag}" if args.tag else ""
    summary.to_csv(OUT_DIR / f"summary{tag}.csv", index=False)
    pd.DataFrame(all_trades).to_csv(OUT_DIR / f"trades{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nNNFX 5-indicator backtest | ${args.equity:,.0f} per symbol | risk {args.risk:.0%} | "
          f"cross window {args.window} | fees {args.fee_bps} bps/side | "
          f"{'long only' if args.long_only else 'long+short'} | "
          f"{'fixed size' if args.no_compound else 'compounding'}\n")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / f'summary{tag}.csv'}  and  {OUT_DIR / f'trades{tag}.csv'}")


if __name__ == "__main__":
    main()
