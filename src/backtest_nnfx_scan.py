#!/usr/bin/env python3
"""
backtest_nnfx_scan.py - NNFX 5-indicator rules on stocks the scanner picks itself (no hard-coded tickers)
=======================================================================================================
Universe:  every US equity Alpaca has ever listed (active AND inactive, so stocks delisted since are
           included), on NYSE / NASDAQ / AMEX / ARCA / BATS, minus ETFs, funds, warrants, units, rights
           and preferreds (by asset name).
Scan:      on the first session of each month, using ONLY daily bars up to the previous close:
             1. price >= $10 and the 100 highest 20-day median dollar volumes ("liquid 100"),
             2. rank them by --pick and keep the top --picks:
                  trend      Kaufman efficiency ratio of the last 63 closes (|net move| / path length;
                             direction-neutral, 1.0 = straight line) - clean trends suit EMA/MACD/RSI rules
                  momentum   |63-day return|
                  volume     20-day / 120-day dollar volume (stocks suddenly in play)
                  volatility 14-day ATR as % of price
New entries are only taken in the current month's picks; open positions are managed until they exit.
Trading:   the rules in backtest_nnfx.py (EMA20 baseline, MACD + RSI-50 crosses, RVOL >= 1.3, 1.5/3.0 ATR
           bracket, 3-loss daily breaker) on 1-hour bars anchored at 09:30 ET, built from Alpaca SIP
           30-minute bars. One shared account: 2% risk per trade, at most --picks open positions, each
           capped at equity / --picks notional.

Usage:
  python src/backtest_nnfx_scan.py                                 # 2022, trend pick, top 5, $10k
  python src/backtest_nnfx_scan.py --pick momentum --picks 8 --long-only
  python src/backtest_nnfx_scan.py --start 2025-01-01 --end 2025-12-31
Outputs: results/nnfx/scan_{summary,trades,picks}_<tag>.csv
"""

import argparse
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_nnfx import add_indicators

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "price_cache" / "scan"
OUT_DIR = REPO / "results" / "nnfx"
ET = "America/New_York"
BENCH = "SPY"                         # benchmark and session calendar only - never picked
EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
NOT_STOCK = re.compile(
    r"\b(ETF|ETN|ETP|FUND|TRUST|PROSHARES|DIREXION|ISHARES|SPDR|INVESCO|VANECK|VANGUARD|GLOBAL X|WISDOMTREE|"
    r"ULTRA|ULTRAPRO|LEVERAGED|INVERSE|2X|3X|-1X|BULL|BEAR|INDEX|WARRANTS?|RIGHTS?|UNITS?|PREFERRED|"
    r"DEPOSITARY SHARES REPRESENTING|NOTES?|DEBENTURES?|ACQUISITION)\b", re.I)


def load_env():
    path = REPO / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def data_client():
    from alpaca.data.historical.stock import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])


# --------------------------------------------------------------------------- universe
def all_stocks() -> list[str]:
    """Common stocks Alpaca has listed on the major exchanges, active or not (cached for 30 days)."""
    path = CACHE / "assets.pkl"
    if path.exists() and datetime.now().timestamp() - path.stat().st_mtime < 30 * 86400:
        return pd.read_pickle(path)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest
    load_env()
    tc = TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=True)
    syms = set()
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        for a in tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=status)):
            if (a.exchange.value in EXCHANGES and re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", a.symbol)
                    and not NOT_STOCK.search(a.name or "")):
                syms.add(a.symbol)
    out = sorted(syms)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


def daily_bars(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Split-adjusted daily bars for every symbol, fetched 500 at a time and cached per batch."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client, frames = None, []
    batches = [symbols[i:i + 500] for i in range(0, len(symbols), 500)]
    for n, batch in enumerate(batches):
        path = CACHE / f"daily_{start}_{end}_{n:03d}_{batch[0]}_{batch[-1]}.pkl"
        if path.exists():
            frames.append(pd.read_pickle(path))
            continue
        client = client or data_client()
        print(f"  daily bars batch {n + 1}/{len(batches)} ({batch[0]}..{batch[-1]})", flush=True)
        req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                               start=datetime.combine(start, time(0), tzinfo=timezone.utc),
                               end=datetime.combine(end, time(23, 59), tzinfo=timezone.utc),
                               feed=DataFeed.SIP, adjustment=Adjustment.SPLIT)
        df = client.get_stock_bars(req).df
        if df.empty:
            df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                              index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]))
        else:
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.to_pickle(path)
        frames.append(df)
    allbars = pd.concat(frames)
    out = {}
    for sym, g in allbars.groupby(level=0):
        g = g.droplevel(0)
        g.index = pd.to_datetime(g.index, utc=True).tz_convert(ET).date
        out[sym] = g
    return out


# --------------------------------------------------------------------------- scanner
def score_table(daily: dict[str, pd.DataFrame], asof: date, min_price: float, liquid_n: int) -> pd.DataFrame:
    """Scores for the liquid-N stocks using bars strictly before `asof`."""
    rows = []
    for sym, d in daily.items():
        if sym == BENCH:
            continue
        d = d[d.index < asof]
        if len(d) < 130 or d.index[-1] < asof - timedelta(days=7):   # too new, or delisted / acquired
            continue
        c, v = d["close"], d["volume"]
        if c.iloc[-1] < min_price:
            continue
        dv = c * v
        tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
        path = c.diff().abs().iloc[-63:].sum()
        rows.append({
            "symbol": sym,
            "dollar_vol_m": dv.iloc[-20:].median() / 1e6,
            "trend": abs(c.iloc[-1] - c.iloc[-64]) / path if path > 0 else 0.0,
            "momentum": abs(c.iloc[-1] / c.iloc[-64] - 1),
            "volume": dv.iloc[-20:].mean() / dv.iloc[-120:].mean(),
            "volatility": tr.iloc[-14:].mean() / c.iloc[-1],
            "ret_63d_pct": (c.iloc[-1] / c.iloc[-64] - 1) * 100,
        })
    t = pd.DataFrame(rows)
    return t.nlargest(liquid_n, "dollar_vol_m")


def monthly_picks(daily, sessions: list[date], pick: str, picks: int, min_price: float, liquid_n: int) -> pd.DataFrame:
    firsts = pd.Series(sessions).groupby(pd.Series([(d.year, d.month) for d in sessions])).first()
    rows = []
    for d in firsts:
        t = score_table(daily, d, min_price, liquid_n).nlargest(picks, pick)
        for rank, r in enumerate(t.itertuples(), 1):
            rows.append({"month": d.strftime("%Y-%m"), "from": d, "rank": rank, "symbol": r.symbol,
                         "score": round(getattr(r, pick), 3), "ret_63d_%": round(r.ret_63d_pct, 1),
                         "dollar_vol_m": round(r.dollar_vol_m)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- hourly bars
def hourly_bars(sym: str, start: date, end: date) -> pd.DataFrame:
    """Regular-session 1-hour bars anchored at 09:30 ET (09:30, 10:30, ... 15:30), split-adjusted, cached."""
    path = CACHE / f"{sym}_1h_{start}_{end}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                           start=datetime.combine(start, time(0), tzinfo=timezone.utc),
                           end=datetime.combine(end + timedelta(days=1), time(0), tzinfo=timezone.utc),
                           feed=DataFeed.SIP, adjustment=Adjustment.SPLIT)
    df = data_client().get_stock_bars(req).df
    if df.empty:
        out = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    else:
        df = df.xs(sym, level=0) if isinstance(df.index, pd.MultiIndex) else df
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        t = df.index.time
        df = df[(t >= time(9, 30)) & (t < time(16, 0))]
        df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        out = df.resample("60min", offset="30min").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        out = out[out.Volume > 0]
    out.to_pickle(path)
    return out


# --------------------------------------------------------------------------- portfolio simulation
def simulate(data: dict[str, pd.DataFrame], allowed: dict[str, set], equity0: float, risk: float, slots: int,
             fee_bps: float, allow_short: bool, start: pd.Timestamp) -> tuple[list[dict], pd.Series]:
    fee = fee_bps / 1e4
    cash = equity0                       # realized equity; positions are marked separately
    positions, pending, trades = {}, {}, []
    streak = {}                          # per-symbol (session, losses, halted) - the breaker is per symbol as in the spec
    timeline = sorted(set().union(*[set(d.index[d.index >= start]) for d in data.values()]))
    rows = {s: d.to_dict("index") for s, d in data.items()}
    last_close = {}
    curve = []

    def close(sym, ts, price, reason):
        nonlocal cash
        p = positions.pop(sym)
        pnl = (price - p["entry"]) * p["units"] * p["side"] - (p["entry"] + price) * p["units"] * fee
        cash += pnl
        trades.append({"symbol": sym, "side": "LONG" if p["side"] == 1 else "SHORT", "entry_time": p["time"],
                       "entry": p["entry"], "stop": p["sl"], "target": p["tp"], "units": p["units"],
                       "exit_time": ts, "exit": price, "reason": reason, "pnl": pnl,
                       "r_multiple": pnl / p["risk_usd"], "equity": cash})
        sess, losses, _ = streak[sym]
        losses = losses + 1 if pnl < 0 else 0
        streak[sym] = (sess, losses, losses >= 3)

    for ts in timeline:
        month = ts.strftime("%Y-%m")
        sess = ts.date()
        for sym, b in ((s, rows[s].get(ts)) for s in data):
            if b is None:
                continue
            if streak.get(sym, (None,))[0] != sess:
                streak[sym] = (sess, 0, False)
            # 1) fill yesterday-bar signal at this bar's open
            if sym in pending and sym not in positions and not streak[sym][2]:
                side, atr = pending[sym]
                entry = b["Open"]
                equity = cash + sum((last_close[s] - p["entry"]) * p["units"] * p["side"] for s, p in positions.items())
                units = min(equity * risk / (1.5 * atr), equity / slots / entry)
                if units > 0 and len(positions) < slots:
                    positions[sym] = {"side": side, "entry": entry, "units": units, "time": ts,
                                      "sl": entry - side * 1.5 * atr, "tp": entry + side * 3.0 * atr,
                                      "risk_usd": units * 1.5 * atr}
            pending.pop(sym, None)
            # 2) manage the open position (stop first if both touched; gaps fill at the open)
            p = positions.get(sym)
            if p is not None:
                o, h, l, s = b["Open"], b["High"], b["Low"], p["side"]
                if s == 1:
                    if o <= p["sl"]:   close(sym, ts, o, "SL gap")
                    elif o >= p["tp"]: close(sym, ts, o, "TP gap")
                    elif l <= p["sl"]: close(sym, ts, p["sl"], "SL")
                    elif h >= p["tp"]: close(sym, ts, p["tp"], "TP")
                else:
                    if o >= p["sl"]:   close(sym, ts, o, "SL gap")
                    elif o <= p["tp"]: close(sym, ts, o, "TP gap")
                    elif h >= p["sl"]: close(sym, ts, p["sl"], "SL")
                    elif l <= p["tp"]: close(sym, ts, p["tp"], "TP")
            last_close[sym] = b["Close"]
            # 3) signal on this bar's close -> next bar's open; only in this month's picks
            if (sym not in positions and not streak[sym][2] and sym in allowed.get(month, ())
                    and np.isfinite(b["atr"]) and b["atr"] > 0):
                if b["long_sig"]:
                    pending[sym] = (1, b["atr"])
                elif b["short_sig"] and allow_short:
                    pending[sym] = (-1, b["atr"])
        curve.append(cash + sum((last_close[s] - p["entry"]) * p["units"] * p["side"] for s, p in positions.items()))

    for sym in list(positions):
        close(sym, timeline[-1], last_close[sym], "end of data")
    if curve:
        curve[-1] = cash
    return trades, pd.Series(curve, index=timeline)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--pick", default="trend", choices=["trend", "momentum", "volume", "volatility"])
    ap.add_argument("--picks", type=int, default=5, help="stocks picked per month = max open positions")
    ap.add_argument("--liquid", type=int, default=100, help="size of the most-liquid pool the picks come from")
    ap.add_argument("--min-price", type=float, default=10.0)
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--risk", type=float, default=0.02)
    ap.add_argument("--window", type=int, default=1, help="bars within which MACD/RSI crosses may occur")
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = all_stocks()
    print(f"Universe: {len(universe)} listed + delisted common stocks")
    daily = daily_bars(universe + [BENCH], start - timedelta(days=270), end)
    sessions = sorted(daily[BENCH].index)
    sessions = [d for d in sessions if start <= d <= end]
    picks = monthly_picks(daily, sessions, args.pick, args.picks, args.min_price, args.liquid)
    allowed = picks.groupby("month")["symbol"].apply(set).to_dict()

    symbols = sorted(picks.symbol.unique())
    print(f"Scanner picked {len(symbols)} distinct stocks over {len(allowed)} months: {' '.join(symbols)}")
    warm = start - timedelta(days=45)
    data = {}
    for s in symbols:
        h = hourly_bars(s, warm, end)
        if len(h) > 60:
            data[s] = add_indicators(h, args.window)
    trades, curve = simulate(data, allowed, args.equity, args.risk, args.picks, args.fee_bps, not args.long_only,
                             pd.Timestamp(start, tz=ET))

    t = pd.DataFrame(trades)
    tag = args.tag or f"{start.year}_{args.pick}{args.picks}{'_long' if args.long_only else ''}"
    picks.to_csv(OUT_DIR / f"scan_picks_{tag}.csv", index=False)
    t.to_csv(OUT_DIR / f"scan_trades_{tag}.csv", index=False)

    spy = daily.get(BENCH)
    spy_ret = (spy.loc[[d for d in spy.index if start <= d <= end], "close"].iloc[[0, -1]].pipe(lambda s: s.iloc[1] / s.iloc[0] - 1) * 100) if spy is not None else np.nan
    n = len(t)
    wins, losses = (t.pnl[t.pnl > 0].sum(), -t.pnl[t.pnl < 0].sum()) if n else (0, 0)
    summary = {
        "period": f"{start}..{end}", "pick": args.pick, "picks": args.picks, "sides": "long" if args.long_only else "long+short",
        "trades": n, "long/short": f"{(t.side == 'LONG').sum() if n else 0}/{(t.side == 'SHORT').sum() if n else 0}",
        "win_rate_%": round((t.pnl > 0).mean() * 100, 1) if n else 0,
        "profit_factor": round(wins / losses, 2) if losses else np.nan,
        "avg_R": round(t.r_multiple.mean(), 2) if n else 0,
        "max_dd_%": round((curve / curve.cummax() - 1).min() * 100, 1),
        "return_%": round((curve.iloc[-1] / args.equity - 1) * 100, 1),
        "spy_%": round(spy_ret, 1),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / f"scan_summary_{tag}.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nMonthly picks (scored on data before the 1st session of the month):")
    print(picks.pivot(index="month", columns="rank", values="symbol").to_string())
    if n:
        print("\nP&L by stock:")
        print(t.groupby("symbol").agg(trades=("pnl", "size"), pnl=("pnl", "sum"), avg_R=("r_multiple", "mean"))
              .round(2).sort_values("pnl", ascending=False).to_string())
    print("\n" + "  ".join(f"{k}: {v}" for k, v in summary.items()))
    print(f"Saved results/nnfx/scan_*_{tag}.csv")


if __name__ == "__main__":
    main()
