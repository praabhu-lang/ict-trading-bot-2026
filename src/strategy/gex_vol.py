"""GEX volatility multiplier: how big today's range should be, from the pre-open net GEX.

Finding (Jan 2-Oct 7 2026): net GEX does not call direction, but it does forecast the day's range, also after
yesterday's range is known (SPY day range 1.17% in the most-negative GEX quartile vs 0.81% in the most-positive).

Model, per ticker-day, everything known before the open:
  base       = mean high-low range (% of prior close) of the previous 20 sessions
  multiplier = pos_mult if pre-open net GEX > 0 else neg_mult   (geometric mean of range/base, Jan 2-Jun 3)
  expected range = base x multiplier

Fitted Jan 2-Jun 3 2026 (SPY+QQQ, 210 days): positive GEX x0.89, negative x1.02.
Out of sample Jun 4-Nov 3 (212 days): actual x0.86 / x0.99; gap 95% CI excludes zero. Forecast error vs the
plain 20-day average improves only ~2%: a real but small effect.
Rejected: a continuous model on the trailing-60-day GEX percentile fitted the train period (+9%) and did
nothing out of sample (+0.3%); the GEX-range slope halved after Jun 4, only the sign held.

Uses: stop/target width x multiplier; risk per trade x clip(1/multiplier, lo, hi).

    python -m src.strategy.gex_vol            # build history, fit on Jan 2-Jun 3, evaluate on Jun 4-today
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..core.settings import env

log = logging.getLogger(__name__)

MODEL_PATH = Path("research/gex_vol_model.json")
HIST_PATH = Path("data/gex_vol_history.csv")
TRAIN_END = date(2026, 6, 3)
BASE_N = 20


@dataclass
class GexVolModel:
    pos_mult: float
    neg_mult: float
    train_days: int
    fitted_on: str

    def multiplier(self, net_gex: float) -> float:
        return self.pos_mult if net_gex > 0 else self.neg_mult

    def predict_range(self, base: float, net_gex: float) -> float:
        return base * self.multiplier(net_gex)

    def size_factor(self, net_gex: float, lo: float = 0.5, hi: float = 1.5) -> float:
        return float(np.clip(1.0 / self.multiplier(net_gex), lo, hi))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "GexVolModel":
        return cls(**json.loads(path.read_text()))


# ------------------------------------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: one row per (ticker, date) with high, low, close, net_gex. Adds range and base (no look-ahead)."""
    out = []
    for _, g in df.sort_values("date").groupby("ticker"):
        g = g.copy()
        prev_close = g.close.shift(1)
        g["range"] = (g.high - g.low) / prev_close * 100
        g["base"] = g["range"].shift(1).rolling(BASE_N, min_periods=BASE_N).mean()
        out.append(g)
    return pd.concat(out)


def fit(df: pd.DataFrame) -> GexVolModel:
    d = df.dropna(subset=["range", "base", "net_gex"])
    d = d[d.range > 0]
    rel = np.log(d.range / d.base)
    pos = d.net_gex > 0
    return GexVolModel(float(np.exp(rel[pos].mean())), float(np.exp(rel[~pos].mean())), len(d),
                       f"{d.date.min()}..{d.date.max()}")


# ------------------------------------------------------------------------------------------- history
def _daily_bars(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    h = {"APCA-API-KEY-ID": env("APCA_API_KEY_ID"), "APCA-API-SECRET-KEY": env("APCA_API_SECRET_KEY")}
    p = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start.isoformat(), "end": end.isoformat(),
         "adjustment": "raw", "feed": "sip", "limit": 10000}
    rows = []
    while True:
        for attempt in range(8):
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=h, params=p, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        js = r.json()
        for s, bars in (js.get("bars") or {}).items():
            rows += [{"ticker": s, "date": pd.Timestamp(b["t"]).tz_convert("America/New_York").date(),
                      "high": b["h"], "low": b["l"], "close": b["c"]} for b in bars]
        if not js.get("next_page_token"):
            return pd.DataFrame(rows)
        p["page_token"] = js["next_page_token"]


def build_history(tickers: list[str], start: date, end: date,
                  windows: list[tuple[date, date]]) -> pd.DataFrame:
    """Daily bars + pre-open proxy net GEX (src/backtest/gex_history.py, the same thing live can compute)."""
    from ..backtest.gex_history import VolumeGexHistory
    bars = _daily_bars(tickers, start - timedelta(days=45), end)
    rows = []
    for t in tickers:
        b = bars[bars.ticker == t].set_index("date").sort_index()
        for w0, w1 in windows:
            g = VolumeGexHistory()
            g.prepare(t, w0, w1, b.close)
            for d in b.index:
                if w0 <= d <= w1 and start <= d <= end:
                    r = g(t, d)
                    rows.append({"ticker": t, "date": d, "net_gex": r.net_gex if r else np.nan,
                                 "high": b.at[d, "high"], "low": b.at[d, "low"], "close": b.at[d, "close"]})
    hist = pd.DataFrame(rows)
    # include the pre-window bars so `range`/`base` have their lookback
    pre = bars[bars.date < start].assign(net_gex=np.nan)
    return pd.concat([pre, hist]).sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")


# ------------------------------------------------------------------------------------------- evaluation
def evaluate(df: pd.DataFrame, m: GexVolModel) -> None:
    d = df.dropna(subset=["range", "base", "net_gex"]).copy()
    d = d[d.range > 0]
    d["pred"] = [m.predict_range(b, g) for b, g in zip(d.base, d.net_gex)]
    d["mult"] = d.pred / d.base
    for label, part in (("train Jan-Jun3", d[d.date <= TRAIN_END]), ("TEST Jun4-now", d[d.date > TRAIN_END])):
        if part.empty:
            continue
        e_base = np.mean(np.abs(np.log(part.range / part.base)))
        e_mod = np.mean(np.abs(np.log(part.range / part.pred)))
        print(f"{label:16s} days={len(part):4d}  abs log error: 20d-average {e_base:.3f}  GEX model {e_mod:.3f}  "
              f"({100 * (1 - e_mod / e_base):+.1f}%)")
        part = part.assign(bucket=np.where(part.net_gex > 0, "positive GEX", "negative GEX"))
        print(part.groupby("bucket").agg(days=("range", "size"), mult=("mult", "mean"),
                                                         predicted=("pred", "mean"), actual=("range", "mean"))
              .round(2).to_string(), "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SPY,QQQ")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    tickers = a.tickers.split(",")
    if HIST_PATH.exists() and not a.rebuild:
        hist = pd.read_csv(HIST_PATH, parse_dates=["date"])
        hist["date"] = hist.date.dt.date
    else:
        hist = build_history(tickers, date(2026, 1, 2), date(2026, 11, 3),
                             [(date(2026, 1, 2), date(2026, 10, 7)), (date(2026, 10, 8), date(2026, 11, 4))])
        HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(HIST_PATH, index=False)
    df = add_features(hist)
    m = fit(df[df.date <= TRAIN_END])
    print(f"fit on {m.fitted_on} ({m.train_days} ticker-days): positive GEX x{m.pos_mult:.3f}, "
          f"negative GEX x{m.neg_mult:.3f} of the 20-day average range\n")
    evaluate(df, m)
    m.save()
    df.to_csv(HIST_PATH.with_name("gex_vol_features.csv"), index=False)


if __name__ == "__main__":
    main()
