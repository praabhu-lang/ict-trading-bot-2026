from datetime import date

import pytest

from src.backtest.engine import Backtester
from src.core.settings import Settings
from src.core.store import LockHeld, Store
from src.data.models import parse_occ, to_schwab_symbol
from src.data.schwab import SchwabClient, SchwabTokenStore
from tests.helpers import TODAY, bear_setup_bars


# The 6-session fixture has no 20-day trend and SPY trades above its VWAP: only VRZ + volume (50 pts) is known.
LOOSE = dict(universe=["SPY"], require_spy_align=False, require_trend_align=False, min_convergence=50, option_min_score=50,
             option_min_dte=0, option_max_dte=0)  # 0DTE: the $100 fixture's 2-week premium is too big to size


def test_backtest_trades_the_bear_setup_and_reports_stats():
    bars = {"SPY": bear_setup_bars(follow_through=12)}
    result = Backtester(Settings(**LOOSE), 10_000, bars).run(TODAY, TODAY)
    trades = result["trades"]
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t.direction == "bear" and t.asset_class == "option" and t.pricing == "model"
    assert t.reason in ("TARGET", "TRAILING_STOP", "EOD_FLATTEN", "MOMENTUM_FADE")
    assert t.pnl > 0                                       # price followed through in the put's favour
    stats = result["stats"]
    assert stats["trades"] == 1 and "max_drawdown_pct" in stats
    assert result["equity"].iloc[-1]["equity"] == pytest.approx(10_000 + t.pnl)


def test_backtest_respects_entry_window():
    bars = {"SPY": bear_setup_bars(follow_through=12)}
    s = Settings(**LOOSE, no_trade_open_minutes=120)  # signal at 11:00 is before 11:30
    assert Backtester(s, 10_000, bars).run(TODAY, TODAY)["stats"]["trades"] == 0


def test_local_store_lock_and_update(tmp_path):
    store = Store(str(tmp_path))
    with store.lock("locks/engine.lock", ttl_seconds=60):
        with pytest.raises(LockHeld):
            with store.lock("locks/engine.lock", ttl_seconds=60):
                pass
    with store.lock("locks/engine.lock", ttl_seconds=60):
        pass
    store.update_json("control.json", lambda c: {**c, "paused": True})
    store.update_json("control.json", lambda c: {**c, "broker": "alpaca_paper"})
    assert store.read_json("control.json") == {"paused": True, "broker": "alpaca_paper"}


def test_occ_symbol_formats():
    assert parse_occ("SPY   261008C00580000") == ("SPY", date(2026, 10, 8), "C", 580.0)
    assert to_schwab_symbol("SPY261008P00580500") == "SPY   261008P00580500"


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text, self.headers = status, body, str(body), {}

    def json(self):
        return self._body


class _Session:
    def __init__(self, token_status=200):
        self.token_status = token_status

    def post(self, url, **kw):
        if self.token_status != 200:
            return _Resp(self.token_status, {"error": "invalid_grant"})
        return _Resp(200, {"access_token": "abc", "expires_in": 1800})

    def request(self, method, url, **kw):
        return _Resp(200, {"underlyingPrice": 580.0, "callExpDateMap": {"2026-10-08:0": {"580.0": [{
            "symbol": "SPY   261008C00580000", "bid": 1.0, "ask": 1.1, "delta": 0.5, "gamma": 0.1,
            "openInterest": 1000, "totalVolume": 50, "volatility": 15.0}]}}, "putExpDateMap": {}})


def _client(monkeypatch, session):
    for k, v in {"SCHWAB_CLIENT_ID": "id", "SCHWAB_CLIENT_SECRET": "sec", "SCHWAB_REFRESH_TOKEN": "rt"}.items():
        monkeypatch.setenv(k, v)
    return SchwabClient(SchwabTokenStore(None), session=session)


def test_schwab_chain_parsing(monkeypatch):
    chain = _client(monkeypatch, _Session()).option_chain("SPY", date(2026, 10, 8), date(2026, 10, 8))
    o = chain.options[0]
    assert chain.spot == 580.0 and o.symbol == "SPY261008C00580000" and o.iv == pytest.approx(0.15)


def test_schwab_expired_token_raises_market_data_unavailable(monkeypatch):
    from src.data.models import MarketDataUnavailable

    with pytest.raises(MarketDataUnavailable):
        _client(monkeypatch, _Session(token_status=400)).option_chain("SPY", date(2026, 10, 8), date(2026, 10, 8))


def test_lock_waits_for_short_holder(tmp_path):
    import threading
    import time as _t

    store = Store(str(tmp_path))
    held = threading.Event()

    def holder():
        with store.lock("locks/engine.lock", ttl_seconds=60):
            held.set()
            _t.sleep(0.5)

    th = threading.Thread(target=holder)
    th.start()
    held.wait()
    with store.lock("locks/engine.lock", ttl_seconds=60, wait_seconds=5, poll_seconds=0.1):
        pass
    th.join()


def test_schwab_relogin_is_pending_until_authenticator_confirms(tmp_path, monkeypatch):
    import src.data.schwab as sch

    monkeypatch.setenv("SCHWAB_CLIENT_ID", "id")
    monkeypatch.setenv("SCHWAB_CLIENT_SECRET", "sec")
    monkeypatch.setenv("SCHWAB_REFRESH_TOKEN", "old-expired")
    monkeypatch.setattr(sch.requests, "post", lambda *a, **k: _Resp(200, {"refresh_token": "fresh", "access_token": "a"}))
    store = Store(str(tmp_path))
    tokens = SchwabTokenStore(store)
    tokens.exchange_to_pending("https://dash/?code=C0DE%40&session=x")
    assert tokens.has_pending()
    assert tokens.load()["refresh_token"] == "old-expired"      # not used until activated
    assert tokens.activate_pending()
    assert tokens.load()["refresh_token"] == "fresh" and not tokens.has_pending()
    assert 6.9 < tokens.days_left() <= 7


def test_backtest_expiry_follows_dte_window():
    bt = Backtester(Settings(), 10_000, {"SPY": bear_setup_bars()})
    assert bt._expiry_for("SPY", date(2026, 10, 7)) == date(2026, 10, 21)    # daily expiries: exactly 14 days
    assert bt._expiry_for("AAPL", date(2026, 10, 7)) == date(2026, 10, 23)   # weeklies: first Friday >= 14 days
    assert bt._expiry_for("AAPL", date(2026, 3, 19)) == date(2026, 4, 2)     # Good Friday Apr 3 -> Thursday
    assert bt._expiry_for("AAPL", date(2026, 3, 20)) == date(2026, 4, 10)    # Thursday Apr 2 is only 13 days out


def test_pick_strike_targets_45_50_delta():
    from src.backtest.engine import pick_strike
    from src.backtest.pricing import bs_delta
    k = pick_strike("SPY", 700.0, 14 / 252, 0.18, "C")
    assert 0.45 <= bs_delta(700.0, k, 14 / 252, 0.18, "C") <= 0.50
