from datetime import date

import pytest

from src.backtest.engine import Backtester
from src.core.settings import Settings
from src.core.store import LockHeld, Store
from src.data.models import parse_occ, to_schwab_symbol
from src.data.schwab import SchwabClient, SchwabTokenStore
from tests.helpers import TODAY, bear_setup_bars


def test_backtest_trades_the_bear_setup_and_reports_stats():
    bars = {"SPY": bear_setup_bars(follow_through=12)}
    result = Backtester(Settings(universe=["SPY"]), 10_000, bars).run(TODAY, TODAY)
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
    s = Settings(universe=["SPY"], no_trade_open_minutes=120)  # signal at 11:00 is before 11:30
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
