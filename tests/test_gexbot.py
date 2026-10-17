import io
import zipfile
from datetime import datetime, time

from src.agents.market_analyst import MarketAnalystAgent
from src.core.clock import ET
from src.core.settings import Settings
from src.core.store import Store
from src.data.gexbot import GexbotClient, archive_eod, eod_session_date, parse_gex
from src.data.models import MarketDataUnavailable
from tests.helpers import TODAY, FakeMarket, FakeNews, bear_setup_bars, make_chain

SAMPLE = {"timestamp": 1791466202, "ticker": "SPY", "spot": 774.93, "zero_gamma": 776.5,
          "major_pos_vol": 778, "major_pos_oi": 780, "major_neg_vol": 775, "major_neg_oi": 770,
          "strikes": [[770, 0.1, -2.5, [0, 0, 0, 0, 0]], [780, 0.2, 3.0, [0, 0, 0, 0, 0]]],
          "sum_gex_vol": 2.3, "sum_gex_oi": -1691.99}


def _zip(name="SPY/classic/gex_full/2026-10-08_SPY_classic_gex_full.json.gz") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, b"x")
    return buf.getvalue()


class FakeGexbot(GexbotClient):
    def __init__(self, covered=("SPY",), eod=None, fail=()):
        super().__init__(api_key="k")
        self._tickers = set(covered)
        self.eod = eod or _zip()
        self.fail = set(fail)
        self.calls = 0

    def gex(self, ticker):
        self.calls += 1
        if ticker in self.fail:
            raise MarketDataUnavailable("down")
        return parse_gex(SAMPLE)

    def eod_zip(self, ticker):
        if ticker in self.fail:
            raise MarketDataUnavailable("down")
        return self.eod


def test_parse_gex_maps_levels():
    g = parse_gex(SAMPLE)
    assert g.gamma_flip == 776.5 and g.call_wall == 780 and g.put_wall == 770
    assert g.regime == "NEGATIVE" and g.by_strike[780.0] == 3.0


def test_archive_saves_once_and_skips_uncovered(tmp_path):
    store = Store(str(tmp_path))
    client = FakeGexbot(covered=("SPY",))
    assert eod_session_date(client.eod) == "2026-10-08"
    first = archive_eod(client, store, ["SPY", "JPM"])
    assert first["SPY"].startswith("saved gexbot/eod/SPY/2026-10-08.zip") and "not covered" in first["JPM"]
    assert store.exists("gexbot/eod/SPY/2026-10-08.zip")
    assert archive_eod(client, store, ["SPY"])["SPY"].startswith("already saved")


def test_archive_failure_is_reported_per_ticker(tmp_path):
    res = archive_eod(FakeGexbot(covered=("SPY", "QQQ"), fail=("QQQ",)), Store(str(tmp_path)), ["SPY", "QQQ"])
    assert res["SPY"].startswith("saved") and res["QQQ"].startswith("FAILED")


class GexMarket(FakeMarket):
    def __init__(self, *a, gexbot=None, **k):
        super().__init__(*a, **k)
        self.gexbot = gexbot

    def gexbot_levels(self, symbol, now):
        if self.gexbot is None or not self.gexbot.supports(symbol):
            return None
        try:
            return self.gexbot.gex(symbol)
        except MarketDataUnavailable:
            return None


def test_analyst_prefers_gexbot_and_trades_stocks_without_schwab_chain():
    now = datetime.combine(TODAY, time(11, 0, 30), tzinfo=ET)
    market = GexMarket({"SPY": bear_setup_bars()}, {}, gexbot=FakeGexbot())   # no Schwab chain at all
    s = Settings(require_spy_align=False, min_convergence=65)
    analysis, chain, _ = MarketAnalystAgent(market, FakeNews()).analyze("SPY", now, s, require_chain=True)
    assert chain is None and analysis.snapshot["gamma_flip"] == 776.5            # GEX from gexbot
    assert analysis.signal is not None                                             # stocks-only: still scans


def test_options_on_still_requires_schwab_chain():
    now = datetime.combine(TODAY, time(11, 0, 30), tzinfo=ET)
    market = GexMarket({"SPY": bear_setup_bars()}, {}, gexbot=FakeGexbot())
    s = Settings(option_min_score=90)
    try:
        MarketAnalystAgent(market, FakeNews()).analyze("SPY", now, s, require_chain=True)
    except MarketDataUnavailable:
        return
    raise AssertionError("expected the missing chain to block a scan that may trade options")


def test_schwab_gex_used_when_gexbot_fails():
    now = datetime.combine(TODAY, time(11, 0, 30), tzinfo=ET)
    market = GexMarket({"SPY": bear_setup_bars()}, {"SPY": make_chain(101.6)}, gexbot=FakeGexbot(fail=("SPY",)))
    analysis, chain, _ = MarketAnalystAgent(market, FakeNews()).analyze("SPY", now, Settings(), require_chain=True)
    assert chain is not None and analysis.snapshot["call_wall"] == 102.0          # computed from the chain


def test_xsp_uses_spx_levels_scaled_by_a_tenth():
    from src.data.market import MarketData

    gb = FakeGexbot(covered=("SPX",))
    market = MarketData(None, None, gexbot=gb)
    g = market.gexbot_levels("XSP", datetime.combine(TODAY, time(11, 0), tzinfo=ET))
    spx = parse_gex(SAMPLE)
    assert g.call_wall == round(spx.call_wall / 10, 2) and g.put_wall == round(spx.put_wall / 10, 2)
    assert g.gamma_flip == round(spx.gamma_flip / 10, 2) and g.regime == spx.regime
    assert market.gexbot_levels("SPY", datetime.combine(TODAY, time(11, 0), tzinfo=ET)) is None  # not covered


def test_xsp_bars_take_spy_volume():
    from src.data.market import MarketData

    class FakeSchwab:
        def price_history(self, symbol, start, end):
            df = bear_setup_bars()
            if symbol == "$XSP":  # index bars: prices but no volume
                df = df.assign(volume=0.0)
            return df

    now = datetime.combine(TODAY, time(15, 0), tzinfo=ET)
    market = MarketData(FakeSchwab(), None)
    xsp, spy = market.bars("XSP", now), market.bars("SPY", now)
    assert xsp["volume"].sum() > 0 and (xsp["volume"] == spy["volume"].reindex(xsp.index)).all()
