import pytest

from src.brokers import BrokerUnavailable, make_broker
from src.brokers.ibkr import IBKRBroker
from src.brokers.platforms import CredentialStore
from src.core.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_LIVE_API_KEY_ID", "APCA_LIVE_API_SECRET_KEY",
              "IBKR_GATEWAY_URL", "IBKR_ACCOUNT_ID", "ALLOW_LIVE_TRADING"):
        monkeypatch.delenv(k, raising=False)
    return Store(str(tmp_path))


def test_credentials_saved_blank_keeps_value_and_env_fallback(store, monkeypatch):
    creds = CredentialStore(store)
    assert not creds.is_configured("alpaca_paper")
    creds.save("alpaca_paper", {"api_key": "K1", "api_secret": "S1"})
    creds.save("alpaca_paper", {"api_key": "K2", "api_secret": ""})       # blank secret = keep
    assert creds.get("alpaca_paper") == {"api_key": "K2", "api_secret": "S1"}
    monkeypatch.setenv("APCA_LIVE_API_KEY_ID", "envkey")
    monkeypatch.setenv("APCA_LIVE_API_SECRET_KEY", "envsecret")
    assert CredentialStore(store).is_configured("alpaca_live")


def test_registry_guards(store):
    with pytest.raises(BrokerUnavailable, match="no official API"):
        make_broker("robinhood", store)
    with pytest.raises(BrokerUnavailable, match="not configured"):
        make_broker("alpaca_paper", store)
    with pytest.raises(BrokerUnavailable, match="Unknown"):
        make_broker("schwab", store)                                       # Schwab is data only
    CredentialStore(store).save("ibkr", {"gateway_url": "https://gw:5000/v1/api", "account_id": "U123", "paper": False})
    with pytest.raises(BrokerUnavailable, match="ALLOW_LIVE_TRADING"):
        make_broker("ibkr", store)


def test_ibkr_paper_account_allowed_without_live_flag(store):
    CredentialStore(store).save("ibkr", {"gateway_url": "https://gw:5000/v1/api", "account_id": "DU1", "paper": True})
    b = make_broker("ibkr", store)
    assert isinstance(b, IBKRBroker) and b.is_paper


class _R:
    def __init__(self, body, status=200):
        self.status_code, self._b, self.text, self.content = status, body, str(body), b"x"

    def json(self):
        return self._b


class FakeGateway:
    """Minimal Client Portal API: auth, contract search, order with one confirmation prompt."""

    def __init__(self):
        self.verify = False
        self.calls = []

    def request(self, method, url, **kw):
        path = url.split("/v1/api")[1]
        self.calls.append((method, path, kw.get("json")))
        if path == "/iserver/auth/status":
            return _R({"authenticated": True})
        if path == "/iserver/accounts":
            return _R({"accounts": ["DU1"]})
        if path == "/iserver/secdef/search":
            return _R([{"conid": 756733, "symbol": "SPY"}])
        if path == "/iserver/secdef/info":
            return _R([{"conid": 111, "maturityDate": "20261008"}, {"conid": 222, "maturityDate": "20261009"}])
        if path == "/iserver/account/DU1/orders":
            return _R([{"id": "reply-1", "message": ["Are you sure?"]}])
        if path == "/iserver/reply/reply-1":
            return _R([{"order_id": "987", "order_status": "Submitted"}])
        if path == "/iserver/account/order/status/987":
            return _R({"order_status": "Filled", "cum_fill": "3", "average_price": "1.23"})
        return _R({}, 404)


def test_ibkr_option_order_confirms_prompt_and_reads_fill():
    gw = FakeGateway()
    b = IBKRBroker("https://gw:5000/v1/api", "DU1", paper=True, session=gw)
    oid = b.submit_limit("SPY261008C00580000", 3, "buy", 1.234)
    assert oid == "987"
    order = next(c[2] for c in gw.calls if c[1] == "/iserver/account/DU1/orders")["orders"][0]
    assert order["conid"] == 111 and order["side"] == "BUY" and order["orderType"] == "LMT" and order["price"] == 1.23
    st = b.get_order("987")
    assert st.status == "filled" and st.filled_qty == 3 and st.filled_avg_price == 1.23
