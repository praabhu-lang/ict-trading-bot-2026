"""Idle cutoff for the Streamlit dashboard.

Cloud Run bills the dashboard for as long as a browser tab holds its WebSocket open, and the
auto-refreshing fragments keep that socket busy forever. A forgotten tab cost ~$55 in October 2026.

After DASHBOARD_IDLE_MINUTES (default 20) with no mouse, key, touch or scroll activity, this closes the
socket, blocks Streamlit's automatic reconnects and health pings, and shows a "Paused" overlay.
"Resume" lets Streamlit reconnect to the same server session (kept for server.disconnectedSessionTTL,
see .streamlit/config.toml), so no new Authenticator login is needed if the instance is still up.
If the instance was recycled in the meantime the page reloads and asks for the login again.
"""
from __future__ import annotations

import json
import os

import streamlit.components.v1 as components

DEFAULT_IDLE_MINUTES = 20.0

_JS = r"""
(function () {
  var W = window;
  if (W.__idleGuard) { return; }
  var IDLE_MS = __IDLE_MS__;
  var DEAD_WS = 'wss://127.0.0.1:9/';
  var g = W.__idleGuard = {last: Date.now(), paused: false, sockets: new Set(), overlay: null};

  ['mousemove', 'mousedown', 'keydown', 'touchstart', 'wheel', 'scroll'].forEach(function (ev) {
    W.document.addEventListener(ev, function () { if (!g.paused) { g.last = Date.now(); } },
                                {capture: true, passive: true});
  });

  // Capture the live socket: Streamlit's auto-rerun timers send on it every 15-60 s.
  var RealWS = W.WebSocket;
  var realSend = RealWS.prototype.send;
  RealWS.prototype.send = function (d) { g.sockets.add(this); return realSend.call(this, d); };
  // While paused, reconnect attempts go to a dead local address instead of waking Cloud Run.
  W.WebSocket = new Proxy(RealWS, {
    construct: function (T, args) {
      args = Array.prototype.slice.call(args);
      if (g.paused) { args[0] = DEAD_WS; }
      var ws = Reflect.construct(T, args);
      if (!g.paused) { g.sockets.add(ws); }
      return ws;
    }
  });
  // Health / host-config pings that precede a reconnect are blocked too while paused.
  var realFetch = W.fetch.bind(W);
  W.fetch = function (input, init) {
    var u = typeof input === 'string' ? input : (input && input.url) || String(input);
    if (g.paused && u.indexOf('_stcore') !== -1) { return Promise.reject(new TypeError('dashboard paused')); }
    return realFetch(input, init);
  };
  var realOpen = W.XMLHttpRequest.prototype.open;
  W.XMLHttpRequest.prototype.open = function (m, u) {
    var a = Array.prototype.slice.call(arguments);
    if (g.paused && String(u).indexOf('_stcore') !== -1) { a[1] = 'https://127.0.0.1:9/'; }
    return realOpen.apply(this, a);
  };

  function overlay(html) {
    if (!g.overlay) {
      g.overlay = W.document.createElement('div');
      g.overlay.id = 'idle-guard-overlay';
      g.overlay.style.cssText = 'position:fixed;inset:0;z-index:2147483647;display:flex;align-items:center;' +
        'justify-content:center;background:rgba(14,17,23,.82);font-family:sans-serif;';
      W.document.body.appendChild(g.overlay);
      // Streamlit's own "connection error" modal marks the rest of the page inert; keep the overlay usable.
      var keepLive = function () {
        if (g.overlay.hasAttribute('inert')) { g.overlay.removeAttribute('inert'); }
        if (g.overlay.getAttribute('aria-hidden')) { g.overlay.removeAttribute('aria-hidden'); }
      };
      new MutationObserver(keepLive).observe(g.overlay, {attributes: true, attributeFilter: ['inert', 'aria-hidden']});
    }
    g.overlay.innerHTML = '<div style="background:#fff;color:#1f2328;border-radius:12px;padding:28px 32px;' +
      'max-width:380px;text-align:center;box-shadow:0 8px 30px rgba(0,0,0,.35)">' + html + '</div>';
    g.overlay.style.display = 'flex';
  }

  function pause() {
    g.paused = true;
    g.sockets.forEach(function (s) { try { s.close(); } catch (e) {} });
    g.sockets.clear();
    overlay('<div style="font-size:20px;font-weight:600;margin-bottom:8px">Dashboard paused</div>' +
      '<div style="font-size:14px;color:#57606a;margin-bottom:20px">No activity for ' + Math.round(IDLE_MS / 60000) +
      ' minutes, so the live connection was closed to stop Cloud Run charges.</div>' +
      '<button id="idle-guard-resume" style="background:#2a78d6;color:#fff;border:0;border-radius:8px;' +
      'padding:10px 22px;font-size:15px;cursor:pointer">Resume</button>');
  }

  // Streamlit's "connection error" modal swallows clicks at the document level, so listen on window
  // in the capture phase (it runs first).
  W.addEventListener('click', function (e) {
    if (g.paused && e.target && e.target.id === 'idle-guard-resume') {
      e.preventDefault(); e.stopImmediatePropagation(); resume();
    }
  }, true);

  function resume() {
    g.paused = false;
    g.last = Date.now();
    overlay('<div style="font-size:16px">Reconnecting…</div>');
    var start = Date.now();
    var t = W.setInterval(function () {
      var open = false;
      g.sockets.forEach(function (s) { if (s.readyState === 1) { open = true; } });
      if (open) { W.clearInterval(t); g.overlay.style.display = 'none'; return; }
      if (Date.now() - start > 10000) { W.clearInterval(t); W.location.reload(); }
    }, 250);
  }

  W.setInterval(function () {
    if (!g.paused && Date.now() - g.last >= IDLE_MS) { pause(); }
  }, 5000);
})();
"""

# components.html renders in a same-origin iframe. The guard must live in the main page's realm (the
# iframe may be re-rendered, which would kill its timers and patched functions), so the iframe only
# appends a <script> to the parent document.
_LOADER = """<script>
(function () {
  var P = window.parent;
  if (!P || P.__idleGuard) { return; }
  var s = P.document.createElement('script');
  s.textContent = %s;
  P.document.head.appendChild(s);
})();
</script>"""


def idle_minutes() -> float:
    try:
        return float(os.getenv("DASHBOARD_IDLE_MINUTES", DEFAULT_IDLE_MINUTES))
    except ValueError:
        return DEFAULT_IDLE_MINUTES


def script(minutes: float) -> str:
    return _LOADER % json.dumps(_JS.replace("__IDLE_MS__", str(int(minutes * 60000))))


def install() -> None:
    """Inject the idle cutoff once per page; a value <= 0 disables it."""
    minutes = idle_minutes()
    if minutes <= 0:
        return
    components.html(script(minutes), height=0)
