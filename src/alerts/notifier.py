"""Email alerts via Resend, de-duplicated through the ledger so a 1-minute loop never spams."""
from __future__ import annotations

import html
import logging

import requests

from ..core.settings import env

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, api_key: str | None, to: str | None, sender: str | None = None, ledger=None):
        self.api_key = api_key
        self.to = to
        self.sender = sender or "ICT Trading Bot <onboarding@resend.dev>"
        self.ledger = ledger
        self.last_error: str | None = None

    @classmethod
    def from_env(cls, ledger=None) -> "Notifier":
        return cls(env("RESEND_API_KEY"), env("ALERT_EMAIL_TO"), env("ALERT_EMAIL_FROM"), ledger)

    def send(self, subject: str, body_html: str, dedupe_key: str | None = None) -> bool:
        if dedupe_key and self.ledger is not None and not self.ledger.alert_once(dedupe_key):
            return False
        if not (self.api_key and self.to):
            self.last_error = "RESEND_API_KEY or ALERT_EMAIL_TO not set"
            log.warning("Email not sent (%s): %s", self.last_error, subject)
            return False
        try:
            resp = requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"from": self.sender, "to": [self.to], "subject": subject, "html": wrap(subject, body_html)},
                timeout=10,
            )
            if resp.status_code in (200, 201):
                return True
            self.last_error = f"Resend {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as exc:
            self.last_error = str(exc)
        log.error("Email failed: %s", self.last_error)
        if self.ledger is not None:
            self.ledger.log("ERROR", f"Email failed ({subject}): {self.last_error}")
        return False


def wrap(title: str, body: str) -> str:
    return (
        '<div style="font-family:Arial,sans-serif;max-width:760px;margin:auto;color:#1a202c">'
        f'<h2 style="border-bottom:2px solid #2c5282;padding-bottom:6px">{html.escape(title)}</h2>{body}'
        '<p style="font-size:12px;color:#718096;margin-top:24px">ICT Trading Bot · automated alert</p></div>'
    )


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "<p><i>None</i></p>"
    columns = columns or list(rows[0].keys())
    head = "".join(f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #cbd5e0">{html.escape(c)}</th>'
                   for c in columns)
    body = "".join(
        "<tr>" + "".join(f'<td style="padding:4px 8px;border-bottom:1px solid #edf2f7">{html.escape(_fmt(r.get(c)))}</td>'
                         for c in columns) + "</tr>"
        for r in rows
    )
    return f'<table style="border-collapse:collapse;font-size:13px">{"<tr>" + head + "</tr>"}{body}</table>'


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return "" if v is None else str(v)
