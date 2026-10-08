"""State store: a local directory, mirrored to a private GCS bucket when GCS_BUCKET_NAME is set.

Ownership rules (prevents the old last-writer-wins corruption):
  ledger.db     - written only by engine jobs, while holding the engine lock
  control.json  - written only by the dashboard (generation-matched read-modify-write)
  secrets/*     - Schwab refresh token, written by the dashboard re-auth page
Everything else only reads.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class LockHeld(RuntimeError):
    pass


class Store:
    def __init__(self, local_dir: str = "data", bucket_name: str | None = None):
        self.local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)
        self.bucket_name = bucket_name
        self._bucket = None
        if bucket_name:
            from google.cloud import storage

            self._bucket = storage.Client().bucket(bucket_name)

    @classmethod
    def from_env(cls, local_dir: str | None = None) -> "Store":
        return cls(local_dir or os.getenv("STATE_DIR", "data"), os.getenv("GCS_BUCKET_NAME") or None)

    @property
    def remote(self) -> bool:
        return self._bucket is not None

    def local_path(self, name: str) -> str:
        path = os.path.join(self.local_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # ---------- whole-file sync ----------
    def download(self, name: str, local_name: str | None = None) -> bool:
        if not self.remote:
            src = self.local_path(name)
            if local_name and local_name != name and os.path.exists(src):
                import shutil

                shutil.copyfile(src, self.local_path(local_name))
            return os.path.exists(src)
        blob = self._bucket.get_blob(name)
        if blob is None:
            return False
        blob.download_to_filename(self.local_path(local_name or name))
        return True

    def upload(self, name: str) -> None:
        if not self.remote:
            return
        path = self.local_path(name)
        if os.path.exists(path):
            self._bucket.blob(name).upload_from_filename(path)

    def exists(self, name: str) -> bool:
        if not self.remote:
            return os.path.exists(os.path.join(self.local_dir, name))
        return self._bucket.blob(name).exists()

    def put_bytes(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Write a whole object (archives); local mode writes under local_dir."""
        if not self.remote:
            with open(self.local_path(name), "wb") as f:
                f.write(data)
            return
        self._bucket.blob(name).upload_from_string(data, content_type=content_type)

    # ---------- JSON documents ----------
    def read_json(self, name: str, default=None):
        if self.remote:
            blob = self._bucket.get_blob(name)
            if blob is None:
                return default
            return json.loads(blob.download_as_text())
        path = self.local_path(name)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    def update_json(self, name: str, mutate, default=None, retries: int = 5):
        """Read-modify-write that never silently overwrites a concurrent change."""
        if not self.remote:
            data = mutate(self.read_json(name, default if default is not None else {}))
            path = self.local_path(name)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
            return data

        from google.api_core.exceptions import PreconditionFailed

        for _ in range(retries):
            blob = self._bucket.get_blob(name)
            current = json.loads(blob.download_as_text()) if blob else (default if default is not None else {})
            generation = blob.generation if blob else 0
            data = mutate(current)
            try:
                self._bucket.blob(name).upload_from_string(
                    json.dumps(data, indent=2, default=str),
                    content_type="application/json",
                    if_generation_match=generation,
                )
                return data
            except PreconditionFailed:
                time.sleep(0.3)
        raise RuntimeError(f"Could not update {name}: concurrent writers")

    # ---------- exclusive lock (one engine process at a time) ----------
    @contextmanager
    def lock(self, name: str, ttl_seconds: int, wait_seconds: float = 0, poll_seconds: float = 5):
        """Exclusive lock; waits up to `wait_seconds` for a short-lived holder to finish."""
        owner = f"{socket.gethostname()}:{os.getpid()}"
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                self._acquire(name, owner, ttl_seconds)
                break
            except LockHeld:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            self._release(name)

    def _acquire(self, name: str, owner: str, ttl: int) -> None:
        payload = json.dumps({"owner": owner, "at": datetime.now(timezone.utc).isoformat()})
        if not self.remote:
            path = self.local_path(name)
            for _ in range(2):
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(fd, "w") as f:
                        f.write(payload)
                    return
                except FileExistsError:
                    if time.time() - os.path.getmtime(path) > ttl:
                        os.remove(path)
                        continue
                    raise LockHeld(name)
            raise LockHeld(name)

        from google.api_core.exceptions import NotFound, PreconditionFailed

        for _ in range(2):
            try:
                self._bucket.blob(name).upload_from_string(payload, if_generation_match=0)
                return
            except PreconditionFailed:
                blob = self._bucket.get_blob(name)
                if blob is None:
                    continue
                age = (datetime.now(timezone.utc) - blob.updated).total_seconds()
                if age > ttl:
                    log.warning("Breaking stale lock %s (age %.0fs)", name, age)
                    try:
                        blob.delete(if_generation_match=blob.generation)
                    except (NotFound, PreconditionFailed):
                        pass
                    continue
                raise LockHeld(name)
        raise LockHeld(name)

    def _release(self, name: str) -> None:
        if not self.remote:
            try:
                os.remove(self.local_path(name))
            except FileNotFoundError:
                pass
            return
        from google.api_core.exceptions import NotFound

        try:
            self._bucket.blob(name).delete()
        except NotFound:
            pass
