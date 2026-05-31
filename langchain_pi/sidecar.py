"""Long-lived Node sidecar process over an NDJSON stdio protocol."""

from __future__ import annotations

import atexit
import itertools
import json
import os
import threading
from pathlib import Path
from queue import Queue
from subprocess import PIPE, Popen
from typing import Iterator, Optional

_SCRIPT = Path(__file__).parent / "_sidecar" / "pi-sidecar.mjs"


class PiSidecar:
    def __init__(
        self,
        node_path: str = "node",
        cwd: Optional[str] = None,
        node_modules_dir: Optional[str] = None,
        script_path: Optional[str] = None,
    ) -> None:
        env = dict(os.environ)
        if node_modules_dir:
            # NODE_PATH is ignored for ESM; the sidecar resolves from here instead.
            env["LANGCHAIN_PI_NODE_MODULES"] = str(Path(node_modules_dir))
        self._proc = Popen(
            [node_path, str(script_path or _SCRIPT)],
            cwd=cwd or None,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            env=env,
            bufsize=0,
        )
        self._queues: dict[str, Queue] = {}
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stderr_chunks: list[bytes] = []
        self._dead = False
        self._closed = False
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        atexit.register(self.close)

    def _drain_stderr(self) -> None:
        stderr = self._proc.stderr
        assert stderr is not None
        for chunk in iter(lambda: stderr.read(4096), b""):
            self._stderr_chunks.append(chunk)

    def _stderr_text(self) -> str:
        return b"".join(self._stderr_chunks).decode("utf-8", "replace").strip()

    def _read_loop(self) -> None:
        buf = b""
        stdout = self._proc.stdout
        assert stdout is not None
        while True:
            chunk = stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:  # split on \n only; never U+2028/U+2029
                raw, buf = buf.split(b"\n", 1)
                raw = raw.rstrip(b"\r")
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                queue = self._queues.get(event.get("id"))
                if queue is not None:
                    queue.put(event)
        # Process exited: fail any in-flight (or future) request instead of
        # blocking forever. Guarded so registration in stream() can't race it.
        with self._lock:
            self._dead = True
            message = self._stderr_text() or "pi sidecar process exited"
            for queue in self._queues.values():
                queue.put({"type": "error", "error": {"errorMessage": message}})
                queue.put({"type": "end"})

    def stream(self, request: dict) -> Iterator[dict]:
        with self._lock:
            if self._closed or self._dead:
                raise RuntimeError(
                    self._stderr_text() or "pi sidecar is not running"
                )
            rid = str(next(self._counter))
            queue: Queue = Queue()
            self._queues[rid] = queue
        self._write({**request, "id": rid})
        try:
            while True:
                event = queue.get()
                if event.get("type") == "end":
                    break
                yield event
        except GeneratorExit:
            self._write({"type": "control", "action": "abort", "id": rid})
            raise
        finally:
            with self._lock:
                self._queues.pop(rid, None)

    def _write(self, payload: dict) -> None:
        if self._closed or self._proc.stdin is None:
            return
        line = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
