#!/usr/bin/env python3
"""
obs_logger.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import atexit
import json
import signal
import threading
import queue
import time
from typing import Optional

import numpy as np


class ObsLogger:
    """Non-blocking logger that writes obs+action samples to a single binary file.

    Per-sample layout (float32, little-endian):
        [obs_size float32] + [action_size float32]  -> (obs_size + action_size) float32

    Metadata written to JSON periodically and on close.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        out_dir: Path | str = Path("logs/obs"),
        base_name: str = "obs",
        queue_max_size: int = 10000,
        json_sync_interval: float = 2.0,  # Write JSON every N seconds
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_name = f"{base_name}_{ts}"
        self.bin_path = self.out_dir / f"{self.base_name}.bin"
        self.meta_path = self.out_dir / f"{self.base_name}.json"

        # Open file for append in binary mode
        self._f = open(self.bin_path, "ab")

        # Queue for non-blocking writes
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_max_size)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()

        self.obs_size = obs_size
        self.action_size = action_size
        self.sample_size = obs_size + action_size  # floats per sample
        self.sample_bytes = self.sample_size * 4
        self._num_written = 0
        self._drops = 0
        self._lock = threading.Lock()
        self.start_time = time.time()
        self._json_sync_interval = json_sync_interval
        self._last_json_sync = time.time()

        # Register cleanup handlers for graceful shutdown
        atexit.register(self._atexit_cleanup)
        self._original_sigint = signal.signal(signal.SIGINT, self._sigint_handler)

    def push(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Enqueue one sample. Non-blocking; drops on full queue."""
        rec = np.concatenate([np.asarray(obs, dtype=np.float32).ravel(), np.asarray(action, dtype=np.float32).ravel()])
        if rec.size != self.sample_size:
            raise ValueError(f"Expected sample size {self.sample_size}, got {rec.size}")
        data = rec.tobytes()
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            with self._lock:
                self._drops += 1
        
        # Periodically sync JSON to disk
        now = time.time()
        if now - self._last_json_sync >= self._json_sync_interval:
            self._write_json()
            self._last_json_sync = now

    def _writer_thread(self) -> None:
        """Background thread that writes queued bytes to disk."""
        while not self._stop.is_set() or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._f.write(data)
                self._f.flush()  # Flush each write to ensure data is on disk
                self._num_written += 1
            except Exception:
                # Swallow IO errors to avoid crashing real-time loop
                pass

    def _sigint_handler(self, signum, frame):
        """Handle Ctrl+C by closing logger then calling original handler."""
        self.close()
        # Re-raise or call original handler
        if self._original_sigint and callable(self._original_sigint):
            self._original_sigint(signum, frame)
        else:
            raise KeyboardInterrupt

    def _atexit_cleanup(self):
        """Cleanup on program exit."""
        self.close()

    def _write_json(self) -> None:
        """Write metadata JSON (can be called multiple times)."""
        meta = {
            "base_name": self.base_name,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "obs_size": int(self.obs_size),
            "action_size": int(self.action_size),
            "sample_size": int(self.sample_size),
            "sample_bytes": int(self.sample_bytes),
            "sample_count": int(self._compute_sample_count()),
            "drops": int(self._drops),
            "version": "obslog_v2",
        }
        try:
            with open(self.meta_path, "w") as mf:
                json.dump(meta, mf, indent=2)
        except Exception:
            pass

    def close(self) -> None:
        """Stop writer thread, flush and close file, write metadata JSON."""
        if self._closed:
            return
        self._closed = True
        
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass

        # Final JSON write
        self._write_json()
        
        # Unregister handlers
        try:
            atexit.unregister(self._atexit_cleanup)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGINT, self._original_sigint or signal.SIG_DFL)
        except Exception:
            pass

    def _compute_sample_count(self) -> int:
        try:
            size = self.bin_path.stat().st_size
            return size // self.sample_bytes
        except Exception:
            return int(self._num_written)
