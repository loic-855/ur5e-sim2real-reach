#!/usr/bin/env python3
"""
obs_logger.py (moved under utils)
This is the same ObsLogger implementation but placed in ``scripts/sim2real/utils`` so it can be imported
with ``from utils.obs_logger import ObsLogger`` like other utilities in this package.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import threading
import queue
import time
from typing import Optional

import numpy as np


class ObsLogger:
    """Non-blocking logger that writes obs+action samples to a single binary file.

    Per-sample layout (float32, little-endian):
        [29 float32 obs] + [7 float32 action]  -> 36 float32 = 144 bytes/sample

    Metadata written to JSON on close.
    """

    def __init__(
        self,
        out_dir: Path | str = Path("logs/sim2real"),
        base_name: str = "obs",
        queue_max_size: int = 10000,
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
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()

        self.sample_size = 36  # floats per sample (29 obs + 7 action)
        self.sample_bytes = self.sample_size * 4
        self._num_written = 0
        self._drops = 0
        self._lock = threading.Lock()
        self.start_time = time.time()

    def push(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Enqueue one sample (obs 29D, action 7D). Non-blocking; drops on full queue."""
        rec = np.concatenate([np.asarray(obs, dtype=np.float32).ravel(), np.asarray(action, dtype=np.float32).ravel()])
        if rec.size != self.sample_size:
            raise ValueError(f"Expected sample size {self.sample_size}, got {rec.size}")
        data = rec.tobytes()
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            with self._lock:
                self._drops += 1

    def _writer_thread(self) -> None:
        """Background thread that writes queued bytes to disk."""
        while not self._stop.is_set() or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._f.write(data)
                self._num_written += 1
            except Exception:
                # Swallow IO errors to avoid crashing real-time loop
                pass
        try:
            self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Stop writer thread, flush and close file, write metadata JSON."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass

        meta = {
            "base_name": self.base_name,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "sample_size": self.sample_size,
            "sample_bytes": self.sample_bytes,
            "sample_count": int(self._compute_sample_count()),
            "drops": int(self._drops),
            "version": "obslog_v1",
        }
        try:
            with open(self.meta_path, "w") as mf:
                json.dump(meta, mf, indent=2)
        except Exception:
            pass

    def _compute_sample_count(self) -> int:
        try:
            size = self.bin_path.stat().st_size
            return size // self.sample_bytes
        except Exception:
            return int(self._num_written)
