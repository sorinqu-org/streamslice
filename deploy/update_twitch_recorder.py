#!/usr/bin/env python3
"""Idempotently upgrade the deployed recorder integration."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


SETTINGS_FIELDS_OLD = """    processor_command: tuple[str, ...]
"""
SETTINGS_FIELDS_NEW = """    processor_command: tuple[str, ...]
    processor_max_attempts: int
    processor_retry_initial_seconds: int
    processor_retry_max_seconds: int
    processor_timeout_seconds: int
"""

SETTINGS_LOAD_OLD = """            processor_command=tuple(str(item) for item in processor_command_raw),
"""
SETTINGS_LOAD_NEW = """            processor_command=tuple(str(item) for item in processor_command_raw),
            processor_max_attempts=max(1, int(processor.get("max_attempts", 8))),
            processor_retry_initial_seconds=max(
                1, int(processor.get("retry_initial_seconds", 60))
            ),
            processor_retry_max_seconds=max(
                1, int(processor.get("retry_max_seconds", 1800))
            ),
            processor_timeout_seconds=max(
                60, int(processor.get("timeout_seconds", 18000))
            ),
"""

MANAGER_STATE_OLD = """        self.threads: list[threading.Thread] = []
"""
MANAGER_STATE_NEW = """        self.threads: list[threading.Thread] = []
        self.retry_attempts: dict[Path, int] = {}
        self.retry_pending: set[Path] = set()
        self.retry_timers: set[threading.Timer] = set()
        self.failed: set[Path] = set()
"""

STOP_OLD = """    def stop(self) -> None:
        for _ in self.threads:
"""
STOP_NEW = """    def stop(self) -> None:
        with self.lock:
            timers = list(self.retry_timers)
        for timer in timers:
            timer.cancel()
        for _ in self.threads:
"""

ENQUEUE_OLD = """            if path in self.queued or self._was_uploaded(path):
                return
            self.queued.add(path)
        self.tasks.put(path)

    def _state_path(self, chunk: Path) -> Path:
"""
ENQUEUE_NEW = """            if path in self.queued or path in self.failed or self._was_uploaded(path):
                return
            self.queued.add(path)
        self.tasks.put(path)

    def _schedule_retry(self, chunk: Path, returncode: int | None = None) -> None:
        with self.lock:
            failures = self.retry_attempts.get(chunk, 0) + 1
            self.retry_attempts[chunk] = failures
            if failures >= self.settings.processor_max_attempts:
                self.failed.add(chunk)
                self.retry_pending.discard(chunk)
                LOG.critical(
                    "StreamSlice failed permanently for %s after %s attempts; "
                    "the chunk is preserved and later chunks will continue",
                    chunk,
                    failures,
                )
                return
            delay = min(
                self.settings.processor_retry_initial_seconds * (2 ** (failures - 1)),
                self.settings.processor_retry_max_seconds,
            )
            self.retry_pending.add(chunk)

        LOG.error(
            "StreamSlice failed for %s (exit %s, attempt %s/%s); retrying in %ss",
            chunk,
            returncode if returncode is not None else "exception",
            failures,
            self.settings.processor_max_attempts,
            delay,
        )

        def requeue() -> None:
            with self.lock:
                self.retry_pending.discard(chunk)
                self.retry_timers.discard(timer)
                should_requeue = (
                    chunk.exists()
                    and chunk not in self.failed
                    and not self._was_uploaded(chunk)
                )
            if should_requeue:
                self.tasks.put(chunk)

        timer = threading.Timer(delay, requeue)
        timer.daemon = True
        with self.lock:
            self.retry_timers.add(timer)
        timer.start()

    def _state_path(self, chunk: Path) -> Path:
"""

PROCESSOR_OLD = """                    LOG.info("Handing %s to StreamSlice", chunk)
                    completed = subprocess.run(command, check=False)
                    if completed.returncode != 0:
                        LOG.error(
                            "StreamSlice failed for %s (exit %s); it will be retried",
                            chunk,
                            completed.returncode,
                        )
                        time.sleep(30)
                        self.tasks.put(chunk)
                        continue
                    self._mark_uploaded(chunk)
                    LOG.info("StreamSlice completed and uploaded render bundle for %s", chunk)
"""
PROCESSOR_NEW = """                    attempt = self.retry_attempts.get(chunk, 0) + 1
                    LOG.info(
                        "Handing %s to StreamSlice (attempt %s/%s)",
                        chunk,
                        attempt,
                        self.settings.processor_max_attempts,
                    )
                    completed = subprocess.run(
                        command,
                        check=False,
                        timeout=self.settings.processor_timeout_seconds,
                    )
                    if completed.returncode != 0:
                        self._schedule_retry(chunk, completed.returncode)
                        continue
                    self._mark_uploaded(chunk)
                    with self.lock:
                        self.retry_attempts.pop(chunk, None)
                        self.retry_pending.discard(chunk)
                    LOG.info("StreamSlice completed and uploaded render bundle for %s", chunk)
"""

EXCEPTION_OLD = """            except Exception:
                LOG.exception("Upload error for %s", chunk)
                time.sleep(30)
                self.tasks.put(chunk)
                continue
            finally:
                with self.lock:
                    if chunk not in list(self.tasks.queue):
                        self.queued.discard(chunk)
"""
EXCEPTION_NEW = """            except Exception:
                LOG.exception("Upload error for %s", chunk)
                if self.settings.processor_command:
                    self._schedule_retry(chunk)
                else:
                    time.sleep(30)
                    self.tasks.put(chunk)
                continue
            finally:
                with self.lock:
                    if (
                        chunk not in list(self.tasks.queue)
                        and chunk not in self.retry_pending
                    ):
                        self.queued.discard(chunk)
"""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"Expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    source = args.path.read_text(encoding="utf-8")
    for old, new, label in (
        (SETTINGS_FIELDS_OLD, SETTINGS_FIELDS_NEW, "settings fields"),
        (SETTINGS_LOAD_OLD, SETTINGS_LOAD_NEW, "settings load"),
        (MANAGER_STATE_OLD, MANAGER_STATE_NEW, "manager state"),
        (STOP_OLD, STOP_NEW, "stop method"),
        (ENQUEUE_OLD, ENQUEUE_NEW, "enqueue method"),
        (PROCESSOR_OLD, PROCESSOR_NEW, "processor branch"),
        (EXCEPTION_OLD, EXCEPTION_NEW, "exception handler"),
    ):
        source = replace_once(source, old, new, label)
    ast.parse(source, filename=str(args.path))
    args.path.write_text(source, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
