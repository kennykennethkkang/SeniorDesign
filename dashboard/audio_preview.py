"""On-demand audio preview generator for the labeling review page.

The dashboard serves audio from a WAVE-cluster NFS mount. The originals are
typically 16 kHz mono 16-bit PCM WAVs, but for a 30-40 min recording that's
still ~70 MB — enough to crash a browser tab when the labeler is editing
many fields while audio is buffering.

This module produces an 8 kHz mono WAV preview (half the bytes) the first
time a given source is requested and caches the result under
``.local_dashboard/audio_preview/<hash>.wav``. Subsequent requests serve the
cached file directly. Originals are never modified — the ML pipeline keeps
reading the 16 kHz WAVs as before.

Pure-stdlib (``wave`` + ``audioop``) so we don't depend on ffmpeg, which
isn't installed on this cluster.
"""
from __future__ import annotations

import audioop
import hashlib
import os
import threading
import wave
from pathlib import Path
from typing import Final


PREVIEW_TARGET_RATE: Final[int] = 8000
PREVIEW_TARGET_CHANNELS: Final[int] = 1
PREVIEW_TARGET_WIDTH: Final[int] = 2  # 16-bit PCM
_PREVIEW_FRAME_CHUNK: Final[int] = 65536


# Per-target-path locks so concurrent requests for the same audio don't
# generate the preview twice. One lock per distinct cache path is enough;
# the dict itself is guarded by ``_LOCKS_GUARD``.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def cache_filename_for(source_path: Path) -> str:
    """Map an audio file to a stable cache filename.

    Includes the source path so two files with the same basename in
    different folders never collide. Includes mtime + size so re-encoded
    originals get a fresh preview automatically.
    """

    try:
        stat = source_path.stat()
        digest_input = f"{source_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    except OSError:
        digest_input = str(source_path)
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:24]
    return f"{digest}.wav"


def preview_path_for(source_path: Path, cache_root: Path) -> Path:
    """Resolve the cache path without touching the disk."""

    return cache_root / cache_filename_for(source_path)


def _generate_preview(source_path: Path, target_path: Path) -> None:
    """Read the source WAV, downsample to 8 kHz mono 16-bit, write target.

    Writes to a sibling temp file first and renames on success so a partial
    file is never visible if generation aborts midway.
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    state = None
    try:
        with wave.open(str(source_path), "rb") as source:
            src_channels = source.getnchannels()
            src_rate = source.getframerate()
            src_width = source.getsampwidth()
            with wave.open(str(tmp_path), "wb") as dst:
                dst.setnchannels(PREVIEW_TARGET_CHANNELS)
                dst.setsampwidth(PREVIEW_TARGET_WIDTH)
                dst.setframerate(PREVIEW_TARGET_RATE)
                while True:
                    raw = source.readframes(_PREVIEW_FRAME_CHUNK)
                    if not raw:
                        break
                    # Normalize bit width first so ratecv has a 16-bit input.
                    if src_width != PREVIEW_TARGET_WIDTH:
                        raw = audioop.lin2lin(raw, src_width, PREVIEW_TARGET_WIDTH)
                    # Fold stereo down to mono before resampling — cheaper
                    # and prevents the ratecv buffer from blowing up on
                    # long stereo files.
                    if src_channels > 1:
                        raw = audioop.tomono(raw, PREVIEW_TARGET_WIDTH, 0.5, 0.5)
                    if src_rate != PREVIEW_TARGET_RATE:
                        raw, state = audioop.ratecv(
                            raw,
                            PREVIEW_TARGET_WIDTH,
                            PREVIEW_TARGET_CHANNELS,
                            src_rate,
                            PREVIEW_TARGET_RATE,
                            state,
                        )
                    dst.writeframes(raw)
        os.replace(tmp_path, target_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def ensure_preview(source_path: Path, cache_root: Path) -> Path:
    """Return the cached preview, generating it if missing.

    Raises ``FileNotFoundError`` if the source doesn't exist.
    """

    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    target = preview_path_for(source_path, cache_root)
    if target.is_file():
        return target
    lock = _lock_for(target)
    with lock:
        # Re-check inside the lock — another thread may have finished while
        # we were waiting.
        if target.is_file():
            return target
        _generate_preview(source_path, target)
    return target
