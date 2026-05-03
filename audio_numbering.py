#!/usr/bin/env python3
"""Ensure project media filenames carry a stable numeric prefix.

The numbering convention is not cosmetic. It keeps input ordering explicit across
manual uploads, YouTube downloads, and subsequent diarization runs so logs and
outputs can be matched back to source files without ambiguity.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
}
NUMBERED_PREFIX = re.compile(r"^(\d{3})_")
DEFAULT_AUDIO_DIR = pathlib.Path(__file__).resolve().parent / "audio_in"


def is_audio_file(path: pathlib.Path) -> bool:
    """Accept project media files while excluding generated helper inputs."""

    if not path.is_file():
        return False
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        return False
    if "_whisper_input" in path.stem:
        return False
    return True


def list_audio_files(audio_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return normalized media candidates from the target directory tree."""

    if not audio_dir.exists():
        return []
    return sorted(
        (entry.resolve() for entry in audio_dir.rglob("*") if is_audio_file(entry)),
        key=lambda p: str(p.relative_to(audio_dir.resolve())).lower(),
    )


def next_available_number(used_numbers: set[int]) -> int:
    """Find the first unused numeric prefix for the current directory state."""

    value = 1
    while value in used_numbers:
        value += 1
    return value


def normalize_audio_dir(audio_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Rename unnumbered files in-place and report every applied rename.

    Numbering is scoped per folder so each library set can start at 001 while
    older top-level files remain compatible with the original flat layout.
    """

    files = list_audio_files(audio_dir)
    renamed_paths: list[tuple[pathlib.Path, pathlib.Path]] = []
    files_by_parent: dict[pathlib.Path, list[pathlib.Path]] = {}
    for audio_path in files:
        files_by_parent.setdefault(audio_path.parent, []).append(audio_path)

    for parent_dir, parent_files in sorted(files_by_parent.items(), key=lambda item: str(item[0]).lower()):
        used_numbers: set[int] = set()
        for audio_path in parent_files:
            match = NUMBERED_PREFIX.match(audio_path.name)
            if match:
                used_numbers.add(int(match.group(1)))

        for audio_path in parent_files:
            if NUMBERED_PREFIX.match(audio_path.name):
                continue

            prefix_num = next_available_number(used_numbers)
            target_path = audio_path.with_name(f"{prefix_num:03d}_{audio_path.name}")
            while target_path.exists():
                prefix_num += 1
                while prefix_num in used_numbers:
                    prefix_num += 1
                target_path = audio_path.with_name(f"{prefix_num:03d}_{audio_path.name}")

            audio_path.rename(target_path)
            used_numbers.add(prefix_num)
            renamed_paths.append((audio_path, target_path.resolve()))

    return renamed_paths


def main() -> int:
    """Provide a small CLI for enforcing the numbering convention."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        default=str(DEFAULT_AUDIO_DIR),
        help="Audio directory to normalize (default: ./audio_in relative to this script)",
    )
    args = parser.parse_args()

    audio_dir = pathlib.Path(args.audio_dir).expanduser().resolve()
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}", file=sys.stderr)
        return 1
    if not audio_dir.is_dir():
        print(f"Audio path is not a directory: {audio_dir}", file=sys.stderr)
        return 1

    files = list_audio_files(audio_dir)
    renamed_paths = normalize_audio_dir(audio_dir)
    for original_path, target_path in renamed_paths:
        print(f"[RENUMBER] {original_path.name} -> {target_path.name}")

    print(
        f"Numbering check complete in {audio_dir}: "
        f"{len(files)} audio file(s), {len(renamed_paths)} renamed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
