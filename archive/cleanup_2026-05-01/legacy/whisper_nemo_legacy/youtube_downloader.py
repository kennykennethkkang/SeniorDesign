#!/usr/bin/env python3
"""
Download YouTube audio as 16 kHz mono WAV into audio_in/ for diarization tests.

Usage:
  python youtube_downloader.py <youtube_url>

Requires: yt-dlp, ffmpeg (available on WAVE nodes)
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import yt_dlp


def resolve_ffmpeg_location() -> str | None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="../audio_in",
        help="Destination directory (default: ../audio_in)",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_location = resolve_ffmpeg_location()
    if not ffmpeg_location:
        print(
            "No ffmpeg binary found. Install ffmpeg or ensure imageio-ffmpeg is installed.",
            file=sys.stderr,
        )
        return 1

    ydl_opts = {
        # Save into output dir with clear, unique names.
        "outtmpl": str(out_dir / "%(title).80s_%(id)s.%(ext)s"),
        # Grab best available audio, then convert to wav mono 16 kHz.
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            },
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "ffmpeg_location": ffmpeg_location,
        "quiet": False,
        "noprogress": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(args.url, download=True)
            # yt-dlp returns final filename in requested format
            downloaded = ydl.prepare_filename(info)
            final_wav = pathlib.Path(downloaded).with_suffix(".wav")
            print(f"\nSaved: {final_wav}")
            return 0
    except Exception as exc:  # pragma: no cover - convenience wrapper
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
