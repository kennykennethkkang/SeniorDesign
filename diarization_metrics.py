#!/usr/bin/env python3
"""Diarization Error Rate + companion metrics for the senior-design dashboard.

We need a way to grade our fine-tuned model's output against the labels we
hand-annotated — ideally without pulling in a full pyannote environment on
the cluster just to run one evaluation. So we rolled our own, keeping it
pure-Python except for the optional scipy Hungarian-algorithm speedup.

Metrics reported:
  - DER  (Diarization Error Rate) — the NIST-convention (miss + false_alarm +
    confusion) / reference_speech. Lower is better.
  - miss / false_alarm / confusion — DER broken into its three components so
    we can tell *why* a run got worse (e.g. high false alarm = VAD too loose).
  - JER  (Jaccard Error Rate) — per-speaker, averaged. Unlike DER, one
    very-talkative speaker can't dominate the number, so it's a better
    sanity-check on over-segmented recordings.
  - speaker_count_diff — |hyp_speakers − ref_speakers|; useful early signal
    for over- or under-clustering.

Speaker assignment uses scipy's linear_sum_assignment when available (optimal
Hungarian); otherwise we fall back to greedy max-overlap, which is within a
few percent of optimal for the small speaker counts we see.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Interval:
    """Holds a single speaker-speech region from an RTTM file."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def parse_rttm_intervals(rttm_path: Path | str) -> list[Interval]:
    """Parse an RTTM file into Interval objects so the rest of the module can work with typed data.

    RTTM is a NIST format: SPEAKER <uri> <ch> <start> <duration> ... <speaker> ...
    We only care about columns 3, 4, and 7 (start, duration, speaker label).
    """

    path = Path(rttm_path)
    intervals: list[Interval] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: bad start/duration: {exc}") from exc
            speaker = parts[7]
            if duration <= 0:
                continue
            intervals.append(Interval(start=start, end=start + duration, speaker=speaker))
    intervals.sort(key=lambda iv: (iv.start, iv.end))
    return intervals


def merge_intervals(intervals: Iterable[Interval]) -> list[tuple[float, float]]:
    """Collapse overlapping intervals down to non-overlapping ranges.

    The DER denominator is total reference speech time with overlap counted
    once, so we need to flatten multi-speaker regions before summing.
    """

    sorted_intervals = sorted(intervals, key=lambda iv: iv.start)
    merged: list[tuple[float, float]] = []
    for iv in sorted_intervals:
        if not merged or iv.start > merged[-1][1]:
            merged.append((iv.start, iv.end))
            continue
        prev_start, prev_end = merged[-1]
        merged[-1] = (prev_start, max(prev_end, iv.end))
    return merged


def total_speech_time(intervals: Iterable[Interval]) -> float:
    """Sum merged speech time — overlap regions are counted once, not once per speaker."""

    return sum(end - start for start, end in merge_intervals(intervals))


def speakers_in(intervals: Iterable[Interval]) -> list[str]:
    """Return a stable-sorted list of unique speaker labels so row/column order is deterministic."""

    return sorted({iv.speaker for iv in intervals})


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute the overlap between two time ranges — used as the building block for the speaker-match matrix."""

    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_matrix(reference: Sequence[Interval], hypothesis: Sequence[Interval]) -> dict[tuple[str, str], float]:
    """Build a {(ref_speaker, hyp_speaker): seconds_of_overlap} table for speaker assignment.

    O(n×m) is fine here — diarization files for our recordings top out at a
    few thousand intervals total, so brute-force pairwise is fast enough.
    """

    matrix: dict[tuple[str, str], float] = {}
    for ref in reference:
        for hyp in hypothesis:
            shared = _interval_overlap(ref.start, ref.end, hyp.start, hyp.end)
            if shared <= 0:
                continue
            key = (ref.speaker, hyp.speaker)
            matrix[key] = matrix.get(key, 0.0) + shared
    return matrix


def best_speaker_mapping(
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
) -> dict[str, str]:
    """Find the optimal ref→hyp speaker assignment so we don't penalize correct speech just because the labels differ.

    Without this, a perfect transcript where ref calls the teacher "SPK_0" and
    the hypothesis calls them "SPK_1" would score 100 % confusion. We solve the
    assignment problem using scipy's Hungarian algorithm when available, and fall
    back to greedy otherwise — WAVE doesn't always have scipy installed.
    """

    ref_speakers = speakers_in(reference)
    hyp_speakers = speakers_in(hypothesis)
    if not ref_speakers or not hyp_speakers:
        return {}
    matrix = overlap_matrix(reference, hypothesis)

    try:
        import numpy as np  # type: ignore
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:  # pragma: no cover - cluster without scipy
        return _greedy_speaker_mapping(ref_speakers, hyp_speakers, matrix)

    cost = np.zeros((len(ref_speakers), len(hyp_speakers)), dtype=np.float64)
    for i, ref in enumerate(ref_speakers):
        for j, hyp in enumerate(hyp_speakers):
            cost[i, j] = -matrix.get((ref, hyp), 0.0)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: dict[str, str] = {}
    for r, c in zip(row_ind, col_ind):
        if matrix.get((ref_speakers[r], hyp_speakers[c]), 0.0) > 0:
            mapping[ref_speakers[r]] = hyp_speakers[c]
    return mapping


def _greedy_speaker_mapping(
    ref_speakers: Sequence[str],
    hyp_speakers: Sequence[str],
    matrix: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Greedy fallback for when scipy isn't around — grab the highest-overlap pair first, repeat."""

    edges = sorted(((overlap, ref, hyp) for (ref, hyp), overlap in matrix.items()), reverse=True)
    used_ref: set[str] = set()
    used_hyp: set[str] = set()
    mapping: dict[str, str] = {}
    for overlap, ref, hyp in edges:
        if overlap <= 0 or ref in used_ref or hyp in used_hyp:
            continue
        mapping[ref] = hyp
        used_ref.add(ref)
        used_hyp.add(hyp)
    return mapping


def compute_der(
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
) -> dict[str, float]:
    """Compute person-time DER (miss + false_alarm + confusion) / reference_speech.

    We use the NIST person-time convention: when two speakers overlap in the
    reference, that 2-second window contributes 2 person-seconds to the
    denominator. The simpler binary-presence version under-penalizes overlapping
    regions, which matters a lot for classroom audio where multiple kids talk
    at once. All three components are returned separately so we can tell which
    one is driving a bad score (miss=VAD missed speech, false_alarm=hallucinated
    speech, confusion=right timing but wrong speaker label).
    """

    if not reference and not hypothesis:
        return {
            "der": 0.0,
            "miss": 0.0,
            "false_alarm": 0.0,
            "confusion": 0.0,
            "reference_speech": 0.0,
            "hypothesis_speech": 0.0,
        }
    if not reference:
        # No reference speech at all but the hypothesis fired: pure false
        # alarm. DER's denominator is zero, so the rate is undefined; we
        # report inf so it can't be confused with a successful run.
        hyp_persontime = sum(iv.duration for iv in hypothesis)
        return {
            "der": float("inf") if hyp_persontime > 0 else 0.0,
            "miss": 0.0,
            "false_alarm": hyp_persontime,
            "confusion": 0.0,
            "reference_speech": 0.0,
            "hypothesis_speech": hyp_persontime,
        }

    mapping = best_speaker_mapping(reference, hypothesis)
    boundaries = sorted({pt for iv in list(reference) + list(hypothesis) for pt in (iv.start, iv.end)})

    miss = 0.0
    false_alarm = 0.0
    confusion = 0.0
    reference_speech = 0.0
    hypothesis_speech = 0.0

    for left, right in zip(boundaries, boundaries[1:]):
        slice_duration = right - left
        if slice_duration <= 0:
            continue
        ref_active = {iv.speaker for iv in reference if iv.start <= left and right <= iv.end}
        hyp_active = {iv.speaker for iv in hypothesis if iv.start <= left and right <= iv.end}
        n_r = len(ref_active)
        n_h = len(hyp_active)
        # "Correct" = both sides of a matched pair are simultaneously active.
        # Everything left on the smaller side is confusion (right timing, wrong speaker).
        n_correct = sum(
            1 for ref_speaker, hyp_speaker in mapping.items()
            if ref_speaker in ref_active and hyp_speaker in hyp_active
        )
        reference_speech += n_r * slice_duration
        hypothesis_speech += n_h * slice_duration
        miss += max(0, n_r - n_h) * slice_duration
        false_alarm += max(0, n_h - n_r) * slice_duration
        confusion += (min(n_r, n_h) - n_correct) * slice_duration

    if reference_speech <= 0:
        # Guard against divide-by-zero when reference intervals are all zero-length
        # (shouldn't happen with valid RTTM, but we don't want to crash on bad input).
        return {
            "der": 0.0 if hypothesis_speech == 0 else float("inf"),
            "miss": miss,
            "false_alarm": false_alarm,
            "confusion": confusion,
            "reference_speech": 0.0,
            "hypothesis_speech": hypothesis_speech,
        }

    return {
        "der": (miss + false_alarm + confusion) / reference_speech,
        "miss": miss,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "reference_speech": reference_speech,
        "hypothesis_speech": hypothesis_speech,
    }


def compute_jer(
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
) -> dict[str, float]:
    """Compute per-speaker Jaccard Error Rate and average it across reference speakers.

    JER is useful alongside DER because DER gets dominated by whoever talks the
    most — JER gives each speaker equal weight. For each reference speaker we
    find the best-matching hypothesis speaker (most overlap) and compute
    1 − intersection/union. A speaker the model never predicted = JER 1.0.
    """

    ref_speakers = speakers_in(reference)
    if not ref_speakers:
        return {"jer": 0.0, "per_speaker": {}}
    hyp_by_speaker: dict[str, list[Interval]] = {}
    for iv in hypothesis:
        hyp_by_speaker.setdefault(iv.speaker, []).append(iv)
    ref_by_speaker: dict[str, list[Interval]] = {}
    for iv in reference:
        ref_by_speaker.setdefault(iv.speaker, []).append(iv)

    per_speaker: dict[str, float] = {}
    for ref_speaker in ref_speakers:
        ref_segments = ref_by_speaker.get(ref_speaker, [])
        ref_total = total_speech_time(ref_segments)
        best_pair: tuple[str, float] | None = None
        for hyp_speaker, hyp_segments in hyp_by_speaker.items():
            shared = sum(
                _interval_overlap(r.start, r.end, h.start, h.end)
                for r in ref_segments
                for h in hyp_segments
            )
            if best_pair is None or shared > best_pair[1]:
                best_pair = (hyp_speaker, shared)
        if best_pair is None:
            per_speaker[ref_speaker] = 1.0
            continue
        hyp_speaker, intersection = best_pair
        hyp_total = total_speech_time(hyp_by_speaker.get(hyp_speaker, []))
        union = ref_total + hyp_total - intersection
        per_speaker[ref_speaker] = 1.0 - (intersection / union) if union > 0 else 1.0

    average = sum(per_speaker.values()) / len(per_speaker)
    return {"jer": average, "per_speaker": per_speaker}


def score_intervals(
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
) -> dict[str, object]:
    """Score two already-parsed interval sequences. Lets callers feed in
    SRT-derived intervals (or in-memory test fixtures) without round-tripping
    through a temp RTTM file."""

    der_metrics = compute_der(reference, hypothesis)
    jer_metrics = compute_jer(reference, hypothesis)
    ref_speakers = speakers_in(reference)
    hyp_speakers = speakers_in(hypothesis)
    return {
        "der": der_metrics["der"],
        "miss_seconds": der_metrics["miss"],
        "false_alarm_seconds": der_metrics["false_alarm"],
        "confusion_seconds": der_metrics["confusion"],
        "reference_speech_seconds": der_metrics["reference_speech"],
        "hypothesis_speech_seconds": der_metrics["hypothesis_speech"],
        "jer": jer_metrics["jer"],
        "per_speaker_jer": jer_metrics["per_speaker"],
        "reference_speaker_count": len(ref_speakers),
        "hypothesis_speaker_count": len(hyp_speakers),
        "speaker_count_diff": abs(len(ref_speakers) - len(hyp_speakers)),
    }


def score_run(
    reference_rttm: Path | str,
    hypothesis_rttm: Path | str,
) -> dict[str, object]:
    """Top-level entry point: read two RTTM files and return a complete metrics dict for the dashboard."""

    reference = parse_rttm_intervals(reference_rttm)
    hypothesis = parse_rttm_intervals(hypothesis_rttm)
    payload = score_intervals(reference, hypothesis)
    payload["reference_rttm"] = str(reference_rttm)
    payload["hypothesis_rttm"] = str(hypothesis_rttm)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a hypothesis RTTM against a reference RTTM.")
    parser.add_argument("--reference", required=True, help="Ground-truth RTTM (the labels).")
    parser.add_argument("--hypothesis", required=True, help="Model-produced RTTM (the prediction).")
    parser.add_argument(
        "--format",
        choices=("json", "human"),
        default="human",
        help="json prints the full payload; human prints a short table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics = score_run(args.reference, args.hypothesis)
    if args.format == "json":
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    print(f"DER: {metrics['der'] * 100:.2f}%")
    print(f"  miss:        {metrics['miss_seconds']:.2f}s")
    print(f"  false alarm: {metrics['false_alarm_seconds']:.2f}s")
    print(f"  confusion:   {metrics['confusion_seconds']:.2f}s")
    print(f"JER: {metrics['jer'] * 100:.2f}%  (averaged across {metrics['reference_speaker_count']} reference speakers)")
    print(f"speakers — reference: {metrics['reference_speaker_count']}, hypothesis: {metrics['hypothesis_speaker_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
