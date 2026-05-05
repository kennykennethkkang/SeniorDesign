#!/usr/bin/env python3
"""Diarization Error Rate + companion metrics for the senior-design dashboard.

The point of this module is to grade a fine-tuned model's predictions against
ground-truth labels — so we can answer "did the new training run actually do
better?" without dragging in a pyannote install on the cluster.

Everything here works on (start, end, speaker) intervals derived from RTTM
files. Numbers it reports:

  - DER (Diarization Error Rate): the canonical (miss + false_alarm + confusion)
    divided by total reference speech time. Lower is better.
  - miss / false_alarm / confusion: the three components of DER, broken out so
    we can tell *why* a run did worse — e.g. a high false-alarm rate usually
    means the VAD threshold is too permissive.
  - JER (Jaccard Error Rate): per-speaker, averaged. Less sensitive to one
    very-talkative speaker dominating the metric, which DER is prone to.
  - speaker_count_diff: |hyp_speakers - ref_speakers|. Useful for spotting
    over-segmentation early.

For optimal speaker mapping we try scipy's linear_sum_assignment when it's
available; if scipy isn't installed we fall back to greedy assignment, which
is within a few percent of optimal on diarization-sized problems.
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
    """One stretch of speech for one speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def parse_rttm_intervals(rttm_path: Path | str) -> list[Interval]:
    """Read an RTTM file into a list of Interval(start, end, speaker).

    The RTTM line format is space-separated: SPEAKER <uri> <ch> <start>
    <duration> <NA> <NA> <speaker> <conf> <NA>. We only need start, duration,
    and speaker columns; everything else is just kept in the file as a NeMo /
    pyannote contract.
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
    """Merge a stream of intervals (any speakers) into non-overlapping ranges.

    Used to compute total speech time of a side regardless of which speaker is
    talking — the denominator in DER is total *reference* speech time, with
    overlap counted once.
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
    """Total speech time across all speakers, with overlap counted once."""

    return sum(end - start for start, end in merge_intervals(intervals))


def speakers_in(intervals: Iterable[Interval]) -> list[str]:
    """Stable-sorted list of unique speaker labels."""

    return sorted({iv.speaker for iv in intervals})


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return how many seconds two intervals share."""

    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_matrix(reference: Sequence[Interval], hypothesis: Sequence[Interval]) -> dict[tuple[str, str], float]:
    """Build the {(ref_speaker, hyp_speaker): overlap_seconds} cost matrix.

    O(n*m) is fine for diarization-sized problems (a couple thousand intervals
    at most). If we ever needed more, the right move would be to sweep both
    sides as sorted event streams.
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
    """Return ref_speaker -> hyp_speaker assignment that maximizes shared time.

    Uses scipy.optimize.linear_sum_assignment if available (optimal Hungarian
    in O((n*m)*sqrt(n+m))). Falls back to a greedy max-overlap pick when scipy
    is not installed — the cluster doesn't always have it. Greedy is within a
    few percent of optimal for diarization speaker counts (typically <8).
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
    """Greedy fallback: at each step take the largest unassigned overlap pair."""

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
    """Return DER plus its three components for one (ref, hyp) RTTM pair.

    This is person-time DER (NIST convention) — when two speakers talk at once
    in the reference, that 2-second slice contributes 2 person-seconds to the
    denominator and to miss/false-alarm/confusion accounting. The simpler
    binary-presence version under-counts errors in overlapping speech, which
    senior-design audio (multiple kids in one clip) hits a lot.

    Definitions:
      - reference_speech    = sum over slices of (active reference speakers) *
                              slice duration. The denominator.
      - hypothesis_speech   = same, on the hypothesis side. Reported for
                              context (e.g. spotting an over-talkative VAD).
      - miss                = slices where reference has more speakers active
                              than hypothesis can cover.
      - false_alarm         = slices where hypothesis has more speakers active
                              than reference, i.e. the model invents speech.
      - confusion           = slices where speaker counts match but the mapped
                              pair isn't simultaneously active.
      - der                 = (miss + false_alarm + confusion) / reference_speech

    No forgiveness collar by default. If we want one later, shrink reference
    intervals before passing them in.
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
        # Count how many mapped (ref -> hyp) pairs have BOTH sides active in
        # this slice. Those are "correct"; everything else on the smaller
        # side becomes confusion.
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
        # Reference intervals were all zero-length / overlapped to nothing.
        # Fall through to the empty-reference branch's semantics so the rate
        # isn't a divide-by-zero.
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
    """Jaccard Error Rate: per-reference-speaker, averaged.

    For each reference speaker s_r, the best hypothesis match is the one with
    largest temporal overlap. The per-speaker error is 1 - |intersection| /
    |union|, and JER is the mean of those errors. A speaker that the model
    never produces shows up as JER 1.0 for that slot; that's intentional —
    forgetting an entire speaker is a real, expensive error.
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


def score_run(
    reference_rttm: Path | str,
    hypothesis_rttm: Path | str,
) -> dict[str, object]:
    """Read two RTTMs and produce a presentation-ready metrics payload."""

    reference = parse_rttm_intervals(reference_rttm)
    hypothesis = parse_rttm_intervals(hypothesis_rttm)
    der_metrics = compute_der(reference, hypothesis)
    jer_metrics = compute_jer(reference, hypothesis)
    ref_speakers = speakers_in(reference)
    hyp_speakers = speakers_in(hypothesis)
    return {
        "reference_rttm": str(reference_rttm),
        "hypothesis_rttm": str(hypothesis_rttm),
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
