#!/usr/bin/env python3
"""Audio stitching workflow for randomized RTTM training samples."""
from __future__ import annotations

import json
import mimetypes
import shutil
import sys
from pathlib import Path

from audio_numbering import AUDIO_EXTENSIONS
from fine_tuning_manager import slugify
from stitch_audio import StitchInput, TrainingTarget, run_stitch
from workflow_background import run_status, utc_now_iso

from dashboard.constants import DASHBOARD_REFRESH_STATUSES
from dashboard.slurm import submit_sbatch_job


class StitchingMixin:
    """Handles the stitched-audio tab, artifacts, and optional training export."""

    def stitched_run_directories(self, *, limit: int = 30) -> list[Path]:
        if not self.stitched_dir.is_dir():
            return []
        run_dirs = [
            path
            for path in self.stitched_dir.iterdir()
            if path.is_dir()
        ]
        run_dirs.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
        return run_dirs[:limit]

    def active_stitched_run_directories(self) -> list[Path]:
        return [
            path
            for path in self.stitched_run_directories(limit=100)
            if run_status(path) in DASHBOARD_REFRESH_STATUSES
        ]

    def stitched_summary(self) -> dict[str, int]:
        rows = self.stitched_run_rows(limit=100, script_name="")
        return {
            "total": len(rows),
            "succeeded": sum(1 for row in rows if row.get("status") == "succeeded"),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
            "submitted": sum(1 for row in rows if row.get("status") in {"submitted", "running"}),
            "training_added": sum(1 for row in rows if row.get("trainingUsage")),
        }

    def read_stitched_metadata(self, run_dir: Path) -> dict[str, object]:
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            return {}
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def first_existing_stitched_artifact(self, run_dir: Path, suffix: str, *, contains: str = "") -> Path | None:
        candidates = sorted(run_dir.glob(f"*{suffix}"), key=lambda path: path.name.lower())
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if contains and contains not in candidate.name:
                continue
            return candidate
        return None

    def stitched_run_rows(self, *, limit: int = 20, script_name: str = "") -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for run_dir in self.stitched_run_directories(limit=limit):
            metadata = self.read_stitched_metadata(run_dir)
            wav_path = self.resolve_stitched_artifact(metadata.get("audio_path"), run_dir, ".wav")
            rttm_path = self.resolve_stitched_artifact(metadata.get("rttm_path"), run_dir, ".rttm")
            srt_path = self.resolve_stitched_artifact(metadata.get("srt_path"), run_dir, ".srt")
            manifest_path = self.resolve_stitched_artifact(metadata.get("manifest_path"), run_dir, ".tsv", contains="_segments")
            transcript_path = self.resolve_stitched_artifact(metadata.get("transcript_path"), run_dir, ".txt")
            review_path = self.resolve_stitched_artifact(metadata.get("review_path"), run_dir, ".html", contains="_review")
            review_flags_path = self.resolve_stitched_artifact(metadata.get("review_flags_path"), run_dir, ".tsv", contains="_review_flags")
            metadata_path = run_dir / "metadata.json"
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            status = run_status(run_dir)
            links = [
                link
                for link in [
                    self.frontend_artifact_link("Stitched WAV", wav_path, script_name),
                    self.frontend_artifact_link("RTTM", rttm_path, script_name),
                    self.frontend_artifact_link("Segment SRT", srt_path, script_name),
                    self.frontend_artifact_link("Timing Review", review_path, script_name),
                    self.frontend_artifact_link("Segment Manifest", manifest_path, script_name),
                    self.frontend_artifact_link("Transcript Notes", transcript_path, script_name),
                    self.frontend_artifact_link("Review Flags", review_flags_path, script_name),
                    self.frontend_artifact_link("metadata.json", metadata_path, script_name),
                    self.frontend_artifact_link("stdout.log", stdout_path, script_name),
                    self.frontend_artifact_link("stderr.log", stderr_path, script_name),
                ]
                if link
            ]
            rows.append(
                {
                    "name": run_dir.name,
                    "path": self.describe_path(run_dir),
                    "status": status,
                    "runDir": self.describe_path(run_dir),
                    "seed": str(metadata.get("seed") or ""),
                    "durationSeconds": metadata.get("duration_seconds", 0),
                    "speakerCount": metadata.get("speaker_count", 0),
                    "segmentCount": metadata.get("segment_count", 0),
                    "inputCount": metadata.get("input_count", 0),
                    "outputName": str(metadata.get("output_name") or run_dir.name),
                    "displayName": str(metadata.get("display_name") or metadata.get("output_name") or run_dir.name),
                    "error": str(metadata.get("error") or ""),
                    "startedAt": str(metadata.get("started_at_utc") or metadata.get("stitch_started_at_utc") or ""),
                    "completedAt": str(metadata.get("completed_at_utc") or ""),
                    "trainingTargets": [str(item) for item in (metadata.get("training_targets") or [])],
                    "trainingUsage": [item for item in (metadata.get("training_usage") or []) if isinstance(item, dict)],
                    "segments": [item for item in (metadata.get("segments") or []) if isinstance(item, dict)],
                    "audioHref": self.file_link(wav_path, script_name) if isinstance(wav_path, Path) and wav_path.is_file() else "",
                    "links": links,
                    "slurmQueue": self.cached_slurm_queue(str(metadata.get("slurm_job_id") or ""), status),
                    "metadata": metadata,
                }
            )
        return rows

    def resolve_stitched_artifact(
        self,
        metadata_value: object,
        run_dir: Path,
        suffix: str,
        *,
        contains: str = "",
    ) -> Path | None:
        value = str(metadata_value or "").strip()
        if value:
            try:
                candidate = self.resolve_under_root(value)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.is_file():
                return candidate
        return self.first_existing_stitched_artifact(run_dir, suffix, contains=contains)

    def stitching_inputs_from_form(self, form) -> tuple[list[StitchInput], list[str]]:
        selected_audio = [self.clean_audio_selection_value(value) for value in form.getlist("selected_audio")]
        selected_audio = [value for value in selected_audio if value]
        speaker_labels = [str(value or "").strip() for value in form.getlist("speaker_labels")]
        if not selected_audio:
            return [], ["Select at least two audio files to stitch."]
        if len(selected_audio) < 2:
            return [], ["Select at least two audio files to stitch."]

        resolved_audio = self.selected_audio_names(selected_audio, run_all=False)
        speaker_by_requested = {
            name: speaker_labels[index] if index < len(speaker_labels) else ""
            for index, name in enumerate(selected_audio)
        }
        inputs: list[StitchInput] = []
        errors: list[str] = []
        for audio_name in resolved_audio:
            audio_path = (self.audio_dir / audio_name).resolve()
            try:
                audio_path.relative_to(self.audio_dir.resolve())
            except ValueError:
                errors.append(f"{audio_name}: path is outside audio_in.")
                continue
            if not audio_path.is_file():
                errors.append(f"{audio_name}: audio file was not found.")
                continue
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                errors.append(f"{audio_name}: unsupported audio type.")
                continue
            speaker = (
                speaker_by_requested.get(audio_name)
                or speaker_by_requested.get(Path(audio_name).name)
                or "Speaker_0"
            )
            safe_speaker = self.rttm_safe_token(speaker, fallback="Speaker_0")
            inputs.append(StitchInput(audio_path=audio_path, audio_name=audio_name, speaker=safe_speaker))
        if len(inputs) < 2 and not errors:
            errors.append("Select at least two audio files that still exist in audio_in.")
        return inputs, errors

    def stitching_training_targets_from_form(self, form) -> tuple[list[TrainingTarget], list[str]]:
        if not form.getfirst("add_to_training"):
            return [], []
        explicit_targets, errors = self.training_label_targets_from_values(form.getlist("training_targets"))
        targets: list[TrainingTarget] = []
        seen: set[str] = set()
        for target in explicit_targets:
            training_target = TrainingTarget(
                backend=target["backend"],
                project_name=target["project_name"],
            )
            if training_target.key in seen:
                continue
            targets.append(training_target)
            seen.add(training_target.key)

        raw_project = (form.getfirst("project_name") or "").strip()
        project_name = slugify(raw_project) if raw_project else ""
        try:
            backends = self.training_label_target_backends(form.getfirst("fine_tuning_backend") or "pyannote")
        except ValueError as exc:
            errors.append(str(exc))
            backends = []
        if project_name:
            for backend in backends:
                training_target = TrainingTarget(backend=backend, project_name=project_name)
                if training_target.key in seen:
                    continue
                targets.append(training_target)
                seen.add(training_target.key)
        if not targets:
            errors.append(
                "Choose at least one existing fine-tuning project or provide a new project name before adding the stitched sample to training."
            )
        return targets, errors

    def write_stitch_inputs_json(self, run_dir: Path, inputs: list[StitchInput]) -> Path:
        inputs_path = run_dir / "selected_audio.json"
        inputs_path.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "audio_path": self.describe_path(item.audio_path),
                            "audio_name": item.audio_name,
                            "speaker": item.speaker,
                        }
                        for item in inputs
                    ]
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return inputs_path

    def stitched_run_from_form(self, form) -> Path:
        raw_run_dir = (form.getfirst("run_dir") or form.getfirst("stitch_run") or "").strip()
        if not raw_run_dir:
            raise ValueError("Pick a stitched output to rename.")
        run_dir = self.resolve_under_root(raw_run_dir)
        stitched_root = self.stitched_dir.resolve()
        try:
            run_dir.resolve().relative_to(stitched_root)
        except ValueError as exc:
            raise ValueError("Stitched output path is outside the stitched workspace.") from exc
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Stitched output was not found: {raw_run_dir}")
        return run_dir

    def handle_stitching_rename(self, environ):
        form = self.parse_form(environ)
        display_name = (form.getfirst("display_name") or form.getfirst("stitch_name") or "").strip()
        if not display_name:
            return self.redirect(environ, "/stitching", message="Provide a new stitched output name.", status="error")
        try:
            run_dir = self.stitched_run_from_form(form)
            metadata = self.read_stitched_metadata(run_dir)
            metadata.update(
                {
                    "display_name": display_name,
                    "renamed_at_utc": utc_now_iso(),
                }
            )
            metadata.setdefault("run_dir", self.describe_path(run_dir))
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        except (OSError, ValueError, FileNotFoundError) as exc:
            return self.redirect(environ, "/stitching", message=str(exc), status="error")
        self.invalidate_dashboard_cache()
        return self.redirect(
            environ,
            "/stitching",
            message=f"Renamed stitched output to '{display_name}'.",
            status="success",
        )

    def handle_stitching_run(self, environ):
        form = self.parse_form(environ)
        inputs, input_errors = self.stitching_inputs_from_form(form)
        training_targets, target_errors = self.stitching_training_targets_from_form(form)
        errors = [*input_errors, *target_errors]
        if errors:
            return self.redirect(
                environ,
                "/stitching",
                message=self.notification_message("The stitch request needs a correction.", *errors[:8]),
                status="error",
            )

        output_name = (form.getfirst("stitch_name") or "").strip() or "stitched-audio"
        seed = (form.getfirst("stitch_seed") or "").strip()
        use_slurm = bool(form.getfirst("stitch_use_slurm"))
        target_name = slugify(output_name)

        if use_slurm:
            if not shutil.which("sbatch"):
                return self.redirect(
                    environ,
                    "/stitching",
                    message="sbatch is not available on this machine. Clear the Slurm checkbox and run the stitch locally.",
                    status="error",
                )
            try:
                run_dir = self.stitched_dir / self.run_directory_name("stitch", count=len(inputs))
                run_dir.mkdir(parents=True, exist_ok=False)
                inputs_path = self.write_stitch_inputs_json(run_dir, inputs)
                metadata = {
                    "workflow": "audio_stitching",
                    "mode": "slurm",
                    "input_count": len(inputs),
                    "seed": seed,
                    "output_name": target_name,
                    "display_name": output_name,
                    "training_targets": [target.key for target in training_targets],
                }
                submit = submit_sbatch_job(
                    sbatch_script=self.site_stitching_sbatch,
                    cwd=self.root,
                    run_dir=run_dir,
                    export_env={
                        "SITE_ROOT_DIR": str(self.root),
                        "SITE_STITCH_RUN_DIR": str(run_dir),
                        "SITE_STITCH_INPUTS_JSON": str(inputs_path),
                        "SITE_STITCH_OUTPUT_NAME": output_name,
                        "SITE_STITCH_SEED": seed,
                        "SITE_STITCH_TRAINING_TARGETS_JSON": json.dumps([target.key for target in training_targets]),
                    },
                    metadata=metadata,
                    job_label="audio stitching",
                )
            except Exception as exc:  # noqa: BLE001
                return self.redirect(environ, "/stitching", message=str(exc), status="error")
            self.invalidate_dashboard_cache()
            return self.redirect(
                environ,
                "/stitching",
                message=(
                    f"Submitted stitching job {submit['slurm_job_id']} for {len(inputs)} file(s). "
                    "The stitched WAV, RTTM, manifest, and review page will land under stitched/."
                ),
                status="success",
            )

        try:
            result = run_stitch(
                inputs=inputs,
                output_name=output_name,
                seed=seed,
                root=self.root,
                training_targets=training_targets,
            )
        except Exception as exc:  # noqa: BLE001
            return self.redirect(environ, "/stitching", message=str(exc), status="error")
        self.invalidate_dashboard_cache()
        training_message = ""
        if result.training_usage:
            training_message = " Added to training: " + ", ".join(str(item["project_key"]) for item in result.training_usage)
        return self.redirect(
            environ,
            "/stitching",
            message=(
                f"Created stitched sample '{result.wav_path.name}' with {len(result.segments)} RTTM segment(s)."
                f"{training_message}"
            ),
            status="success",
        )
