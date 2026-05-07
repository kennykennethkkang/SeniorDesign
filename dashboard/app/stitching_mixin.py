#!/usr/bin/env python3
"""Audio stitching workflow for randomized RTTM training samples."""
from __future__ import annotations

import json
import mimetypes
import os
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

    def mirror_stitched_into_media(self, run_dir: Path, *, force: bool = False) -> str:
        """Drop the stitched WAV into audio_in/ so it shows up in Media + the
        stitching/diarization tabs, and write a pre-completed label_status
        record so the user doesn't have to label it by hand.

        Idempotent — once metadata.media_mirror_completed is True we skip on
        every subsequent dashboard render. Pass ``force=True`` only when
        re-running cleanup tests.

        Returns the audio_in-relative path the stitched file landed at, or
        an empty string if nothing was mirrored (run not yet succeeded, no
        WAV/RTTM, etc.).
        """

        metadata = self.read_stitched_metadata(run_dir)
        if str(metadata.get("submission_status") or "").lower() != "succeeded":
            return ""
        if metadata.get("media_mirror_completed") and not force:
            return str(metadata.get("media_mirror_path") or "")

        wav_source = self.resolve_stitched_artifact(metadata.get("audio_path"), run_dir, ".wav")
        rttm_source = self.resolve_stitched_artifact(metadata.get("rttm_path"), run_dir, ".rttm")
        if not (isinstance(wav_source, Path) and wav_source.is_file()):
            return ""
        if not (isinstance(rttm_source, Path) and rttm_source.is_file()):
            return ""
        transcript_source = self.resolve_stitched_artifact(metadata.get("transcript_path"), run_dir, ".txt")

        # The user-visible folder under audio_in is "audioStitching" — same
        # name on the dashboard's media library + diarization tabs so it's
        # easy to find. Keep one sub-folder per run so multiple stitches
        # don't collide on filenames.
        mirror_root = self.audio_dir / "audioStitching" / run_dir.name
        mirror_root.mkdir(parents=True, exist_ok=True)
        wav_target = mirror_root / wav_source.name
        rttm_target = mirror_root / rttm_source.name
        transcript_target = mirror_root / transcript_source.name if transcript_source and transcript_source.is_file() else None

        # Hardlinks let the audio_in/ entry share storage with the stitched/
        # original; for a multi-GB stitched WAV this avoids burning disk on a
        # full copy. Fall back to copy when the FS rejects the link (e.g.
        # cross-device on networked filesystems).
        def _link_or_copy(source: Path, target: Path) -> None:
            if target.exists() and target.samefile(source):
                return
            if target.exists():
                target.unlink()
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)

        try:
            _link_or_copy(wav_source, wav_target)
            _link_or_copy(rttm_source, rttm_target)
            if transcript_source and transcript_target:
                _link_or_copy(transcript_source, transcript_target)
        except OSError as exc:
            metadata["media_mirror_error"] = str(exc)
            try:
                (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            except OSError:
                pass
            return ""

        relative_audio_name = self.audio_relative_path(wav_target)

        # Translate the run's segments into the same RTTM-line shape that
        # manual labels store, so the Training Labels page renders the
        # entry like any other completed sample.
        segments = [item for item in (metadata.get("segments") or []) if isinstance(item, dict)]
        label_segment_lines = []
        for seg in segments:
            try:
                start = float(seg.get("start") or 0.0)
                duration = float(seg.get("duration") or 0.0)
            except (TypeError, ValueError):
                continue
            if duration <= 0:
                continue
            speaker = str(seg.get("speaker") or "Speaker_0")
            label_segment_lines.append(f"{start:.3f} {start + duration:.3f} {speaker}")
        label_segments_text = "\n".join(label_segment_lines)

        usages = [item for item in (metadata.get("training_usage") or []) if isinstance(item, dict)]
        target_projects = [
            f"{usage.get('backend')}/{usage.get('project_name')}"
            for usage in usages
            if usage.get("backend") and usage.get("project_name")
        ]
        backends = sorted({str(usage.get("backend") or "") for usage in usages if usage.get("backend")})
        primary_usage = usages[0] if usages else {}

        try:
            self.upsert_training_label_record(
                relative_audio_name,
                {
                    "audio_file": relative_audio_name,
                    "backend": (backends[0] if len(backends) == 1 else "both") if backends else "both",
                    "target_backends": backends or ["nemo", "pyannote"],
                    "target_projects": target_projects,
                    "training_projects": target_projects,
                    "training_usage": usages,
                    "training_audio_path": str(primary_usage.get("sample_audio_path") or ""),
                    "training_rttm_path": str(primary_usage.get("sample_rttm_path") or ""),
                    "training_transcript_path": str(primary_usage.get("sample_transcript_path") or ""),
                    "project_name": str(primary_usage.get("project_name") or ""),
                    "include_transcript": False,
                    "issue_questions": "",
                    "label_segments": label_segments_text,
                    "transcript_text": "",
                    "segment_count": len(segments),
                    "speaker_count": metadata.get("speaker_count", 0),
                    "source": "audio_stitching",
                    "status": "completed",
                    "system_questions": [],
                    "completed_at_utc": str(metadata.get("completed_at_utc") or utc_now_iso()),
                },
            )
        except Exception as exc:  # noqa: BLE001
            metadata["media_mirror_label_error"] = str(exc)

        metadata["media_mirror_path"] = str(wav_target.resolve().relative_to(self.root.resolve())) if wav_target.is_file() else ""
        metadata["media_mirror_audio_name"] = relative_audio_name
        metadata["media_mirror_completed"] = True
        metadata["media_mirror_completed_at_utc"] = utc_now_iso()
        try:
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
        self.invalidate_dashboard_cache()
        return relative_audio_name

    def trigger_stitch_auto_train(self, run_dir: Path, *, force: bool = False) -> list[str]:
        """Queue an auto-train sbatch for every project the stitched run added a sample to.

        Idempotent — stamps ``auto_train_triggered_at_utc`` and the queued
        target list into the run's metadata.json so subsequent passes (e.g.
        when the dashboard re-renders or the slurm path completes after the
        local handler has already returned) don't double-queue. Pass
        ``force=True`` to retry even if the stamp is present (currently used
        only by tests).
        """

        metadata = self.read_stitched_metadata(run_dir)
        if str(metadata.get("submission_status") or "").lower() != "succeeded":
            return []
        if metadata.get("auto_train_triggered") and not force:
            return list(metadata.get("auto_train_targets") or [])
        usages = [item for item in (metadata.get("training_usage") or []) if isinstance(item, dict)]
        if not usages:
            return []

        # Lazy import keeps the dashboard import graph un-tangled — auto_train
        # pulls in fine_tuning_manager, which is heavyweight to import at
        # startup.
        from dashboard import auto_train as _auto_train

        seen: set[str] = set()
        queued: list[str] = []
        for usage in usages:
            backend = str(usage.get("backend") or "").strip()
            project_name = str(usage.get("project_name") or "").strip()
            if not backend or not project_name:
                continue
            key = f"{backend}/{project_name}"
            if key in seen:
                continue
            seen.add(key)
            try:
                prepare_options = self.auto_train_prepare_options(project_name, backend)
                extra_env = self.auto_train_extra_env(backend)
                _auto_train.queue_auto_train(
                    project_name,
                    backend=backend,
                    root=self.root,
                    prepare_options=prepare_options,
                    extra_env=extra_env,
                    version_name="stitched-auto",
                )
                queued.append(key)
            except Exception as exc:  # noqa: BLE001
                metadata.setdefault("auto_train_errors", []).append(
                    {"target": key, "error": str(exc)}
                )

        metadata["auto_train_triggered"] = True
        metadata["auto_train_triggered_at_utc"] = utc_now_iso()
        metadata["auto_train_targets"] = queued
        try:
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass
        return queued

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
            # Catch the slurm-async case: a stitch finished in a cluster job
            # after the immediate handler had already returned. Both the
            # media mirror and the auto-train queue are idempotent —
            # subsequent renders no-op once the metadata has the stamp.
            try:
                self.mirror_stitched_into_media(run_dir)
            except Exception:
                pass
            try:
                self.trigger_stitch_auto_train(run_dir)
            except Exception:
                pass
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
        # Pair selected_audio with speaker_labels BEFORE filtering empties out,
        # otherwise dropping a malformed selection shifts every speaker index
        # one slot to the left and you get mislabeled segments. The form
        # always submits the two lists in the same order so positional
        # zipping is correct.
        raw_audio = list(form.getlist("selected_audio"))
        raw_speakers = list(form.getlist("speaker_labels"))
        paired: list[tuple[str, str]] = []
        for index, raw_value in enumerate(raw_audio):
            cleaned = self.clean_audio_selection_value(raw_value)
            if not cleaned:
                continue
            speaker = str(raw_speakers[index] if index < len(raw_speakers) else "").strip()
            paired.append((cleaned, speaker))
        if len(paired) < 2:
            return [], ["Select at least two audio files to stitch."]

        selected_audio = [name for name, _ in paired]
        resolved_audio = self.selected_audio_names(selected_audio, run_all=False)
        speaker_by_requested: dict[str, str] = {}
        for name, speaker in paired:
            # First label wins if the same file is submitted twice — same
            # behavior the old code had through dedup, just made explicit.
            speaker_by_requested.setdefault(name, speaker)
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

    def handle_stitching_delete(self, environ):
        """Wipe a stitched run dir and any sample copies it pushed into fine-tuning projects.

        We read the run's metadata before nuking the folder so we can locate
        the per-project audio/rttm/transcript copies that ``add_to_training``
        wrote out — leaving those behind would make the project look like it
        still has the stitched sample even after the run is gone.
        """

        form = self.parse_form(environ)
        try:
            run_dir = self.stitched_run_from_form(form)
        except (OSError, ValueError, FileNotFoundError) as exc:
            return self.redirect(environ, "/stitching", message=str(exc), status="error")

        metadata = self.read_stitched_metadata(run_dir)
        removed_files = 0
        try:
            workspace_root = self.root.resolve()
        except OSError:
            workspace_root = self.root

        for usage in metadata.get("training_usage") or []:
            if not isinstance(usage, dict):
                continue
            for key in ("sample_audio_path", "sample_rttm_path", "sample_transcript_path"):
                raw = str(usage.get(key) or "").strip()
                if not raw:
                    continue
                try:
                    candidate = self.resolve_local_path(raw).resolve()
                    candidate.relative_to(workspace_root)
                except (OSError, ValueError):
                    continue
                if candidate.is_file():
                    try:
                        candidate.unlink()
                        removed_files += 1
                    except OSError:
                        pass

        # Clear the audio_in mirror + the matching label_status entry that
        # mirror_stitched_into_media wrote, otherwise the deleted run keeps
        # haunting the Media Library and the Training Labels page.
        mirror_audio_name = str(metadata.get("media_mirror_audio_name") or "").strip()
        if mirror_audio_name:
            try:
                records = self.load_training_label_records()
                if mirror_audio_name in records:
                    del records[mirror_audio_name]
                    self.save_training_label_records(records)
            except Exception:  # noqa: BLE001
                pass
        # Clear both the new audioStitching/ mirror and the old stitched/
        # mirror so deletes still work for runs created before the rename.
        for mirror_parent in ("audioStitching", "stitched"):
            mirror_dir = self.audio_dir / mirror_parent / run_dir.name
            if not mirror_dir.exists():
                continue
            try:
                mirror_dir.resolve().relative_to(self.audio_dir.resolve())
                shutil.rmtree(mirror_dir)
            except (OSError, ValueError):
                pass

        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            return self.redirect(environ, "/stitching", message=f"Could not delete run: {exc}", status="error")

        self.invalidate_dashboard_cache()
        suffix = f" Also removed {removed_files} fine-tuning sample file(s)." if removed_files else ""
        return self.redirect(
            environ,
            "/stitching",
            message=f"Deleted stitched run '{run_dir.name}'.{suffix}",
            status="success",
        )

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
        # Local stitch finished synchronously; mirror the output into audio_in
        # (so Media Library + the project sample list both surface it) and
        # queue auto-train so the Fine-Tuning tab shows "submitted" runs
        # without a second click. The slurm path is covered by
        # stitched_run_rows on the next dashboard render.
        media_message = ""
        try:
            mirrored_name = self.mirror_stitched_into_media(result.run_dir)
        except Exception:  # noqa: BLE001
            mirrored_name = ""
        if mirrored_name:
            media_message = f" Visible in Media Library at audio_in/{mirrored_name}."
        auto_train_message = ""
        try:
            queued_targets = self.trigger_stitch_auto_train(result.run_dir)
        except Exception:  # noqa: BLE001
            queued_targets = []
        if queued_targets:
            auto_train_message = " Auto-train queued for: " + ", ".join(queued_targets)
        return self.redirect(
            environ,
            "/stitching",
            message=(
                f"Created stitched sample '{result.wav_path.name}' with {len(result.segments)} RTTM segment(s)."
                f"{training_message}{media_message}{auto_train_message}"
            ),
            status="success",
        )
