"""Bounded voice acquisition and profile state for CosyVoice zero-shot prompts."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class VoiceWorkflowError(RuntimeError):
    def __init__(self, stage: str, code: str, summary: str, *, retryable: bool = False, choices: list[dict[str, Any]] | None = None, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(summary)
        self.stage, self.code, self.summary = stage, code, summary
        self.retryable, self.choices, self.diagnostics = retryable, choices or [], diagnostics or {}

    def as_result(self) -> dict[str, Any]:
        return {"status": "error", "stage": self.stage, "code": self.code, "summary": self.summary, "retryable": self.retryable, "choices": self.choices, "diagnostics": self.diagnostics}


def _slug(value: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:48] or f"voice-{uuid.uuid4().hex[:8]}")


def _character_label(query: str, name: str) -> str:
    """Choose a human identity, never a source URL or generic request phrase."""
    for candidate in (query, name):
        value = " ".join(str(candidate or "").split()).strip()
        if not value or re.search(r"https?://|(?:youtube\.com|youtu\.be)", value, re.IGNORECASE):
            continue
        value = re.sub(r"\s+voice(?:\s+(?:clone|profile))?$", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"\s*\([^)]{1,40}\)\s*$", "", value).strip()
        if value and value.casefold() not in {"custom", "custom voice", "the selected voice"}:
            return value[:120]
    return "the selected voice"


_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PROCESS_LOCK = threading.RLock()

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development fallback
    fcntl = None


class VoiceManager:
    SEARCH_LIMIT = 5
    MAX_SOURCE_SECONDS = 300
    PREFERRED_SOURCE_SECONDS = 24
    # Twelve seconds stays inside the preferred 10-15 second cloning window
    # while giving CosyVoice more speaker detail than the minimum sample.
    TARGET_SECONDS = 12
    MIN_REFERENCE_SECONDS = 10
    MAX_REFERENCE_SECONDS = 15
    TARGET_SAMPLE_RATE = 24000
    PREPARE_LOCK_MAX_AGE_SECONDS = 600
    RECENT_SOURCE_MAX_AGE_SECONDS = 86400
    CANDIDATE_MAX_AGE_SECONDS = 86400

    def __init__(self, root: Path, ytdlp: str) -> None:
        self.root, self.ytdlp = Path(root), ytdlp
        self.profiles, self.candidates = self.root / "profiles", self.root / "candidates"
        self.state_file, self.logs = self.root / "state.json", self.root / "logs"
        self.recent_sources_file = self.root / "recent_sources.json"
        for path in (self.profiles, self.candidates, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self._write_json(self.state_file, {"session_profile": None, "session_candidate": False, "default_profile": None})
        self._prune_stale_candidates()

    def _prune_stale_candidates(self) -> list[str]:
        """Remove abandoned previews while preserving the currently active candidate."""
        state = self._read_json(self.state_file)
        active_id = str(state.get("session_profile") or "") if state.get("session_candidate") else ""
        now = time.time()
        removed = []
        for directory in self.candidates.iterdir():
            if not directory.is_dir() or directory.name == active_id:
                continue
            metadata = self._read_json(directory / "profile.json")
            created_at = float(metadata.get("created_at") or directory.stat().st_mtime)
            if now - created_at <= self.CANDIDATE_MAX_AGE_SECONDS:
                continue
            shutil.rmtree(directory)
            removed.append(directory.name)
        return removed

    @contextmanager
    def _state_lock(self):
        """Serialize state read-modify-write across Hermes and the TTS process."""
        lock_path = self.root / ".state.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _PROCESS_LOCK, lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def dispatch(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "status": self.status, "list": self.list_profiles,
            "search": lambda: self.search(str(args.get("query") or "")),
            "create": lambda: self.create(
                str(args.get("query") or ""),
                str(args.get("name") or args.get("query") or "Custom voice"),
                bool(args.get("make_default", False)),
                bool(args.get("enabled", True)),
                str(args.get("source_url") or ""),
                args.get("start_seconds"),
                bool(args.get("allow_variant", False)),
            ),
            "prepare": lambda: self.prepare(
                str(args.get("source_url") or ""), str(args.get("query") or ""),
                str(args.get("name") or args.get("query") or "Custom voice"), args.get("start_seconds"),
                str(args.get("prompt_text") or ""), str(args.get("profile_id") or ""),
            ),
            "refine": lambda: self.refine(str(args.get("profile_id") or ""), str(args.get("style_choice") or "")),
            "set_personality": lambda: self.set_personality(
                str(args.get("profile_id") or ""),
                bool(args.get("enabled", True)),
                str(args.get("personality_prompt") or ""),
            ),
            "accept": lambda: self.accept(str(args.get("profile_id") or ""), str(args.get("style_choice") or "original"), True, False),
            "set_default": lambda: self.accept(str(args.get("profile_id") or ""), str(args.get("style_choice") or "original"), True, True),
            "reset": self.reset, "discard": lambda: self.discard(str(args.get("profile_id") or "")),
        }
        if action not in handlers:
            raise VoiceWorkflowError("dispatch", "UNKNOWN_ACTION", f"Unknown action: {action}")
        return handlers[action]()

    def create(
        self,
        query: str,
        name: str,
        make_default: bool = False,
        personality_enabled: bool = True,
        source_url: str = "",
        start_seconds: Any = None,
        allow_variant: bool = False,
    ) -> dict[str, Any]:
        """Select, prepare, and save a bounded source for an explicit request.

        This compound operation avoids asking the model to copy a source URL from
        one tool result into a later call. One alternate source is attempted when
        acquisition or deterministic signal validation rejects the first choice.
        """
        duplicate = self._find_duplicate_profile(query, name, source_url)
        if duplicate:
            profile_id = str(duplicate["profile"]["id"])
            self.set_personality(profile_id, personality_enabled)
            saved = self.accept(
                profile_id,
                str(duplicate["profile"].get("selected_style") or "original"),
                True,
                make_default,
            )
            saved.update({
                "reused_existing": True,
                "duplicate_reason": duplicate["reason"],
                "source_selection": {
                    "mode": "existing_profile",
                    "selected": duplicate["profile"],
                    "attempted_sources": 0,
                    "failures": [],
                },
            })
            return saved

        if source_url:
            prepared = self.prepare(source_url, query, name, start_seconds)
            return self._finish_create(
                prepared,
                personality_enabled=personality_enabled,
                make_default=make_default,
                source_selection={
                    "mode": "provided_url",
                    "selected": {"source_url": source_url},
                    "attempted_sources": 1,
                    "failures": [],
                },
            )

        search_query = query if re.search(r"\bvoice\b", query, re.IGNORECASE) else f"{query} voice"
        search_result = self.search(search_query)
        candidates = [
            choice for choice in search_result.get("choices", [])
            if choice.get("suitability") != "reject" and choice.get("source_url")
        ]
        if not candidates:
            raise VoiceWorkflowError(
                "source_selection", "NO_SUITABLE_SOURCE",
                "Search returned no unflagged source suitable for automatic preparation.",
                choices=search_result.get("choices", []),
            )

        failures: list[dict[str, Any]] = []
        for choice in candidates[: self.SEARCH_LIMIT]:
            try:
                result = self.prepare(
                    str(choice["source_url"]), query, name,
                )
            except VoiceWorkflowError as exc:
                failures.append({
                    "source_id": choice.get("id"), "stage": exc.stage,
                    "code": exc.code, "summary": exc.summary,
                })
                if exc.stage not in {"download", "segment_scan", "extract", "signal_validation", "prompt_transcript"}:
                    raise
                continue

            return self._finish_create(
                result,
                personality_enabled=personality_enabled,
                make_default=make_default,
                source_metadata={
                    "title": str(choice.get("label") or "")[:160],
                    "channel": str(choice.get("channel") or "")[:120],
                    "search_query": str(search_result.get("query_used") or search_query)[:200],
                },
                source_selection={
                    "mode": "automatic", "selected": choice,
                    "attempted_sources": 1 + len(failures), "failures": failures,
                },
                search_metadata_ms=search_result.get("timings_ms", {}).get("search_metadata", 0),
            )

        raise VoiceWorkflowError(
            "auto_prepare", "AUTOMATIC_SOURCES_REJECTED",
            "All unflagged sources in the bounded search result failed preparation.",
            retryable=True,
            choices=search_result.get("choices", []),
            diagnostics={"failures": failures},
        )

    def _finish_create(
        self,
        prepared: dict[str, Any],
        *,
        personality_enabled: bool,
        make_default: bool,
        source_selection: dict[str, Any],
        source_metadata: dict[str, Any] | None = None,
        search_metadata_ms: int = 0,
    ) -> dict[str, Any]:
        profile_id = str((prepared.get("profile") or {}).get("id") or "")
        if not profile_id:
            raise VoiceWorkflowError("auto_prepare", "MISSING_PROFILE_ID", "Prepared voice did not return a profile id.")
        source, metadata = self._profile_source(profile_id)
        if source_metadata:
            metadata.setdefault("source", {}).update(source_metadata)
            self._write_json(source / "profile.json", metadata)
        if not personality_enabled:
            self.set_personality(profile_id, False)
        saved = self.accept(profile_id, "original", True, make_default)
        saved["source_selection"] = source_selection
        saved["timings_ms"] = dict(prepared.get("timings_ms") or {})
        if search_metadata_ms:
            saved["timings_ms"]["search_metadata"] = search_metadata_ms
        saved["refinements"] = (saved.get("profile") or {}).get("refinements", [])
        return saved

    def _run(self, stage: str, command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise VoiceWorkflowError(stage, "COMMAND_TIMEOUT", f"{stage.replace('_', ' ').title()} timed out.", retryable=True, choices=[{"action": "retry", "label": "Try again"}]) from exc
        if result.returncode:
            detail = (result.stderr or result.stdout or "Unknown command failure").strip()[-1200:]
            raise VoiceWorkflowError(stage, "SOURCE_ACCESS_FAILED" if stage in {"search", "download"} else "MEDIA_PROCESSING_FAILED", detail, retryable=True, choices=[{"action": "retry", "label": "Try again"}, {"action": "provide_url", "label": "Provide another source"}], diagnostics={"exit_code": result.returncode})
        return result

    def search(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        if not query.strip():
            raise VoiceWorkflowError("search", "QUERY_REQUIRED", "A voice search phrase is required.")
        if not Path(self.ytdlp).exists():
            raise VoiceWorkflowError("search", "YTDLP_UNAVAILABLE", "The isolated yt-dlp helper is not installed.", choices=[{"action": "provide_url", "label": "Provide a known source URL"}])
        attempts = self._search_queries(query)
        entries: list[dict[str, Any]] = []
        used_query = attempts[0]
        for used_query in attempts:
            result = self._run("search", [self.ytdlp, "--js-runtimes", "node:/usr/bin/node", "--dump-single-json", "--skip-download", f"ytsearch{self.SEARCH_LIMIT}:{used_query}"], 45)
            entries = [item for item in (json.loads(result.stdout).get("entries") or []) if item]
            if entries:
                break
        choices = []
        for rank, entry in enumerate(entries):
            url = str(entry.get("webpage_url") or entry.get("url") or "")
            if not url:
                continue
            flags = self._source_flags(entry)
            choices.append({"id": str(entry.get("id") or rank + 1), "label": str(entry.get("title") or "Untitled")[:160], "source_url": url, "duration_seconds": entry.get("duration"), "channel": entry.get("channel") or entry.get("uploader"), "confidence": self._source_confidence(entry, query, rank), "suitability": "reject" if flags else "review", "flags": flags})
        choices.sort(key=lambda item: (item["suitability"] != "reject", item["confidence"]), reverse=True)
        self._remember_recent_sources(choices, used_query)
        return {"status": "needs_choice", "stage": "source_selection", "summary": "Ranked source candidates. Explicit creation requests select and validate one automatically.", "query_used": used_query, "fallback_used": used_query != attempts[0], "choices": choices, "timings_ms": {"search_metadata": round((time.perf_counter() - started) * 1000)}}

    def _remember_recent_sources(self, choices: list[dict[str, Any]], query: str) -> None:
        """Persist bounded search provenance so a selected URL survives chat turns."""
        sources = [
            {
                "id": str(choice.get("id") or ""),
                "source_url": str(choice.get("source_url") or ""),
                "suitability": str(choice.get("suitability") or "review"),
            }
            for choice in choices[: self.SEARCH_LIMIT]
            if choice.get("source_url")
        ]
        self._write_json(self.recent_sources_file, {
            "created_at": int(time.time()),
            "query": query,
            "sources": sources,
        })

    def source_was_recently_searched(self, source_url: str) -> bool:
        """Allow only exact, unflagged URLs returned by a recent plugin search."""
        if not source_url:
            return False
        record = self._read_json(self.recent_sources_file)
        created_at = int(record.get("created_at") or 0)
        if created_at <= 0 or time.time() - created_at > self.RECENT_SOURCE_MAX_AGE_SECONDS:
            return False
        return any(
            item.get("source_url") == source_url and item.get("suitability") != "reject"
            for item in (record.get("sources") or [])
            if isinstance(item, dict)
        )

    @staticmethod
    def _search_queries(query: str) -> list[str]:
        compact = re.sub(r"\b(?:clean|clear|direct|official|solo|speech|voice|sample|audio|quality)\b", "", query, flags=re.I)
        compact = re.sub(r"\s+", " ", compact).strip()
        return [query] if not compact or compact.casefold() == query.casefold() else [query, compact]

    @staticmethod
    def _source_confidence(entry: dict[str, Any], query: str, rank: int) -> float:
        title = str(entry.get("title") or "").casefold()
        query_terms = [term for term in re.findall(r"[a-z0-9]{3,}", query.casefold()) if term not in {"voice", "clean", "sample", "audio"}]
        score = 0.65 - rank * 0.07 + min(0.2, 0.05 * sum(term in title for term in query_terms))
        duration = float(entry.get("duration") or 0)
        if 30 <= duration <= 900:
            score += 0.08
        if any(term in title for term in ("music", "song", "trailer", "film", "movie", "animation", "compilation", "reaction", "gameplay")):
            score -= 0.25
        return round(max(0.05, min(0.95, score)), 2)

    @staticmethod
    def _source_flags(entry: dict[str, Any]) -> list[str]:
        text = f"{entry.get('title') or ''} {entry.get('channel') or entry.get('uploader') or ''}".casefold()
        rules = {
            "outtake_or_rant": ("loses it", "meltdown", "profane", "rant", "rage", "uncensored", "blooper", "outtake"),
            "imitation": ("parody", "impression", "impersonation", "snl"),
            "compilation_or_music": ("montage", "compilation", "collection", "playlist", "sampler", "music", "song"),
            "commentary_or_review": ("problem with", "review", "reaction", "analysis", "breakdown", "explained", "why i", "my thoughts"),
            "objectionable_source": ("racist", "sexist", "hate speech", "slur"),
        }
        return [label for label, terms in rules.items() if any(term in text for term in terms)]

    def prepare(self, source_url: str, query: str, name: str, start_seconds: Any = None, prompt_text: str = "", profile_id: str = "") -> dict[str, Any]:
        if profile_id and not source_url:
            if not prompt_text.strip():
                raise VoiceWorkflowError("prompt_transcript", "TRANSCRIPT_REQUIRED", "An exact spoken transcript is required for CosyVoice.")
            return self._supply_verified_transcript(profile_id, prompt_text)
        lock = self.root / ".prepare.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            if time.time() - lock.stat().st_mtime <= self.PREPARE_LOCK_MAX_AGE_SECONDS:
                raise VoiceWorkflowError("prepare", "PREPARE_BUSY", "Another source is already being prepared.", retryable=True, choices=[{"action": "retry", "label": "Retry after preparation finishes"}])
            shutil.rmtree(lock, ignore_errors=True)
            lock.mkdir()
        try:
            return self._prepare_source(source_url, query, name, start_seconds, prompt_text)
        finally:
            shutil.rmtree(lock, ignore_errors=True)

    def _prepare_source(self, source_url: str, query: str, name: str, start_seconds: Any, prompt_text: str) -> dict[str, Any]:
        started, timings = time.perf_counter(), {}
        if not source_url:
            return self.search(query) if query else (_ for _ in ()).throw(VoiceWorkflowError("prepare", "SOURCE_REQUIRED", "A selected source URL is required."))
        parsed = urlparse(source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise VoiceWorkflowError("prepare", "INVALID_SOURCE_URL", "The source must be an http(s) URL.")
        if (parsed.hostname or "").casefold() not in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}:
            raise VoiceWorkflowError("prepare", "UNSUPPORTED_SOURCE", "The automated source fetcher currently accepts YouTube URLs only.")
        candidate_id = f"{_slug(name)}-{uuid.uuid4().hex[:8]}"
        candidate_dir = self.candidates / candidate_id
        candidate_dir.mkdir(parents=True)
        source_wav = candidate_dir / "source.wav"
        raw_reference_wav = candidate_dir / "reference.raw.wav"
        reference_wav = candidate_dir / "reference.wav"
        start = max(0.0, float(start_seconds or 0.0))
        try:
            leg = time.perf_counter()
            source_seconds = self.PREFERRED_SOURCE_SECONDS if start_seconds is not None else self.MAX_SOURCE_SECONDS
            self._run("download", [self.ytdlp, "--js-runtimes", "node:/usr/bin/node", "--no-playlist", "--download-sections", f"*{start}-{start + source_seconds}", "--force-keyframes-at-cuts", "-x", "--audio-format", "wav", "--audio-quality", "0", "-o", str(candidate_dir / "download.%(ext)s"), source_url], 180)
            timings["download"] = round((time.perf_counter() - leg) * 1000)
            downloads = sorted(candidate_dir.glob("download*.wav"))
            if not downloads:
                raise VoiceWorkflowError("download", "AUDIO_NOT_CREATED", "The source downloaded but produced no WAV audio.")
            downloads[0].replace(source_wav)
            leg = time.perf_counter()
            segment = self._choose_segment(source_wav)
            timings["segment_scan"] = round((time.perf_counter() - leg) * 1000)
            leg = time.perf_counter()
            self._run("extract", ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(segment["start_seconds"]), "-t", str(self.TARGET_SECONDS), "-i", str(source_wav), "-ac", "1", "-ar", str(self.TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(raw_reference_wav)], 45)
            cleanup = self._clean_reference(raw_reference_wav, reference_wav)
            metrics = self.analyze_wav(reference_wav)
            self._validate_metrics(metrics)
            timings["extract_and_signal_validation"] = round((time.perf_counter() - leg) * 1000)
            source_wav.unlink(missing_ok=True)
        except VoiceWorkflowError as exc:
            timings["prepare_total"] = round((time.perf_counter() - started) * 1000)
            exc.diagnostics.setdefault("timings_ms", timings)
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise
        # CosyVoice needs a transcript for zero-shot conditioning. This private,
        # single-user deployment accepts the local transcript automatically.
        supplied_transcript = " ".join(prompt_text.split())
        draft_transcript, transcription_error = self._draft_transcript(reference_wav)
        transcript = supplied_transcript or draft_transcript
        if not transcript:
            shutil.rmtree(candidate_dir, ignore_errors=True)
            raise VoiceWorkflowError(
                "prompt_transcript", "TRANSCRIPTION_UNAVAILABLE",
                "The selected source passed signal checks but produced no usable local transcript.",
                retryable=True, diagnostics={"transcription_error": transcription_error, "timings_ms": timings},
            )
        transcript_source = "request" if supplied_transcript else "automatic_asr"
        validation = {"status": "signal_validated", "reason": "Deterministic signal checks passed; the transcript is accepted for conditioning but may require human correction.", **metrics}
        personality_label = _character_label(query, name)
        metadata = {"id": candidate_id, "name": personality_label, "status": "preview", "created_at": int(time.time()), "source": {"url": source_url, "segment_start": round(start + segment["start_seconds"], 2), "segment_duration": metrics["duration_seconds"], "reference_cleanup": cleanup}, "reference_wav": "reference.wav", "prompt_text": transcript, "transcript": {"text": transcript, "accepted": True, "verified": bool(supplied_transcript), "source": transcript_source, "error": transcription_error}, "cosyvoice": {"reference_wav": "reference.wav", "sample_rate": self.TARGET_SAMPLE_RATE}, "validation": validation, "refinements": self._refinements(metrics), "selected_style": "original", "delivery_prompt": "", "personality": {"enabled": True, "label": personality_label, "mode": "character", "paired_with_voice": True, "prompt": self._personality_prompt(personality_label)}}
        self._refresh_style_prompt(metadata)
        self._write_json(candidate_dir / "profile.json", metadata)
        self._set_state(session_profile=candidate_id, candidate=True)
        timings["prepare_total"] = round((time.perf_counter() - started) * 1000)
        metadata["timings_ms"] = timings
        self._write_json(candidate_dir / "profile.json", metadata)
        return {"status": "preview_ready", "stage": "automatic_transcript", "summary": "Signal checks passed and the local transcript was accepted automatically.", "profile": metadata, "timings_ms": timings}

    def _clean_reference(self, input_path: Path, output_path: Path) -> dict[str, Any]:
        """Run an optional isolated cleaner and safely retain the raw reference."""
        command_template = os.getenv("COSYVOICE_CLEAN_REFERENCE_COMMAND", "").strip()
        if not command_template:
            os.replace(input_path, output_path)
            return {"status": "not_configured", "applied": False}

        command = [
            part.replace("{input_path}", str(input_path)).replace("{output_path}", str(output_path))
            for part in shlex.split(command_template)
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
            if result.returncode != 0 or not output_path.is_file():
                detail = (result.stderr or result.stdout or "cleaner produced no output").strip()[-300:]
                raise RuntimeError(detail)
            cleaned_metrics = self.analyze_wav(output_path)
            self._validate_metrics(cleaned_metrics)
            try:
                cleaner_report = json.loads(result.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                cleaner_report = {}
            input_path.unlink(missing_ok=True)
            return {
                "status": "applied",
                "applied": True,
                "method": str(cleaner_report.get("method") or "external_reference_cleaner"),
                "pause_compaction_applied": bool(cleaner_report.get("pause_compaction_applied", False)),
                "metrics": cleaned_metrics,
            }
        except (OSError, RuntimeError, subprocess.TimeoutExpired, VoiceWorkflowError) as exc:
            output_path.unlink(missing_ok=True)
            os.replace(input_path, output_path)
            return {
                "status": "fallback_raw",
                "applied": False,
                "reason": str(exc)[:300],
            }

    def _supply_verified_transcript(self, profile_id: str, prompt_text: str) -> dict[str, Any]:
        source, metadata = self._profile_source(profile_id)
        transcript = " ".join(prompt_text.split())
        if not transcript:
            raise VoiceWorkflowError("prompt_transcript", "TRANSCRIPT_REQUIRED", "An exact spoken transcript is required for CosyVoice.")
        metadata["prompt_text"] = transcript
        metadata["transcript"] = {"text": transcript, "accepted": True, "verified": True, "source": "user", "error": None}
        metadata.setdefault("cosyvoice", {})["reference_wav"] = "reference.wav"
        metadata["cosyvoice"]["sample_rate"] = self.TARGET_SAMPLE_RATE
        self._write_json(source / "profile.json", metadata)
        self._set_state(session_profile=profile_id, candidate=source.parent == self.candidates)
        return {"status": "preview_ready", "stage": "transcript_updated", "summary": "The corrected transcript is recorded and ready to save.", "profile": metadata}

    @staticmethod
    def _draft_transcript(reference_wav: Path) -> tuple[str, str | None]:
        """Use a configured public adapter, then Hermes' optional local fallback."""
        command_template = os.getenv("COSYVOICE_TRANSCRIBE_COMMAND", "").strip()
        packaged_adapter = Path("/srv/cosyvoice2/app/transcribe.py")
        packaged_python = Path("/srv/cosyvoice2/venv/bin/python")
        if not command_template and packaged_adapter.is_file() and packaged_python.is_file():
            command_template = f"{packaged_python} {packaged_adapter} --input {{input_path}}"
        if command_template:
            try:
                command = [part.replace("{input_path}", str(reference_wav)) for part in shlex.split(command_template)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
                if result.returncode != 0:
                    return "", f"Configured transcription command failed: {result.stderr.strip()[:300]}"
                output = result.stdout.strip()
                try:
                    payload = json.loads(output)
                    output = str(payload.get("text") or payload.get("transcript") or "")
                except json.JSONDecodeError:
                    pass
                text = " ".join(output.split())
                return (text, None) if text else ("", "Configured transcription command returned no text.")
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                return "", f"Configured transcription command failed: {str(exc)[:300]}"
        try:
            from tools.transcription_tools import transcribe_audio_local_fallback
        except ImportError:
            return "", "Hermes local transcription helper is unavailable."
        try:
            result = transcribe_audio_local_fallback(str(reference_wav))
        except Exception as exc:
            return "", f"Local transcription failed: {str(exc)[:300]}"
        if isinstance(result, dict):
            text = result.get("text") or result.get("transcript") or ""
        else:
            text = getattr(result, "text", result if isinstance(result, str) else "")
        text = " ".join(str(text).split())
        return (text, None) if text else ("", "Local transcription returned no text.")

    def refine(self, profile_id: str, style_choice: str) -> dict[str, Any]:
        source, metadata = self._profile_source(profile_id)
        styles = {item["id"]: item for item in metadata.get("refinements", [])}
        if style_choice == "original":
            prompt = ""
        elif style_choice in styles:
            prompt = styles[style_choice]["prompt"]
        else:
            return {"status": "needs_choice", "stage": "style_selection", "summary": "Choose one of the clip-specific refinements.", "choices": [{"id": "original", "label": "Original delivery"}, *styles.values()], "recommended": "original"}
        metadata["selected_style"], metadata["delivery_prompt"] = style_choice, prompt
        self._refresh_style_prompt(metadata)
        self._write_json(source / "profile.json", metadata)
        self._set_state(session_profile=profile_id, candidate=source.parent == self.candidates)
        return {"status": "preview_ready", "stage": "style_preview", "summary": "The refinement is selected for the next CosyVoice synthesis request.", "profile": metadata}

    def set_personality(self, profile_id: str, enabled: bool, personality_prompt: str = "") -> dict[str, Any]:
        source, metadata = self._profile_source(profile_id)
        personality = metadata.setdefault("personality", {})
        personality["enabled"] = enabled
        custom_prompt = " ".join(personality_prompt.split()).strip()
        if custom_prompt:
            if len(custom_prompt) > 2000:
                raise VoiceWorkflowError(
                    "personality", "PERSONALITY_PROMPT_TOO_LONG",
                    "The profile personality prompt must be 2,000 characters or fewer.",
                )
            personality["prompt"] = custom_prompt
            personality["prompt_source"] = "custom"
        self._refresh_style_prompt(metadata)
        self._write_json(source / "profile.json", metadata)
        return {"status": "ok", "stage": "personality", "summary": "Voice-associated mannerisms " + ("enabled." if enabled else "disabled."), "profile": metadata}

    @staticmethod
    def _personality_prompt(label: str) -> str:
        return (
            f"Use {label} as the sole presentation persona. Use its conversational temperament, cadence, "
            "vocabulary, values, and restrained occasional signature phrasing. Let those traits shape "
            "the response naturally while answering the user directly. Never announce, label, or explain "
            f"the persona, and never begin with phrases such as 'As {label}'. Do not turn routine answers "
            "into speeches or add theatrical grandeur where it does not fit. Use recognizable catchphrases "
            "sparingly, only when they fit the conversation naturally. Jarvis remains the operational role "
            "and name, not a second presentation style; do not blend in another assistant or character "
            "persona. Preserve factual accuracy, tool discipline, and safety behavior."
        )

    @staticmethod
    def _refresh_style_prompt(metadata: dict[str, Any]) -> None:
        parts = [str(metadata.get("delivery_prompt") or "").strip()]
        personality = metadata.get("personality") or {}
        if personality.get("enabled", True):
            parts.append(str(personality.get("prompt") or "").strip())
        metadata["style_prompt"] = " ".join(part for part in parts if part)

    def personality_context(self, profile_id: str = "") -> dict[str, Any] | None:
        if not profile_id:
            # A prepared candidate can be previewed, but is not an active voice
            # or personality layer until accept/set_default saves it.
            profile_id = str(self._selection_report()["selected_profile_id"] or "")
        if not profile_id:
            return None
        try:
            _, metadata = self._profile_source(profile_id)
        except VoiceWorkflowError:
            return None
        personality = metadata.get("personality") or {}
        return {"profile_id": profile_id, "prompt": personality.get("prompt", "")} if personality.get("enabled", True) and personality.get("prompt") else None

    def accept(self, profile_id: str, style_choice: str, rights_confirmed: bool, make_default: bool) -> dict[str, Any]:
        # Retained in the Python signature for compatibility with older callers.
        # Authentication plus the explicit request is the authorization boundary.
        del rights_confirmed
        source, metadata = self._profile_source(profile_id)
        transcript = metadata.get("transcript") or {}
        transcript_accepted = transcript.get("accepted", transcript.get("verified", False))
        if not str(metadata.get("prompt_text") or "").strip() or not transcript_accepted:
            return {
                "status": "needs_choice", "stage": "prompt_transcript",
                "summary": "CosyVoice cannot save this profile until usable conditioning text is available.",
                "choices": [{"id": "prepare_with_prompt_text", "label": "Provide or correct the transcript for this prepared profile"}],
                "selection": self._selection_report(),
            }
        if style_choice != metadata.get("selected_style", "original"):
            self.refine(profile_id, style_choice)
            source, metadata = self._profile_source(profile_id)
        metadata["status"] = "saved"
        destination = self.profiles / profile_id
        if source != destination:
            if destination.exists():
                shutil.rmtree(destination)
            source.replace(destination)
        self._write_json(destination / "profile.json", metadata)
        self._set_state(session_profile=profile_id, candidate=False, default_profile=profile_id if make_default else None, preserve_default=not make_default)
        state = self._read_json(self.state_file)
        selection = self._selection_report()
        persistence_verified = (
            state.get("session_profile") == profile_id
            and not state.get("session_candidate")
            and (not make_default or state.get("default_profile") == profile_id)
            and selection.get("selected_profile_id") == profile_id
        )
        return {
            "status": "saved", "stage": "complete",
            "summary": f"Saved {metadata['name']}" + (" and made it the default voice." if make_default else " for the current session."),
            "profile": metadata, "default": make_default, "selection": selection,
            "persistence_verified": persistence_verified,
        }

    def status(self) -> dict[str, Any]:
        selection = self._selection_report()
        return {
            "status": "ok", "stage": "status", "state": self._read_json(self.state_file),
            "selection": selection, "profiles": self._profile_summaries(),
            "integration": "CosyVoice TTS must resolve `selection.selected_profile_id`; `is_fallback=true` means no saved selection is available.",
        }

    def list_profiles(self) -> dict[str, Any]:
        profiles = self._profile_summaries()
        return {"status": "ok", "stage": "list", "summary": f"{len(profiles)} saved CosyVoice profiles.", "profiles": profiles}

    def _profile_summaries(self) -> list[dict[str, Any]]:
        state = self._read_json(self.state_file)
        result = []
        for path in sorted(self.profiles.glob("*/profile.json")):
            data = self._read_json(path)
            personality = data.get("personality") or {}
            source = data.get("source") or {}
            profile_id = data.get("id")
            result.append({
                "id": profile_id,
                "name": data.get("name"),
                "personality_label": personality.get("label"),
                "personality_enabled": bool(personality.get("enabled", True)),
                "personality_mode": personality.get("mode"),
                "paired_with_voice": bool(personality.get("paired_with_voice", False)),
                "selected_style": data.get("selected_style"),
                "created_at": data.get("created_at"),
                "prompt_text_accepted": bool((data.get("transcript") or {}).get("accepted", True)),
                "prompt_text_verified": bool((data.get("transcript") or {}).get("verified")),
                "source_url": source.get("url"),
                "source_title": source.get("title"),
                "is_session": profile_id == state.get("session_profile") and not state.get("session_candidate"),
                "is_default": profile_id == state.get("default_profile"),
            })
        result.sort(
            key=lambda item: (
                bool(item.get("is_session")),
                bool(item.get("is_default")),
                int(item.get("created_at") or 0),
            ),
            reverse=True,
        )
        return result

    @staticmethod
    def _identity_key(value: str) -> str:
        words = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
        ignored = {"voice", "profile", "clone", "character"}
        return " ".join(word for word in words if word not in ignored)

    @staticmethod
    def _source_key(value: str) -> str:
        try:
            parsed = urlparse(value)
            host = parsed.netloc.casefold().removeprefix("www.").removeprefix("m.")
            if host == "youtu.be":
                return f"youtube:{parsed.path.strip('/').split('/')[0]}"
            if host.endswith("youtube.com"):
                query = dict(item.split("=", 1) for item in parsed.query.split("&") if "=" in item)
                if query.get("v"):
                    return f"youtube:{query['v']}"
            return value.strip().casefold().rstrip("/")
        except (TypeError, ValueError):
            return value.strip().casefold().rstrip("/")

    def _find_duplicate_profile(self, query: str, name: str, source_url: str) -> dict[str, Any] | None:
        summaries = self._profile_summaries()
        if source_url:
            source_key = self._source_key(source_url)
            for profile in summaries:
                if profile.get("source_url") and self._source_key(str(profile["source_url"])) == source_key:
                    return {"reason": "same_source", "profile": profile}
        requested = {self._identity_key(query), self._identity_key(name)} - {""}
        for profile in summaries:
            existing = {
                self._identity_key(str(profile.get("name") or "")),
                self._identity_key(str(profile.get("personality_label") or "")),
            } - {""}
            if requested & existing:
                return {"reason": "same_identity", "profile": profile}
        return None

    def reset(self) -> dict[str, Any]:
        with self._state_lock():
            state = self._read_json(self.state_file)
            state.update({"session_profile": None, "session_candidate": False, "updated_at": int(time.time())})
            self._write_json(self.state_file, state)
        return {"status": "ok", "stage": "reset", "summary": "Session voice cleared; the saved default remains selected.", "selection": self._selection_report()}

    def discard(self, profile_id: str) -> dict[str, Any]:
        target = self.candidates / profile_id
        if not target.exists():
            raise VoiceWorkflowError("discard", "CANDIDATE_NOT_FOUND", f"Prepared candidate not found: {profile_id}")
        shutil.rmtree(target)
        with self._state_lock():
            state = self._read_json(self.state_file)
            if state.get("session_profile") == profile_id:
                state.update({"session_profile": None, "session_candidate": False, "updated_at": int(time.time())})
                self._write_json(self.state_file, state)
        return {"status": "ok", "stage": "discard", "summary": "Prepared CosyVoice candidate was discarded.", "selection": self._selection_report()}

    def workflow_context(self) -> dict[str, Any]:
        """Return restart-safe routing facts without treating a candidate as active."""
        selection = self._selection_report()
        state = self._read_json(self.state_file)
        candidate_id = str(state.get("session_profile") or "") if state.get("session_candidate") else ""
        if candidate_id and not self._profile_exists(candidate_id, self.candidates):
            candidate_id = ""
        return {
            "selection": selection,
            "candidate_profile_id": candidate_id or None,
            "actionable_profile_id": candidate_id or selection["selected_profile_id"],
        }

    def _selection_report(self) -> dict[str, Any]:
        """Describe the saved selection independently of a TTS adapter.

        `selected_profile_id` is intentionally null for a prepared candidate;
        callers can distinguish a real saved selection from backend fallback.
        """
        state = self._read_json(self.state_file)
        session_id = str(state.get("session_profile") or "")
        default_id = str(state.get("default_profile") or "")
        if session_id and not state.get("session_candidate") and self._profile_exists(session_id, self.profiles):
            scope = "session+default" if session_id == default_id else "session"
            return {"selected_profile_id": session_id, "scope": scope, "is_fallback": False, "fallback_reason": None}
        if default_id and self._profile_exists(default_id, self.profiles):
            return {"selected_profile_id": default_id, "scope": "default", "is_fallback": False, "fallback_reason": None}
        reason = "candidate_requires_acceptance" if state.get("session_candidate") else "no_saved_selection"
        return {"selected_profile_id": None, "scope": "fallback", "is_fallback": True, "fallback_reason": reason}

    @classmethod
    def _profile_exists(cls, profile_id: str, parent: Path) -> bool:
        """Match the runtime's minimum complete-profile contract."""
        if not profile_id or not _PROFILE_ID.fullmatch(profile_id):
            return False
        directory = parent / profile_id
        metadata = cls._read_json(directory / "profile.json")
        prompt_text = str(
            metadata.get("prompt_text")
            or (metadata.get("transcript") or {}).get("text")
            or ""
        ).strip()
        return (directory / "reference.wav").is_file() and bool(prompt_text)

    def _profile_source(self, profile_id: str) -> tuple[Path, dict[str, Any]]:
        if not profile_id:
            raise VoiceWorkflowError("profile", "PROFILE_REQUIRED", "A prepared or saved profile id is required.")
        if not _PROFILE_ID.fullmatch(profile_id):
            raise VoiceWorkflowError("profile", "INVALID_PROFILE_ID", "The voice profile id is invalid.")
        for parent in (self.candidates, self.profiles):
            source = parent / profile_id
            if source.is_dir():
                metadata = self._read_json(source / "profile.json")
                if metadata:
                    return source, metadata
        raise VoiceWorkflowError("profile", "PROFILE_NOT_FOUND", f"CosyVoice profile not found: {profile_id}")

    def _set_state(self, *, session_profile: str, candidate: bool = False, default_profile: str | None = None, preserve_default: bool = True) -> None:
        with self._state_lock():
            state = self._read_json(self.state_file)
            state.update({"session_profile": session_profile, "session_candidate": candidate, "updated_at": int(time.time())})
            if not preserve_default or default_profile is not None:
                state["default_profile"] = default_profile
            self._write_json(self.state_file, state)

    def _choose_segment(self, path: Path) -> dict[str, Any]:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise VoiceWorkflowError("segment_scan", "UNSUPPORTED_PCM", "The downloaded WAV is not 16-bit PCM.")
            rate, frames, channels = wav.getframerate(), wav.getnframes(), wav.getnchannels()
            raw = array("h", wav.readframes(frames))
        if not raw or rate <= 0:
            raise VoiceWorkflowError("segment_scan", "EMPTY_AUDIO", "The downloaded source contains no readable audio.")
        # WAV samples are interleaved by channel. Scan frame-aligned mono data
        # so a 12-second window and its offset mean the same thing to ffmpeg.
        if channels > 1:
            raw = array("h", (
                sum(raw[index:index + channels]) // channels
                for index in range(0, len(raw) - channels + 1, channels)
            ))
        target = min(self.TARGET_SECONDS * rate, len(raw))
        if target < rate * self.MIN_REFERENCE_SECONDS:
            raise VoiceWorkflowError(
                "segment_scan", "SOURCE_TOO_SHORT",
                f"The source has less than {self.MIN_REFERENCE_SECONDS} seconds of usable audio.",
            )
        step = max(rate, target // 2)
        best = {"start_seconds": 0.0, "score": -1.0, "metrics": {}}
        for start in range(0, max(1, len(raw) - target + 1), step):
            window = raw[start:start + target]
            metrics = self._samples_metrics(window)
            score = metrics["voiced_ratio"] * (1.0 - min(0.8, metrics["clipping_ratio"] * 4)) * min(1.0, metrics["rms"] / 5000)
            if score > best["score"]:
                best = {
                    "start_seconds": round(start / rate, 2),
                    "score": round(score, 3),
                    "metrics": metrics,
                }
        metrics = best["metrics"]
        if (
            metrics.get("rms", 0) < 250
            or metrics.get("voiced_ratio", 0) < 0.12
            or metrics.get("clipping_ratio", 1) > 0.05
        ):
            raise VoiceWorkflowError("segment_scan", "NO_SPEECH_LIKE_SEGMENT", "No sufficiently voiced segment was found.", choices=[{"action": "try_next_video", "label": "Use another source"}, {"action": "provide_offset", "label": "Choose an offset"}])
        return best

    @staticmethod
    def _samples_metrics(samples: array) -> dict[str, float]:
        if not samples:
            return {"rms": 0.0, "clipping_ratio": 1.0, "voiced_ratio": 0.0}
        rms = math.sqrt(sum(value * value for value in samples) / len(samples))
        clipping = sum(abs(value) >= 32000 for value in samples) / len(samples)
        voiced = sum(abs(value) >= 450 for value in samples) / len(samples)
        return {"rms": round(rms, 2), "clipping_ratio": round(clipping, 4), "voiced_ratio": round(voiced, 4)}

    @classmethod
    def analyze_wav(cls, path: Path) -> dict[str, float]:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise VoiceWorkflowError("signal_validation", "NORMALIZATION_FAILED", "Reference audio must be mono 16-bit PCM.")
            samples = array("h", wav.readframes(wav.getnframes()))
            metrics = cls._samples_metrics(samples)
            metrics.update({"sample_rate": wav.getframerate(), "duration_seconds": round(wav.getnframes() / wav.getframerate(), 2)})
            return metrics

    @staticmethod
    def _validate_metrics(metrics: dict[str, float]) -> None:
        if (
            metrics["duration_seconds"] < VoiceManager.MIN_REFERENCE_SECONDS
            or metrics["duration_seconds"] > VoiceManager.MAX_REFERENCE_SECONDS
            or metrics["rms"] < 250
            or metrics["voiced_ratio"] < 0.12
            or metrics["clipping_ratio"] > 0.05
        ):
            raise VoiceWorkflowError("signal_validation", "REFERENCE_QUALITY_FAILED", "The reference audio failed deterministic signal checks.", choices=[{"action": "try_next_segment", "label": "Try another segment"}, {"action": "try_next_video", "label": "Use another source"}])

    @staticmethod
    def _refinements(metrics: dict[str, float]) -> list[dict[str, str]]:
        if metrics["rms"] < 2500:
            return [{"id": "clearer", "label": "Clearer and more projected", "prompt": "Speak with slightly clearer projection while keeping the reference voice."}, {"id": "warmer", "label": "Warmer and more intimate", "prompt": "Use a warm, intimate delivery while keeping the reference voice."}]
        return [{"id": "calmer", "label": "Calmer and more measured", "prompt": "Speak a little slower and more calmly while keeping the reference voice."}, {"id": "expressive", "label": "More expressive", "prompt": "Use slightly more expressive phrasing while keeping the reference voice."}]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
