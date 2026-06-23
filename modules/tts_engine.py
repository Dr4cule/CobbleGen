from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import subprocess
import unicodedata
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import edge_tts
import requests

from config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_BASE_URL,
    ELEVENLABS_MAX_CHARS,
    ELEVENLABS_MODEL,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_SIMILARITY,
    ELEVENLABS_SPEAKER_BOOST,
    ELEVENLABS_STABILITY,
    ELEVENLABS_STYLE,
    ELEVENLABS_VOICE_ID,
    ENABLE_TTS_FALLBACK,
    NVIDIA_API_KEY,
    NVIDIA_DEFAULT_EMOTION,
    NVIDIA_DYNAMIC_EMOTION,
    NVIDIA_EMOTION_MODE,
    NVIDIA_MAGPIE_FUNCTION_ID,
    NVIDIA_RIVA_SERVER,
    NVIDIA_RIVA_USE_SSL,
    NVIDIA_TTS_CHUNK_CROSSFADE_MS,
    NVIDIA_TTS_GRPC_MAX_MESSAGE_MB,
    NVIDIA_TTS_LANGUAGE_CODE,
    NVIDIA_TTS_MAX_ATTEMPTS,
    NVIDIA_TTS_MAX_CHARS,
    NVIDIA_TTS_NORMALIZE_CHUNKS,
    NVIDIA_TTS_SAMPLE_RATE_HZ,
    NVIDIA_TTS_TARGET_LUFS,
    NVIDIA_TTS_VOICE,
    PARLER_EMOTION,
    PARLER_CHUNK_CROSSFADE_MS,
    PARLER_LOCAL_CHUNKING_ENABLED,
    PARLER_LOCAL_CHUNK_CHARS,
    PARLER_MAX_ATTEMPTS,
    PARLER_MAX_NEW_TOKENS,
    PARLER_MIN_NEW_TOKENS,
    PARLER_SPEAKER_ID,
    PARLER_TEMPERATURE,
    PARLER_TOKEN_MARGIN,
    PARLER_TOKENS_PER_SECOND,
    PARLER_TTS_URL,
    PARLER_VOICE_DESCRIPTION,
    STORY_TARGET_WPM,
    TEMP_DIR,
    TTS_AUDIO_QUALITY_GUARD,
    TTS_FALLBACK_BACKEND,
    TTS_BACKEND,
    TTS_REQUEST_TIMEOUT,
    TTS_STATIC_ENERGY_COV_MAX,
    TTS_STATIC_MAX_SECONDS,
    TTS_STATIC_RMS_DB_THRESHOLD,
    TTS_STATIC_ZCR_THRESHOLD,
    TTS_VOICE,
    USE_WHISPER_ALIGN,
    VOX_PROMPT_CHANNELS,
    VOX_PROMPT_SAMPLE_RATE,
    VOX_SILENCE_PAD_MS,
    VOX_SILENCE_THRESHOLD,
    VOX_CPM_MAX_CHARS,
    VOX_CPM_PROMPT_TEXT,
    VOX_CPM_PROMPT_WAV,
    VOX_CPM_SPEAKER,
    VOX_CPM_URL,
    WHISPER_MODEL,
)


LOGGER = logging.getLogger(__name__)


def _run_command(command: list[str], context: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        input=input_text,
        check=False,
    )
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-2000:]
        LOGGER.error("%s failed: %s", context, stderr_tail)
        raise RuntimeError(f"{context} failed with exit code {completed.returncode}")
    return completed


def _strip_punctuation_token(token: str) -> str:
    return re.sub(r"^\W+|\W+$", "", token).strip()


def _normalise_tts_text(text: str) -> str:
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "—": ", ",
        "–": ", ",
        "…": "...",
        "&": " and ",
    }
    cleaned = text.replace("\ufeff", "")
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = "".join(char for char in cleaned if not unicodedata.combining(char))
    pronunciation_replacements = {
        r"\bAITA\b": "Am I the asshole",
        r"\bAITAH\b": "Am I the asshole",
        r"\bTIFU\b": "Today I messed up",
        r"\bTLDR\b": "Too long, didn't read",
        r"\bTL;DR\b": "Too long, didn't read",
        r"\bOP\b": "original poster",
        r"\bMIL\b": "mother in law",
        r"\bFIL\b": "father in law",
        r"\bSIL\b": "sister in law",
        r"\bBIL\b": "brother in law",
        r"\bD\.C\.(?=\W|$)": "D C",
        r"\bUS\b": "U S",
        r"\bUSA\b": "U S A",
        r"\bAI\b": "A I",
    }
    for pattern, replacement in pronunciation_replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(r"\b(\d+)\s*-\s*year\s*-\s*old\b", r"\1 year old", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalise_word_sequence(words: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    cleaned: list[dict[str, float | str]] = []
    for item in words:
        token = str(item["word"]).strip()
        if not token:
            continue
        start = float(item["start"])
        end = max(float(item["end"]), start)
        cleaned.append({"word": token, "start": round(start, 3), "end": round(end, 3)})

    for index, item in enumerate(cleaned):
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            if index + 1 < len(cleaned):
                item["end"] = float(cleaned[index + 1]["start"])
            else:
                item["end"] = round(start + 0.18, 3)
        if float(item["end"]) <= start:
            item["end"] = round(start + 0.18, 3)
    return cleaned


def _nvidia_available_voice_names(service: Any) -> list[str]:
    import riva.client

    config_response = service.stub.GetRivaSynthesisConfig(
        riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
    )

    voice_names: list[str] = []
    for model_config in config_response.model_config:
        parameters = model_config.parameters
        voice_name = str(parameters.get("voice_name") or "").strip()
        if not voice_name:
            continue

        subvoices = [voice.split(":")[0].strip() for voice in str(parameters.get("subvoices", "")).split(",")]
        appended_subvoice = False
        if subvoices:
            for subvoice in subvoices:
                if subvoice:
                    voice_names.append(f"{voice_name}.{subvoice}")
                    appended_subvoice = True

        if appended_subvoice:
            continue

        voice_names.append(voice_name)

    return voice_names


def _resolve_nvidia_voice_name(service: Any, preferred_voice: str) -> str:
    preferred = preferred_voice.strip()
    if not preferred:
        return preferred_voice

    available_voices = _nvidia_available_voice_names(service)
    if not available_voices:
        return preferred_voice

    preferred_lower = preferred.lower()
    preferred_suffix = preferred_lower.split(".")[-1]
    available_map = {voice.lower(): voice for voice in available_voices}

    if preferred_lower in available_map:
        return available_map[preferred_lower]

    for voice in available_voices:
        voice_lower = voice.lower()
        if voice_lower.endswith(f".{preferred_lower}") or voice_lower.endswith(f".{preferred_suffix}"):
            return voice

    if "houzhen" in preferred_suffix:
        for voice in available_voices:
            voice_lower = voice.lower()
            if "hi-in" in voice_lower and "houzhen" in voice_lower:
                return voice
        for voice in available_voices:
            if "houzhen" in voice.lower():
                return voice

    return preferred_voice


def _nvidia_word_timestamps_from_meta(meta: Any, audio_path: Path) -> list[dict[str, float | str]]:
    processed_text = str(getattr(meta, "processed_text", "") or "").strip()
    predicted_durations = [float(duration) for duration in getattr(meta, "predicted_durations", []) if duration is not None]

    if not processed_text or not predicted_durations:
        return []

    tokens = [token for token in processed_text.split() if token.strip()]
    if len(tokens) != len(predicted_durations):
        tokens = [token for token in _normalise_tts_text(processed_text).split() if token.strip()]
        if len(tokens) != len(predicted_durations):
            return []

    total_predicted_duration = sum(max(duration, 0.0) for duration in predicted_durations)
    if total_predicted_duration <= 0:
        return []

    actual_duration = get_audio_duration(audio_path)
    if actual_duration <= 0:
        actual_duration = total_predicted_duration

    duration_scale = actual_duration / total_predicted_duration if total_predicted_duration > 0 else 1.0
    word_timestamps: list[dict[str, float | str]] = []
    current_start = 0.0

    for token, predicted_duration in zip(tokens, predicted_durations):
        scaled_duration = max(float(predicted_duration), 0.0) * duration_scale
        if scaled_duration <= 0:
            scaled_duration = actual_duration / max(1, len(tokens))

        current_end = current_start + scaled_duration
        word_timestamps.append(
            {
                "word": token,
                "start": round(current_start, 3),
                "end": round(current_end, 3),
            }
        )
        current_start = current_end

    if word_timestamps:
        word_timestamps[-1]["end"] = round(actual_duration, 3)
    return _normalise_word_sequence(word_timestamps)


def _resolve_api_url(base_url: str, path: str) -> str:
    base_parts = urlsplit(base_url)
    origin = f"{base_parts.scheme}://{base_parts.netloc}" if base_parts.scheme and base_parts.netloc else base_url.rstrip("/")
    return f"{origin}{path}"


def _download_audio_file(audio_url: str, output_path: Path) -> Path:
    response = requests.get(audio_url, timeout=TTS_REQUEST_TIMEOUT)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def _extract_timestamps_from_payload(payload: dict[str, Any]) -> list[dict[str, float | str]]:
    word_timestamps = payload.get("word_timestamps", [])
    if not isinstance(word_timestamps, list):
        return []
    return _normalise_word_sequence(
        [
            {
                "word": str(item.get("word", "")).strip(),
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", item.get("start", 0.0))),
            }
            for item in word_timestamps
            if isinstance(item, dict)
        ]
    )


def _download_api_audio(payload: dict[str, Any], api_url: str, output_path: Path) -> Path:
    audio_url = str(payload.get("audio_url") or "").strip()
    if not audio_url:
        raise RuntimeError(f"TTS response did not include audio_url: {json.dumps(payload)[:1000]}")

    if audio_url.startswith("/"):
        audio_url = _resolve_api_url(api_url, audio_url)

    return _download_audio_file(audio_url, output_path)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def _estimate_spoken_duration(text: str) -> float:
    words = _word_count(text)
    return (words / max(STORY_TARGET_WPM, 1)) * 60 if words else 0.0


def _parler_tokens_for_text(text: str) -> int:
    expected_seconds = max(_estimate_spoken_duration(text), 1.0)
    estimated_tokens = math.ceil(expected_seconds * PARLER_TOKENS_PER_SECOND * PARLER_TOKEN_MARGIN)
    return max(PARLER_MIN_NEW_TOKENS, min(PARLER_MAX_NEW_TOKENS, estimated_tokens))


def _audio_health_report(audio_path: Path, text: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "healthy": True,
        "score": 0.0,
        "warnings": [],
        "duration": 0.0,
        "expected_duration": round(_estimate_spoken_duration(text), 3),
    }
    if not TTS_AUDIO_QUALITY_GUARD:
        return report

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        report["healthy"] = False
        report["score"] = 100.0
        report["warnings"].append(f"unreadable wav: {exc}")
        return report

    duration = frame_count / sample_rate if sample_rate else 0.0
    expected_duration = float(report["expected_duration"])
    report["duration"] = round(duration, 3)

    if sample_width != 2 or not raw_frames or duration <= 0:
        report["healthy"] = False
        report["score"] = 100.0
        report["warnings"].append("empty or unsupported wav")
        return report

    import array

    samples = array.array("h")
    samples.frombytes(raw_frames)
    if channel_count > 1:
        samples = array.array("h", samples[::channel_count])

    window_size = max(1, sample_rate)
    static_seconds = 0
    static_streak = 0
    max_static_streak = 0
    active_windows = 0
    total_windows = 0
    overall_peak = 0

    sub_frame_size = max(1, sample_rate // 50)  # ~20ms energy frames

    for start in range(0, len(samples), window_size):
        segment = samples[start : start + window_size]
        if not segment:
            continue
        total_windows += 1
        peak = max(abs(sample) for sample in segment)
        overall_peak = max(overall_peak, peak)
        rms = math.sqrt(sum(sample * sample for sample in segment) / len(segment))
        rms_db = 20 * math.log10(max(rms, 1.0) / 32768)
        peak_db = 20 * math.log10(max(peak, 1) / 32768)
        zero_crossing_rate = sum(
            1
            for left, right in zip(segment, segment[1:])
            if (left < 0 <= right) or (left >= 0 > right)
        ) / max(1, len(segment) - 1)

        if rms_db > -45.0 or peak_db > -22.0:
            active_windows += 1

        # Energy modulation across ~20ms sub-frames. True static (white noise /
        # buzz) holds a near-constant energy, so its coefficient of variation is
        # tiny. Speech bursts loud on vowels and drops between words, keeping the
        # CoV high. This is what separates genuine static from loud, consonant-
        # heavy narration that merely has a high zero-crossing rate.
        sub_rms_values: list[float] = []
        for sub_start in range(0, len(segment), sub_frame_size):
            sub = segment[sub_start : sub_start + sub_frame_size]
            if not sub:
                continue
            sub_rms_values.append(math.sqrt(sum(s * s for s in sub) / len(sub)))
        if sub_rms_values:
            mean_rms = sum(sub_rms_values) / len(sub_rms_values)
            if mean_rms > 0:
                variance = sum((value - mean_rms) ** 2 for value in sub_rms_values) / len(sub_rms_values)
                energy_cov = math.sqrt(variance) / mean_rms
            else:
                energy_cov = 1.0
        else:
            energy_cov = 1.0

        is_static = (
            zero_crossing_rate >= TTS_STATIC_ZCR_THRESHOLD
            and rms_db >= TTS_STATIC_RMS_DB_THRESHOLD
            and energy_cov <= TTS_STATIC_ENERGY_COV_MAX
        )
        if is_static:
            static_seconds += 1
            static_streak += 1
            max_static_streak = max(max_static_streak, static_streak)
        else:
            static_streak = 0

    active_ratio = active_windows / total_windows if total_windows else 0.0
    peak_db = 20 * math.log10(max(overall_peak, 1) / 32768)
    report.update(
        {
            "static_seconds": static_seconds,
            "max_static_streak": max_static_streak,
            "active_ratio": round(active_ratio, 3),
            "peak_db": round(peak_db, 2),
        }
    )

    score = 0.0
    if max_static_streak > TTS_STATIC_MAX_SECONDS:
        report["warnings"].append(f"sustained static-like audio for {max_static_streak}s")
        score += 40.0 + max_static_streak

    if expected_duration >= 2.0:
        if duration > max(expected_duration * 2.4, expected_duration + 12.0):
            report["warnings"].append(
                f"duration {duration:.1f}s is too long for expected {expected_duration:.1f}s"
            )
            score += 20.0 + min(duration - expected_duration, 30.0)
        if duration < max(expected_duration * 0.35, 1.0):
            report["warnings"].append(
                f"duration {duration:.1f}s is too short for expected {expected_duration:.1f}s"
            )
            score += 25.0

    if duration > 2.0 and active_ratio < 0.05 and peak_db < -28.0:
        report["warnings"].append("audio is mostly silence or extremely low-level signal")
        score += 30.0

    report["score"] = round(score, 3)
    report["healthy"] = score == 0.0
    return report


def _report_is_salvageable(report: dict[str, Any]) -> bool:
    """Decide whether unhealthy audio is still good enough to ship.

    A reel should only be aborted for genuinely broken audio: an unreadable or
    empty WAV, or a clip that is mostly silence. Soft issues -- a brief
    static-like blip, or a duration that merely runs long/short -- are tolerated
    when the signal is otherwise loud and full-length, so one borderline chunk
    never fails an entire generation.
    """
    if report.get("healthy"):
        return True

    warnings = " ".join(report.get("warnings", [])).lower()
    if "unreadable wav" in warnings or "empty or unsupported wav" in warnings:
        return False
    if "mostly silence" in warnings:
        return False

    # Real speech keeps the active ratio high and produces a peak well above the
    # noise floor; require both before accepting a flagged clip.
    active_ratio = float(report.get("active_ratio", 0.0))
    peak_db = float(report.get("peak_db", -120.0))
    duration = float(report.get("duration", 0.0))
    return active_ratio >= 0.4 and peak_db >= -20.0 and duration >= 1.0


def _request_parler_tts_audio(
    text: str,
    output_path: Path,
    temperature: float,
    max_new_tokens: int,
) -> tuple[Path, list[dict[str, float | str]], dict[str, Any]]:
    normalised_text = _normalise_tts_text(text)
    if not normalised_text:
        raise ValueError("Cannot generate speech for empty text.")

    form_data: dict[str, str] = {
        "text": normalised_text,
        "speaker_id": PARLER_SPEAKER_ID,
        "emotion": PARLER_EMOTION,
        "temperature": str(temperature),
        "max_new_tokens": str(max_new_tokens),
    }
    if PARLER_VOICE_DESCRIPTION.strip():
        form_data["voice_description"] = PARLER_VOICE_DESCRIPTION.strip()

    response = requests.post(PARLER_TTS_URL, data=form_data, timeout=TTS_REQUEST_TIMEOUT)
    if not response.ok:
        error_body = response.text[-1000:]
        raise RuntimeError(f"Parler narration request failed with {response.status_code}: {error_body}")

    payload = response.json()
    _download_api_audio(payload, PARLER_TTS_URL, output_path)
    return output_path, _extract_timestamps_from_payload(payload), payload


def _generate_parler_tts_audio(text: str, output_path: Path) -> tuple[Path, list[dict[str, float | str]]]:
    normalised_text = _normalise_tts_text(text)
    max_new_tokens = _parler_tokens_for_text(normalised_text)
    temperatures = []
    for value in (PARLER_TEMPERATURE, min(PARLER_TEMPERATURE, 0.4), 0.32):
        if value not in temperatures:
            temperatures.append(value)

    attempts: list[tuple[float, Path, list[dict[str, float | str]], dict[str, Any]]] = []
    max_attempts = max(1, PARLER_MAX_ATTEMPTS)

    for attempt_index in range(max_attempts):
        temperature = temperatures[min(attempt_index, len(temperatures) - 1)]
        attempt_path = output_path
        if attempt_index:
            attempt_path = output_path.with_name(f"{output_path.stem}_retry_{attempt_index}{output_path.suffix}")

        generated_path, timestamps, _payload = _request_parler_tts_audio(
            normalised_text,
            attempt_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        candidate_path = _trim_wave_silence(generated_path)
        report = _audio_health_report(candidate_path, normalised_text)
        attempts.append((float(report["score"]), candidate_path, timestamps, report))

        if report["healthy"]:
            if candidate_path != output_path:
                output_path.write_bytes(candidate_path.read_bytes())
                candidate_path = output_path
            return candidate_path, timestamps

        LOGGER.warning(
            "Parler chunk attempt %d/%d looked unhealthy: %s",
            attempt_index + 1,
            max_attempts,
            "; ".join(report.get("warnings", [])),
        )

    _score, best_path, best_timestamps, best_report = min(attempts, key=lambda item: item[0])
    if TTS_AUDIO_QUALITY_GUARD and not _report_is_salvageable(best_report):
        raise RuntimeError(f"Parler audio failed quality guard: {best_report}")

    LOGGER.warning("Using best available Parler chunk despite health warnings: %s", best_report)
    if best_path != output_path:
        output_path.write_bytes(best_path.read_bytes())
        best_path = output_path
    return best_path, best_timestamps


async def _generate_edge_tts_audio(text: str, output_path: Path) -> tuple[Path, list[dict[str, float | str]]]:
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
    word_timestamps: list[dict[str, float | str]] = []

    with output_path.open("wb") as audio_file:
        async for chunk in communicate.stream():
            chunk_type = chunk.get("type")
            if chunk_type == "audio":
                chunk_data = chunk.get("data")
                if chunk_data is not None:
                    audio_file.write(chunk_data)
            elif chunk_type == "WordBoundary":
                word = str(chunk.get("text", "")).strip()
                if not word:
                    continue
                start = float(chunk.get("offset", 0)) / 10_000_000
                duration = float(chunk.get("duration", 0)) / 10_000_000
                word_timestamps.append(
                    {
                        "word": word,
                        "start": start,
                        "end": start + duration if duration > 0 else start,
                    }
                )

    return output_path, _normalise_word_sequence(word_timestamps)


def _nvidia_synthesis_service() -> tuple[Any, Any]:
    api_key = (NVIDIA_API_KEY or "").strip()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY must be set for the NVIDIA Magpie TTS backend.")

    try:
        import riva.client
        from riva.client.proto.riva_audio_pb2 import AudioEncoding
    except ImportError as exc:
        raise ImportError("Install nvidia-riva-client to use the NVIDIA Magpie TTS backend.") from exc

    max_message_bytes = max(1, NVIDIA_TTS_GRPC_MAX_MESSAGE_MB) * 1024 * 1024
    auth = riva.client.Auth(
        use_ssl=NVIDIA_RIVA_USE_SSL,
        uri=NVIDIA_RIVA_SERVER,
        metadata_args=[
            ["function-id", NVIDIA_MAGPIE_FUNCTION_ID],
            ["authorization", f"Bearer {api_key}"],
        ],
        options=[
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        ],
    )
    return riva.client.SpeechSynthesisService(auth), AudioEncoding


def _write_linear_pcm_wav(audio_bytes: bytes, output_path: Path, sample_rate_hz: int) -> Path:
    if not audio_bytes:
        raise RuntimeError("NVIDIA Magpie returned an empty audio payload.")

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(audio_bytes)
    return output_path


def _request_nvidia_magpie_audio(
    service: Any,
    audio_encoding: Any,
    text: str,
    output_path: Path,
    voice_name: str,
) -> tuple[Path, Any]:
    response = service.synthesize(
        text=text,
        voice_name=voice_name,
        language_code=NVIDIA_TTS_LANGUAGE_CODE,
        sample_rate_hz=NVIDIA_TTS_SAMPLE_RATE_HZ,
        encoding=audio_encoding.LINEAR_PCM,
    )
    return _write_linear_pcm_wav(response.audio, output_path, NVIDIA_TTS_SAMPLE_RATE_HZ), getattr(response, "meta", None)


def _generate_nvidia_magpie_audio(
    service: Any,
    audio_encoding: Any,
    text: str,
    output_path: Path,
    voice_name: str,
) -> tuple[Path, Any]:
    normalised_text = _normalise_tts_text(text)
    if not normalised_text:
        raise ValueError("Cannot generate speech for empty text.")

    attempts: list[tuple[float, Path, dict[str, Any], Any]] = []
    max_attempts = max(1, NVIDIA_TTS_MAX_ATTEMPTS)
    for attempt_index in range(max_attempts):
        attempt_path = output_path
        if attempt_index:
            attempt_path = output_path.with_name(f"{output_path.stem}_retry_{attempt_index}{output_path.suffix}")

        try:
            generated_path, response_meta = _request_nvidia_magpie_audio(
                service,
                audio_encoding,
                normalised_text,
                attempt_path,
                voice_name,
            )
            candidate_path = _trim_wave_silence(generated_path)
            report = _audio_health_report(candidate_path, normalised_text)
            attempts.append((float(report["score"]), candidate_path, report, response_meta))
            if report["healthy"]:
                if candidate_path != output_path:
                    output_path.write_bytes(candidate_path.read_bytes())
                    candidate_path = output_path
                return candidate_path, response_meta

            LOGGER.warning(
                "NVIDIA Magpie chunk attempt %d/%d looked unhealthy: %s",
                attempt_index + 1,
                max_attempts,
                "; ".join(report.get("warnings", [])),
            )
        except Exception as exc:
            LOGGER.warning(
                "NVIDIA Magpie chunk attempt %d/%d failed: %s",
                attempt_index + 1,
                max_attempts,
                exc,
            )
            if attempt_index + 1 == max_attempts and not attempts:
                raise

    _score, best_path, best_report, best_meta = min(attempts, key=lambda item: item[0])
    if TTS_AUDIO_QUALITY_GUARD and not _report_is_salvageable(best_report):
        raise RuntimeError(f"NVIDIA Magpie audio failed quality guard: {best_report}")

    LOGGER.warning("Using best available NVIDIA Magpie chunk despite health warnings: %s", best_report)
    if best_path != output_path:
        output_path.write_bytes(best_path.read_bytes())
        best_path = output_path
    return best_path, best_meta


def _trim_wave_silence(audio_path: Path) -> Path:
    with wave.open(str(audio_path), "rb") as wav_file:
        params = wav_file.getparams()
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw_frames = wav_file.readframes(frame_count)

    if sample_width != 2 or frame_count == 0:
        return audio_path

    import struct

    samples = struct.unpack("<" + ("h" * (len(raw_frames) // 2)), raw_frames)
    pad_frames = int(frame_rate * (VOX_SILENCE_PAD_MS / 1000))

    def _frame_peak(index: int) -> int:
        start = index * channel_count
        frame_samples = samples[start : start + channel_count]
        return max(abs(sample) for sample in frame_samples) if frame_samples else 0

    start_frame = 0
    while start_frame < frame_count and _frame_peak(start_frame) < VOX_SILENCE_THRESHOLD:
        start_frame += 1

    end_frame = frame_count - 1
    while end_frame > start_frame and _frame_peak(end_frame) < VOX_SILENCE_THRESHOLD:
        end_frame -= 1

    if start_frame == 0 and end_frame == frame_count - 1:
        return audio_path

    start_frame = max(0, start_frame - pad_frames)
    end_frame = min(frame_count, end_frame + pad_frames + 1)
    trimmed_frames = samples[start_frame * channel_count : end_frame * channel_count]
    if not trimmed_frames:
        return audio_path

    trimmed_path = audio_path.with_name(f"{audio_path.stem}_trimmed.wav")
    with wave.open(str(trimmed_path), "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(struct.pack("<" + ("h" * len(trimmed_frames)), *trimmed_frames))
    return trimmed_path


def _generate_f5_tts_audio(text: str, output_path: Path) -> tuple[Path, list[dict[str, float | str]]]:
    prompt_wav = Path(VOX_CPM_PROMPT_WAV).expanduser().resolve() if VOX_CPM_PROMPT_WAV else None
    normalised_text = _normalise_tts_text(text)
    if not prompt_wav:
        raise ValueError("VOX_CPM_PROMPT_WAV must be configured for the F5-TTS API.")
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Configured VOX_CPM_PROMPT_WAV does not exist: {prompt_wav}")
    if not VOX_CPM_PROMPT_TEXT.strip():
        raise ValueError("VOX_CPM_PROMPT_TEXT must be configured for the F5-TTS API.")

    with prompt_wav.open("rb") as prompt_file:
        files = {
            "prompt_wav": (prompt_wav.name, prompt_file, "audio/wav"),
        }
        response = requests.post(VOX_CPM_URL, data={"text": normalised_text}, files=files, timeout=TTS_REQUEST_TIMEOUT)

    if not response.ok:
        error_body = response.text[-1000:]
        raise RuntimeError(f"F5-TTS request failed with {response.status_code}: {error_body}")

    payload = response.json()
    _download_api_audio(payload, VOX_CPM_URL, output_path)
    return output_path, _extract_timestamps_from_payload(payload)


def _split_text_by_words(text: str, max_chars: int) -> list[str]:
    max_chars = max(1, max_chars)
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
            chunks.extend(word[index : index + max_chars] for index in range(0, len(word), max_chars))
            continue
        candidate = " ".join(current + [word]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _chunk_text_for_vox(text: str, max_chars: int) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = [sentence] if len(sentence) <= max_chars else _split_text_by_words(sentence, max_chars)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if not current:
                current = piece
                continue
            candidate = f"{current} {piece}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _offset_timestamps(
    word_timestamps: list[dict[str, float | str]],
    offset_seconds: float,
) -> list[dict[str, float | str]]:
    adjusted: list[dict[str, float | str]] = []
    for item in word_timestamps:
        adjusted.append(
            {
                "word": item["word"],
                "start": float(item["start"]) + offset_seconds,
                "end": float(item["end"]) + offset_seconds,
            }
        )
    return adjusted


def _concat_audio_files(audio_paths: list[Path], output_path: Path, crossfade_ms: int = 0) -> Path:
    if not audio_paths:
        raise ValueError("No audio files were provided for concatenation.")
    if len(audio_paths) == 1:
        output_path.write_bytes(audio_paths[0].read_bytes())
        return output_path

    command = ["ffmpeg", "-y"]
    for audio_path in audio_paths:
        command.extend(["-i", str(audio_path)])

    crossfade_seconds = max(0.0, crossfade_ms / 1000)
    if crossfade_seconds > 0:
        filter_parts = []
        previous_label = "[0:a]"
        for index in range(1, len(audio_paths)):
            output_label = f"[a{index}]"
            filter_parts.append(
                f"{previous_label}[{index}:a]acrossfade=d={crossfade_seconds:.3f}:c1=tri:c2=tri{output_label}"
            )
            previous_label = output_label
        filter_complex = ";".join(filter_parts)
        output_label = previous_label
    else:
        stream_inputs = "".join(f"[{index}:a]" for index in range(len(audio_paths)))
        filter_complex = f"{stream_inputs}concat=n={len(audio_paths)}:v=0:a=1[aout]"
        output_label = "[aout]"

    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            output_label,
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
    )
    _run_command(command, "FFmpeg audio concat")
    return output_path


def _generate_vox_cpm_audio_with_timestamps(text: str) -> tuple[Path, list[dict[str, float | str]]]:
    normalised_text = _normalise_tts_text(text)
    chunks = _chunk_text_for_vox(normalised_text, VOX_CPM_MAX_CHARS)
    if not chunks:
        raise ValueError("Cannot generate speech for empty text.")

    chunk_audio_paths: list[Path] = []
    combined_timestamps: list[dict[str, float | str]] = []
    current_offset = 0.0

    for index, chunk in enumerate(chunks, start=1):
        chunk_path = TEMP_DIR / f"narration_chunk_{index:02d}.wav"
        generated_path, chunk_timestamps = _generate_f5_tts_audio(chunk, chunk_path)
        chunk_audio_paths.append(generated_path)

        if not chunk_timestamps:
            chunk_timestamps = _timestamps_with_fallback(chunk, generated_path)
        combined_timestamps.extend(_offset_timestamps(chunk_timestamps, current_offset))
        current_offset += get_audio_duration(generated_path)

    final_audio_path = TEMP_DIR / "narration.wav"
    _concat_audio_files(chunk_audio_paths, final_audio_path)
    return final_audio_path, _normalise_word_sequence(combined_timestamps)


def _generate_parler_chunked_audio_with_timestamps(text: str) -> tuple[Path, list[dict[str, float | str]]]:
    normalised_text = _normalise_tts_text(text)
    chunks = _chunk_text_for_vox(normalised_text, PARLER_LOCAL_CHUNK_CHARS)
    if not chunks:
        raise ValueError("Cannot generate speech for empty text.")

    if len(chunks) == 1:
        audio_path = TEMP_DIR / "narration.wav"
        generated_path, timestamps = _generate_parler_tts_audio(chunks[0], audio_path)
        if not timestamps:
            timestamps = _timestamps_with_fallback(chunks[0], generated_path)
        return generated_path, timestamps

    LOGGER.info("Generating Parler narration in %d local chunks.", len(chunks))
    chunk_audio_paths: list[Path] = []
    combined_timestamps: list[dict[str, float | str]] = []
    current_offset = 0.0
    crossfade_seconds = max(0.0, PARLER_CHUNK_CROSSFADE_MS / 1000)

    for index, chunk in enumerate(chunks, start=1):
        LOGGER.info("Generating Parler chunk %d/%d (%d chars).", index, len(chunks), len(chunk))
        chunk_path = TEMP_DIR / f"narration_chunk_{index:02d}.wav"
        generated_path, chunk_timestamps = _generate_parler_tts_audio(chunk, chunk_path)
        chunk_audio_paths.append(generated_path)

        if not chunk_timestamps:
            chunk_timestamps = _timestamps_with_fallback(chunk, generated_path)
        combined_timestamps.extend(_offset_timestamps(chunk_timestamps, current_offset))

        duration = get_audio_duration(generated_path)
        current_offset += max(0.0, duration - crossfade_seconds)

    final_audio_path = TEMP_DIR / "narration.wav"
    _concat_audio_files(chunk_audio_paths, final_audio_path, crossfade_ms=PARLER_CHUNK_CROSSFADE_MS)
    return final_audio_path, _normalise_word_sequence(combined_timestamps)


# Emotion lexicon: cue words/markers -> Magpie emotion label. Scored per chunk so
# the strongest signal wins. Order does not matter; ties fall back to neutral.
_EMOTION_CUES: dict[str, tuple[str, ...]] = {
    "angry": (
        "angry", "furious", "rage", "yelled", "screamed", "shouted", "snapped", "how dare",
        "betrayed", "liar", "lied", "cheated", "unfair", "hate", "disgusting", "fed up",
        "enough", "slammed", "stormed", "fuming", "outraged", "pissed", "argument", "fight",
    ),
    "happy": (
        "happy", "laughed", "laughing", "smiled", "smiling", "grinned", "joy", "excited",
        "thrilled", "delighted", "celebrate", "wonderful", "amazing", "best day", "hilarious",
        "funny", "cheered", "grateful", "relief", "relieved", "yay", "love it",
    ),
    "sad": (
        "sad", "cried", "crying", "tears", "sobbed", "heartbroken", "grief", "lost", "loss",
        "funeral", "died", "death", "alone", "lonely", "miss", "missed", "sorry", "regret",
        "broke down", "devastated", "mourning", "empty",
    ),
    "fearful": (
        "scared", "afraid", "terrified", "fear", "horror", "panic", "shaking", "trembling",
        "creepy", "ghost", "shadow", "footsteps", "knock", "darkness", "scream", "danger",
        "something was", "no one there", "wasn't there", "behind me", "watching", "blood",
    ),
    "calm": (
        "calm", "quiet", "peaceful", "gently", "slowly", "softly", "relaxed", "breathe",
        "still", "serene", "settled", "whispered", "thought", "remember", "looking back",
    ),
}

_EMOTION_PUNCT = {
    "angry": lambda t: t.count("!"),
    "fearful": lambda t: t.count("..."),
}

# Preferred substitutes when a speaker lacks the exact emotion variant. The
# requested emotion itself is appended afterwards; "neutral" is the final floor.
_EMOTION_FALLBACK_CHAIN: dict[str, list[str]] = {
    "fearful": ["fearful", "fear", "calm", "sad"],
    "fear": ["fear", "fearful", "calm", "sad"],
    "sad": ["sad", "calm"],
    "happy": ["happy", "calm"],
    "angry": ["angry"],
    "calm": ["calm", "neutral"],
    "neutral": ["neutral"],
}


def _detect_chunk_emotion(text: str) -> str:
    """Pick the dominant emotion for a narration chunk from its wording.

    Returns a Magpie emotion label (angry/happy/sad/fearful/calm) or
    NVIDIA_DEFAULT_EMOTION when no signal is strong enough.
    """
    lowered = text.lower()
    scores: dict[str, float] = {emotion: 0.0 for emotion in _EMOTION_CUES}
    for emotion, cues in _EMOTION_CUES.items():
        for cue in cues:
            if cue in lowered:
                scores[emotion] += 1.0
    if "!" in text:
        scores["angry"] += 0.5 * text.count("!")
        scores["happy"] += 0.25 * text.count("!")
    if "..." in text:
        scores["fearful"] += 0.5
        scores["sad"] += 0.25

    best_emotion = max(scores, key=lambda key: scores[key])
    if scores[best_emotion] < 1.0:
        return (NVIDIA_DEFAULT_EMOTION or "neutral").lower()
    return best_emotion


def _voice_base_and_emotion(voice_name: str) -> tuple[str, str | None]:
    """Split a Magpie voice into its base (Locale.Speaker) and emotion suffix.

    Recognises a trailing emotion segment only when it matches a known emotion,
    so non-emotion subvoices are left attached to the base.
    """
    known = {"neutral", "calm", "happy", "sad", "angry", "fearful", "fear", "disgust", "surprise"}
    parts = voice_name.split(".")
    if len(parts) >= 2 and parts[-1].lower() in known:
        return ".".join(parts[:-1]), parts[-1]
    return voice_name, None


def _emotion_voice_resolver(service: Any, base_voice: str) -> Any:
    """Return a function mapping an emotion label to a concrete available voice.

    Resolves against the voices the API actually exposes for this speaker, with
    graceful fallback: requested emotion -> default emotion -> plain base voice.
    Results are cached per process.
    """
    available = set(_nvidia_available_voice_names(service))
    base, _current_emotion = _voice_base_and_emotion(base_voice)
    default_emotion = (NVIDIA_DEFAULT_EMOTION or "neutral").lower()

    def _available(candidate: str) -> str | None:
        if candidate in available:
            return candidate
        lowered = candidate.lower()
        for voice in available:
            if voice.lower() == lowered:
                return voice
        return None

    cache: dict[str, str] = {}

    def resolve(emotion: str) -> str:
        emotion = (emotion or default_emotion).lower()
        if emotion in cache:
            return cache[emotion]
        # Degrade each emotion toward the nearest available mood before falling
        # back to neutral, so e.g. "fearful" on a voice without a Fearful variant
        # becomes the tense "calm" rather than a flat neutral read.
        preference = _EMOTION_FALLBACK_CHAIN.get(emotion, []) + [emotion, default_emotion, "neutral"]
        seen_labels: set[str] = set()
        for label in preference:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            resolved = _available(f"{base}.{label}")
            if resolved:
                if label != emotion:
                    LOGGER.info("Emotion '%s' is using available voice fallback '%s'.", emotion, resolved)
                cache[emotion] = resolved
                return resolved
        for candidate in (base, base_voice):
            resolved = _available(candidate)
            if resolved:
                cache[emotion] = resolved
                return resolved
        # Last resort: ask the existing resolver (handles fuzzy matches).
        fallback = _resolve_nvidia_voice_name(service, base_voice)
        cache[emotion] = fallback
        return fallback

    return resolve


def _generate_nvidia_magpie_audio_with_timestamps(text: str) -> tuple[Path, list[dict[str, float | str]]]:
    normalised_text = _normalise_tts_text(text)
    chunks = _chunk_text_for_vox(normalised_text, NVIDIA_TTS_MAX_CHARS)
    if not chunks:
        raise ValueError("Cannot generate speech for empty text.")

    oversized_chunks = [len(chunk) for chunk in chunks if len(chunk) > NVIDIA_TTS_MAX_CHARS]
    if oversized_chunks:
        raise RuntimeError(
            f"NVIDIA TTS chunking produced chunks over {NVIDIA_TTS_MAX_CHARS} chars: {oversized_chunks}"
        )

    service, audio_encoding = _nvidia_synthesis_service()
    resolved_voice_name = _resolve_nvidia_voice_name(service, NVIDIA_TTS_VOICE)

    # Emotion strategy. "dominant" keeps ONE voice variant for the whole reel so
    # the timbre never jumps mid-narration (the main cause of "inconsistent"
    # audio); "dynamic" switches per chunk; "off" uses the plain voice.
    emotion_resolver = None
    mode = NVIDIA_EMOTION_MODE if NVIDIA_DYNAMIC_EMOTION else "off"
    if mode in {"dominant", "dynamic"}:
        try:
            emotion_resolver = _emotion_voice_resolver(service, resolved_voice_name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not enable emotion voices, using single voice: %s", exc)
            emotion_resolver = None

    fixed_voice = resolved_voice_name
    if emotion_resolver is not None and mode == "dominant":
        dominant = _detect_chunk_emotion(normalised_text)
        fixed_voice = emotion_resolver(dominant)
        LOGGER.info("NVIDIA dominant emotion for reel: %s -> %s", dominant, fixed_voice)

    LOGGER.info(
        "Generating NVIDIA Magpie narration in %d chunk(s), max %d chars each [emotion=%s].",
        len(chunks), NVIDIA_TTS_MAX_CHARS, mode if emotion_resolver else "off",
    )

    chunk_audio_paths: list[Path] = []
    combined_timestamps: list[dict[str, float | str]] = []
    current_offset = 0.0
    crossfade_seconds = max(0.0, NVIDIA_TTS_CHUNK_CROSSFADE_MS / 1000)

    for index, chunk in enumerate(chunks, start=1):
        if emotion_resolver is not None and mode == "dynamic":
            emotion = _detect_chunk_emotion(chunk)
            chunk_voice = emotion_resolver(emotion)
            LOGGER.info(
                "Generating NVIDIA Magpie chunk %d/%d (%d chars) [emotion=%s].",
                index, len(chunks), len(chunk), emotion,
            )
        else:
            chunk_voice = fixed_voice
            LOGGER.info("Generating NVIDIA Magpie chunk %d/%d (%d chars).", index, len(chunks), len(chunk))
        chunk_path = TEMP_DIR / f"narration_chunk_{index:02d}.wav"
        generated_path, response_meta = _generate_nvidia_magpie_audio(
            service,
            audio_encoding,
            chunk,
            chunk_path,
            chunk_voice,
        )
        # Normalise each chunk to a consistent loudness so volume doesn't jump at
        # the stitch points.
        if NVIDIA_TTS_NORMALIZE_CHUNKS and len(chunks) > 1:
            generated_path = _normalize_loudness(generated_path, NVIDIA_TTS_TARGET_LUFS)
        chunk_audio_paths.append(generated_path)

        chunk_timestamps = _nvidia_word_timestamps_from_meta(response_meta, generated_path)
        if not chunk_timestamps:
            fallback_text = str(getattr(response_meta, "processed_text", "") or chunk)
            chunk_timestamps = _timestamps_with_fallback(fallback_text, generated_path)
        combined_timestamps.extend(_offset_timestamps(chunk_timestamps, current_offset))

        duration = get_audio_duration(generated_path)
        current_offset += max(0.0, duration - crossfade_seconds)

    final_audio_path = TEMP_DIR / "narration.wav"
    _concat_audio_files(chunk_audio_paths, final_audio_path, crossfade_ms=NVIDIA_TTS_CHUNK_CROSSFADE_MS)
    return final_audio_path, _normalise_word_sequence(combined_timestamps)


# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------
# Curated default voices that work on the free tier (the API key cannot list
# voices, so we ship a known-good catalogue). IDs are ElevenLabs' public
# premade voices. Library/community voices require a paid plan.
ELEVENLABS_VOICE_CATALOG: list[dict[str, str]] = [
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "gender": "male", "desc": "warm, mature storyteller"},
    {"id": "nPczCjzI2devNBz1zQrb", "name": "Brian", "gender": "male", "desc": "deep, calm narrator"},
    {"id": "pqHfZKP75CvOlQylNhV4", "name": "Bill", "gender": "male", "desc": "documentary baritone"},
    {"id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel", "gender": "male", "desc": "authoritative news read"},
    {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam", "gender": "male", "desc": "youthful, energetic"},
    {"id": "cjVigY5qzO86Huf0OWal", "name": "Eric", "gender": "male", "desc": "friendly, conversational"},
    {"id": "iP95p4xoKVk53GoZ742B", "name": "Chris", "gender": "male", "desc": "casual, relatable"},
    {"id": "bIHbv24MWmeRgasZH58o", "name": "Will", "gender": "male", "desc": "chill, easy-going"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "gender": "female", "desc": "soft, warm"},
    {"id": "FGY2WhTYpPnrIDTdsKH5", "name": "Laura", "gender": "female", "desc": "bright, upbeat"},
    {"id": "XB0fDUnXU5powFXDhCwa", "name": "Charlotte", "gender": "female", "desc": "expressive, dramatic"},
    {"id": "pFZP5JQG7iQjIQuC4Bku", "name": "Lily", "gender": "female", "desc": "gentle, youthful"},
    {"id": "cgSgspJ2msm6clMCkdW9", "name": "Jessica", "gender": "female", "desc": "playful, modern"},
    {"id": "Xb7hH8MSUJpSbSDYk0k2", "name": "Alice", "gender": "female", "desc": "clear, confident"},
    {"id": "XrExE9yKIg1WjnnlVkGX", "name": "Matilda", "gender": "female", "desc": "warm, friendly"},
]


def list_elevenlabs_voices() -> list[dict[str, Any]]:
    """Return the curated ElevenLabs voice catalogue for the UI."""
    return [
        {"ShortName": f"{v['id']}|{v['name']}", "Gender": v["gender"], "Locale": v["desc"]}
        for v in ELEVENLABS_VOICE_CATALOG
    ]


def _elevenlabs_voice_id(voice: str) -> str:
    """Accept a bare id, an "id|Name" pair, or a friendly name; return the id."""
    candidate = (voice or "").strip()
    if not candidate:
        return ELEVENLABS_VOICE_ID
    if "|" in candidate:
        candidate = candidate.split("|", 1)[0].strip()
    # If it looks like a 20-char ElevenLabs id, use as-is.
    if re.fullmatch(r"[A-Za-z0-9]{16,}", candidate):
        return candidate
    for v in ELEVENLABS_VOICE_CATALOG:
        if v["name"].lower() == candidate.lower():
            return v["id"]
    return candidate or ELEVENLABS_VOICE_ID


def _words_from_char_alignment(
    characters: list[str],
    start_times: list[float],
    end_times: list[float],
    base_offset: float = 0.0,
) -> list[dict[str, float | str]]:
    """Convert ElevenLabs character-level alignment into word timestamps."""
    words: list[dict[str, float | str]] = []
    current = ""
    word_start: float | None = None
    word_end = 0.0
    for char, start, end in zip(characters, start_times, end_times):
        if char.isspace():
            if current.strip():
                words.append({"word": current, "start": (word_start or 0.0) + base_offset, "end": word_end + base_offset})
            current = ""
            word_start = None
            continue
        if word_start is None:
            word_start = float(start)
        current += char
        word_end = float(end)
    if current.strip():
        words.append({"word": current, "start": (word_start or 0.0) + base_offset, "end": word_end + base_offset})
    return words


def _elevenlabs_chunks(text: str, max_chars: int) -> list[str]:
    """Split narration on sentence boundaries, packing up to max_chars.

    Reels almost always fit in a single request (best for consistency); this
    only splits very long scripts so each request stays under the API limit.
    """
    if len(text) <= max_chars:
        return [text]
    return _chunk_text_for_vox(text, max_chars)


def _request_elevenlabs_audio(
    text: str,
    voice_id: str,
    previous_text: str | None,
    next_text: str | None,
) -> tuple[bytes, list[dict[str, Any]]]:
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY must be set to use the ElevenLabs backend.")

    url = f"{ELEVENLABS_BASE_URL}/v1/text-to-speech/{voice_id}/with-timestamps"
    payload: dict[str, Any] = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": ELEVENLABS_STABILITY,
            "similarity_boost": ELEVENLABS_SIMILARITY,
            "style": ELEVENLABS_STYLE,
            "use_speaker_boost": ELEVENLABS_SPEAKER_BOOST,
        },
    }
    # Continuity context keeps prosody consistent across multi-chunk narrations.
    if previous_text:
        payload["previous_text"] = previous_text[-500:]
    if next_text:
        payload["next_text"] = next_text[:500]

    response = requests.post(
        url,
        params={"output_format": ELEVENLABS_OUTPUT_FORMAT},
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=TTS_REQUEST_TIMEOUT,
    )
    if response.status_code == 401:
        raise RuntimeError(f"ElevenLabs auth/permission error: {response.text[:300]}")
    if response.status_code == 402:
        raise RuntimeError("ElevenLabs quota exhausted or voice requires a paid plan.")
    if response.status_code == 429:
        raise RuntimeError("ElevenLabs rate limit hit; try again shortly.")
    if not response.ok:
        raise RuntimeError(f"ElevenLabs request failed ({response.status_code}): {response.text[:300]}")

    body = response.json()
    audio_b64 = body.get("audio_base64")
    if not audio_b64:
        raise RuntimeError("ElevenLabs response did not include audio.")
    import base64

    audio_bytes = base64.b64decode(audio_b64)
    alignment = body.get("alignment") or body.get("normalized_alignment") or {}
    return audio_bytes, alignment


def _generate_elevenlabs_audio_with_timestamps(text: str) -> tuple[Path, list[dict[str, float | str]]]:
    normalised_text = _normalise_tts_text(text)
    if not normalised_text:
        raise ValueError("Cannot generate speech for empty text.")

    voice_id = _elevenlabs_voice_id(_active_elevenlabs_voice())
    chunks = _elevenlabs_chunks(normalised_text, ELEVENLABS_MAX_CHARS)
    LOGGER.info("Generating ElevenLabs narration in %d chunk(s) with voice %s.", len(chunks), voice_id)

    mp3_paths: list[Path] = []
    combined_timestamps: list[dict[str, float | str]] = []
    current_offset = 0.0

    for index, chunk in enumerate(chunks, start=1):
        previous_text = chunks[index - 2] if index >= 2 else None
        next_text = chunks[index] if index < len(chunks) else None
        audio_bytes, alignment = _request_elevenlabs_audio(chunk, voice_id, previous_text, next_text)
        chunk_mp3 = TEMP_DIR / f"narration_el_{index:02d}.mp3"
        chunk_mp3.write_bytes(audio_bytes)
        mp3_paths.append(chunk_mp3)

        characters = alignment.get("characters") or []
        starts = alignment.get("character_start_times_seconds") or []
        ends = alignment.get("character_end_times_seconds") or []
        if characters and starts and ends:
            combined_timestamps.extend(_words_from_char_alignment(characters, starts, ends, current_offset))
        else:
            combined_timestamps.extend(_offset_timestamps(_proportional_timestamps(chunk, chunk_mp3), current_offset))
        current_offset += get_audio_duration(chunk_mp3)

    # Convert to a single WAV so downstream mastering/concat behaves identically
    # to the other backends.
    final_wav = TEMP_DIR / "narration.wav"
    if len(mp3_paths) == 1:
        _convert_to_wav(mp3_paths[0], final_wav)
    else:
        _concat_audio_files(mp3_paths, final_wav)
    return final_wav, _normalise_word_sequence(combined_timestamps)


# Holds the per-job ElevenLabs voice override (env-driven). Defined as a tiny
# helper so the dispatcher can pass a voice without threading it everywhere.
def _active_elevenlabs_voice() -> str:
    import os

    return os.getenv("ELEVENLABS_VOICE_ID") or ELEVENLABS_VOICE_ID


def _convert_to_wav(src: Path, dst: Path) -> Path:
    command = ["ffmpeg", "-y", "-i", str(src), "-acodec", "pcm_s16le", "-ac", "1", str(dst)]
    _run_command(command, "FFmpeg mp3->wav conversion")
    return dst


def _normalize_loudness(audio_path: Path, target_lufs: float) -> Path:
    """Loudness-normalise a WAV to a target LUFS so chunk volumes match.

    Returns the normalised file, or the original on failure (best-effort).
    """
    try:
        out_path = audio_path.with_name(f"{audio_path.stem}_norm.wav")
        command = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", str(NVIDIA_TTS_SAMPLE_RATE_HZ), "-ac", "1",
            "-c:a", "pcm_s16le", str(out_path),
        ]
        _run_command(command, "FFmpeg loudness normalisation")
        return out_path
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Loudness normalisation failed, using raw chunk: %s", exc)
        return audio_path


def _generate_piper_audio(text: str, output_path: Path) -> Path:
    command = ["piper", "--model", TTS_VOICE, "--output_file", str(output_path)]
    _run_command(command, "Piper synthesis", input_text=text)
    return output_path


def _ensure_wave_audio(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    wav_path = TEMP_DIR / "timing_source.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        str(wav_path),
    ]
    _run_command(command, "FFmpeg audio conversion")
    return wav_path


def get_audio_duration(audio_path: str | Path) -> float:
    """Return the audio duration in seconds using ffprobe.

    Args:
        audio_path: Audio file path to inspect.

    Returns:
        Audio duration as a float in seconds.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(Path(audio_path).expanduser().resolve()),
    ]
    completed = _run_command(command, "ffprobe duration lookup")
    return float((completed.stdout or "0").strip() or 0.0)


def _proportional_timestamps(text: str, audio_path: Path) -> list[dict[str, float | str]]:
    wav_path = _ensure_wave_audio(audio_path)
    with wave.open(str(wav_path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = frame_count / frame_rate if frame_rate else 0.0

    words = [word for word in text.split() if word.strip()]
    if not words:
        return []
    if duration <= 0:
        duration = float(len(words)) * 0.3

    step = duration / len(words)
    timestamps = []
    for index, word in enumerate(words):
        start = index * step
        end = (index + 1) * step
        timestamps.append({"word": word, "start": start, "end": end})
    return _normalise_word_sequence(timestamps)


@lru_cache(maxsize=1)
def _load_whisper_model() -> Any:
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def _whisper_align(text: str, audio_path: Path) -> list[dict[str, float | str]]:
    model = _load_whisper_model()
    segments, _ = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        initial_prompt=text,
        vad_filter=False,
    )

    words: list[dict[str, float | str]] = []
    for segment in segments:
        for word in segment.words or []:
            token = _strip_punctuation_token(getattr(word, "word", "") or "")
            if not token:
                continue
            start = float(getattr(word, "start", 0.0) or 0.0)
            end = float(getattr(word, "end", start) or start)
            words.append({"word": token, "start": start, "end": end})

    if not words:
        raise RuntimeError("Whisper alignment returned no word timestamps.")
    return _normalise_word_sequence(words)


def _timestamps_with_fallback(text: str, audio_path: Path) -> list[dict[str, float | str]]:
    if USE_WHISPER_ALIGN:
        try:
            return _whisper_align(text, audio_path)
        except ImportError:
            LOGGER.warning("faster-whisper is not installed; using proportional timing fallback.")
        except Exception as exc:
            LOGGER.warning("Whisper alignment failed; using proportional timing fallback: %s", exc)
    return _proportional_timestamps(text, audio_path)


def generate_speech(text: str) -> tuple[Path, list[dict[str, float | str]]]:
    """Generate narration audio and aligned word timestamps.

    Args:
        text: Narration text to synthesise.

    Returns:
        A tuple of `(audio_path, word_timestamps)`.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    requested_backend = TTS_BACKEND.strip().lower()
    normalised_text = _normalise_tts_text(text)

    def _generate_with_backend(backend_name: str) -> tuple[Path, list[dict[str, float | str]]]:
        if backend_name in {"elevenlabs", "eleven_labs", "eleven", "11labs", "elabs"}:
            return _generate_elevenlabs_audio_with_timestamps(normalised_text)

        if backend_name in {"nvidia", "nvidia_magpie", "nvidia-magpie", "magpie", "magpie_tts", "riva"}:
            return _generate_nvidia_magpie_audio_with_timestamps(normalised_text)

        if backend_name == "edge_tts":
            audio_path = TEMP_DIR / "narration.mp3"
            generated_path, timestamps = asyncio.run(_generate_edge_tts_audio(normalised_text, audio_path))
            if timestamps:
                return generated_path, timestamps
            return generated_path, _timestamps_with_fallback(normalised_text, generated_path)

        if backend_name in {"parler_tts", "parler-tts", "story_narration", "story-narration"}:
            if PARLER_LOCAL_CHUNKING_ENABLED:
                return _generate_parler_chunked_audio_with_timestamps(normalised_text)
            audio_path = TEMP_DIR / "narration.wav"
            generated_path, timestamps = _generate_parler_tts_audio(normalised_text, audio_path)
            if timestamps:
                return generated_path, timestamps
            return generated_path, _timestamps_with_fallback(normalised_text, generated_path)

        if backend_name in {"vox_cpm", "f5_tts", "f5-tts"}:
            LOGGER.info(
                "Using Parler story narration API for backend '%s'. "
                "Set TTS_BACKEND=f5_clone to use the legacy prompt_wav API.",
                backend_name,
            )
            if PARLER_LOCAL_CHUNKING_ENABLED:
                return _generate_parler_chunked_audio_with_timestamps(normalised_text)
            audio_path = TEMP_DIR / "narration.wav"
            generated_path, timestamps = _generate_parler_tts_audio(normalised_text, audio_path)
            if timestamps:
                return generated_path, timestamps
            return generated_path, _timestamps_with_fallback(normalised_text, generated_path)

        if backend_name in {"f5_clone", "f5-clone", "vox_cpm_legacy", "vox-cpm-legacy"}:
            return _generate_vox_cpm_audio_with_timestamps(normalised_text)

        if backend_name == "piper":
            audio_path = TEMP_DIR / "narration.wav"
            generated_path = _generate_piper_audio(normalised_text, audio_path)
            return generated_path, _timestamps_with_fallback(normalised_text, generated_path)

        raise ValueError(f"Unsupported TTS backend: {backend_name}")

    try:
        return _generate_with_backend(requested_backend)
    except Exception as exc:
        fallback_backend = TTS_FALLBACK_BACKEND.strip().lower()
        if ENABLE_TTS_FALLBACK and fallback_backend and fallback_backend != requested_backend:
            LOGGER.warning(
                "Primary TTS backend '%s' failed (%s). Falling back to '%s'.",
                requested_backend,
                exc,
                fallback_backend,
            )
            return _generate_with_backend(fallback_backend)
        raise


async def _list_edge_voices_async() -> list[dict[str, Any]]:
    voices = await edge_tts.list_voices()
    return [dict(voice) for voice in voices]


def list_edge_tts_voices() -> list[dict[str, Any]]:
    """Return the available Edge TTS voices.

    Args:
        None.

    Returns:
        A list of voice dictionaries from the edge-tts package.
    """
    return asyncio.run(_list_edge_voices_async())


def list_nvidia_tts_voices() -> list[dict[str, Any]]:
    """Return NVIDIA Riva TTS voices visible to the configured API key."""
    service, _audio_encoding = _nvidia_synthesis_service()
    import riva.client

    config_response = service.stub.GetRivaSynthesisConfig(
        riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
    )
    voices: list[dict[str, Any]] = []
    for model_config in config_response.model_config:
        parameters = model_config.parameters
        language_code = parameters.get("language_code", "")
        voice_name = parameters.get("voice_name", "")
        subvoices = [voice.split(":")[0] for voice in parameters.get("subvoices", "").split(",") if voice.strip()]
        if subvoices:
            for subvoice in subvoices:
                voices.append(
                    {
                        "ShortName": f"{voice_name}.{subvoice}",
                        "Gender": "",
                        "Locale": language_code,
                    }
                )
        elif voice_name:
            voices.append({"ShortName": voice_name, "Gender": "", "Locale": language_code})
    return voices


def list_tts_voices(backend: str | None = None) -> list[dict[str, Any]]:
    """Return voices for the given (or configured) backend when supported."""
    backend_name = (backend or TTS_BACKEND).strip().lower()
    if backend_name in {"elevenlabs", "eleven_labs", "eleven", "11labs", "elabs"}:
        return list_elevenlabs_voices()
    if backend_name in {"nvidia", "nvidia_magpie", "nvidia-magpie", "magpie", "magpie_tts", "riva"}:
        return list_nvidia_tts_voices()
    return list_edge_tts_voices()
