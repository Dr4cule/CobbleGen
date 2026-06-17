from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import requests

from config import *


LOGGER = logging.getLogger(__name__)

_CONTENT_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z']+\b")

_CONTENT_STOPWORDS = {
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
}


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in _CONTENT_WORD_RE.findall(text.lower())
        if len(word) > 2 and word not in _CONTENT_STOPWORDS
    }


def _max_overlap_ratio(candidate: str, references: list[str]) -> float:
    candidate_words = _content_words(candidate)
    if not candidate_words:
        return 0.0

    best_ratio = 0.0
    for reference in references:
        reference_words = _content_words(reference)
        if not reference_words:
            continue
        overlap = len(candidate_words & reference_words)
        denom = max(1, min(len(candidate_words), len(reference_words)))
        best_ratio = max(best_ratio, overlap / denom)
    return best_ratio


def _pick_creative_variant(seed_text: str, options: list[str], references: list[str]) -> str:
    if not options:
        return ""

    seed_value = sum(ord(char) for char in seed_text)
    best_candidate = options[seed_value % len(options)]
    best_score = float("inf")

    for offset in range(len(options)):
        candidate = options[(seed_value + offset) % len(options)]
        score = _max_overlap_ratio(candidate, references)
        if score <= 0.2:
            return candidate
        if score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _normalise_whitespace(text: str) -> str:
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€”": ", ",
        "â€“": ", ",
        "â€¦": "...",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": ", ",
        "\u2013": ", ",
        "\u2026": "...",
    }
    cleaned = text.replace("\ufeff", "")
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return re.sub(r"\s+", " ", cleaned).strip()


def _estimate_duration_seconds(text: str) -> float:
    word_count = len(re.findall(r"\b\w+\b", text))
    return (word_count / STORY_TARGET_WPM) * 60 if word_count else 0.0


def _scene_count_for_text(text: str) -> int:
    estimated_duration = _estimate_duration_seconds(text)
    if estimated_duration <= 0:
        return max(1, DEFAULT_SCENE_COUNT)
    target = math.ceil(estimated_duration / max(SECONDS_PER_SCENE_IMAGE, 1.0))
    return max(MIN_SCENE_COUNT, min(MAX_SCENE_COUNT, target))


def _first_sentence(text: str) -> str:
    match = re.search(r"(.+?[.!?])(?:\s|$)", text.strip())
    return match.group(1).strip() if match else text.strip()[:160]


def _trim_to_word_budget(text: str, target_words: int) -> str:
    words = text.split()
    if len(words) <= target_words:
        return text.strip()

    candidate = " ".join(words[:target_words]).strip()
    sentence_matches = list(re.finditer(r"[.!?](?:['\"])?(?:\s|$)", candidate))
    minimum_end = int(len(candidate) * 0.65)
    for match in reversed(sentence_matches):
        if match.end() >= minimum_end:
            return candidate[: match.end()].strip()

    return candidate.rstrip(" ,;:-") + "."


def _has_sentence_ending(text: str) -> bool:
    return bool(re.search(r"[.!?](?:['\"])?$", text.strip()))


def _acceptable_condensed_text(text: str, target_words: int) -> bool:
    word_count = len(text.split())
    min_words = max(12, int(target_words * 0.55))
    max_words = max(min_words, int(target_words * 1.2) + 20)
    return min_words <= word_count <= max_words and _has_sentence_ending(text)


def _split_long_sentence(sentence: str) -> list[str]:
    words = sentence.split()
    if len(words) <= NARRATION_MAX_SENTENCE_WORDS:
        return [sentence.strip()]

    clauses = re.split(r"(?<=,)\s+|(?<=;)\s+|(?<=:)\s+|\s+-\s+", sentence)
    if len(clauses) > 1:
        parts: list[str] = []
        current = ""
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            candidate = f"{current} {clause}".strip()
            if current and len(candidate.split()) > NARRATION_MAX_SENTENCE_WORDS:
                parts.append(current.strip().rstrip(",;:"))
                current = clause
            else:
                current = candidate
        if current:
            parts.append(current.strip().rstrip(",;:"))
        return [part for part in parts if part]

    midpoint = len(words) // 2
    split_index = midpoint
    for index in range(midpoint, min(len(words), midpoint + 6)):
        token = words[index].strip(",")
        if token.lower() in {"but", "and", "then", "so"}:
            split_index = index
            break

    left = " ".join(words[:split_index]).strip().rstrip(",")
    right = " ".join(words[split_index:]).strip()
    if left and not re.search(r"[.!?]$", left):
        left += "."
    if right:
        right = right[0].upper() + right[1:]
    return [part for part in (left, right) if part]


def _infer_story_tone(text: str) -> str:
    lowered = text.lower()
    tone_keywords = {
        "tense": {
            "emergency",
            "alert",
            "siren",
            "creature",
            "creatures",
            "basement",
            "outside",
            "window",
            "horror",
            "panic",
            "danger",
            "invasion",
            "fear",
            "scared",
            "dark",
            "urgent",
        },
        "playful": {
            "funny",
            "joke",
            "laugh",
            "laughed",
            "ridiculous",
            "bizarre",
            "weird",
            "awkward",
            "prank",
            "underwear",
            "bon jovi",
            "nostalgic",
            "claimed",
            "claiming",
            "amused",
        },
        "angry": {
            "asshole",
            "angry",
            "rage",
            "furious",
            "yelled",
            "shouted",
            "fight",
            "argument",
            "rude",
            "cheated",
            "betrayed",
            "payback",
        },
        "somber": {
            "sad",
            "sorrow",
            "cry",
            "cried",
            "lost",
            "loss",
            "funeral",
            "grief",
            "heartbroken",
            "mourning",
        },
        "warm": {
            "warm",
            "wholesome",
            "kind",
            "gentle",
            "grateful",
            "thankful",
            "love",
            "loving",
            "comfort",
            "sweet",
            "tender",
        },
    }

    tone_scores = {
        tone: sum(1 for keyword in keywords if keyword in lowered)
        for tone, keywords in tone_keywords.items()
    }
    tone, score = max(tone_scores.items(), key=lambda item: item[1])
    return tone if score else "dramatic"


def _story_style_hint(story_tone: str) -> str:
    hints = {
        "tense": "tense storyteller with measured pauses, quiet dread, and escalating suspense",
        "playful": "wry, amused storyteller with conversational timing and dry comedic beats",
        "angry": "heated storyteller with clipped delivery, sharp emphasis, and visible frustration",
        "somber": "somber storyteller with soft pacing, reflective pauses, and emotional restraint",
        "warm": "warm storyteller with gentle emphasis, human detail, and affectionate pacing",
        "dramatic": NARRATION_STYLE_HINT,
    }
    return hints.get(story_tone, NARRATION_STYLE_HINT)


def _heuristic_speech_styling(story_text: str, story_tone: str = "dramatic") -> str:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", story_text) if segment.strip()]
    styled_sentences: list[str] = []
    tone = (story_tone or "dramatic").strip().lower()

    if tone in {"tense", "somber"}:
        max_sentence_words = max(8, int(NARRATION_MAX_SENTENCE_WORDS * 0.65))
        sentence_joiner = " ... "
    elif tone == "angry":
        max_sentence_words = max(8, int(NARRATION_MAX_SENTENCE_WORDS * 0.6))
        sentence_joiner = " "
    elif tone == "playful":
        max_sentence_words = max(10, int(NARRATION_MAX_SENTENCE_WORDS * 0.75))
        sentence_joiner = " "
    elif tone == "warm":
        max_sentence_words = max(10, int(NARRATION_MAX_SENTENCE_WORDS * 0.85))
        sentence_joiner = " "
    else:
        max_sentence_words = NARRATION_MAX_SENTENCE_WORDS
        sentence_joiner = " "

    for sentence in sentences:
        sentence = re.sub(r"\s+,", ",", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence).strip()
        if len(sentence.split()) > max_sentence_words:
            styled_sentences.extend(_split_long_sentence(sentence))
        else:
            styled_sentences.append(sentence)

    styled_text = sentence_joiner.join(styled_sentences)
    styled_text = re.sub(r"\bbut\b", "but", styled_text, flags=re.IGNORECASE)
    styled_text = re.sub(r"\bnow\b", "now", styled_text, flags=re.IGNORECASE)
    styled_text = styled_text.replace(" -- ", ", ")
    styled_text = re.sub(r"\s+", " ", styled_text).strip()
    return styled_text


def _pick_variant(seed_text: str, options: list[str]) -> str:
    if not options:
        return ""
    seed_value = sum(ord(char) for char in seed_text)
    return options[seed_value % len(options)]


def _fallback_intro_text(story_text: str, title: str, hook: str) -> str:
    tone = _infer_story_tone(story_text)
    title_text = _normalise_whitespace(title).rstrip(".!?") or "this story"
    hook_text = _normalise_whitespace(hook).rstrip(".!?") or "you need to hear this"
    options_by_tone = {
        "tense": [
            "This starts normal, which is exactly why it feels wrong.",
            "A routine moment is about to tilt sideways.",
            "You can feel the story go off-script almost immediately.",
            "Everything looks calm right before it stops being calm.",
            "This is the kind of setup that never stays harmless.",
        ],
        "playful": [
            "This starts polite and ends up acting like a rumor.",
            "What begins as a normal errand turns into folklore.",
            "This story has the energy of a joke that escaped.",
            "It starts with a straight face and ends in disbelief.",
            "You can tell it’s going sideways because it gets funny first.",
        ],
        "angry": [
            "This starts with somebody already testing the limit.",
            "The patience is gone before the story even heats up.",
            "It opens like a favor and closes like a warning.",
            "You can feel the bad decision arriving early.",
            "This is the kind of setup that dares you to stay calm.",
        ],
        "somber": [
            "It begins quietly, which makes the weight hit harder.",
            "This one gets heavy in a way you feel coming.",
            "A small moment turns into something that lingers.",
            "The calm part ends faster than it should.",
            "This starts soft and ends with a quiet bruise.",
        ],
        "warm": [
            "It starts ordinary and ends up more human than expected.",
            "A small moment turns into something you remember later.",
            "This one sneaks up on you by being kind first.",
            "It begins simple, then lands with a lot of heart.",
            "What looks routine at first turns unexpectedly personal.",
        ],
        "dramatic": [
            "This starts like a normal day and ends like a legend.",
            "One ordinary moment is enough to tip everything.",
            "The story gets weird the moment it starts acting normal.",
            "This is the part where the whole mood changes.",
            "It feels harmless until it absolutely does not.",
        ],
    }
    options = options_by_tone.get(tone, options_by_tone["dramatic"])
    return _pick_creative_variant(
        f"{story_text}|{title_text}|{hook_text}|intro",
        options,
        [story_text, title_text, hook_text],
    )


def _fallback_outro_text(story_text: str, title: str, hook: str) -> str:
    tone = _infer_story_tone(story_text)
    title_text = _normalise_whitespace(title).rstrip(".!?") or "this story"
    hook_text = _normalise_whitespace(hook).rstrip(".!?") or "this one"
    options_by_tone = {
        "tense": [
            "Tell me where you would have drawn the line.",
            "Would you have kept going after that?",
            "That ending is why quiet jobs never stay quiet.",
            "Now I want your honest verdict.",
            "What would you have done once it shifted?",
        ],
        "playful": [
            "Would you have laughed first or denied the whole thing?",
            "Tell me which part you’d retell to your friends.",
            "That ending belongs in the group chat forever.",
            "Now be honest, would you have doubled down?",
            "What part of that would you have believed?",
        ],
        "angry": [
            "Who crossed the line first in your opinion?",
            "Would you have walked out sooner?",
            "That ending leaves a pretty clear argument.",
            "Now I want the real verdict.",
            "What would you have shut down first?",
        ],
        "somber": [
            "Would you have handled that any differently?",
            "That ending settles in a way you feel later.",
            "Tell me how you would have answered.",
            "What would you have kept, and what would you have let go?",
            "That’s the kind of ending that lingers.",
        ],
        "warm": [
            "Would you have answered the same way?",
            "Tell me how you would have moved through that moment.",
            "That ending leaves a human-sized mark.",
            "What would you have done differently, if anything?",
            "Now I want your version of it.",
        ],
        "dramatic": [
            "And that’s the moment the whole story turns.",
            "Would you have made the same call?",
            "Tell me where you land on this one.",
            "That ending is why people keep arguing.",
            "Now I want the honest answer.",
        ],
    }
    options = options_by_tone.get(tone, options_by_tone["dramatic"])
    return _pick_creative_variant(
        f"{story_text}|{title_text}|{hook_text}|outro",
        options,
        [story_text, title_text, hook_text],
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    start_index = cleaned.find("{")
    if start_index == -1:
        raise ValueError("No JSON object found in Gemini response.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(cleaned)):
        char = cleaned[index]
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start_index : index + 1])
        escaped = False

    raise ValueError("Could not extract a complete JSON object from Gemini response.")


def _gemini_model_candidates() -> list[str]:
    candidates = [GEMINI_MODEL, *GEMINI_MODEL_FALLBACKS]
    deduped: list[str] = []
    seen: set[str] = set()
    for model in candidates:
        cleaned = model.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _gemini_response_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            text_parts = [str(part.get("text", "")).strip() for part in parts if isinstance(part, dict) and part.get("text")]
            combined = "\n".join(part for part in text_parts if part)
            if combined.strip():
                return combined.strip()

    feedback = body.get("promptFeedback")
    if isinstance(feedback, dict):
        block_reason = feedback.get("blockReason") or feedback.get("blockReasonMessage")
        if block_reason:
            raise ValueError(f"Gemini blocked the prompt: {block_reason}")

    error = body.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("status") or error)
        raise requests.RequestException(message)

    raise ValueError("Gemini response did not include any text.")


def _gemini_model_not_found(response: requests.Response) -> bool:
    if response.status_code not in {400, 404}:
        return False

    try:
        body = response.json()
    except ValueError:
        body = {}

    error = body.get("error") if isinstance(body, dict) else None
    message = response.text.lower()
    if isinstance(error, dict):
        message = f"{message} {str(error.get('message', '')).lower()} {str(error.get('status', '')).lower()}"

    keywords = ("model", "not found", "does not exist", "unsupported")
    return any(keyword in message for keyword in keywords)


def _gemini_should_try_next_model(response: requests.Response) -> bool:
    if response.status_code in {429, 500, 502, 503, 504}:
        return True
    return _gemini_model_not_found(response)


def _call_gemini(
    prompt: str,
    expect_json: bool = False,
    timeout_seconds: int = 60,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    from config import DISABLE_GEMINI

    if DISABLE_GEMINI:
        raise requests.RequestException("Gemini disabled via DISABLE_GEMINI")
    if not GEMINI_API_KEY:
        raise requests.RequestException("GEMINI_API_KEY must be set for Gemini requests.")

    model_candidates = _gemini_model_candidates()
    last_error: Exception | None = None
    for index, model in enumerate(model_candidates):
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "candidateCount": 1,
            },
        }
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        if max_output_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
        if expect_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=timeout_seconds,
        )

        if _gemini_should_try_next_model(response) and index < len(model_candidates) - 1:
            LOGGER.warning(
                "Gemini model %s returned %s; trying fallback.",
                model,
                response.status_code,
            )
            last_error = requests.HTTPError(response.text)
            continue

        response.raise_for_status()

        try:
            body = response.json()
        except ValueError as exc:
            last_error = exc
            raise requests.RequestException("Gemini response was not valid JSON.") from exc

        return _gemini_response_text(body).strip()

    if last_error is not None:
        raise requests.RequestException(str(last_error)) from last_error
    raise requests.RequestException("Gemini request failed without a response.")


def _ollama_response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = str(message.get("content") or "").strip()
                if content:
                    return content
            text = str(choice.get("text") or "").strip()
            if text:
                return text

    error = body.get("error")
    if error:
        raise requests.RequestException(str(error))
    raise ValueError("Ollama response did not include any text.")


def _call_ollama(
    prompt: str,
    expect_json: bool = False,
    timeout_seconds: int | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    if not ENABLE_OLLAMA_FALLBACK:
        raise requests.RequestException("Ollama fallback is disabled.")

    endpoints = OLLAMA_ENDPOINTS or ((OLLAMA_BASE_URL, OLLAMA_MODEL),)
    full_timeout = timeout_seconds or OLLAMA_REQUEST_TIMEOUT
    # Earlier hosts get a shorter timeout so a hung host fails over fast; the
    # last host keeps the full window to absorb a genuine cloud cold start.
    quick_timeout = min(full_timeout, 45)
    last_error: Exception | None = None
    for index, (host, model) in enumerate(endpoints):
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_tokens"] = max(max_output_tokens, 4096)
        if expect_json:
            payload["response_format"] = {"type": "json_object"}

        is_last = index == len(endpoints) - 1
        attempt_timeout = full_timeout if is_last else quick_timeout
        try:
            response = requests.post(
                f"{host}/v1/chat/completions",
                json=payload,
                timeout=attempt_timeout,
            )
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError as exc:
                raise requests.RequestException("Ollama response was not valid JSON.") from exc
            return _ollama_response_text(body).strip()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if index < len(endpoints) - 1:
                LOGGER.warning("Ollama endpoint %s (%s) failed (%s); trying next.", host, model, exc)
                continue

    raise requests.RequestException(f"All Ollama endpoints failed: {last_error}")


def _is_rate_limited_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    message = str(error).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message


def _call_story_model(
    prompt: str,
    expect_json: bool = False,
    timeout_seconds: int = 60,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    return _call_ollama(
        prompt,
        expect_json=expect_json,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _fallback_scenes(story_text: str) -> list[dict[str, Any]]:
    scene_count = _scene_count_for_text(story_text)
    fractions = [round(index / (scene_count + 1), 2) for index in range(1, scene_count + 1)]
    lower_story = story_text.lower()

    if any(keyword in lower_story for keyword in ("emergency alert", "extraterrestrial", "invasion", "creatures", "sirens")):
        defaults = [
            ("emergency alert siren", "Emergency alert"),
            ("locked bathroom fear", "Stay inside"),
            ("alien invasion city", "The invasion"),
            ("survivor guilt aftermath", "Aftermath"),
            ("empty city street", "After the sirens"),
            ("dark apartment hallway", "Someone knocks"),
            ("destroyed city skyline", "The world changes"),
            ("survivor alone room", "Still alive"),
        ]
    else:
        defaults = [
            ("minecraft parkour", "Story starts"),
            ("tense decision", "Things escalate"),
            ("awkward argument", "Conflict hits"),
            ("emotional ending", "Final reveal"),
            ("family text message", "Receipts"),
            ("wedding dinner argument", "The line gets crossed"),
            ("quiet bedroom phone", "Waiting for replies"),
            ("dramatic confession", "The verdict"),
        ]

    return [
        {"fraction": fraction, "query": defaults[index % len(defaults)][0], "caption": defaults[index % len(defaults)][1]}
        for index, fraction in enumerate(fractions)
    ]


def _make_query_from_text(segment: str) -> str:
    """Create a 2-4 word Unsplash-friendly query from a sentence segment.

    This is a lightweight heuristic that strips common stopwords and short words,
    then returns the first 2-4 content words to increase relevance to the scene.
    """
    stopwords = {
        "the",
        "and",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "with",
        "at",
        "by",
        "from",
        "that",
        "this",
        "it",
        "is",
        "was",
        "were",
        "are",
        "be",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
    }
    cleaned = re.sub(r"[^\w\s'-]", " ", segment or "")
    words = [w.strip("'\"- ") for w in cleaned.split() if w.strip()]
    content = [w for w in words if w.lower() not in stopwords and len(w) > 2]
    if not content:
        # Fallback to the first substantive words from the raw sentence
        content = [w for w in words if len(w) > 2]
    # Return 2-4 words joined; prefer contiguous phrase when possible
    query_words = content[:4]
    if len(query_words) == 1 and len(content) > 1:
        query_words = content[:2]
    if not query_words and words:
        query_words = words[:2]
    query = " ".join(query_words)
    return query or "dramatic scene"


def _scene_context_for_fraction(sentences: list[str], fraction: float) -> str:
    if not sentences:
        return ""

    clamped_fraction = max(0.0, min(1.0, fraction))
    center_index = min(len(sentences) - 1, max(0, int(round(clamped_fraction * max(len(sentences) - 1, 0)))))
    start_index = max(0, center_index - 1)
    end_index = min(len(sentences), center_index + 2)
    return _normalise_whitespace(" ".join(sentences[start_index:end_index]))


def _dedupe_text_items(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalise_whitespace(str(value))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _build_scene_search_terms(
    query: str,
    caption: str,
    story_excerpt: str,
    theme_query: str = "",
    theme_caption: str = "",
) -> list[str]:
    story_query = _make_query_from_text(story_excerpt) if story_excerpt else ""
    caption_query = _make_query_from_text(caption) if caption else ""
    candidates = _dedupe_text_items(
        [
            story_query,
            query,
            caption_query,
            theme_query,
            theme_caption,
            f"{story_query} {caption_query}" if story_query and caption_query else "",
            f"{query} {caption_query}" if query and caption_query else "",
            f"{story_query} {caption}" if story_query and caption else "",
            f"{query} {story_query}" if query and story_query else "",
        ]
    )
    return candidates[:6]


def _fallback_meta(story_text: str) -> dict[str, Any]:
    title_source = _first_sentence(story_text).replace('"', "")
    title = title_source[:67].rstrip() + ("..." if len(title_source) > 67 else "")
    # Build scene queries from story sentences to make Unsplash results story-specific
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", story_text) if s.strip()]
    if not sentences:
        sentences = [title_source]
    scene_count = _scene_count_for_text(story_text)
    fractions = [round(index / (scene_count + 1), 2) for index in range(1, scene_count + 1)]
    scenes: list[dict[str, Any]] = []
    themed_defaults = _fallback_scenes(story_text)
    # sample up to scene_count sentences across the story (start, mid, end...)
    for i, fraction in enumerate(fractions):
        story_excerpt = _scene_context_for_fraction(sentences, fraction) or title_source
        story_query = _make_query_from_text(story_excerpt)[:40]
        caption = _first_sentence(story_excerpt)[:60]
        theme_query = themed_defaults[i]["query"] if i < len(themed_defaults) else ""
        theme_caption = themed_defaults[i]["caption"] if i < len(themed_defaults) else ""
        query = story_query or theme_query or "dramatic scene"
        search_terms = _build_scene_search_terms(query, caption, story_excerpt, theme_query, theme_caption)
        scenes.append(
            {
                "fraction": fraction,
                "query": query,
                "caption": caption or theme_caption or "Story beat",
                "story_excerpt": story_excerpt,
                "search_terms": search_terms,
            }
        )

    title_text = title or "Reddit story you won't believe"
    hook_text = _first_sentence(story_text) or "This story spiraled fast."

    return {
        "title": title or "Reddit story you won't believe",
        "description": (
            "This Reddit story gets messy fast. Watch the full reel and drop your verdict in the comments."
        ),
        "hashtags": [
            "reddit",
            "redditstories",
            "aita",
            "storytime",
            "minecraft",
            "parkour",
            "shorts",
            "tiktokstory",
            "dramastory",
            "viralstory",
            "familydrama",
            "fyp",
        ],
        "hook": hook_text,
        "intro_text": _fallback_intro_text(story_text, title_text, hook_text),
        "outro_text": _fallback_outro_text(story_text, title_text, hook_text),
        "scenes": scenes,
    }





def _normalise_meta(meta: dict[str, Any], story_text: str) -> dict[str, Any]:
    fallback = _fallback_meta(story_text)
    title = _normalise_whitespace(str(meta.get("title") or fallback["title"]))[:70]
    description = _normalise_whitespace(str(meta.get("description") or fallback["description"]))
    hashtags = [str(tag).strip().lstrip("#") for tag in meta.get("hashtags", []) if str(tag).strip()]
    if len(hashtags) < 12:
        for tag in fallback["hashtags"]:
            if tag not in hashtags:
                hashtags.append(tag)
            if len(hashtags) == 12:
                break
    hashtags = hashtags[:12]
    hook = _normalise_whitespace(str(meta.get("hook") or fallback["hook"]))
    intro_text = _normalise_whitespace(str(meta.get("intro_text") or fallback["intro_text"]))
    outro_text = _normalise_whitespace(str(meta.get("outro_text") or fallback["outro_text"]))

    scenes: list[dict[str, Any]] = []
    story_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", story_text) if s.strip()]
    raw_scenes = meta.get("scenes", [])
    if isinstance(raw_scenes, list):
        for index, item in enumerate(raw_scenes):
            if not isinstance(item, dict):
                continue
            try:
                fraction = max(0.0, min(1.0, float(item.get("fraction", 0.0))))
            except (TypeError, ValueError):
                fraction = 0.0
            query = _normalise_whitespace(str(item.get("query") or "minecraft parkour"))[:40]
            caption = _normalise_whitespace(str(item.get("caption") or "Story beat"))[:60]
            story_excerpt = _normalise_whitespace(str(item.get("story_excerpt") or ""))[:120]
            if not story_excerpt:
                story_excerpt = _scene_context_for_fraction(story_sentences, fraction)
            raw_terms = item.get("search_terms", [])
            search_terms: list[str] = []
            if isinstance(raw_terms, list):
                search_terms.extend(str(term) for term in raw_terms if str(term).strip())
            elif isinstance(raw_terms, str) and raw_terms.strip():
                search_terms.append(raw_terms)

            fallback_scene = fallback["scenes"][min(index, len(fallback["scenes"]) - 1)] if fallback["scenes"] else {}
            search_terms = _build_scene_search_terms(
                query,
                caption,
                story_excerpt,
                str(fallback_scene.get("query", "")),
                str(fallback_scene.get("caption", "")),
            ) + search_terms
            scenes.append(
                {
                    "fraction": fraction,
                    "query": query,
                    "caption": caption,
                    "story_excerpt": story_excerpt,
                    "search_terms": _dedupe_text_items(search_terms),
                }
            )
    if not scenes:
        scenes = fallback["scenes"]
    scenes = sorted(scenes, key=lambda item: item["fraction"])
    target_scene_count = _scene_count_for_text(story_text)
    if len(scenes) > target_scene_count:
        scenes = scenes[:target_scene_count]
    elif len(scenes) < target_scene_count:
        seen = {(scene["query"], scene["caption"]) for scene in scenes}
        for fallback_scene in fallback["scenes"]:
            key = (fallback_scene["query"], fallback_scene["caption"])
            if key in seen:
                continue
            scenes.append(fallback_scene)
            seen.add(key)
            if len(scenes) >= target_scene_count:
                break
        scenes = sorted(scenes, key=lambda item: item["fraction"])

    return {
        "title": title or fallback["title"],
        "description": description or fallback["description"],
        "hashtags": hashtags,
        "hook": hook or fallback["hook"],
        "intro_text": intro_text or fallback["intro_text"],
        "outro_text": outro_text or fallback["outro_text"],
        "scenes": scenes,
    }


def condense_story(raw_text: str) -> str:
    """Condense a story to fit the configured reel duration when needed.

    Args:
        raw_text: Original story text.

    Returns:
        The original or condensed story text ready for narration.
    """
    cleaned = _normalise_whitespace(raw_text)
    if not cleaned:
        return ""

    estimated_duration = _estimate_duration_seconds(cleaned)
    if MAX_REEL_DURATION == 0 or estimated_duration <= MAX_REEL_DURATION:
        return cleaned

    target_seconds = MAX_REEL_DURATION
    if SHOW_INTRO_CARD:
        target_seconds = max(1.0, target_seconds - INTRO_DURATION)
    target_words = max(1, math.floor((target_seconds / 60) * STORY_TARGET_WPM))
    prompt = f"""
You are adapting a Reddit story into a short-form spoken narration.
Condense the story below so it fits within about {target_words} words.
Requirements:
- Keep the strongest emotional beats and the original meaning.
- Write in natural first-person speaking style.
- Keep it concise and easy to read aloud.
- Do not add commentary or labels.

Story:
{cleaned}
""".strip()

    try:
        condensed = _normalise_whitespace(
            _call_story_model(
                prompt,
                timeout_seconds=150,
                temperature=0.25,
                max_output_tokens=max(256, min(1536, target_words * 2)),
            )
        )
        if condensed:
            if _acceptable_condensed_text(condensed, target_words):
                return condensed
            LOGGER.warning(
                "Gemini condensation looked truncated or off-budget (%d words for target %d); using heuristic trim.",
                len(condensed.split()),
                target_words,
            )
    except requests.RequestException as exc:
        LOGGER.warning("Story-model condense request failed, using heuristic trim: %s", exc)

    heuristic = _trim_to_word_budget(cleaned, target_words)
    return heuristic if heuristic else cleaned


def analyse_story(story_text: str) -> dict[str, Any]:
    """Analyse a story and generate reel metadata plus scene prompts.

    Args:
        story_text: Final narration text.

    Returns:
        A metadata dictionary with title, description, hashtags, hook, and scenes.
    """
    target_scene_count = _scene_count_for_text(story_text)
    story_tone = _infer_story_tone(story_text)
    prompt = f"""
Return a JSON object with exactly these keys:
- title: punchy video title, max 70 chars
- description: 2-3 sentence YouTube/TikTok description with CTA
- hashtags: array of 12 strings without #
- hook: the single most gripping sentence from the story
- intro_text: a short spoken opening for the reel, 1 sentence max, 6-16 words, specific to this story, not a generic stock opener, and not a repeat of the hook
- intro_text: a creative cold open that feels like a teaser or trailer line, not a summary; avoid reusing the hook or the story's exact phrasing
- outro_text: a creative closer that feels like a verdict, reaction, or punchy question, not a summary; keep it distinct from the intro and avoid reusing the hook or the story's exact phrasing
- scenes: array of scene objects with keys fraction, query, caption

Scene rules:
- fraction must be a float between 0 and 1
- query must be 2-4 words for Unsplash, concrete, story-specific, and visually searchable
- caption must be a short on-screen label
- include exactly {target_scene_count} scenes, spaced across the story
- avoid generic stock phrases when possible; use nouns and events from the story
- keep the intro and outro distinct from each other and distinct from the hook
- the story tone is {story_tone}

Respond with JSON only.

Story:
{story_text}
""".strip()

    try:
        raw_response = _call_story_model(prompt, expect_json=True, timeout_seconds=150, temperature=0.2, max_output_tokens=2048)
        return _normalise_meta(_extract_json_object(raw_response), story_text)
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        LOGGER.warning("Falling back to heuristic metadata because story-model analysis failed: %s", exc)
        return _fallback_meta(story_text)


def style_story_for_narration(story_text: str) -> str:
    """Lightly adapt story punctuation and sentence cadence for more expressive TTS.

    Args:
        story_text: Final narration text before speech generation.

    Returns:
        A TTS-friendly narration string with the same story facts and stronger pacing.
    """
    cleaned = _normalise_whitespace(story_text)
    if not cleaned or not ENABLE_EXPRESSIVE_NARRATION:
        return cleaned

    story_tone = _infer_story_tone(cleaned)
    style_hint = _story_style_hint(story_tone)

    prompt = f"""
Rewrite the story below for spoken narration in the style of a {style_hint}.

Rules:
- Keep the same story facts, order, and point of view.
- Do not add new events or change the meaning.
- Improve pacing and emotion using only natural sentence breaks and punctuation.
- Keep the wording very close to the original.
- Use plain ASCII punctuation only.
- Return only the narration text.

The story tone is {story_tone}. Match that tone while keeping the narration natural.

Story:
{cleaned}
""".strip()

    try:
        candidate = _normalise_whitespace(
            _call_story_model(
                prompt,
                timeout_seconds=150,
                temperature=0.4,
                max_output_tokens=max(512, min(2048, len(cleaned.split()) * 2)),
            )
        )
        if candidate:
            original_words = len(cleaned.split())
            candidate_words = len(candidate.split())
            if (
                _has_sentence_ending(candidate)
                and (original_words == 0 or abs(candidate_words - original_words) <= max(6, original_words * 0.15))
            ):
                return candidate
            LOGGER.warning(
                "Gemini narration styling looked truncated or too different (%d -> %d words); using heuristic pacing.",
                original_words,
                candidate_words,
            )
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("Expressive narration styling failed, using heuristic pacing: %s", exc)

    return _heuristic_speech_styling(cleaned, story_tone)


def process_story_file(story_path: str | Path) -> dict[str, Any]:
    """Read, condense, and analyse a story file.

    Args:
        story_path: Path to the source `.txt` story file.

    Returns:
        A dictionary containing the raw story text, final narration text, and metadata.
    """
    path = Path(story_path).expanduser().resolve()
    raw_text = path.read_text(encoding="utf-8").strip()
    analysis_text = condense_story(raw_text)
    speech_text = style_story_for_narration(analysis_text)
    meta = analyse_story(analysis_text)
    return {
        "raw_text": raw_text,
        "final_text": analysis_text,
        "speech_text": speech_text,
        "meta": meta,
    }
