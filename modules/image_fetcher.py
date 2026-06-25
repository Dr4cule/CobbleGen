from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from config import OUTPUT_WIDTH, TEMP_DIR, UNSPLASH_ACCESS_KEY, UNSPLASH_MAX_IMAGES, UNSPLASH_ORIENTATION, UNSPLASH_PER_PAGE, UNSPLASH_REQUEST_DELAY, UNSPLASH_SEARCH_URL


LOGGER = logging.getLogger(__name__)
UNSPLASH_SEARCH_CACHE: dict[str, dict[str, Any]] = {}

SEARCH_STOPWORDS = {
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
    "not",
    "do",
    "did",
    "does",
    "can",
    "could",
    "would",
    "should",
    "will",
    "just",
    "story",
    "scene",
    "photo",
    "image",
    "shot",
    "people",
    "person",
    "thing",
    "things",
}


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "scene"


def _search_unsplash(query: str) -> dict[str, Any]:
    cache_key = _normalise_text(query).lower()
    if cache_key in UNSPLASH_SEARCH_CACHE:
        return UNSPLASH_SEARCH_CACHE[cache_key]

    params = {
        "query": query,
        "per_page": UNSPLASH_PER_PAGE,
        "orientation": UNSPLASH_ORIENTATION,
        "client_id": UNSPLASH_ACCESS_KEY,
    }
    response = requests.get(UNSPLASH_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    time.sleep(UNSPLASH_REQUEST_DELAY)
    payload = response.json()
    UNSPLASH_SEARCH_CACHE[cache_key] = payload
    return payload


def _download_image(url: str, output_path: Path) -> None:
    response = requests.get(url, params={"w": OUTPUT_WIDTH}, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    time.sleep(UNSPLASH_REQUEST_DELAY)


def _is_unsplash_access_error(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code in {401, 403, 429}


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", str(value or ""))).strip()


def _search_phrase(value: str) -> str:
    cleaned = _normalise_text(value)
    if not cleaned:
        return ""

    words = [word.strip("'\"-") for word in cleaned.split() if word.strip()]
    content = [word for word in words if len(word) > 2 and word.lower() not in SEARCH_STOPWORDS]
    if not content:
        content = [word for word in words if len(word) > 2]
    return " ".join(content[:4]).strip()


def _dedupe_queries(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalise_text(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _scene_candidate_queries(scene: dict[str, Any]) -> list[str]:
    query = str(scene.get("query") or "").strip()
    caption = str(scene.get("caption") or "").strip()
    story_excerpt = str(scene.get("story_excerpt") or "").strip()
    query_phrase = _search_phrase(query)
    caption_phrase = _search_phrase(caption)
    excerpt_phrase = _search_phrase(story_excerpt)

    # The LLM's `query` is the cleanest, most story-specific 2-4 word phrase, so
    # try it first. Combining it with the caption sharpens relevance. Only after
    # those do we fall back to the noisier auto-extracted search_terms and
    # excerpt fragments, so a junk fragment never crowds out the good query.
    raw_terms = scene.get("search_terms", [])
    extra_terms: list[str] = []
    if isinstance(raw_terms, str):
        extra_terms.append(_search_phrase(raw_terms))
    elif isinstance(raw_terms, list):
        extra_terms.extend(_search_phrase(str(term)) for term in raw_terms if str(term).strip())

    terms = [
        query,
        query_phrase,
        f"{query_phrase} {caption_phrase}" if query_phrase and caption_phrase else "",
        caption_phrase,
        f"{query_phrase} {excerpt_phrase}" if query_phrase and excerpt_phrase else "",
        *extra_terms,
        excerpt_phrase,
    ]
    return _dedupe_queries(terms)[:5]


def _result_search_text(result: dict[str, Any]) -> str:
    fields: list[str] = [str(result.get("alt_description") or ""), str(result.get("description") or "")]
    tags = result.get("tags", [])
    if isinstance(tags, list):
        for tag in tags[:6]:
            if isinstance(tag, dict):
                fields.append(str(tag.get("title") or tag.get("name") or ""))
            else:
                fields.append(str(tag))
    return _normalise_text(" ".join(fields))


def _tokenise(value: str) -> set[str]:
    return {
        word.lower()
        for word in _normalise_text(value).split()
        if len(word) > 2 and word.lower() not in SEARCH_STOPWORDS
    }


def _score_result(scene: dict[str, Any], candidate_query: str, result: dict[str, Any]) -> float:
    candidate_tokens = _tokenise(candidate_query)
    scene_tokens = _tokenise(" ".join(str(scene.get(field) or "") for field in ("query", "caption", "story_excerpt")))
    result_tokens = _tokenise(_result_search_text(result))
    result_text = _result_search_text(result).lower()

    if not result_tokens:
        return -10.0

    candidate_overlap = candidate_tokens & result_tokens
    scene_overlap = scene_tokens & result_tokens

    score = 0.0
    # Heavy weight on the LLM's chosen search phrase hitting the result.
    score += len(candidate_overlap) * 4.0
    # Story-specific tokens (from caption/excerpt) matching the result.
    score += len(scene_overlap) * 2.0

    # Big bonus when the exact candidate phrase appears verbatim in the result
    # text — this means the photo was tagged with the story's own subject.
    candidate_phrase = _normalise_text(candidate_query).lower()
    if candidate_phrase and len(candidate_phrase) >= 6 and candidate_phrase in result_text:
        score += 6.0

    # Penalise results that share zero story-specific tokens with the scene —
    # those are the ones that look like a random stock photo.
    if scene_tokens and not scene_overlap:
        score -= 5.0

    # Mild bonus for subset match (rare on Unsplash, but rewarding when it happens).
    if candidate_tokens and candidate_tokens <= result_tokens:
        score += 3.0

    if result.get("alt_description"):
        score += 0.75
    if result.get("description"):
        score += 0.5
    if result.get("tags"):
        score += 0.5

    return score


def _best_unsplash_result(scene: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    best_choice: tuple[float, str, dict[str, Any]] | None = None
    for candidate_query in _scene_candidate_queries(scene):
        try:
            payload = _search_unsplash(candidate_query)
        except requests.RequestException as exc:
            LOGGER.warning("Unsplash search failed for query '%s': %s", candidate_query, exc)
            continue

        results = payload.get("results", [])
        if not results:
            continue

        for result in results[:UNSPLASH_PER_PAGE]:
            score = _score_result(scene, candidate_query, result)
            if best_choice is None or score > best_choice[0]:
                best_choice = (score, candidate_query, result)

        if best_choice is not None and best_choice[0] >= 12.0:
            break

    if best_choice is None:
        return None
    return best_choice[1], best_choice[2]


def fetch_images_for_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fetch Unsplash context images for the provided scene prompts.

    Args:
        scenes: Scene dictionaries containing at least `query`, `caption`, and `fraction`.

    Returns:
        Scene dictionaries enriched with `image_path` and `photographer`.
    """
    if not UNSPLASH_ACCESS_KEY:
        LOGGER.warning("UNSPLASH_ACCESS_KEY is not set; skipping context image fetch.")
        return [{**scene, "image_path": None, "photographer": None} for scene in scenes]

    image_dir = TEMP_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    # Clear any previous images so each story run fetches fresh context images
    for existing in list(image_dir.iterdir()):
        try:
            if existing.is_file():
                existing.unlink()
        except Exception:
            # Non-fatal: if a file cannot be removed, continue and let new files overwrite
            pass

    enriched_scenes: list[dict[str, Any]] = []
    max_images = len(scenes) if UNSPLASH_MAX_IMAGES <= 0 else min(UNSPLASH_MAX_IMAGES, len(scenes))
    for index, scene in enumerate(scenes):
        if index >= max_images:
            enriched_scenes.append({**scene, "image_path": None, "photographer": None})
            continue

        query = str(scene.get("query") or "").strip()
        if not query and not scene.get("search_terms") and not scene.get("story_excerpt"):
            enriched_scenes.append({**scene, "image_path": None, "photographer": None})
            continue

        try:
            best_result = _best_unsplash_result(scene)
            if best_result is None:
                LOGGER.warning("No Unsplash results found for query '%s'.", query)
                enriched_scenes.append({**scene, "image_path": None, "photographer": None})
                continue

            resolved_query, result = best_result
            output_path = image_dir / f"{index + 1:02d}_{_safe_stem(resolved_query)}.jpg"
            _download_image(result["urls"]["regular"], output_path)

            download_location = result.get("links", {}).get("download_location")
            if download_location:
                try:
                    tracking = requests.get(download_location, params={"client_id": UNSPLASH_ACCESS_KEY}, timeout=30)
                    tracking.raise_for_status()
                    time.sleep(UNSPLASH_REQUEST_DELAY)
                except requests.RequestException as exc:
                    LOGGER.warning("Unsplash download tracking failed for query '%s': %s", resolved_query, exc)

            photographer_name = result.get("user", {}).get("name") or "Unknown"
            LOGGER.info(
                "Selected Unsplash image for scene %d with query '%s': %s",
                index + 1,
                resolved_query,
                result.get("alt_description") or result.get("description") or resolved_query,
            )
            enriched_scenes.append(
                {
                    **scene,
                    "image_path": output_path,
                    "photographer": f"{photographer_name} on Unsplash",
                }
            )
        except requests.RequestException as exc:
            LOGGER.warning("Unsplash fetch failed for query '%s': %s", query, exc)
            enriched_scenes.append({**scene, "image_path": None, "photographer": None})
            if _is_unsplash_access_error(exc):
                LOGGER.warning("Stopping Unsplash fetches after access/rate-limit error.")
                for remaining_scene in scenes[index + 1 :]:
                    enriched_scenes.append({**remaining_scene, "image_path": None, "photographer": None})
                break

    return enriched_scenes
