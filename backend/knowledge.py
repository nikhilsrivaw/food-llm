"""Safe, optional enrichment for a predicted food label.

The knowledge layer deliberately does not classify images.  It accepts the
label supplied by the team's model (or by a demo caller), normalises it for a
local lookup, and returns a stable response even when no local record or
network is available.

Only Python's standard library is used here.  Live requests are limited to
the public Wikipedia REST summary endpoint and TheMealDB's free ``1`` API
key endpoint.  TheMealDB data is treated as a recipe suggestion rather than
proof that a meal exactly matches a classifier label.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FOODS_PATH = ROOT / "data" / "foods.json"

FETCH_TIMEOUT_SECONDS = 4.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_CACHE_TTL_SECONDS = 300.0

WIKIPEDIA_HOST = "en.wikipedia.org"
MEALDB_HOST = "www.themealdb.com"
WIKIPEDIA_SOURCE_LABEL = "Wikipedia REST summary"
MEALDB_SOURCE_LABEL = "TheMealDB recipe suggestion"
LOCAL_SOURCE_LABEL = "Local foods.json illustrative record"
LOCAL_NUTRITION_SOURCE_LABEL = "Local foods.json illustrative estimate (unverified)"
RECIPE_NOTE = "Recipe suggestion; it is not guaranteed to exactly match the predicted class."

_TAG_RE = re.compile(r"<[^>]*>", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(?:script|style)\b.*?</(?:script|style)\s*>", re.IGNORECASE | re.DOTALL)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")
_NORMALIZE_RE = re.compile(r"[^\w]+", re.UNICODE)

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _load_food_data() -> tuple[dict[str, Any], str]:
    """Load local data without making module import failure fatal."""
    try:
        with FOODS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}, "Illustrative estimates only; no exact serving size is implied."

    if not isinstance(payload, dict):
        return {}, "Illustrative estimates only; no exact serving size is implied."
    foods = payload.get("foods", {})
    if not isinstance(foods, dict):
        foods = {}
    basis = payload.get("servingBasis")
    if not isinstance(basis, str) or not basis.strip():
        basis = "Illustrative estimates only; no exact serving size is implied."
    # Make the non-verified nature explicit while retaining the project's
    # serving-basis wording.  This is intentionally not a precise portion.
    if "unverif" not in basis.lower() and "illustrative" not in basis.lower():
        basis = f"{basis.rstrip('.')} Illustrative and unverified; no exact serving size is implied."
    return foods, basis


FOODS, SERVING_BASIS = _load_food_data()


def normalize_label(label: str) -> str:
    """Return a safe class id for spaces, hyphens, punctuation, and case.

    The function intentionally has no class allow-list.  A future model can
    pass a new label and still receive the same explicit unavailable response.
    """
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    stripped = label.strip()
    if not stripped:
        raise ValueError("label must not be empty")
    class_id = _NORMALIZE_RE.sub("_", stripped.casefold()).strip("_")
    class_id = re.sub(r"_+", "_", class_id)
    if not class_id:
        raise ValueError("label must contain at least one usable character")
    return class_id


def _human_label(class_id: str) -> str:
    return class_id.replace("_", " ").title()


def _clean_text(value: Any, limit: int = 4000) -> str:
    """Convert external text to bounded, plain text suitable for JSON/UI use."""
    if not isinstance(value, str):
        return ""
    text = _SCRIPT_STYLE_RE.sub(" ", value)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _CONTROL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text[:limit]


def _clean_lines(value: Any, limit: int = 4000) -> list[str]:
    if not isinstance(value, str):
        return []
    text = _SCRIPT_STYLE_RE.sub(" ", value)
    text = _TAG_RE.sub("\n", text)
    text = html.unescape(text)
    text = _CONTROL_RE.sub(" ", text)
    lines = []
    for line in re.split(r"\r?\n+", text):
        cleaned = _SPACE_RE.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned[:limit])
    return lines


def _clean_url(value: Any, *, hosts: set[str] | None = None) -> str | None:
    """Return a safe display URL without ever accepting credentials or JS."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 2048 or any(char.isspace() for char in candidate):
        return None
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not hostname or parsed.username or parsed.password or port:
        return None
    if hosts is not None and hostname not in hosts:
        return None
    return candidate


def _allowed_request_url(url: str) -> bool:
    """Permit only the two exact API families this module is allowed to call."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
            return False
    except ValueError:
        return False

    if hostname == WIKIPEDIA_HOST:
        return parsed.path.startswith("/api/rest_v1/page/summary/") and not parsed.query
    if hostname == MEALDB_HOST:
        return parsed.path == "/api/json/v1/1/search.php" and parsed.query.startswith("s=")
    return False


def _cache_ttl() -> float:
    raw = os.environ.get("FOODVISION_KNOWLEDGE_CACHE_TTL", str(DEFAULT_CACHE_TTL_SECONDS))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_CACHE_TTL_SECONDS


def clear_cache() -> None:
    """Clear the in-process public-response cache; useful for tests and reloads."""
    with _cache_lock:
        _cache.clear()


def _fetch_json(url: str) -> Any:
    """Fetch one allow-listed JSON endpoint with a strict timeout."""
    if not _allowed_request_url(url):
        raise ValueError("URL is not on the knowledge layer allow-list")

    now = time.monotonic()
    ttl = _cache_ttl()
    if ttl:
        with _cache_lock:
            cached = _cache.get(url)
            if cached and now - cached[0] < ttl:
                return cached[1]
            if cached:
                _cache.pop(url, None)

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "FoodVision knowledge layer/1.0",
        },
        method="GET",
    )
    response = urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS)
    try:
        raw = response.read()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("knowledge response is invalid or too large")
    payload = json.loads(bytes(raw).decode("utf-8"))
    if ttl:
        with _cache_lock:
            _cache[url] = (time.monotonic(), payload)
    return payload


def _safe_fetch(url: str) -> tuple[Any | None, bool]:
    """Turn every live request error into a simple unavailable result."""
    try:
        return _fetch_json(url), True
    except Exception:
        # Do not return exception text: network errors may contain URLs,
        # proxies, or other implementation details that are not user data.
        return None, False


def _wiki_url(human_label: str) -> str:
    page_name = human_label.replace(" ", "_")
    return f"https://{WIKIPEDIA_HOST}/api/rest_v1/page/summary/{quote(page_name, safe='')}"


def _mealdb_url(human_label: str) -> str:
    return f"https://{MEALDB_HOST}/api/json/v1/1/search.php?{urlencode({'s': human_label})}"


def _local_overview(food: dict[str, Any] | None, human_label: str) -> dict[str, Any]:
    if food:
        text = _clean_text(food.get("origin")) or _clean_text(food.get("description"))
        if text:
            return {
                "text": text,
                "sourceUrl": None,
                "sourceLabel": f"{LOCAL_SOURCE_LABEL}; origin note not independently verified",
                "status": "local_illustrative",
            }
    return {
        "text": f"No verified overview is available for {human_label}.",
        "sourceUrl": None,
        "sourceLabel": "No verified overview source available",
        "status": "unavailable",
    }


def _online_overview(payload: Any, request_url: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    text = _clean_text(payload.get("extract"))
    if not text:
        return None
    content_urls = payload.get("content_urls")
    page_url = None
    if isinstance(content_urls, dict):
        desktop = content_urls.get("desktop")
        if isinstance(desktop, dict):
            page_url = _clean_url(desktop.get("page"), hosts={WIKIPEDIA_HOST})
    source_url = page_url or request_url
    return {
        "text": text,
        "sourceUrl": source_url,
        "sourceLabel": WIKIPEDIA_SOURCE_LABEL,
        "status": "online",
    }


def _local_ingredients(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []
    return [cleaned for item in values if (cleaned := _clean_text(item, limit=500))]


def _local_steps(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [cleaned for item in value if (cleaned := _clean_text(item, limit=1000))]


def _local_recipe(food: dict[str, Any] | None, human_label: str) -> dict[str, Any]:
    if food:
        ingredients = _local_ingredients(food.get("ingredients"))
        steps = _local_steps(food.get("steps"))
        if ingredients or steps:
            return {
                "title": f"{human_label} recipe suggestion",
                "ingredients": ingredients,
                "steps": steps,
                "sourceUrl": None,
                "sourceLabel": f"{LOCAL_SOURCE_LABEL}; recipe is illustrative",
                "status": "local_illustrative",
                "note": RECIPE_NOTE,
            }
    return {
        "title": f"{human_label} recipe suggestion unavailable",
        "ingredients": [],
        "steps": [],
        "sourceUrl": None,
        "sourceLabel": "No verified recipe source available",
        "status": "unavailable",
        "note": RECIPE_NOTE,
    }


def _meal_ingredients(meal: dict[str, Any]) -> list[str]:
    ingredients: list[str] = []
    for index in range(1, 21):
        ingredient = _clean_text(meal.get(f"strIngredient{index}"), limit=300)
        measure = _clean_text(meal.get(f"strMeasure{index}"), limit=200)
        if ingredient:
            ingredients.append(f"{measure} {ingredient}".strip())
    return ingredients


def _meal_steps(meal: dict[str, Any]) -> list[str]:
    instructions = meal.get("strInstructions")
    lines = _clean_lines(instructions, limit=1200)
    if len(lines) > 1:
        return lines
    if lines:
        # Preserve a provider's single instruction as one safe step.  Splitting
        # arbitrary prose into invented steps would add information not given.
        return lines
    return []


def _online_recipe(payload: Any, request_url: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("meals"), list):
        return None
    for meal in payload["meals"]:
        if not isinstance(meal, dict):
            continue
        title = _clean_text(meal.get("strMeal"), limit=300)
        ingredients = _meal_ingredients(meal)
        steps = _meal_steps(meal)
        if not title or not (ingredients or steps):
            continue
        source_url = _clean_url(meal.get("strSource")) or request_url
        image_url = _clean_url(meal.get("strMealThumb"), hosts={MEALDB_HOST})
        recipe: dict[str, Any] = {
            "title": title,
            "ingredients": ingredients,
            "steps": steps,
            "sourceUrl": source_url,
            "sourceLabel": f"{MEALDB_SOURCE_LABEL} (not guaranteed exact class match)",
            "status": "online",
            "note": RECIPE_NOTE,
        }
        if image_url:
            recipe["imageUrl"] = image_url
        return recipe
    return None


def _nutrition(food: dict[str, Any] | None) -> dict[str, Any]:
    values: dict[str, int | float] = {}
    if food and isinstance(food.get("nutrition"), dict):
        for key in ("calories", "protein", "carbs", "fat"):
            value = food["nutrition"].get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values[key] = value
    if values:
        return {
            "status": "illustrative",
            "values": values,
            "basis": SERVING_BASIS,
            "sourceLabel": LOCAL_NUTRITION_SOURCE_LABEL,
        }
    return {
        "status": "unavailable",
        "basis": "No verified nutrition data is available; no exact serving size is implied.",
        "sourceLabel": "No verified nutrition source available",
    }


def _add_source(sources: list[dict[str, Any]], label: str, url: str | None, status: str) -> None:
    candidate = {"label": label, "url": url, "status": status}
    if candidate not in sources:
        sources.append(candidate)


def build_insights(label: str, online: bool = True) -> dict[str, Any]:
    """Build a stable explanatory response for any predicted class label.

    Args:
        label: The raw class label supplied by the classifier.
        online: When true, optionally query only the allow-listed public APIs.
            When false, no network request is attempted.

    Raises:
        TypeError: If ``label`` is not a string.
        ValueError: If ``label`` is empty or contains no usable characters.
    """
    started = time.perf_counter()
    class_id = normalize_label(label)
    human_label = _human_label(class_id)
    record = FOODS.get(class_id)
    food = record if isinstance(record, dict) else None

    overview = _local_overview(food, human_label)
    recipe = _local_recipe(food, human_label)
    sources: list[dict[str, Any]] = []
    if food:
        _add_source(sources, LOCAL_SOURCE_LABEL, None, "illustrative")

    successful_live_sources = 0
    attempted_live_sources = 0
    if online:
        wiki_url = _wiki_url(human_label)
        wiki_payload, wiki_ok = _safe_fetch(wiki_url)
        attempted_live_sources += 1
        online_overview = _online_overview(wiki_payload, wiki_url) if wiki_ok else None
        if online_overview:
            overview = online_overview
            successful_live_sources += 1
            _add_source(sources, WIKIPEDIA_SOURCE_LABEL, overview["sourceUrl"], "online")
        else:
            _add_source(sources, WIKIPEDIA_SOURCE_LABEL, wiki_url, "unavailable")

        meal_url = _mealdb_url(human_label)
        meal_payload, meal_ok = _safe_fetch(meal_url)
        attempted_live_sources += 1
        online_recipe = _online_recipe(meal_payload, meal_url) if meal_ok else None
        if online_recipe:
            recipe = online_recipe
            successful_live_sources += 1
            _add_source(sources, MEALDB_SOURCE_LABEL, recipe["sourceUrl"], "online")
        else:
            _add_source(sources, MEALDB_SOURCE_LABEL, meal_url, "unavailable")

    if not online:
        retrieval_status = "local"
        retrieval_mode = "offline"
    elif successful_live_sources == attempted_live_sources:
        retrieval_status = "online"
        retrieval_mode = "online"
    elif successful_live_sources:
        retrieval_status = "partial"
        retrieval_mode = "online"
    elif food:
        retrieval_status = "fallback_local"
        retrieval_mode = "online"
    else:
        retrieval_status = "unavailable"
        retrieval_mode = "online"

    return {
        "classId": class_id,
        "label": human_label,
        "overview": overview,
        "recipe": recipe,
        "nutrition": _nutrition(food),
        "sources": sources,
        "retrieval": {
            "mode": retrieval_mode,
            "status": retrieval_status,
            "elapsedMs": round((time.perf_counter() - started) * 1000),
        },
        "restaurants": {
            "status": "search_only",
            "note": "Restaurant search is not performed by this knowledge layer.",
        },
    }


__all__ = ["build_insights", "clear_cache", "normalize_label"]
