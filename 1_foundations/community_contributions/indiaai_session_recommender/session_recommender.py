#!/usr/bin/env python3
"""Recommend IndiaAI sessions based on user interests.

This script reads live session data from:
https://impact.indiaai.gov.in/sessions

It uses the same server-action endpoints as the website, then ranks sessions
against free-text interests (for example: "responsible AI in healthcare").
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

SESSIONS_PAGE_URL = "https://impact.indiaai.gov.in/sessions"

# Stable fallbacks in case bundle parsing fails.
KNOWN_ACTION_IDS = {
    "get_filter_options": "004550b504748b9d6e2093aade9689e9c5cc9bacd3",
    "get_session_cards": "7fa64a2aeb1a08c80313a85bda4d63ec4330d74972",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


@dataclass
class Recommendation:
    session: dict[str, Any]
    score: float
    matched_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.session.get("id"),
            "title": self.session.get("title"),
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
            "date": self.session.get("date"),
            "formatted_date": self.session.get("formattedDate"),
            "start_time": self.session.get("formattedStartTime"),
            "end_time": self.session.get("formattedEndTime"),
            "venue": self.session.get("venue"),
            "auditorium": self.session.get("auditorium"),
            "speakers": extract_speakers(self.session),
            "watch_live_url": extract_watch_live_url(self.session),
            "description": normalize_space(self.session.get("description") or ""),
        }


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def extract_speakers(session: dict[str, Any], limit: int = 5) -> list[str]:
    speakers: list[str] = []
    for speaker in session.get("speakers") or []:
        raw = (speaker or {}).get("heading") or (speaker or {}).get("title") or ""
        cleaned = normalize_space(raw)
        if cleaned:
            speakers.append(cleaned)
    return speakers[:limit]


def extract_watch_live_url(session: dict[str, Any]) -> str | None:
    buttons = (session.get("buttons") or {}).get("watchLiveButton") or []
    for button in buttons:
        url = (button or {}).get("url")
        if url:
            return url
    return None


class IndiaAISessionClient:
    """Fetches session data from IndiaAI sessions page actions."""

    def __init__(
        self,
        sessions_url: str = SESSIONS_PAGE_URL,
        timeout_seconds: int = 30,
        retries: int = 3,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self.sessions_url = sessions_url
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.http = requests.Session()
        self._action_ids_cache: dict[str, str] | None = None

    def discover_action_ids(self) -> dict[str, str]:
        if self._action_ids_cache is not None:
            return self._action_ids_cache

        action_ids = dict(KNOWN_ACTION_IDS)
        try:
            html = self._http_get(self.sessions_url)
            chunk_src = self._extract_sessions_chunk_src(html)
            if chunk_src:
                chunk_url = urljoin(self.sessions_url, chunk_src)
                chunk_text = self._http_get(chunk_url)
                resolved = self._extract_action_ids_from_chunk(chunk_text)
                action_ids.update(resolved)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: could not discover action IDs dynamically ({exc}). "
                "Using fallback action IDs.",
                file=sys.stderr,
            )

        self._action_ids_cache = action_ids
        return action_ids

    def _extract_sessions_chunk_src(self, html: str) -> str | None:
        patterns = [
            r'<script[^>]+src="([^"]*app/\(main\)/sessions/page-[^"]+\.js)"',
            r'<script[^>]+src="([^"]*app/%28main%29/sessions/page-[^"]+\.js)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1)
        return None

    def _extract_action_ids_from_chunk(self, chunk_text: str) -> dict[str, str]:
        # Example in bundle:
        # createServerReference)("...id...",a.callServer,void 0,...,"getSessionCards")
        mapping = {
            "getFilterOptionsAction": "get_filter_options",
            "getSessionCards": "get_session_cards",
        }
        discovered: dict[str, str] = {}
        for fn_name, key in mapping.items():
            pattern = (
                r'createServerReference\)\("([a-f0-9]{20,})",[^)]*?"'
                + re.escape(fn_name)
                + r'"\)'
            )
            match = re.search(pattern, chunk_text)
            if match:
                discovered[key] = match.group(1)
        return discovered

    def _http_get(self, url: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.http.get(url, timeout=self.timeout_seconds)
                response.encoding = "utf-8"
                response.raise_for_status()
                return response.text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                break
        raise RuntimeError(f"GET failed for {url}: {last_exc}") from last_exc

    def _invoke_server_action(self, action_id: str, args: list[Any]) -> str:
        payload = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        headers = {
            "Accept": "text/x-component",
            "Content-Type": "text/plain;charset=UTF-8",
            "next-action": action_id,
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.http.post(
                    self.sessions_url,
                    headers=headers,
                    data=payload.encode("utf-8"),
                    timeout=self.timeout_seconds,
                )
                response.encoding = "utf-8"
                if response.status_code != 200:
                    excerpt = normalize_space((response.text or "")[:400])
                    raise RuntimeError(
                        f"Server action failed with status {response.status_code}: "
                        f"{excerpt}"
                    )
                return response.text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                break
        raise RuntimeError(f"Server action call failed: {last_exc}") from last_exc

    def _extract_json_objects(self, rsc_text: str) -> list[Any]:
        """Parse JSON payloads out of a React Server Components stream."""
        objects: list[Any] = []
        decoder = json.JSONDecoder()

        # We search for frame headers like "1:{" or "2:[", then ask a JSON decoder
        # to parse from that exact offset. This remains reliable even when non-JSON
        # chunks are streamed in between.
        for match in re.finditer(r"(\d+):(?=[{\[])", rsc_text):
            json_start = match.end()
            try:
                obj, _ = decoder.raw_decode(rsc_text, json_start)
            except json.JSONDecodeError:
                continue
            objects.append(obj)
        return objects

    def get_filter_options(self) -> dict[str, Any]:
        action_id = self.discover_action_ids()["get_filter_options"]
        raw = self._invoke_server_action(action_id, [])
        for obj in self._extract_json_objects(raw):
            if (
                isinstance(obj, dict)
                and "dates" in obj
                and "times" in obj
                and "venues" in obj
            ):
                return obj
        raise RuntimeError("Could not parse filter options response.")

    def get_session_page(
        self,
        page: int,
        page_size: int = 25,
        filters: dict[str, Any] | None = None,
        search_query: str = "",
    ) -> dict[str, Any]:
        action_id = self.discover_action_ids()["get_session_cards"]
        args: list[Any] = [
            filters or {},
            {"page": page, "pageSize": page_size},
            search_query or "",
        ]
        for attempt in range(1, self.retries + 1):
            raw = self._invoke_server_action(action_id, args)
            for obj in self._extract_json_objects(raw):
                if isinstance(obj, dict) and "sessions" in obj:
                    return obj
            if attempt < self.retries:
                time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"Could not parse session page response for page {page}.")

    def get_all_sessions(
        self,
        *,
        refresh: bool = False,
        cache_path: Path | None = None,
        max_cache_age_seconds: int = 1800,
        page_size: int = 25,
        max_workers: int = 8,
    ) -> list[dict[str, Any]]:
        cache_file = cache_path or default_cache_path()
        if not refresh:
            cached = self._load_cache(cache_file, max_cache_age_seconds)
            if cached is not None:
                return cached

        first_page = self.get_session_page(page=1, page_size=page_size)
        pagination = first_page.get("pagination") or {}
        page_count = int(pagination.get("pageCount") or 1)

        page_payloads: dict[int, dict[str, Any]] = {1: first_page}
        if page_count > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(max_workers, page_count - 1)
            ) as executor:
                future_map = {
                    executor.submit(
                        self.get_session_page, page=page, page_size=page_size
                    ): page
                    for page in range(2, page_count + 1)
                }
                for future in concurrent.futures.as_completed(future_map):
                    page = future_map[future]
                    try:
                        page_payloads[page] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"Warning: failed to fetch page {page}: {exc}",
                            file=sys.stderr,
                        )

        missing_pages = [p for p in range(1, page_count + 1) if p not in page_payloads]
        for page in missing_pages:
            try:
                page_payloads[page] = self.get_session_page(page=page, page_size=page_size)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Warning: still failed to fetch page {page}: {exc}",
                    file=sys.stderr,
                )

        sessions: list[dict[str, Any]] = []
        for page in sorted(page_payloads):
            sessions.extend(page_payloads[page].get("sessions") or [])

        deduped = self._dedupe_sessions_by_id(sessions)
        self._save_cache(cache_file, deduped)
        return deduped

    def _dedupe_sessions_by_id(self, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_ids: set[Any] = set()
        output: list[dict[str, Any]] = []
        for session in sessions:
            session_id = session.get("id")
            if session_id in seen_ids:
                continue
            seen_ids.add(session_id)
            output.append(session)
        return output

    def _load_cache(
        self, cache_file: Path, max_cache_age_seconds: int
    ) -> list[dict[str, Any]] | None:
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            fetched_at = float(payload.get("fetched_at", 0))
            age = time.time() - fetched_at
            if age > max_cache_age_seconds:
                return None
            sessions = payload.get("sessions")
            if isinstance(sessions, list):
                return sessions
            return None
        except Exception:  # noqa: BLE001
            return None

    def _save_cache(self, cache_file: Path, sessions: list[dict[str, Any]]) -> None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fetched_at": time.time(),
                "session_count": len(sessions),
                "sessions": sessions,
            }
            cache_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to write cache: {exc}", file=sys.stderr)


def default_cache_path() -> Path:
    return Path(__file__).resolve().parent / ".cache" / "sessions_cache.json"


def build_interest_profile(interests: str) -> tuple[list[str], list[str], str]:
    normalized_query = normalize_space(interests.lower())
    phrases = [
        p.strip()
        for p in re.split(r"[,;\n|]+", normalized_query)
        if p.strip()
    ]
    if normalized_query and normalized_query not in phrases:
        phrases.insert(0, normalized_query)

    tokens = [t for t in tokenize(normalized_query) if len(t) > 2 and t not in STOPWORDS]
    if not tokens:
        tokens = [t for t in tokenize(normalized_query) if len(t) > 1]

    # Keep phrase list short and unique while preserving order.
    seen: set[str] = set()
    unique_phrases: list[str] = []
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        unique_phrases.append(phrase)

    return tokens, unique_phrases, normalized_query


def session_text_blob(session: dict[str, Any]) -> str:
    fields = [
        session.get("title") or "",
        session.get("heading") or "",
        session.get("subHeading") or "",
        session.get("descriptionTitle") or "",
        session.get("description") or "",
        session.get("venue") or "",
        session.get("auditorium") or "",
    ]
    speaker_text = " ".join(extract_speakers(session, limit=20))
    partner_text = " ".join(
        normalize_space((partner or {}).get("title") or "")
        for partner in (session.get("knowledgePartners") or [])
    )
    fields.extend([speaker_text, partner_text])
    return normalize_space(" ".join(fields)).lower()


def compute_idf(tokens_per_session: list[list[str]]) -> dict[str, float]:
    doc_count = max(len(tokens_per_session), 1)
    df: Counter[str] = Counter()
    for session_tokens in tokens_per_session:
        for token in set(session_tokens):
            df[token] += 1

    idf: dict[str, float] = {}
    for token, freq in df.items():
        idf[token] = math.log((1 + doc_count) / (1 + freq)) + 1.0
    return idf


def score_session(
    session: dict[str, Any],
    query_tokens: list[str],
    query_phrases: list[str],
    query_text: str,
    idf: dict[str, float],
) -> tuple[float, list[str]]:
    weighted_counts: dict[str, float] = defaultdict(float)

    field_weights: list[tuple[str, float]] = [
        ("title", 4.0),
        ("heading", 2.5),
        ("subHeading", 2.0),
        ("descriptionTitle", 2.0),
        ("description", 1.6),
        ("venue", 0.8),
        ("auditorium", 0.8),
    ]
    for field, weight in field_weights:
        for token in tokenize(str(session.get(field) or "")):
            weighted_counts[token] += weight

    for speaker in extract_speakers(session, limit=20):
        for token in tokenize(speaker):
            weighted_counts[token] += 1.8

    for partner in session.get("knowledgePartners") or []:
        partner_title = normalize_space((partner or {}).get("title") or "")
        for token in tokenize(partner_title):
            weighted_counts[token] += 1.2

    score = 0.0
    matched: dict[str, float] = {}
    for token in query_tokens:
        if token in weighted_counts:
            contribution = weighted_counts[token] * idf.get(token, 1.0)
            score += contribution
            matched[token] = matched.get(token, 0.0) + contribution

    title_lower = normalize_space(str(session.get("title") or "")).lower()
    full_text = session_text_blob(session)
    for phrase in query_phrases:
        if len(phrase) < 3:
            continue
        if phrase in title_lower:
            score += 6.0
            matched[phrase] = max(matched.get(phrase, 0.0), 6.0)
        elif phrase in full_text:
            score += 2.5
            matched[phrase] = max(matched.get(phrase, 0.0), 2.5)

    if query_text:
        title_ratio = SequenceMatcher(None, query_text, title_lower).ratio()
        body_ratio = SequenceMatcher(None, query_text, full_text[:1200]).ratio()
        score += (2.0 * title_ratio) + (0.8 * body_ratio)

    ranked_matches = [
        term
        for term, _ in sorted(matched.items(), key=lambda item: item[1], reverse=True)
    ]
    return score, ranked_matches[:6]


def recommend_sessions(
    sessions: list[dict[str, Any]],
    interests: str,
    *,
    top_n: int = 10,
) -> list[Recommendation]:
    query_tokens, query_phrases, query_text = build_interest_profile(interests)
    if not query_text:
        return []

    tokenized_sessions = [tokenize(session_text_blob(session)) for session in sessions]
    idf = compute_idf(tokenized_sessions)

    ranked: list[Recommendation] = []
    for session in sessions:
        score, matched_terms = score_session(
            session, query_tokens, query_phrases, query_text, idf
        )
        if score <= 0:
            continue
        ranked.append(
            Recommendation(session=session, score=score, matched_terms=matched_terms)
        )

    ranked.sort(key=lambda rec: rec.score, reverse=True)
    return ranked[:top_n]


def format_recommendations(
    recommendations: list[Recommendation], interests: str, total_sessions: int
) -> str:
    if not recommendations:
        return (
            "No strong matches were found.\n"
            "Try broader interests (for example: 'healthcare AI, policy, governance')."
        )

    lines: list[str] = []
    lines.append(
        f"Top {len(recommendations)} recommended sessions for: \"{interests}\""
    )
    lines.append(f"Scanned unique sessions: {total_sessions}")
    lines.append("")

    for idx, rec in enumerate(recommendations, start=1):
        session = rec.session
        title = normalize_space(str(session.get("title") or "Untitled session"))
        when = (
            f"{session.get('formattedDate') or session.get('date') or 'Date TBD'} | "
            f"{session.get('formattedStartTime') or session.get('startTime') or 'TBD'}"
        )
        if session.get("formattedEndTime") or session.get("endTime"):
            when += (
                f" - {session.get('formattedEndTime') or session.get('endTime') or ''}"
            )

        where = " | ".join(
            [part for part in [session.get("venue"), session.get("auditorium")] if part]
        )
        speakers = ", ".join(extract_speakers(session))
        description = normalize_space(str(session.get("description") or ""))[:260]
        watch_live_url = extract_watch_live_url(session)

        lines.append(f"{idx}. {title}")
        lines.append(f"   Score: {rec.score:.2f}")
        lines.append(f"   When: {when}")
        if where:
            lines.append(f"   Where: {where}")
        if speakers:
            lines.append(f"   Speakers: {speakers}")
        if rec.matched_terms:
            lines.append(f"   Why matched: {', '.join(rec.matched_terms)}")
        if description:
            lines.append(f"   About: {description}")
        if watch_live_url:
            lines.append(f"   Watch: {watch_live_url}")
        lines.append("")

    return "\n".join(lines).strip()


def apply_optional_filters(
    sessions: list[dict[str, Any]],
    *,
    date_filter: str | None = None,
    venue_filter: str | None = None,
) -> list[dict[str, Any]]:
    filtered = sessions
    if date_filter:
        filtered = [s for s in filtered if (s.get("date") or "") == date_filter]
    if venue_filter:
        needle = venue_filter.lower().strip()
        filtered = [s for s in filtered if needle in (s.get("venue") or "").lower()]
    return filtered


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan IndiaAI sessions and recommend sessions for your interests."
    )
    parser.add_argument(
        "interests",
        nargs="?",
        default="",
        help='Interests text. Example: "responsible AI, healthcare, skilling"',
    )
    parser.add_argument(
        "-i",
        "--interest",
        dest="interest_option",
        default="",
        help="Interests text (same as positional argument).",
    )
    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=10,
        help="Number of recommendations to show. Default: 10",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Optional exact date filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--venue",
        default="",
        help="Optional venue keyword filter (example: 'Bharat Mandapam').",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cache and fetch fresh session data.",
    )
    parser.add_argument(
        "--cache-minutes",
        type=int,
        default=30,
        help="Cache freshness window in minutes. Default: 30",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print recommendations in JSON format.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save recommendation JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    interests = args.interest_option or args.interests
    interests = normalize_space(interests)
    if not interests:
        print(
            "Please provide interests. Example:\n"
            '  python3 session_recommender.py "responsible AI, healthcare, governance"'
        )
        return 2

    client = IndiaAISessionClient()
    sessions = client.get_all_sessions(
        refresh=args.refresh,
        max_cache_age_seconds=max(0, args.cache_minutes) * 60,
    )

    filtered_sessions = apply_optional_filters(
        sessions,
        date_filter=args.date or None,
        venue_filter=args.venue or None,
    )
    if not filtered_sessions:
        print("No sessions found with the requested date/venue filters.")
        return 1

    recommendations = recommend_sessions(
        filtered_sessions,
        interests,
        top_n=max(1, args.top),
    )
    recommendation_payload = [rec.to_dict() for rec in recommendations]

    if args.save_json:
        output_file = Path(args.save_json).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(recommendation_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON recommendations to: {output_file}")

    if args.json:
        print(json.dumps(recommendation_payload, ensure_ascii=False, indent=2))
    else:
        print(
            format_recommendations(
                recommendations,
                interests=interests,
                total_sessions=len(filtered_sessions),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
