#!/usr/bin/env python3
"""Build a tabular catalog of videos from a YouTube channel.

This script scans all videos from a channel URL, enriches each entry with
metadata, creates concise summaries from descriptions, and exports the result
to CSV, JSON, and Markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


DEFAULT_CHANNEL_URL = "https://www.youtube.com/channel/UCiV0zikSWzC0nx5HFy-C3lg"
DEFAULT_OUTPUT_DIR = Path("tools/ai_sumit_video_catalog/output")
YOUTUBE_BROWSE_API = "https://www.youtube.com/youtubei/v1/browse"
YOUTUBE_NEXT_API = "https://www.youtube.com/youtubei/v1/next"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass
class VideoStub:
    """Lightweight metadata captured from channel listing."""

    video_id: str
    title: str
    published_relative: str
    duration_text: str
    views_text: str

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def normalize_channel_url(channel_url: str) -> str:
    """Normalize a user-provided channel URL."""
    raw = channel_url.strip()
    if not raw:
        raise ValueError("Channel URL is required.")
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    if "youtube.com" not in parsed.netloc:
        raise ValueError(f"Expected a youtube.com URL, got: {channel_url}")

    path = parsed.path.rstrip("/")
    if path.endswith("/videos"):
        path = path[: -len("/videos")]

    return f"https://www.youtube.com{path}"


def safe_int(value: Any) -> int | None:
    """Convert value to int safely."""
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_human_count(value: str) -> int | None:
    """Parse compact counters like '1.2K views' into integers."""
    if not value:
        return None
    text = str(value).upper().replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*([KMB])?", text)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2)
    multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
    return int(number * multiplier)


def duration_text_to_seconds(duration_text: str) -> int | None:
    """Convert 'HH:MM:SS' or 'MM:SS' style duration to seconds."""
    if not duration_text:
        return None
    text = duration_text.strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", text):
        return None
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return None


def parse_published_date(date_text: str) -> str:
    """Parse watch-page date text into ISO date format where possible."""
    if not date_text:
        return ""
    clean = compact_whitespace(date_text)
    match = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", clean)
    if match:
        clean = match.group(1)
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(clean, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def compact_whitespace(value: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return re.sub(r"\s+", " ", value).strip()


def extract_text(node: Any) -> str:
    """Extract display text from YouTube-style text structures."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    if "simpleText" in node:
        return str(node["simpleText"])
    runs = node.get("runs", [])
    if isinstance(runs, list):
        return "".join(str(run.get("text", "")) for run in runs if isinstance(run, dict))
    return ""


def read_balanced_json_object(source: str, start_idx: int) -> dict[str, Any]:
    """Read one JSON object from source starting at opening brace index."""
    if start_idx < 0 or start_idx >= len(source) or source[start_idx] != "{":
        raise ValueError("Invalid JSON object start index.")

    depth = 0
    in_string = False
    escape = False
    end_idx = None

    for i in range(start_idx, len(source)):
        ch = source[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    if end_idx is None:
        raise ValueError("Could not determine end of JSON object.")

    return json.loads(source[start_idx:end_idx])


def extract_json_after_marker(source: str, marker: str) -> dict[str, Any] | None:
    """Find and parse first JSON object occurring after marker."""
    marker_idx = source.find(marker)
    if marker_idx < 0:
        return None
    object_start = source.find("{", marker_idx + len(marker))
    if object_start < 0:
        return None
    try:
        return read_balanced_json_object(source, object_start)
    except (ValueError, json.JSONDecodeError):
        return None


def parse_presented_by(title: str, description: str, fallback_author: str) -> str:
    """Infer 'Presented by' from description or title patterns."""
    patterns = [
        r"(?im)^\s*(?:presented by|hosted by|speaker|guest|instructor)\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*(?:host|presenter)\s*[:\-]\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, description)
        if not match:
            continue
        candidate = compact_whitespace(match.group(1))
        if 2 <= len(candidate) <= 80:
            return candidate

    title_patterns = [
        r"(?i)\b(?:exclusive interview with|interview with|conversation with|talk with)\s+([^|,\-]{2,80})",
        r"(?i)\bwith\s+([A-Za-z][A-Za-z .'\-]{1,60})$",
    ]
    for pattern in title_patterns:
        title_match = re.search(pattern, title.strip())
        if not title_match:
            continue
        candidate = compact_whitespace(title_match.group(1))
        if 2 <= len(candidate) <= 80:
            return candidate

    return fallback_author or "AI Sumit"


def summarize_description(description: str, title: str) -> str:
    """Create a concise summary from the public description."""
    text = description or ""
    if not text.strip():
        return f"Overview video about: {title}."

    skip_line_re = re.compile(
        r"(https?://|www\.|subscribe|follow|instagram|telegram|whatsapp|discord|"
        r"contact|link|coupon|promo|referral|join now|disclaimer|video courtesy|"
        r"podcast courtesy|copyright)",
        re.IGNORECASE,
    )

    kept_lines: list[str] = []
    for line in text.splitlines():
        clean = compact_whitespace(line)
        if not clean:
            continue
        if clean.startswith("#"):
            continue
        if skip_line_re.search(clean) and len(clean) < 140:
            continue
        kept_lines.append(clean)
        if len(" ".join(kept_lines)) > 550:
            break

    merged = compact_whitespace(" ".join(kept_lines)) or compact_whitespace(text)
    merged = re.sub(r"#\w+", "", merged)
    merged = compact_whitespace(merged)

    if not merged:
        return f"Overview video about: {title}."

    sentences = [s.strip(" -") for s in re.split(r"(?<=[.!?])\s+", merged) if s.strip()]
    selected: list[str] = []
    for sentence in sentences:
        if len(sentence) < 20:
            continue
        selected.append(sentence)
        if len(selected) == 2:
            break

    summary = " ".join(selected) if selected else merged
    if len(summary) > 320:
        summary = summary[:317].rsplit(" ", 1)[0] + "..."
    return summary


class YouTubeChannelScanner:
    """Scanner for public YouTube channel videos."""

    def __init__(self, channel_url: str, timeout_seconds: int = 30) -> None:
        self.channel_url = normalize_channel_url(channel_url)
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.api_key: str | None = None
        self.context: dict[str, Any] | None = None

    @property
    def videos_page_url(self) -> str:
        return f"{self.channel_url}/videos?view=0&sort=dd&flow=grid"

    def _get(self, url: str) -> str:
        backoff = 1.5
        for attempt in range(5):
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.text
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Unexpected GET retry failure.")

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        backoff = 1.5
        for attempt in range(5):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Unexpected POST retry failure.")

    def _bootstrap(self) -> dict[str, Any]:
        html = self._get(self.videos_page_url)

        ytcfg = extract_json_after_marker(html, "ytcfg.set(") or {}
        initial_data = (
            extract_json_after_marker(html, "var ytInitialData = ")
            or extract_json_after_marker(html, 'window["ytInitialData"] = ')
            or extract_json_after_marker(html, "ytInitialData = ")
        )
        if not initial_data:
            raise RuntimeError(
                "Could not extract ytInitialData from channel page. "
                "YouTube likely changed page structure."
            )

        api_key = ytcfg.get("INNERTUBE_API_KEY")
        if not api_key:
            api_match = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
            api_key = api_match.group(1) if api_match else None
        if not api_key:
            raise RuntimeError("Could not determine YouTube INNERTUBE API key.")

        context = ytcfg.get("INNERTUBE_CONTEXT")
        if not context:
            client_version_match = re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', html)
            client_version = (
                client_version_match.group(1) if client_version_match else "2.20260101.00.00"
            )
            context = {
                "client": {
                    "clientName": "WEB",
                    "clientVersion": client_version,
                    "hl": "en",
                    "gl": "US",
                }
            }

        self.api_key = api_key
        self.context = context
        return initial_data

    def _extract_videos_tab_content(self, initial_data: dict[str, Any]) -> dict[str, Any]:
        tabs = (
            initial_data.get("contents", {})
            .get("twoColumnBrowseResultsRenderer", {})
            .get("tabs", [])
        )
        for tab in tabs:
            tab_renderer = tab.get("tabRenderer", {})
            if not tab_renderer:
                continue
            title = str(tab_renderer.get("title", "")).lower()
            if tab_renderer.get("selected") or title == "videos":
                return tab_renderer.get("content", {})
        raise RuntimeError("Could not find selected Videos tab in channel page data.")

    def _extract_video_stubs_from_items(
        self, items: list[dict[str, Any]]
    ) -> tuple[list[VideoStub], str | None]:
        stubs: list[VideoStub] = []
        continuation_token: str | None = None

        for item in items:
            continuation = item.get("continuationItemRenderer")
            if continuation:
                continuation_token = (
                    continuation.get("continuationEndpoint", {})
                    .get("continuationCommand", {})
                    .get("token")
                )
                continue

            rich_item = item.get("richItemRenderer", {}).get("content", {})
            video_renderer = rich_item.get("videoRenderer")
            if video_renderer:
                video_id = str(video_renderer.get("videoId", "")).strip()
                if not video_id:
                    continue

                title = extract_text(video_renderer.get("title"))
                stubs.append(
                    VideoStub(
                        video_id=video_id,
                        title=title,
                        published_relative=extract_text(video_renderer.get("publishedTimeText")),
                        duration_text=extract_text(video_renderer.get("lengthText")),
                        views_text=extract_text(video_renderer.get("viewCountText")),
                    )
                )
                continue

            reel_renderer = rich_item.get("reelItemRenderer")
            if reel_renderer:
                video_id = str(reel_renderer.get("videoId", "")).strip()
                if not video_id:
                    continue
                stubs.append(
                    VideoStub(
                        video_id=video_id,
                        title=extract_text(reel_renderer.get("headline")),
                        published_relative="",
                        duration_text="",
                        views_text=extract_text(reel_renderer.get("viewCountText")),
                    )
                )

        return stubs, continuation_token

    def _extract_continuation_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("onResponseReceivedActions", "onResponseReceivedEndpoints"):
            actions = payload.get(key, [])
            for action in actions:
                append_items = action.get("appendContinuationItemsAction", {}).get(
                    "continuationItems"
                )
                if append_items:
                    return append_items
                reload_items = action.get("reloadContinuationItemsCommand", {}).get(
                    "continuationItems"
                )
                if reload_items:
                    return reload_items
        return []

    def fetch_video_stubs(self, max_videos: int | None = None) -> list[VideoStub]:
        """Fetch all visible videos from the channel's videos tab."""
        initial_data = self._bootstrap()
        tab_content = self._extract_videos_tab_content(initial_data)
        rich_grid = tab_content.get("richGridRenderer", {})
        items = rich_grid.get("contents", [])
        stubs, continuation_token = self._extract_video_stubs_from_items(items)

        seen = {stub.video_id for stub in stubs}

        while continuation_token and (max_videos is None or len(stubs) < max_videos):
            if not self.api_key or not self.context:
                raise RuntimeError("Scanner bootstrap is incomplete.")
            payload = {"context": self.context, "continuation": continuation_token}
            response = self._post_json(f"{YOUTUBE_BROWSE_API}?key={self.api_key}", payload)
            continuation_items = self._extract_continuation_items(response)
            if not continuation_items:
                break

            new_stubs, continuation_token = self._extract_video_stubs_from_items(continuation_items)
            for stub in new_stubs:
                if stub.video_id in seen:
                    continue
                stubs.append(stub)
                seen.add(stub.video_id)
                if max_videos is not None and len(stubs) >= max_videos:
                    break

        return stubs[:max_videos] if max_videos is not None else stubs

    def _find_renderer(self, payload: dict[str, Any], renderer_key: str) -> dict[str, Any]:
        """Depth-first search for the first renderer key in payload."""
        stack: list[Any] = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                renderer = node.get(renderer_key)
                if isinstance(renderer, dict):
                    return renderer
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return {}

    def _extract_watch_renderers(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract primary and secondary info renderers from watch-next payload."""
        contents = (
            payload.get("contents", {})
            .get("twoColumnWatchNextResults", {})
            .get("results", {})
            .get("results", {})
            .get("contents", [])
        )
        primary: dict[str, Any] = {}
        secondary: dict[str, Any] = {}
        for item in contents:
            if not primary:
                maybe_primary = item.get("videoPrimaryInfoRenderer")
                if isinstance(maybe_primary, dict):
                    primary = maybe_primary
            if not secondary:
                maybe_secondary = item.get("videoSecondaryInfoRenderer")
                if isinstance(maybe_secondary, dict):
                    secondary = maybe_secondary

        if not primary:
            primary = self._find_renderer(payload, "videoPrimaryInfoRenderer")
        if not secondary:
            secondary = self._find_renderer(payload, "videoSecondaryInfoRenderer")

        return primary, secondary

    def _extract_like_count(self, primary: dict[str, Any]) -> int | None:
        """Extract like count if present in primary renderer."""
        buttons = (
            primary.get("videoActions", {}).get("menuRenderer", {}).get("topLevelButtons", [])
        )
        for button in buttons:
            model = button.get("segmentedLikeDislikeButtonViewModel", {})
            title = (
                model.get("likeButtonViewModel", {})
                .get("likeButtonViewModel", {})
                .get("toggleButtonViewModel", {})
                .get("toggleButtonViewModel", {})
                .get("defaultButtonViewModel", {})
                .get("buttonViewModel", {})
                .get("title", "")
            )
            like_count = parse_human_count(title)
            if like_count is not None:
                return like_count
        return None

    def fetch_video_details(self, video_id: str) -> dict[str, Any]:
        """Fetch richer metadata for a single video via YouTube watch-next API."""
        if not self.api_key or not self.context:
            raise RuntimeError("Scanner is not initialized. Call fetch_video_stubs first.")

        payload = {"context": self.context, "videoId": video_id}
        response = self._post_json(f"{YOUTUBE_NEXT_API}?key={self.api_key}", payload)
        primary, secondary = self._extract_watch_renderers(response)

        owner_renderer = secondary.get("owner", {}).get("videoOwnerRenderer", {})
        description = (
            secondary.get("attributedDescription", {}).get("content")
            or extract_text(secondary.get("description"))
        )
        title = extract_text(primary.get("title"))
        date_text = extract_text(primary.get("dateText"))
        view_count_text = extract_text(
            primary.get("viewCount", {}).get("videoViewCountRenderer", {}).get("viewCount")
        ) or extract_text(
            primary.get("viewCount", {}).get("videoViewCountRenderer", {}).get("shortViewCount")
        )
        subscriber_count_text = extract_text(owner_renderer.get("subscriberCountText"))
        like_count = self._extract_like_count(primary)
        author = extract_text(owner_renderer.get("title"))
        is_live = "watching" in view_count_text.lower() or "streamed live" in date_text.lower()

        return {
            "title": title,
            "author": author,
            "description": description,
            "view_count": parse_human_count(view_count_text),
            "view_count_text": view_count_text,
            "like_count": like_count,
            "subscriber_count_text": subscriber_count_text,
            "duration_seconds": None,
            "publish_date": parse_published_date(date_text),
            "publish_date_text": date_text,
            "category": "",
            "keywords": [],
            "is_live": is_live,
        }


def to_row(stub: VideoStub, details: dict[str, Any]) -> dict[str, Any]:
    """Merge channel listing and per-video details into one tabular row."""
    title = details.get("title") or stub.title or ""
    description = details.get("description") or ""
    author = details.get("author") or ""
    presenter = parse_presented_by(title=title, description=description, fallback_author=author)
    summary = summarize_description(description=description, title=title)

    duration_seconds = details.get("duration_seconds")
    if not isinstance(duration_seconds, int):
        duration_seconds = duration_text_to_seconds(stub.duration_text)
    duration_minutes: float | None = None
    if isinstance(duration_seconds, int) and duration_seconds > 0:
        duration_minutes = round(duration_seconds / 60, 2)

    keywords = details.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []

    normalized_description = compact_whitespace(description)
    description_excerpt = normalized_description[:220]
    if len(normalized_description) > 220:
        description_excerpt += "..."

    return {
        "Date": details.get("publish_date") or "",
        "Date (text)": details.get("publish_date_text") or "",
        "Title": title,
        "Presented by": presenter,
        "Summary": summary,
        "Video URL": stub.url,
        "Video ID": stub.video_id,
        "Duration (min)": duration_minutes if duration_minutes is not None else "",
        "Views": details.get("view_count") if details.get("view_count") is not None else "",
        "Views (text)": details.get("view_count_text") or stub.views_text,
        "Likes": details.get("like_count") if details.get("like_count") is not None else "",
        "Subscribers": details.get("subscriber_count_text") or "",
        "Published (relative)": stub.published_relative,
        "Duration (text)": stub.duration_text,
        "Category": details.get("category") or "",
        "Tags": "; ".join(str(tag) for tag in keywords),
        "Channel": details.get("author") or "AI Sumit",
        "Is Live": details.get("is_live", False),
        "Description excerpt": description_excerpt,
    }


def markdown_escape(value: Any) -> str:
    """Escape Markdown table separators and normalize whitespace."""
    return compact_whitespace(str(value)).replace("|", r"\|")


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows to CSV file."""
    if not rows:
        raise ValueError("No rows available to write CSV.")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows to JSON file."""
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def write_markdown(rows: list[dict[str, Any]], output_path: Path, limit: int) -> None:
    """Write Markdown table preview."""
    if not rows:
        output_path.write_text("No data available.\n", encoding="utf-8")
        return

    selected = rows if limit <= 0 else rows[:limit]
    columns = list(selected[0].keys())

    lines = []
    lines.append(f"# Video catalog preview ({len(selected)} rows)\n")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")

    for row in selected:
        lines.append("| " + " | ".join(markdown_escape(row.get(col, "")) for col in columns) + " |")

    if 0 < limit < len(rows):
        lines.append(
            f"\nShowing first {limit} rows out of {len(rows)}. "
            "Use CSV/JSON for the full dataset."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan YouTube channel videos and export a tabular catalog."
    )
    parser.add_argument(
        "--channel-url",
        default=DEFAULT_CHANNEL_URL,
        help="YouTube channel URL, e.g. https://www.youtube.com/channel/<id>",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV/JSON/Markdown outputs will be written.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit for number of videos to process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of worker threads for per-video metadata enrichment.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Delay per completed video request to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--markdown-limit",
        type=int,
        default=100,
        help="Rows to include in Markdown preview table (0 = all rows).",
    )
    return parser.parse_args()


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows by Date descending; unknown dates are pushed last."""

    def key(row: dict[str, Any]) -> tuple[int, str]:
        date_value = str(row.get("Date", "")).strip()
        if date_value:
            return (1, date_value)
        return (0, "")

    return sorted(rows, key=key, reverse=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scanner = YouTubeChannelScanner(channel_url=args.channel_url)

    print(f"Scanning channel videos from: {scanner.channel_url}")
    stubs = scanner.fetch_video_stubs(max_videos=args.max_videos)
    print(f"Discovered {len(stubs)} videos from channel listing.")

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_to_stub = {pool.submit(scanner.fetch_video_details, stub.video_id): stub for stub in stubs}

        completed = 0
        total = len(stubs)
        for future in as_completed(future_to_stub):
            stub = future_to_stub[future]
            completed += 1
            try:
                details = future.result()
                rows.append(to_row(stub, details))
            except Exception as exc:  # noqa: BLE001
                failures.append((stub.video_id, str(exc)))
                rows.append(
                    to_row(
                        stub,
                        {
                            "title": stub.title,
                            "author": "AI Sumit",
                            "description": "",
                            "view_count": None,
                            "duration_seconds": None,
                            "publish_date": "",
                            "category": "",
                            "keywords": [],
                            "is_live": False,
                        },
                    )
                )

            if completed % 25 == 0 or completed == total:
                print(f"Processed {completed}/{total} videos...")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    rows = sort_rows(rows)

    csv_path = output_dir / "ai_sumit_video_catalog.csv"
    json_path = output_dir / "ai_sumit_video_catalog.json"
    md_path = output_dir / "ai_sumit_video_catalog.md"

    write_csv(rows, csv_path)
    write_json(rows, json_path)
    write_markdown(rows, md_path, limit=args.markdown_limit)

    print("\nDone.")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Rows:     {len(rows)}")
    if failures:
        print(f"Warnings: {len(failures)} videos had metadata fetch issues.")


if __name__ == "__main__":
    main()
