import asyncio
import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
import scrapetube
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from ..utils.config import Config

logger = logging.getLogger(__name__)


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


@dataclass
class ChannelVideo:
    """Basic metadata for a YouTube video from a channel feed."""

    video_id: str
    title: str
    url: str
    published: Optional[str] = None
    duration: Optional[str] = None
    description_snippet: Optional[str] = None


@dataclass
class VideoSummary:
    """Summary record generated for a channel video."""

    video_id: str
    title: str
    url: str
    published: Optional[str]
    duration: Optional[str]
    summary: str
    content_source: str
    summary_method: str
    transcript_available: bool
    input_characters: int
    notes: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class YouTubeChannelSummarizer:
    """
    End-to-end channel summarization pipeline.

    Workflow:
    1) Scrape every video from a channel via scrapetube
    2) For each video, try to fetch transcript text
    3) Summarize transcript (or fallback to title + description)
    4) Export JSON + Markdown reports
    """

    def __init__(
        self,
        config: Config,
        summary_mode: str = "auto",
        preferred_languages: Optional[Sequence[str]] = None,
        request_timeout: int = 20,
        max_summary_sentences: int = 6,
        fetch_transcripts: bool = True,
    ):
        self.config = config
        self.summary_mode = summary_mode.lower().strip()
        if self.summary_mode not in {"auto", "llm", "extractive"}:
            raise ValueError("summary_mode must be one of: auto, llm, extractive")

        self.preferred_languages = list(preferred_languages or ["en", "hi"])
        self.request_timeout = request_timeout
        self.max_summary_sentences = max_summary_sentences
        self.fetch_transcripts = fetch_transcripts

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

        self._llm_summarizer: Optional[Any] = None
        self._llm_available = False
        self._initialize_llm_if_needed()

    def _initialize_llm_if_needed(self) -> None:
        """Initialize LLM summarization when configured to use it."""
        if self.summary_mode == "extractive":
            return

        try:
            from .summarizer import TranscriptSummarizer

            self._llm_summarizer = TranscriptSummarizer(self.config)
            health = self._llm_summarizer.check_service_health()
            self._llm_available = bool(
                health.get("connection_ok") and health.get("model_available")
            )
            if not self._llm_available:
                logger.warning(
                    "LLM health check failed; falling back to extractive summaries. "
                    "Health payload: %s",
                    health,
                )
        except Exception as exc:
            logger.warning(
                "Could not initialize LLM summarizer. Falling back to extractive mode: %s",
                exc,
            )
            self._llm_available = False

        if self.summary_mode == "llm" and not self._llm_available:
            raise RuntimeError(
                "LLM mode requested, but no healthy LLM backend was detected."
            )

    def fetch_channel_videos(
        self, channel_url: str, max_videos: Optional[int] = None
    ) -> List[ChannelVideo]:
        """Fetch channel videos via scrapetube."""
        kwargs: Dict[str, Any] = {"content_type": "videos"}
        if max_videos:
            kwargs["limit"] = max_videos

        channel_id = self.extract_channel_id(channel_url)
        if channel_id:
            kwargs["channel_id"] = channel_id
        else:
            kwargs["channel_url"] = channel_url

        logger.info("Fetching videos from channel: %s", channel_url)
        raw_videos = scrapetube.get_channel(**kwargs)
        videos: List[ChannelVideo] = []
        seen: set = set()

        for raw in raw_videos:
            video_id = raw.get("videoId")
            if not video_id or video_id in seen:
                continue

            seen.add(video_id)
            title = self._extract_text(raw.get("title")) or "Untitled"
            published = self._extract_text(raw.get("publishedTimeText"))
            duration = self._extract_text(raw.get("lengthText"))
            description_snippet = self._extract_text(raw.get("descriptionSnippet"))
            videos.append(
                ChannelVideo(
                    video_id=video_id,
                    title=title,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    published=published,
                    duration=duration,
                    description_snippet=description_snippet,
                )
            )

        logger.info("Discovered %d videos for channel", len(videos))
        return videos

    def summarize_channel(
        self,
        channel_url: str,
        output_dir: str,
        max_videos: Optional[int] = None,
        save_every: int = 20,
    ) -> Dict[str, Any]:
        """Run the full channel summarization workflow."""
        videos = self.fetch_channel_videos(channel_url=channel_url, max_videos=max_videos)
        summaries: List[VideoSummary] = []

        for idx, video in enumerate(videos, start=1):
            logger.info("Processing video %d/%d: %s", idx, len(videos), video.title)
            summaries.append(self.summarize_video(video))

            if save_every > 0 and idx % save_every == 0:
                self._write_outputs(channel_url, output_dir, summaries, len(videos))

        output = self._write_outputs(channel_url, output_dir, summaries, len(videos))
        return output

    def summarize_video(self, video: ChannelVideo) -> VideoSummary:
        """Summarize a single video with transcript-first fallback logic."""
        notes: List[str] = []
        transcript_text = ""
        description = (video.description_snippet or "").strip()

        if self.fetch_transcripts:
            try:
                player_response = self._fetch_player_response(video.video_id)
                watch_description = (
                    player_response.get("videoDetails", {}).get("shortDescription", "") or ""
                ).strip()
                if watch_description:
                    description = watch_description
                transcript_text, transcript_notes = self._fetch_best_transcript(
                    video.video_id, player_response
                )
                notes.extend(transcript_notes)
            except Exception as exc:
                notes.append(f"Failed to fetch watch-page metadata: {exc}")
        else:
            notes.append("Transcript fetching disabled by configuration.")

        if transcript_text:
            input_text = transcript_text
            content_source = "transcript"
        elif description:
            input_text = f"{video.title}\n\n{description}"
            content_source = "description"
            notes.append("Transcript unavailable; summarized title + description.")
        else:
            input_text = video.title
            content_source = "title"
            notes.append("Transcript and description unavailable; summarized title only.")

        try:
            summary_text, summary_method = self._summarize_text(input_text)
            return VideoSummary(
                video_id=video.video_id,
                title=video.title,
                url=video.url,
                published=video.published,
                duration=video.duration,
                summary=summary_text,
                content_source=content_source,
                summary_method=summary_method,
                transcript_available=content_source == "transcript",
                input_characters=len(input_text),
                notes=notes,
                error=None,
            )
        except Exception as exc:
            return VideoSummary(
                video_id=video.video_id,
                title=video.title,
                url=video.url,
                published=video.published,
                duration=video.duration,
                summary="",
                content_source=content_source,
                summary_method="failed",
                transcript_available=content_source == "transcript",
                input_characters=len(input_text),
                notes=notes,
                error=str(exc),
            )

    def _summarize_text(self, text: str) -> Tuple[str, str]:
        """Summarize text using LLM when available, else extractive fallback."""
        if self._llm_available and self._llm_summarizer is not None:
            try:
                result = asyncio.run(self._llm_summarizer.summarize_text(text))
                if not result.error and result.summary.strip():
                    return result.summary.strip(), "llm"
                logger.warning("LLM summarization failed, falling back: %s", result.error)
            except Exception as exc:
                logger.warning("LLM summarization exception, falling back: %s", exc)

        return self.extractive_summary(text, self.max_summary_sentences), "extractive"

    def _fetch_best_transcript(
        self, video_id: str, player_response: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Attempt transcript extraction through multiple methods:
        1) youtube-transcript-api
        2) caption track URLs from watch page
        """
        notes: List[str] = []

        # Method 1: youtube-transcript-api
        transcript, error = self._fetch_transcript_with_api(video_id)
        if transcript:
            return transcript, notes
        if error:
            notes.append(f"youtube-transcript-api unavailable: {error}")

        # Method 2: watch-page caption tracks
        tracks = (
            player_response.get("captions", {})
            .get("playerCaptionsTracklistRenderer", {})
            .get("captionTracks", [])
        )
        if not tracks:
            notes.append("No caption tracks exposed in watch page response.")
            return "", notes

        track = self._pick_caption_track(tracks, self.preferred_languages)
        if not track:
            notes.append("Caption tracks available but no preferred language matched.")
            return "", notes

        caption_text, caption_error = self._download_caption_track(track.get("baseUrl", ""))
        if caption_text:
            return caption_text, notes
        if caption_error:
            notes.append(f"Caption URL retrieval failed: {caption_error}")
        return "", notes

    def _fetch_transcript_with_api(self, video_id: str) -> Tuple[str, Optional[str]]:
        """Try transcript retrieval using youtube-transcript-api."""
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=self.preferred_languages)
            raw = fetched.to_raw_data()
            lines = [item.get("text", "").strip() for item in raw if item.get("text", "").strip()]
            transcript = self._normalize_transcript_lines(lines)
            if transcript:
                return transcript, None
            return "", "Transcript API returned an empty payload."
        except (
            RequestBlocked,
            IpBlocked,
            NoTranscriptFound,
            TranscriptsDisabled,
            CouldNotRetrieveTranscript,
            VideoUnavailable,
        ) as exc:
            return "", str(exc).splitlines()[0]
        except Exception as exc:
            return "", str(exc)

    def _fetch_player_response(self, video_id: str) -> Dict[str, Any]:
        """Fetch watch page and parse ytInitialPlayerResponse JSON."""
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        response = self.session.get(watch_url, timeout=self.request_timeout)
        response.raise_for_status()
        player_response = self.extract_json_after_marker(
            response.text, "ytInitialPlayerResponse = "
        )
        if player_response is None:
            raise ValueError("ytInitialPlayerResponse marker not found.")
        return player_response

    def _download_caption_track(self, base_url: str) -> Tuple[str, Optional[str]]:
        """Download caption track and parse its payload format."""
        if not base_url:
            return "", "Caption track has no baseUrl."

        # Try multiple payload formats because availability can vary by video.
        candidates = [
            self._set_query_param(base_url, "fmt", "json3"),
            self._set_query_param(base_url, "fmt", "srv3"),
            self._set_query_param(base_url, "fmt", "vtt"),
            base_url,
        ]

        last_error: Optional[str] = None
        for url in candidates:
            try:
                response = self.session.get(url, timeout=self.request_timeout)
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    continue
                payload = (response.text or "").strip()
                if not payload:
                    last_error = "Caption payload was empty."
                    continue

                transcript = self._parse_caption_payload(payload)
                if transcript:
                    return transcript, None
                last_error = "Caption payload could not be parsed into text."
            except Exception as exc:
                last_error = str(exc)

        return "", last_error

    def _parse_caption_payload(self, payload: str) -> str:
        """Parse caption payload from json3/xml/vtt and return normalized transcript."""
        if not payload:
            return ""

        stripped = payload.lstrip()
        if stripped.startswith("{"):
            return self._parse_json3_caption_payload(payload)
        if stripped.startswith("<"):
            return self._parse_xml_caption_payload(payload)
        if "WEBVTT" in stripped[:100]:
            return self._parse_vtt_caption_payload(payload)
        return ""

    def _parse_json3_caption_payload(self, payload: str) -> str:
        """Parse YouTube json3 caption format."""
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ""

        lines: List[str] = []
        for event in data.get("events", []):
            segments = event.get("segs")
            if not segments:
                continue
            line = "".join(segment.get("utf8", "") for segment in segments)
            line = html.unescape(line).replace("\n", " ").strip()
            if line:
                lines.append(line)
        return self._normalize_transcript_lines(lines)

    def _parse_xml_caption_payload(self, payload: str) -> str:
        """Parse XML caption payload format."""
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return ""

        lines: List[str] = []
        for node in root.findall(".//text"):
            node_text = "".join(node.itertext())
            node_text = html.unescape(node_text).replace("\n", " ").strip()
            if node_text:
                lines.append(node_text)
        return self._normalize_transcript_lines(lines)

    def _parse_vtt_caption_payload(self, payload: str) -> str:
        """Parse WEBVTT caption payload format."""
        lines: List[str] = []
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if (
                not line
                or line == "WEBVTT"
                or "-->" in line
                or re.fullmatch(r"\d+", line)
                or line.startswith("Kind:")
                or line.startswith("Language:")
            ):
                continue
            lines.append(line)
        return self._normalize_transcript_lines(lines)

    def _normalize_transcript_lines(self, lines: List[str]) -> str:
        """Clean transcript lines and remove immediate duplicates."""
        cleaned: List[str] = []
        previous = ""
        for line in lines:
            normalized = re.sub(r"\s+", " ", line).strip()
            if not normalized:
                continue
            if normalized == previous:
                continue
            previous = normalized
            cleaned.append(normalized)
        return " ".join(cleaned).strip()

    def _pick_caption_track(
        self, tracks: List[Dict[str, Any]], preferred_languages: Sequence[str]
    ) -> Optional[Dict[str, Any]]:
        """Select best caption track based on preferred languages and manual over ASR."""
        if not tracks:
            return None

        def language_priority(track: Dict[str, Any]) -> int:
            language_code = (track.get("languageCode") or "").lower()
            for idx, preferred in enumerate(preferred_languages):
                preferred_code = preferred.lower()
                if language_code == preferred_code or language_code.startswith(
                    preferred_code + "-"
                ):
                    return idx
            return len(preferred_languages) + 1

        ranked = sorted(
            tracks,
            key=lambda track: (
                language_priority(track),
                1 if track.get("kind") == "asr" else 0,
            ),
        )
        return ranked[0] if ranked else None

    @staticmethod
    def extractive_summary(text: str, max_sentences: int = 6) -> str:
        """Generate a lightweight extractive summary without external model calls."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""

        sentence_candidates = re.split(r"(?<=[\.\!\?\u0964])\s+|\n+", normalized)
        sentences = [sentence.strip() for sentence in sentence_candidates if sentence.strip()]
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        words = re.findall(r"\b[\w']+\b", normalized.lower(), flags=re.UNICODE)
        words = [
            word
            for word in words
            if len(word) > 2 and word not in _ENGLISH_STOPWORDS and not word.isdigit()
        ]
        if not words:
            return " ".join(sentences[:max_sentences])

        frequencies = Counter(words)
        scored_sentences: List[Tuple[float, int, str]] = []
        total_sentences = max(len(sentences), 1)
        for idx, sentence in enumerate(sentences):
            sentence_words = re.findall(
                r"\b[\w']+\b", sentence.lower(), flags=re.UNICODE
            )
            filtered = [
                word
                for word in sentence_words
                if len(word) > 2 and word not in _ENGLISH_STOPWORDS and not word.isdigit()
            ]
            if not filtered:
                score = 0.0
            else:
                score = sum(frequencies[word] for word in filtered) / len(filtered)

            # Slightly prioritize opening and closing context.
            position_ratio = idx / total_sentences
            if position_ratio <= 0.2 or position_ratio >= 0.8:
                score += 0.15
            scored_sentences.append((score, idx, sentence))

        top = sorted(scored_sentences, key=lambda item: item[0], reverse=True)[
            :max_sentences
        ]
        top_in_order = sorted(top, key=lambda item: item[1])
        summary = " ".join(item[2] for item in top_in_order)
        return summary.strip()

    def _write_outputs(
        self,
        channel_url: str,
        output_dir: str,
        summaries: List[VideoSummary],
        total_videos_found: int,
    ) -> Dict[str, Any]:
        """Write JSON and Markdown reports."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        slug = self._channel_slug(channel_url)
        json_path = output_path / f"{slug}_video_summaries.json"
        markdown_path = output_path / f"{slug}_video_summaries.md"

        payload = {
            "channel_url": channel_url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_videos_found": total_videos_found,
            "processed_videos": len(summaries),
            "summary_mode": self.summary_mode,
            "videos": [summary.to_dict() for summary in summaries],
        }

        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        markdown_content = self._build_markdown_report(payload)
        with markdown_path.open("w", encoding="utf-8") as fp:
            fp.write(markdown_content)

        logger.info("Wrote JSON report: %s", json_path)
        logger.info("Wrote Markdown report: %s", markdown_path)

        return {
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
            "payload": payload,
        }

    @staticmethod
    def _build_markdown_report(payload: Dict[str, Any]) -> str:
        """Build a human-readable Markdown report."""
        lines: List[str] = []
        lines.append("# YouTube Channel Video Summaries")
        lines.append("")
        lines.append(f"- **Channel:** {payload.get('channel_url', '')}")
        lines.append(f"- **Generated (UTC):** {payload.get('generated_at', '')}")
        lines.append(
            f"- **Videos Processed:** {payload.get('processed_videos', 0)} / "
            f"{payload.get('total_videos_found', 0)}"
        )
        lines.append(f"- **Summary Mode:** {payload.get('summary_mode', 'unknown')}")
        lines.append("")

        videos = payload.get("videos", [])
        for index, video in enumerate(videos, start=1):
            lines.append(f"## {index}. {video.get('title', 'Untitled')}")
            lines.append(f"- URL: {video.get('url', '')}")
            lines.append(f"- Video ID: {video.get('video_id', '')}")
            lines.append(f"- Published: {video.get('published') or 'Unknown'}")
            lines.append(f"- Duration: {video.get('duration') or 'Unknown'}")
            lines.append(f"- Content source: {video.get('content_source', 'unknown')}")
            lines.append(f"- Summary method: {video.get('summary_method', 'unknown')}")
            lines.append(
                f"- Transcript available: {video.get('transcript_available', False)}"
            )
            if video.get("notes"):
                lines.append("- Notes:")
                for note in video.get("notes", []):
                    lines.append(f"  - {note}")
            if video.get("error"):
                lines.append(f"- Error: {video.get('error')}")
            lines.append("")
            lines.append(video.get("summary", ""))
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def extract_json_after_marker(
        text: str, marker: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract first JSON object that appears directly after a marker string.

        This uses brace matching with string-awareness, so nested JSON objects
        and escaped quotes are handled correctly.
        """
        marker_index = text.find(marker)
        if marker_index == -1:
            return None

        start = marker_index + len(marker)
        while start < len(text) and text[start].isspace():
            start += 1

        if start >= len(text) or text[start] != "{":
            return None

        depth = 0
        in_string = False
        escaped = False
        end = None

        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break

        if end is None:
            return None

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def extract_channel_id(channel_url: str) -> Optional[str]:
        """Extract channel ID from a classic /channel/ URL."""
        match = re.search(r"/channel/([A-Za-z0-9_-]+)", channel_url)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_text(data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract text from YouTube renderer formats: simpleText/runs."""
        if not data:
            return None
        if "simpleText" in data:
            return data.get("simpleText")
        runs = data.get("runs", [])
        if runs:
            return "".join(item.get("text", "") for item in runs).strip()
        return None

    @staticmethod
    def _set_query_param(url: str, key: str, value: str) -> str:
        """Set or replace a query-string parameter in a URL."""
        parsed = urlparse(url)
        query = parse_qs(parsed.query, keep_blank_values=True)
        query[key] = [value]
        new_query = urlencode(query, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    @staticmethod
    def _channel_slug(channel_url: str) -> str:
        """Create filesystem-safe slug from channel URL."""
        clean = re.sub(r"^https?://", "", channel_url).strip("/")
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", clean)
        return clean or "channel"
