#!/usr/bin/env python3
"""CLI for scraping a YouTube channel and summarizing each video."""

import argparse
import logging
from typing import List

from src.core.youtube_channel_summarizer import YouTubeChannelSummarizer
from src.utils.config import Config


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the channel summarization CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Scrape all videos from a YouTube channel and generate one summary per video."
        )
    )
    parser.add_argument(
        "--channel-url",
        required=True,
        help="YouTube channel URL (e.g. https://www.youtube.com/channel/UC...)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for JSON and Markdown reports (default: output)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on number of videos to process.",
    )
    parser.add_argument(
        "--summary-mode",
        choices=["auto", "llm", "extractive"],
        default="auto",
        help=(
            "Summary strategy: auto (LLM with fallback), llm (LLM only), "
            "extractive (local heuristic only)."
        ),
    )
    parser.add_argument(
        "--languages",
        default="en,hi",
        help=(
            "Preferred transcript languages in priority order, comma-separated "
            "(default: en,hi)."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Persist intermediate results every N videos (default: 20).",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=20,
        help="HTTP request timeout in seconds for YouTube requests (default: 20).",
    )
    parser.add_argument(
        "--disable-transcripts",
        action="store_true",
        help=(
            "Skip transcript retrieval and summarize using title + description only. "
            "Useful when YouTube blocks transcript requests on cloud IPs."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser


def parse_languages(raw: str) -> List[str]:
    """Parse --languages input into list."""
    parts = [item.strip() for item in raw.split(",")]
    languages = [item for item in parts if item]
    return languages or ["en"]


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = Config()
    summarizer = YouTubeChannelSummarizer(
        config=config,
        summary_mode=args.summary_mode,
        preferred_languages=parse_languages(args.languages),
        request_timeout=args.request_timeout,
        fetch_transcripts=not args.disable_transcripts,
    )

    output = summarizer.summarize_channel(
        channel_url=args.channel_url,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        save_every=args.save_every,
    )

    payload = output["payload"]
    videos = payload.get("videos", [])
    failed = sum(1 for video in videos if video.get("error"))
    transcript_backed = sum(
        1 for video in videos if bool(video.get("transcript_available"))
    )

    print("")
    print("Channel summarization complete")
    print(f"Channel: {payload.get('channel_url')}")
    print(f"Processed: {payload.get('processed_videos')} videos")
    print(f"Transcript-backed summaries: {transcript_backed}")
    print(f"Failures: {failed}")
    print(f"JSON report: {output['json_path']}")
    print(f"Markdown report: {output['markdown_path']}")


if __name__ == "__main__":
    main()
