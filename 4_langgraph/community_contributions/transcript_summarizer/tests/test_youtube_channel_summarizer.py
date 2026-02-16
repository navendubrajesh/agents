import json

from src.core.youtube_channel_summarizer import YouTubeChannelSummarizer
from src.utils.config import Config


class TestYouTubeChannelSummarizerHelpers:
    def setup_method(self):
        self.summarizer = YouTubeChannelSummarizer(
            config=Config(),
            summary_mode="extractive",
            preferred_languages=["en", "hi"],
        )

    def test_extract_channel_id(self):
        channel_url = "https://www.youtube.com/channel/UCiV0zikSWzC0nx5HFy-C3lg"
        assert (
            YouTubeChannelSummarizer.extract_channel_id(channel_url)
            == "UCiV0zikSWzC0nx5HFy-C3lg"
        )

    def test_extract_json_after_marker(self):
        payload = {
            "a": 1,
            "nested": {"b": "value with \"quotes\""},
            "list": [1, 2, 3],
        }
        text = (
            "prefix data "
            + "ytInitialPlayerResponse = "
            + json.dumps(payload)
            + "; trailing script"
        )
        parsed = YouTubeChannelSummarizer.extract_json_after_marker(
            text, "ytInitialPlayerResponse = "
        )
        assert parsed == payload

    def test_pick_caption_track_prefers_manual_over_asr(self):
        tracks = [
            {
                "languageCode": "en",
                "kind": "asr",
                "baseUrl": "https://example.com/asr",
            },
            {
                "languageCode": "en",
                "baseUrl": "https://example.com/manual",
            },
            {
                "languageCode": "hi",
                "baseUrl": "https://example.com/hi",
            },
        ]
        picked = self.summarizer._pick_caption_track(tracks, ["en", "hi"])
        assert picked is not None
        assert picked["baseUrl"] == "https://example.com/manual"

    def test_parse_json3_caption_payload(self):
        payload = {
            "events": [
                {"segs": [{"utf8": "Hello "}, {"utf8": "world"}]},
                {"segs": [{"utf8": "This is line two."}]},
                {"wWinId": 0},  # No segs, should be ignored
            ]
        }
        transcript = self.summarizer._parse_json3_caption_payload(json.dumps(payload))
        assert "Hello world" in transcript
        assert "This is line two." in transcript

    def test_extractive_summary(self):
        text = (
            "Sentence one introduces the topic. "
            "Sentence two adds supporting details. "
            "Sentence three provides more context. "
            "Sentence four gives an example. "
            "Sentence five wraps up the ideas."
        )
        summary = YouTubeChannelSummarizer.extractive_summary(text, max_sentences=3)
        assert summary
        # Ensure we got a condensed result rather than the full text.
        assert len(summary) < len(text)
