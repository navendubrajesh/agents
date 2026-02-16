# AI Sumit YouTube Video Catalog

This utility scans videos from a YouTube channel and exports a tabular dataset with:

- Date
- Title
- Presented by
- Summary
- Video URL
- Duration / Views
- Category / Tags
- Additional metadata

## Run

From repo root:

```bash
python3 tools/ai_sumit_video_catalog/channel_video_catalog.py \
  --channel-url "https://www.youtube.com/channel/UCiV0zikSWzC0nx5HFy-C3lg" \
  --output-dir "tools/ai_sumit_video_catalog/output"
```

Optional flags:

- `--max-videos 50` limit number of processed videos
- `--workers 6` concurrent metadata fetch workers
- `--markdown-limit 100` rows shown in markdown preview (`0` = all rows)

## Output

The script writes:

- `ai_sumit_video_catalog.csv` (full table for spreadsheets)
- `ai_sumit_video_catalog.json` (structured data)
- `ai_sumit_video_catalog.md` (readable markdown preview table)

All output files are written into the folder passed via `--output-dir`.
