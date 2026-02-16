# IndiaAI Session Recommender

Scan live sessions from **https://impact.indiaai.gov.in/sessions** and get personalized recommendations based on your interests.

## What this app does

- Fetches live session data from the IndiaAI sessions page
- Scores sessions against your interests (title, description, speakers, partners, venue)
- Returns top matching sessions with:
  - date/time
  - location
  - speakers
  - watch link (when available)
  - why each session matched

## Quick start

From this folder:

```bash
python3 session_recommender.py "responsible AI, healthcare, governance"
```

## Useful options

```bash
# top 5 recommendations
python3 session_recommender.py "AI for education and skilling" --top 5

# filter by date
python3 session_recommender.py "robotics, manufacturing" --date 2026-02-17

# filter by venue keyword
python3 session_recommender.py "cybersecurity, sovereign ai" --venue "Bharat Mandapam"

# force fresh fetch (ignore cache)
python3 session_recommender.py "public sector AI" --refresh

# output JSON
python3 session_recommender.py "AI in agriculture" --json

# save JSON to file
python3 session_recommender.py "AI in agriculture" --save-json ./recommendations.json
```

## Notes

- The app caches fetched sessions for 30 minutes by default.
- Use `--refresh` to bypass cache.
- Use `--cache-minutes <N>` to change cache duration.
- Duplicate session IDs from the source feed are automatically de-duplicated.
