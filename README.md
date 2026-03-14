# Backend

Transcript evaluator backend for 6 criteria scoring:

1. Greetings (`0` or `1`)
2. Closing statement (`0` or `1`)
3. Summery (`0` or `1`)
4. Legal disclamer (`0` or `1`)
5. Empathy (`0` to `3`)
6. Slang severity (`0` to `5`, mapped from average impact using bands: `15+ -> 0`, `10-14 -> 1`, `6-9 -> 2`, `3-5 -> 3`, `1-2 -> 4`, `0 -> 5`)

Total score is out of `12`, where slang points are added to positive criteria score.

## Setup

```bash
npm install
npm run dev
```

Server runs on `http://localhost:8000` by default.

## Environment variables

Copy `.env.example` to `.env` and set values for your environment.

Required:

- `OPENAI_API_KEY`: OpenAI key used by fallback evaluators.

Optional:

- `OPENAI_MODEL`: model override (default is `gpt-5.3-codex`).
- `CORS_ORIGIN`: comma-separated allowed frontend origins.
- `PORT`: local port override (hosts usually provide this automatically).

Production example:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.3-codex
CORS_ORIGIN=https://your-frontend-domain.com
PORT=8000
```

## API

`POST /api/evaluate`

Request body:

```json
{
	"transcript": "Hello... [full transcript text or structured object]"
}
```

Response includes:

- score summary (`total_score`, `max_score`, `percentage`)
- per-section status for:
	- greetings
	- closingStatement
	- summery
	- legalDisclamer
	- empathy
	- slangSeverity

Evaluation is returned in API response only (no output files are saved by the backend logic).

## Dictionary files (unchanged)

Source phrase dictionaries are read from `backend/Json`:

- `greetings_output.json`
- `closing_statement_output.json`
- `summarization_output.json`
- `LegalDisclaimer.json`
