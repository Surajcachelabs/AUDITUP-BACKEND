# Backend

Transcript evaluator backend for 4 criteria scoring:

1. Greetings (`0` or `1`)
2. Closing statement (`0` or `1`)
3. Summery (`0` or `1`)
4. Legal disclamer (`0` or `1`)

Total score is out of `4`.

## Setup

```bash
npm install
npm run dev
```

Server runs on `http://localhost:8000` by default.

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

Evaluation is returned in API response only (no output files are saved by the backend logic).

## Dictionary files (unchanged)

Source phrase dictionaries are read from `backend/Json`:

- `greetings_output.json`
- `closing_statement_output.json`
- `summarization_output.json`
- `LegalDisclaimer.json`
