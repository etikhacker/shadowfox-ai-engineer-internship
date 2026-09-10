# Week 7 Build Log — Put an LLM behind your API

## Chosen job

The new `POST /triage` endpoint classifies one support message into a closed category and suggests the team that should receive it. This is a single request/response judgement, not a chatbot.

The categories are `billing`, `bug`, `feature`, and `other`. Urgency is `low`, `normal`, or `high`; the suggested team is `support`, `engineering`, `product`, or `other`.

## Build environment

The existing FastAPI API in `level-2-intermediate/backend/main.py` was extended. The provider lane is **OpenRouter** using the OpenAI-compatible HTTP request shape and the free model identifier `openrouter/free`. Secrets are loaded from `.env` and are not committed.

## Iteration log

### 1. Repository inspection

The existing project was a FastAPI document Q&A API with Gemini and FAISS. It already had Pydantic, `requests`, and environment loading, so the new feature was kept in the same API instead of creating a second service.

### 2. Contract before model call

A `JOB-CARD.md` file was written before implementation. The response schema was defined with Pydantic `Literal` fields and numeric bounds before connecting to the provider. This prevents arbitrary model categories and unvalidated free text from escaping the endpoint.

### 3. Tests first

Eight tests were written before the new implementation. The first test run failed because the existing environment did not have the repo's FAISS dependency installed. Installing `requirements.txt` fixed the test environment. The next run failed because the new `call_llm` function did not exist, which confirmed the tests were exercising missing feature behavior rather than passing accidentally.

### 4. Minimal implementation

The endpoint, Pydantic schema, versioned prompt file, OpenRouter client, explicit timeout, bounded retry policy, one repair attempt, JSONL cost logging, and kill switch were implemented. The local stub mode was then added so development and evaluation could run without spending provider quota.

### 5. Validation and local evaluation

All eight tests pass. The deterministic eight-case stub evaluation produced `8/8`, score `1.0`. The test cases cover a valid closed-schema response, blank input, overlong input, invalid category validation, retryable-status policy, repair after invalid JSON, kill-switch behavior, and cost-log creation.

### 6. Real provider smoke test

A synthetic message was sent through the real OpenRouter path:

`Synthetic test: I was charged twice for a made-up subscription.`

The provider returned valid JSON that passed Pydantic validation:

```json
{
  "category": "billing",
  "urgency": "high",
  "suggested_team": "support",
  "confidence": 0.9,
  "reason": "User reports duplicate charge for a fictional subscription, indicating a billing issue."
}
```

No real personal, employer, or confidential data was sent.

## Reliability controls

The provider call has an explicit eight-second timeout. It retries only HTTP 429, HTTP 5xx, and request exceptions, with a maximum of two retries and bounded backoff. Invalid JSON or schema output gets exactly one repair attempt. If the provider still fails, the endpoint returns a clear 502 rather than raw model text. `LLM_ENABLED=0` returns 503 and acts as the kill switch. Each request writes provider, model, repair flag, timestamp, endpoint, and estimated cost to a JSONL cost log.

## Known limitations

The real provider smoke test proves one live path, but the eight-case score is intentionally a deterministic stub score to avoid consuming free-provider quota during repeated test runs. A future evaluation can run the same synthetic cases against OpenRouter and record a separate provider score. The existing Gemini document Q&A code was not refactored because it is outside this endpoint's scope.

## Files added or changed

| File | Purpose |
|---|---|
| `backend/JOB-CARD.md` | Fixed input/output contract and refusal behavior |
| `backend/prompts/triage_v1.txt` | Versioned prompt specification |
| `backend/main.py` | `/triage`, schema validation, provider client, retries, repair, cost log, kill switch |
| `backend/tests/test_triage.py` | Eight automated test cases |
| `backend/eval_cases.json` | Eight synthetic evaluation inputs |
| `backend/run_eval.py` | Deterministic local score runner |
| `backend/smoke_real_provider.py` | One live provider smoke test |
| `README.md` | Curl, setup, output contract, and evaluation documentation |
