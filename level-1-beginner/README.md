# Unpack — AI Concept Explainer (ShadowFox AI Engineer Internship, Beginner Level)

A single-page student utility app that explains any concept at a chosen
difficulty level, using the Gemini API.

## Language support

The app ships with an EN / AZ toggle in the header. Switching it:

- Re-renders every UI label, placeholder, and error message from a small
  `I18N` dictionary (no page reload).
- Adds a language instruction to the Gemini system prompt, so the
  generated explanation itself — including the section headers — comes
  back in the selected language.
- The output renderer's "is this line a section header" check uses a
  Unicode-aware regex (`\p{L}`) so Azerbaijani letters (ə, ş, ç, ğ, ı,
  ö, ü) are recognized the same way Latin letters are.

## How it works

1. The student types a topic or question, optionally adds context (e.g.
   "this is for a first-year DSA course"), and picks a depth level
   (beginner / intermediate / advanced).
2. On submit, the app validates the input, then sends one request to
   Gemini's `generateContent` endpoint directly from the browser
   (no backend — this is intentional for the beginner-level scope).
3. The response is rendered into a structured explanation panel
   (Definition → Key Points → Example).

## Prompt structure

The request is split into two parts, matching Gemini's `systemInstruction` /
`contents` fields:

- **System instruction** (fixed): tells the model its role (study
  assistant), the required output shape (`Definition:`, `Key Points:`,
  `Example:` sections), and the tone rules (simple language, no
  unexplained jargon, no filler/apologies).
- **User prompt** (built per request): the topic, the selected depth
  level, and any optional student-provided context, e.g.

  ```
  Explain this concept: "binary search"
  Depth level: beginner
  Student's context: first-year data structures course
  ```

Splitting a fixed system instruction from a dynamic user prompt keeps the
output format consistent across topics and depth levels, instead of
depending on the model to infer structure each time — this is the "basic
prompt design thinking" the task asks for.

## Input validation

- Empty topic → blocked before any API call, focus returned to the field.
- Empty API key → blocked before any API call.
- Both cases show an inline message in the output panel rather than a
  browser `alert()`, so the flow stays inside the app.

## Error handling

All API calls are wrapped in `try/catch` with additional handling for
non-2xx responses:

| Case | Handling |
|---|---|
| `400` / `403` (bad/invalid key) | Shows the API's own error message with a hint to check the key |
| `429` (rate limited) | Asks the user to wait and retry |
| Other non-2xx | Shows the raw error message from the API body |
| Network failure (`fetch` throws) | Shows a generic "could not reach the API" message |
| `200` but empty `candidates` | Shows an "empty response, try rephrasing" message |

The submit button is disabled and shows "Thinking…" while a request is in
flight, so the student can't fire duplicate requests.

## Output handling

Gemini returns plain text following the requested `Definition: / Key
Points: / Example:` shape. A small renderer (no markdown library) turns
that into semantic HTML: `**bold**` → `<strong>`, `- ` lines → `<li>`
inside `<ul>`, and short `Label:` lines → `<h3>` section headers. All
model output is escaped before insertion, so nothing in the response can
break the page.

## Why this isn't "just a chatbot"

There's no open-ended chat: the interface only accepts a topic + depth +
optional context, and every request goes through the same fixed system
instruction and output shape. The app is scoped to one workflow —
"explain this concept at this level" — which is what a student utility
feature needs, rather than a general-purpose conversation surface.

## Running it

Open `index.html` directly in a browser. Paste a Gemini API key (free,
from [Google AI Studio](https://aistudio.google.com/apikey)) into the key
field. No build step, no dependencies, no server.

**Note on the API key:** this beginner-level version calls Gemini directly
from the browser, so the key lives only in the page (and optionally
`localStorage` if "Remember key" is checked) — it is never sent anywhere
but Google's API. For a production setup, the key would move to a small
backend so it's never exposed client-side; that separation is exactly
what the intermediate/advanced levels of this task introduce.
