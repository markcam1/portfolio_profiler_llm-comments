# Development Spec: AI Analysis Layer for Portfolio Profiler

**Status:** Ready for implementation
**Target language:** Python 3.10+
**Base project:** Fork of `engineerinvestor/Portfolio-Analysis` — the "Portfolio Analyzer" web app
**Spec revision:** 3 — restructured around per-item commentary (see Section 1.1)

---

## 0. Prerequisites — what the v1 app does and does NOT compute

The coding assistant must read this before starting.

### 0.1 Functionality that EXISTS in v1 (confirmed from the UI)

The v1 "Portfolio Analyzer" is a multi-tab web app (Streamlit/Plotly style) with tabs:
**Performance, Monte Carlo, Optimization, Benchmark, About.** The **Performance** tab
computes and displays:

| Item | Type | Notes |
|---|---|---|
| Annual Return | metric | e.g. 7.35% |
| Annual Volatility | metric | e.g. 10.57% |
| Sharpe Ratio | metric | e.g. 0.32 |
| Sortino Ratio | metric | e.g. 0.46 |
| Max Drawdown | metric | e.g. -21.96% |
| Allocation donut | chart | **By TICKER, not asset class** — e.g. VTI 40% / BND 40% / VXUS 20% |
| Cumulative Returns | chart | Growth-of-$1 for the blended portfolio |
| Individual Asset Returns | chart | Growth-of-$1 per ticker |

Other tabs (Monte Carlo, Optimization, Benchmark) exist as available infrastructure.

### 0.2 Functionality that does NOT exist in v1

No source in the v1 codebase for: **asset-class allocation** (the donut is by ticker
only), **sector weights**, **sector comparison vs. a benchmark ETF**, **domestic/foreign
split**, **holdings-level metadata**, **top-concentration view**.

### 0.3 Consequence: a numeric prerequisite for Phase 2

The AI layer cannot comment on numbers the app never computes. A small **classification
& sector module** must be added to the toolkit's *numeric* code (NOT the AI layer —
see Section 3). For a fund-of-ETFs portfolio this is mostly lookups: each ticker maps
to an asset class and region; equity-ETF sector weights come from each fund's published
holdings via `yfinance`.

Work is split into **Phase 1** (items that exist in v1 today) and **Phase 2** (items
unlocked by the classification & sector module).

---

## 1. Purpose

Add an **AI Analysis** feature to the Portfolio Profiler. The portfolio profile —
all charts and tables — is computed and rendered by existing toolkit code with **no
LLM involvement**. The AI layer adds an optional commentary pass on top.

### 1.1 Interaction model (the defining design decision)

This is **NOT an interactive chat agent.** There is no chat shell, no user-typed
questions, no tool-calling loop. The model is:

1. The profile renders normally. No LLM call happens. No resources are spent.
2. The page shows an **"AI Analysis"** button.
3. When (and only when) the user clicks it, the frontend gathers the data behind
   **every item** on the profile and sends it to the AI layer in **one batched call**.
4. The LLM returns **one short comment per item**.
5. The frontend attaches each comment to the item (chart or metric or table) it
   describes.

Each comment is **anchored on its own item's data** but is also given a **brief
whole-portfolio summary** for context, so a comment can make one connecting
observation (e.g. volatility commentary may reference the equity weighting) without
drifting off-topic.

Key consequences:
- If the user never clicks the button, **zero LLM cost** is incurred.
- One batched call per click — not one call per item — for cost, speed, and so the
  model can keep comments from repeating each other.
- No conversation state. Each click is a fresh, stateless batched call.

---

## 2. Core architecture principle (non-negotiable)

> **The LLM operates only at the summary layer. It never computes financial numbers.**

All metrics, allocations, sector weights, returns, and drawdowns are computed by
deterministic Python code (the existing toolkit, plus the Section 0.3 classification
module). The LLM receives finished numbers and produces only natural-language
commentary. It never classifies a ticker, derives a weight, or does arithmetic.

Rationale: keeps the feature cheap, fast, auditable, and free of hallucinated figures.

---

## 3. Project structure

```
portfolio_analysis/
    ...existing v1 modules (performance, monte carlo, optimization, benchmark)...
    classification/          # PHASE 2 numeric prerequisite (Section 0.3) — NOT AI code
        __init__.py
        asset_class.py       # ticker -> asset_class, region
        sectors.py           # equity-ETF sector weights via yfinance; portfolio rollup
        benchmark_sectors.py # total-market ETF sector weights for comparison
    ai_analysis/
        __init__.py
        schema.py            # ProfileItem, ProfileSnapshot, ItemComment — data contracts
        analyze.py           # the single batched-call entry point
        prompts.py           # system prompt, kept separate for easy iteration
        config.py            # model name, API key handling, limits
    tests/
        test_classification.py
        test_schema.py
        test_analyze.py
```

---

## 4. Data contracts (`schema.py`)

Three dataclasses. The frontend/profiler produces a `ProfileSnapshot`; the AI layer
returns a list of `ItemComment`.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ProfileItem:
    """One commentable element on the profile page."""
    item_id: str               # stable unique id, e.g. "annual_volatility"
    item_type: str             # "metric" | "chart" | "table"
    title: str                 # human label, e.g. "Annual Volatility"
    data: dict[str, Any]        # the numbers behind this item — see 4.1
    phase: str = "v1"          # "v1" or "p2" — see Section 0

@dataclass
class ProfileSnapshot:
    """Everything the AI layer needs for one batched call."""
    as_of_date: str
    base_currency: str          # default "USD"
    items: list[ProfileItem]
    portfolio_summary: str      # short plain-text whole-portfolio context — see 4.2

@dataclass
class ItemComment:
    """One comment, returned per item."""
    item_id: str               # must match a ProfileItem.item_id
    comment: str               # the natural-language commentary
    status: str = "ok"         # "ok" | "skipped_no_data" | "error"
```

### 4.1 What goes in `ProfileItem.data`

Only finished numbers — never raw price series the model would have to crunch.
Keep each payload small. Examples:

- Annual Volatility (metric): `{"value_pct": 10.57}`
- Allocation donut (chart): `{"by_ticker": {"VTI": 40.0, "BND": 40.0, "VXUS": 20.0}}`
- Cumulative Returns (chart): `{"total_growth_pct": 41.2, "period": "2021-05 to 2026-05",
  "end_drawdown_pct": -21.96}` — i.e. a few summary stats describing the curve, NOT
  the full daily series.
- Sector weights (chart, **Phase 2**): `{"by_sector": {"Technology": 28.1, ...}}`

Rule: if an item is a chart, send the handful of numbers that characterize it, not the
underlying time series. The comment describes the shape; it does not re-derive it.

### 4.2 `portfolio_summary`

A short (1–3 sentence) plain-text description of the whole portfolio, generated by
deterministic code, not the LLM. Example: `"Three-fund portfolio: VTI 40%, BND 40%,
VXUS 20%. Annual return 7.35%, volatility 10.57%, max drawdown -21.96%."` This is
included in the prompt so each per-item comment can make one connecting observation
without needing every other item's full payload.

### 4.3 Item percentages and `None`

- All percentages are numbers in 0–100 (not 0–1). Document this in the glossary.
- Phase 2 items simply are not included in `ProfileSnapshot.items` until the
  classification module exists — there is no need for placeholder `None` items.
- If an item's `data` is empty or unusable, the returned `ItemComment.status` is
  `"skipped_no_data"` and `comment` explains briefly.

---

## 5. The batched analysis call (`analyze.py`)

```python
def analyze_profile(snapshot: ProfileSnapshot) -> list[ItemComment]:
    """One batched LLM call. Returns one ItemComment per input item.

    Triggered by the frontend's 'AI Analysis' button. Stateless.
    """
```

**Behavior:**
- Exactly **one** LLM call per invocation, regardless of item count.
- The prompt contains: the system prompt from `prompts.py`, the `portfolio_summary`,
  and the full list of items (id, type, title, data).
- The model is instructed to return **structured JSON**: a list of objects, each with
  `item_id` and `comment`. `analyze.py` parses this and maps it to `ItemComment`s.
- **Every input item must get exactly one output comment.** After parsing, verify the
  returned `item_id`s match the input set: any missing item gets a synthesized
  `ItemComment(status="error")`; any unexpected `item_id` is dropped.
- The function is pure: same snapshot in, comparable comments out. No global state.

### 5.1 Structured output handling

- The system prompt must instruct the model to return **only** valid JSON, no prose,
  no markdown fences.
- `analyze.py` must defensively strip ```json fences if present, then `json.loads`.
- On parse failure, retry once; if it still fails, return one `ItemComment(status=
  "error")` per item rather than throwing — the profile must still render.

### 5.2 Caching (recommended)

Key a cache on `(portfolio inputs + date range)` — i.e. the inputs that define the
profile. If the user clicks "AI Analysis" again on an unchanged profile, serve the
cached comments and skip the LLM entirely. Invalidate when any portfolio input changes.

---

## 6. System prompt requirements (`prompts.py`)

The system prompt must explicitly:
- State that every number is provided in the item payloads and the model must **not**
  invent, estimate, or recompute any figure.
- Instruct the model to write **one short comment per item** (≈1–3 sentences each),
  scoped to that item, optionally making **one** connecting observation using the
  `portfolio_summary`.
- Require **specific, quantified** observations ("a -21.96% max drawdown indicates a
  meaningful peak-to-trough decline over the period") rather than generic filler.
- Forbid buy/sell/hold recommendations and any personalized financial advice. Comments
  describe items; they do not tell the user what to do.
- State that `by_ticker` allocation is a per-ticker breakdown and **not** an asset-class
  breakdown — do not infer "60% equity" from ticker data alone.
- Require the exact JSON output shape (list of `{item_id, comment}`), no extra text.
- Instruct the model that if an item's data is empty it should still return an entry
  for that `item_id` with a brief "insufficient data" comment.

---

## 7. Acceptance criteria

Phase 1 (against existing v1 items — the VTI/VXUS/BND example):
- [ ] Rendering the profile triggers **no** LLM call.
- [ ] Clicking "AI Analysis" triggers exactly **one** LLM call.
- [ ] Every profile item receives exactly one comment; counts match.
- [ ] Each comment cites only numbers from that item's payload or `portfolio_summary`.
- [ ] Comments make at most one connecting observation; they do not wander into
      unrelated items.
- [ ] No comment infers asset-class allocation from `by_ticker` data.
- [ ] No buy/sell/hold language appears.
- [ ] Malformed model output is handled: profile still renders, items show an error
      status rather than crashing.
- [ ] Clicking again on an unchanged profile serves cached comments, no new LLM call.

Phase 2 (after the Section 0.3 classification module exists):
- [ ] Sector-weight and asset-allocation items appear in `ProfileSnapshot.items` and
      receive correct, data-grounded comments.

---

## 8. Configuration (`config.py`)

- API key read from an environment variable, never hardcoded.
- Model name configurable in one place.
- `max_tokens` and timeout configurable.
- Reasonable defaults so the feature runs out of the box once the key is set.

---

## 9. Out of scope

- **Any interactive chat / agentic Q&A.** There is no chat shell and no tool-calling
  loop. The feature is a single batched commentary pass behind a button. (This is a
  deliberate change from an earlier design.)
- Any new financial calculation — belongs in the numeric toolkit, including the
  Section 0.3 classification & sector module.
- Personalized investment advice or recommendations.
- Streaming responses (synchronous batched call only).

---

## 10. Suggested build order

**Phase 1 — ships against existing v1 items:**
1. `schema.py` + `test_schema.py` — lock the three data contracts.
2. An adapter that builds a `ProfileSnapshot` from the v1 Performance tab's existing
   output (the five metrics + three charts), plus a deterministic `portfolio_summary`.
3. `prompts.py` system prompt + `analyze.py` with the single batched call and JSON
   parsing/validation.
4. Validate against the real VTI/VXUS/BND profile; iterate on the prompt.
5. Wire the "AI Analysis" button in the frontend: on click, build snapshot → call
   `analyze_profile` → attach each comment to its item.
6. Add the cache (Section 5.2).

**Phase 2 — unlocks sector / asset-class items:**
7. Build the `classification/` module (Section 0.3).
8. Add sector / asset-class / region items to the `ProfileSnapshot` builder; they
   flow through the same batched call automatically.

---

## 11. Notes for the implementer

- **Read Sections 0 and 1.1 first.** v1 computes performance metrics and a by-ticker
  donut only; there is no sector or asset-class data yet. And the feature is a
  one-shot batched commentary pass behind a button — explicitly not a chat agent.
- The hardest part is prompt discipline: stopping the model inventing figures and
  making it return clean JSON. Treat `prompts.py` as a primary artifact and test it.
- Keep `prompts.py` separate from logic so prompt iteration doesn't touch working code.
- `by_ticker` allocation (v1 donut) and asset-class allocation (Phase 2) are distinct.
  Never conflate them.
- Charts are summarized by a few characterizing numbers, never by their full time
  series. The model comments on shape; it does not re-derive the curve.
- The `ProfileSnapshot` → `list[ItemComment]` boundary is the seam between the toolkit
  and the AI layer. If an item needs a number that doesn't exist, add it to the
  numeric toolkit — do not have the LLM compute it.
- This is informational software, not financial advice. Comments describe the
  portfolio; they never direct the user's decisions.
