# Player profiles for the football AI coach: design

Date: 2026-11-02 · Project 20 · Owner choice: option C (one profile format, filled from open data now, from a live
sports-data API later), chat 2026-11-02.

## Goal

Before a match, load what is known about each player and team, so the coach adapts its advice: a player who
completes more passes than expected gets his risky passes rated a little higher; a player who finishes better than
his chances gets the shot option rated a little higher. The adjustment must be **earned on held-out data**: it ships
only if it improves the models on matches the profiles were not built from.

## Profile format (provider-independent)

`profiles.json` for one match, loaded with the match:

```json
{
  "source": "open-data: IDSSE events (other matches) + SkillCorner season aggregates",
  "built_without": ["J03WOH"],
  "players": {
    "<player key>": {
      "name": "D. Ginczek", "team": 0, "position": "STZ",
      "pass_skill":  {"logit": 0.12, "n": 212, "raw": 0.31},
      "finishing":   {"logit": -0.05, "n": 14, "raw": -0.40},
      "dangerous_passes_p30": {"value": 1.8, "n_minutes": 1240}
    }
  },
  "teams": {"0": {"name": "...", "directness": {"value": 0.41, "n": 3}}}
}
```

- Every skill is an **offset on the logit scale** of an existing model, with its sample size and the raw estimate.
- Offsets are **shrunk** towards 0 (the position average) by empirical Bayes: `logit = Σ(y − p) / (n + k)` style,
  with `k` chosen by held-out log-loss. A player with 5 passes stays near average.
- Missing → field absent → the coach behaves exactly as today.

## Providers

| Provider | Fills | Status |
|---|---|---|
| `open_data` (this phase) | `pass_skill` from our tracked matches (completion vs the pass model's expectation, excluding the match being replayed); `finishing` from shots vs xG where xG exists (IDSSE provider xG; SkillCorner has none); passing / runs descriptors from SkillCorner season CSVs (A-League players) | build now |
| `api` (later) | finishing, passing, form from a live provider (e.g. API-Football) with the owner's key in an environment variable; local use only (provider terms usually forbid republishing) | interface + a fake provider in tests only; no calls, no key handling now |

Player keys: provider ids are mapped to our tracking ids per match (IDSSE `PersonId`, SkillCorner player id); the
mapping is stored, and unmapped players simply have no profile.

## How the coach uses it

- Pass: `logit(p) += a_pass · (pass_skill(passer))` (receiver term only if it helps held-out).
- Shot: `logit(xG) += a_shot · finishing(shooter)`.
- `a_*` fitted on training matches (1 = use the offset as estimated); both default to 0 when a profile is missing.
- Browser: the same two lines in `src/lib/coach/advice.ts`; the option list shows a small "player form" note when an
  offset changed the ranking (e.g. "+3 % passing vs expected, 212 passes").

## Evaluation (must pass to ship)

- Leave-one-match-out on IDSSE (and on SkillCorner for pass skill): profiles built only from the training matches;
  compare held-out log-loss / Brier of the pass model with vs without offsets, and the shot model likewise.
- Report how many held-out passes had a profile, the distribution of offsets, and the best `k`.
- If the gain is within noise (match-bootstrap 95 % CI overlaps 0), the profiles ship as **display-only** player
  cards (no effect on advice) and the README says so.

## Phases

1. Library: `coach/profiles.py` (format, shrinkage, open-data provider, API provider interface + fake), evaluation,
   README section, export `profiles.json` with the website match.
2. Site: load `profiles.json`, apply offsets in the TS coach (parity fixtures extended), player-form note in the
   option list.

## Out of scope

Calling a real API, storing API keys, scraping websites, player ratings or rankings published as judgements of
people (the page shows model offsets with sample sizes, not "good/bad player" labels).
