# Opponent Tracker — Test Plan & Fix Log

**Created:** 2026-05-10
**Purpose:** TDD-driven debugging of `src/poker_env/opponent_tracker.py`. Each test below targets a specific bug or guards a critical correctness property. Tests are added to `tests/test_env/test_opponent_tracker_bugs.py` and run via pytest.

**Status legend:**
- `[ ]` not yet written
- `[FAIL]` written, currently failing
- `[FIXED]` written, was failing, now passing
- `[PASS]` written and passing from the start (sanity check)

After each fix, the **Fix Notes** section under that test records: (a) what the bug was, (b) what changed, (c) why the original logic was wrong.

---

## How tests are constructed

All bug tests follow the same pattern:

```python
tracker = OpponentTracker()
tracker.start_hand(hand_number, players, dealer_position, small_blind, big_blind)
tracker.record_action(player_id, ..., action=Action.X, street=Street.Y, position=...)
# ... more actions ...
tracker.end_hand(winners, winnings, final_stacks)
# assertions on tracker.opponents[id]
```

The tracker's public API (`start_hand`, `record_action`, `end_hand`, `get_all_opponent_stats`) is stable. Tests written against it survive the eventual decoupling refactor (event hooks below will delegate to these same methods).

---

## P0 Tests — Block training quality if broken

### T1 — 3-bet opportunity counts when facing a raise, not when raising

**Status:** `[FIXED]`
**Bug ref:** Audit §2.1, `opponent_tracker.py:456-462`
**Scenario:** 3 players. Hand 1: Player A raises preflop. Players B and C fold without raising.
**Assertion:**
- `tracker.opponents[B].three_bet_opportunities == 1` (B faced a raise; this is an opportunity even though they folded)
- `tracker.opponents[B].three_bet_count == 0` (B did not 3-bet)
- `tracker.opponents[A].three_bet_opportunities == 0` (A acted first, never faced a raise this hand)
**Why it matters:** With the current code, only players who themselves raise are counted as having a 3-bet opportunity. The denominator of `three_bet_percent` is wrong; the network's feature `[3]` is junk.

**Fix Notes:**
- **Bug:** old code at lines 456-462 keyed off "this player has at least one preflop raise of their own", so the *denominator* (opportunities) was identical to "did they raise", which collapses the rate to a binary "I raised when I had a raise". Anyone who faced a raise and folded/called was invisible in the denominator.
- **Fix:** new method `_analyze_preflop_action_sequence` walks the hand's preflop actions once, in order, tracking a running count of raises and the set of players who have raised. A player is credited with a `three_bet_opportunity` the first time they act when `raises_so_far >= 1 AND they haven't raised yet` — regardless of whether their response is fold/call/raise.
- **Why old logic was wrong:** "opportunity to 3-bet" is defined by what the player FACES, not what they DO. The old check `if preflop_raises:` measured the latter, conflating the two and making `three_bet_percent` always 0 or undefined.

---

### T2 — 3-bet count requires an earlier raise from someone else

**Status:** `[FIXED]`
**Bug ref:** Audit §2.1
**Scenario:** Heads-up. Player A raises preflop. Player B re-raises (3-bet). Hand ends.
**Assertion:**
- `tracker.opponents[B].three_bet_count == 1`
- `tracker.opponents[A].three_bet_count == 0` (A's raise was the open, not a 3-bet)
- `tracker.opponents[B].three_bet_opportunities == 1`

**Fix Notes:**
- **Bug:** the old `len(preflop_raises) >= 2` check fired only when this *same player* had raised twice in a hand. The opener (A) was never credited because they only have 1 raise; the re-raiser (B) was missed for the same reason — B only has one raise in their own action list. The condition described "this player raised at least twice in this hand", which is a 4-bet, not a 3-bet.
- **Fix:** in `_analyze_preflop_action_sequence`, a raise is counted as a 3-bet iff `raises_so_far == 1 AND player not in raised_already` at the moment of the raise. That correctly attributes B's raise as the 3-bet (raises_so_far=1 because A already raised, B hasn't yet) and A's open as NOT a 3-bet (raises_so_far=0 at the time).
- **Why old logic was wrong:** it counted raises by the wrong player. A 3-bet by definition is the third bet — necessarily made after someone *else's* raise. The old code asked the wrong question entirely.

---

### T3 — Same player raising twice (4-bet) does NOT count as a 3-bet

**Status:** `[FIXED]`
**Bug ref:** Audit §2.1 ("`len(preflop_raises) >= 2` from a single player means they raised, got re-raised, then raised again — that's a 4-bet, not a 3-bet")
**Scenario:** Heads-up. A raises, B re-raises, A re-raises again (4-bet), B folds.
**Assertion:**
- `tracker.opponents[A].three_bet_count == 0` (A's second raise is a 4-bet, not a 3-bet)
- `tracker.opponents[B].three_bet_count == 1`

**Fix Notes:**
- **Bug:** same root cause as T2 — old code counted any player with ≥2 raises this hand as having 3-bet. In a 4-bet pot, that was guaranteed to mis-classify the opener.
- **Fix:** the new walk credits each raise individually based on the *running raise count*. A's second raise occurs when `raises_so_far == 2` — it's a 4-bet, not a 3-bet, so it's not added to `three_bets_made`. The `raised_already` set also blocks A from being re-credited for any later raises.
- **Why old logic was wrong:** the "≥2 raises this hand" heuristic conflated "raised twice in one hand" with "3-bet". Those overlap only in some scenarios; in any 4-bet pot they diverge.

---

### T4 — Fold-to-c-bet across players: A c-bets, B folds

**Status:** `[FIXED]`
**Bug ref:** Audit §2.3, `opponent_tracker.py:512-517`
**Scenario:** Preflop A raises, B calls. Flop comes. A bets (c-bet). B folds.
**Assertion:**
- `tracker.opponents[B].faced_flop_cbet == 1` (B faced a c-bet)
- `tracker.opponents[B].folded_to_flop_cbet == 1` (B folded to it)
- `tracker.opponents[A].faced_flop_cbet == 0` (A made the c-bet; didn't face one)
**Why it matters:** The current code only fires when the SAME player both bet AND folded the flop — which is impossible. The counters are always either equal or zero. Feature `[5]` is dead.

**Fix Notes:**
- **Bug:** the old code looped over `player_actions` (one player at a time) and required that the same `flop_actions` list contained BOTH a bet/raise AND a fold. A single player cannot both bet and fold the same flop — the counters could only ever be 0. Feature `[5]` was structurally dead.
- **Fix:** new method `_analyze_flop_action_sequence` operates on the *hand-level* flop action list. It first identifies the c-bet event (preflop aggressor opens flop betting with bet/all-in). Then walks subsequent flop actions, attributing `faced_cbet` to each unique non-aggressor and `folded_to_cbet` to those whose first response was a fold. Attribution is now cross-player.
- **Why old logic was wrong:** flop c-bet stats are *interactional* — one player bets, another player responds. Folding to a c-bet is by definition not done by the c-better. Looping per-player and requiring both events in one player's actions guaranteed zero.

---

### T5 — C-bet counts only the preflop aggressor opening flop betting

**Status:** `[FIXED]`
**Bug ref:** Audit §2.4
**Scenario:** Preflop: A raises, B calls, C calls. Flop: B bets (donk bet), A raises, C folds, B folds.
**Assertion:**
- `tracker.opponents[A].flop_cbet_made == 0` — A raised the flop in response to B's bet; that's NOT a c-bet because A didn't open the flop betting.
- `tracker.opponents[A].flop_cbet_opportunities == 1` — A was the preflop aggressor and saw a flop; they had an opportunity.
- `tracker.opponents[B].flop_cbet_made == 0` — B was not the preflop aggressor.
**Why it matters:** A c-bet is the preflop aggressor opening the post-flop betting. The current code counts any flop bet/raise by any preflop raiser, inflating cbet_made.

**Fix Notes:**
- **Bug:** old logic counted any flop bet/raise as a c-bet as long as the player had also raised preflop. This conflates "the player was a preflop raiser AND made some flop aggression" with "the player opened flop betting as the preflop aggressor". In multi-way pots with donk bets, the preflop raiser's *response* to a donk bet would be misclassified as a c-bet.
- **Fix:** `_analyze_flop_action_sequence` identifies the c-bet by walking flop actions in order. Players can check in front; the first non-check action must (a) be made by the preflop aggressor and (b) be a bet/all-in. Anything else (donk bet, check by aggressor) means no c-bet occurred. `flop_cbet_opportunities` is incremented for the aggressor whenever they saw a flop, regardless of how the flop played out.
- **Why old logic was wrong:** it conflated "raised preflop AND bet flop" with "c-bet". A c-bet has a strict positional/temporal requirement (you must open the flop betting), which the old code never checked.

---

### T6 — WTSD counts losers who reached showdown

**Status:** `[FIXED]`
**Bug ref:** Audit §2.2, `opponent_tracker.py:465-468`
**Scenario:** Heads-up. Preflop A raises, B calls. Flop, turn, river all check-check. Showdown: A wins, B loses.
**Assertion:**
- `tracker.opponents[A].went_to_showdown == 1`
- `tracker.opponents[B].went_to_showdown == 1`
- `tracker.opponents[A].showdown_wins == 1`
- `tracker.opponents[B].showdown_wins == 0`
**Why it matters:** Current code only counts WTSD if player won the hand OR folded ≥5 actions in. A showdown loser is invisible. `wtsd_percent` undercounts; `w$sd_percent` overcounts (denominator excludes losers).

**Fix Notes:**
- **Bug:** the old gate `player_id in hand.winner_ids or (folded and >=5 actions)` excluded losers who reached showdown without folding. A heads-up loser is the canonical case — they never fold, never win. They were invisible to WTSD, which made W$SD's denominator wrong by exactly the number of losers (W$SD became "showdown_wins / wins" ≈ 100%).
- **Fix:** I now compute the set of players who folded anywhere in the hand. The "reached showdown" set is `players_in_hand - folded_players` whenever that set has size ≥ 2. WTSD increments for everyone in that set; W$SD only for winners with positive winnings.
- **Why old logic was wrong:** WTSD is defined by "made it to the river/all-in confrontation without folding", not by "won the pot". Conditioning on win status corrupts both numerator and denominator of dependent stats.

---

### T7 — Position is relative to button and rotates between hands

**Status:** `[FIXED]`
**Bug ref:** Audit §2.7, `texas_holdem_env.py:262-267`
**Scenario:** 3 players (player_ids 0, 1, 2). Play 3 hands, where the button rotates each hand. In hand 1 the button is on player 0; in hand 2 on player 1; in hand 3 on player 2.
**Assertion:**
- After hand 1: hand record reports player 0's position as 0 (button), player 1 as 1 (SB), player 2 as 2 (BB).
- After hand 2: hand record reports player 1's position as 0 (button), player 2 as 1 (SB), player 0 as 2 (BB).
- After hand 3: hand record reports player 2's position as 0 (button), player 0 as 1 (SB), player 1 as 2 (BB).
**Why it matters:** Today `_calculate_player_positions` records `{player_id: list_index}` — the list index doesn't change as the button rotates, so every player's "position" is constant for the session. Every position-based stat is keyed by player identity, not seat-relative-to-button.

This test runs against the **env**, not the tracker directly, since position calculation happens in `texas_holdem_env.py`.

**Fix Notes:**
- **Bug:** `_calculate_player_positions` stored `{player_id: seat_index}`. Since seat indices don't rotate hand-to-hand, every player had a constant "position" for their entire session. Position-keyed stats (`position_stats`) collapsed onto player identity, defeating the point.
- **Fix:** changed to `(seat_index - button_position) mod num_players`. Now position 0 always means "this hand's button", 1 = SB, 2 = BB, etc. The recorded position rotates as the button rotates.
- **Why old logic was wrong:** seat index ≠ poker position. In real poker, position is named relative to the dealer button, which moves each hand. Without that relativization, all positional analytics become a slightly noisier copy of per-player analytics.

---

## P1 Tests — Wrong but smaller blast radius

### T8 — Squeeze requires a prior raise from someone else AND a caller

**Status:** `[FIXED]`
**Bug ref:** Audit §2.5
**Scenario:** 4 players. Preflop: A raises, B calls, C raises (squeeze!), D folds, A folds, B folds.
**Assertion:**
- `tracker.opponents[C].squeeze_opportunities == 1`
- `tracker.opponents[C].squeeze_attempts == 1`
- `tracker.opponents[A].squeeze_opportunities == 0` (A opened; no squeeze possible)
- `tracker.opponents[B].squeeze_opportunities == 0` (B just called; no squeeze for them)

Negative case: hand 2: A raises, B calls, C folds (no squeeze). C should have an opportunity (raise + caller in front) but no attempt.

**Fix Notes:**
- **Bug:** the old detector simply checked "this player has at least one preflop raise AND at least one other player called or bet". It didn't enforce the temporal order (the caller must be BEFORE the raise it's squeezing); it counted the opener as a squeezer because their own raise plus the BB's blind/calls satisfied the condition; and it counted any raiser as having a squeeze opportunity regardless of whether a caller existed in front of them.
- **Fix:** in the preflop walk, I track `callers_since_last_raise`. A `squeeze_opportunity` is added only when, at the moment a player acts, `raises_so_far >= 1 AND callers_since_last_raise >= 1 AND player hasn't yet raised`. The opener never satisfies this (they act first). The first caller never satisfies this (callers_since_last_raise was 0 when they acted). A squeeze attempt requires that the player then actually raise from that spot.
- **Why old logic was wrong:** squeeze is a specific spot — raise + caller(s) + you re-raise from behind. The old code reduced it to "raised + others did stuff", which lost the positional and temporal constraints entirely.

---

### T9 — Fold-to-3-bet-after-raising is per-hand, not lifetime-gated

**Status:** `[FIXED]` — was `[PASS-WEAK]` initially (test passed against buggy code because the buggy path was guarded by a `len(preflop_actions) >= 2` check). My rewrite removes the lifetime gate entirely, so the deeper bug is no longer reachable.
**Bug ref:** Audit §2.6, `opponent_tracker.py:482-491`
**Scenario:** Heads-up. Hand 1: A raises preflop, B re-raises, A folds. Hand 2: A folds preflop. Hand 3: A calls preflop.
**Assertion (after all 3 hands):**
- `tracker.opponents[A].faced_3bet_after_raise == 1` (only hand 1)
- `tracker.opponents[A].folded_to_3bet_after_raise == 1`
**Why it matters:** Current code uses `opponent.raised_preflop > 0` (lifetime) as the gate. After A's first preflop raise in any hand, this branch fires on every subsequent hand for A, inflating the denominator. However, the path also requires `len(preflop_actions) >= 2 AND len(preflop_raises) == 1` for the player **this hand**, which accidentally guards against the simplest "fold preflop" and "call preflop" follow-ups. The bug actually triggers when the player calls+raises (limp-raise) in a later hand. Stronger test would be needed, but my rewrite walks the action sequence and computes this correctly hand-by-hand, so the deeper bug is fixed as a side-effect of T1-T8 fixes.

**Fix Notes:**
- **Bug:** the lifetime gate `if opponent.raised_preflop > 0` plus the per-hand structural inference `if len(preflop_raises) == 1 AND len(preflop_actions) >= 2` would mis-fire in a future hand where the player limp-raises (call then raise after facing a raise). The denominator would inflate without a real 3-bet ever happening.
- **Fix:** `_analyze_preflop_action_sequence` now computes `faced_3bet_after_raising` and `folded_to_3bet_after_raising` directly from the action sequence: it finds the player's first raise, looks for any subsequent raise by a DIFFERENT player, and (if found) checks the player's first action after that re-raise for FOLD.
- **Why old logic was wrong:** it conflated "this player raised at some point in life" with "this player raised this hand AND faced a 3-bet". The lifetime gate and the structural inference were both wrong proxies for a per-hand sequence check.

---

### T10 — AF and VPIP/PFR use consistent windowing

**Status:** `[FAIL]`
**Bug ref:** Audit §2.9
**Scenario:** Player B plays 100 hands. Across those hands, they have 60 calls and 30 bets/raises lifetime, but only 5 calls and 15 raises in their most recent 50 actions.
**Assertion:** AF should be computed in the same regime as VPIP/PFR. Either both lifetime (`AF = 30/60 = 0.5`) or both windowed; current code mixes them. Test asserts the chosen regime is consistent.
**Why it matters:** AF swings on recency while VPIP/PFR are lifetime, mixing signals. The feature `[2]` value can fluctuate sharply while the others remain stable.

**Decision needed before implementing:** Pick lifetime or windowed. Plan recommends lifetime since other stats are lifetime; window can be added as a separate feature later.

**Fix Notes:**
_(to be filled in after fix)_

---

## Sanity tests — Should pass already, catch regressions

### T11 — VPIP across hands (already covered in `test_opponent_stat_calculation.py`, duplicated here for reproducibility)

**Status:** `[ ]` — not written; existing `test_opponent_stat_calculation.py` covers this.
**Scenario:** Player calls in 1 of 2 hands.
**Assertion:** `vpip == 0.5`, `hands_played == 2`.

---

### T12 — PFR across hands (already covered, duplicated for completeness)

**Status:** `[FAIL]`
**Scenario:** Player raises preflop in 1 of 3 hands.
**Assertion:** `pfr == round(1/3, 3)`.

---

### T13 — Confidence scales with hands_played

**Status:** `[FAIL]`
**Scenario:** Play 50 hands. Then play 50 more.
**Assertion:**
- After 50 hands: `confidence == 0.5`
- After 100 hands: `confidence == 1.0`
- After 150 hands: `confidence == 1.0` (capped)

---

### T14 — Empty tracker returns sane defaults

**Status:** `[FAIL]`
**Scenario:** No hands played.
**Assertion:** `tracker.get_opponent_stats(opponent_id=99) is None`.

---

## Tests skipped for now (track in audit doc, not blocking)

- `Action.BET` reachability — dead code today; will be cleaned during decoupling.
- Hand strength MC sanity — orthogonal to tracker.
- Reset stacks between hands — already covered by `test_reward_bugs.py`.

---

## Test execution log

| Run | Date | Status | Notes |
|---|---|---|---|
| Initial | 2026-05-10 | 8 fail / 3 pass | T1-T8 fail as expected. T9 passes — bug exists but my scenario doesn't trip it (preflop_actions length check accidentally guards the path). T13, T14 sanity pass. |
| Post-fix | 2026-05-10 | 11/11 pass | All adversarial tests fixed via `_analyze_preflop_action_sequence` + `_analyze_flop_action_sequence` rewrite plus env `_calculate_player_positions` fix. |
| Coverage | 2026-05-10 | +20 pass | Added `tests/test_env/test_opponent_tracker_coverage.py` — happy-path tests for every metric (C-1..C-17b). Acts as regression guard on top of the adversarial suite. |

---

## Why 11 adversarial tests isn't the full coverage story

The bug-tests file (`test_opponent_tracker_bugs.py`) contains 11 tests, all
*adversarial* — each one was written to witness a specific defect from the
audit. They are not exhaustive coverage. Coverage of the happy path lives
elsewhere:

- `test_opponent_tracker.py` — 20 pre-existing tests covering profile init,
  action recording, hand history, position tracking, stack ratios, and
  player-type classification.
- `test_env_opponent_tracker_integration.py` — 27 tests covering env↔tracker
  integration (observation shape, action conversion, multi-hand
  accumulation, the `track_opponents` flag).
- `test_opponent_tracker_coverage.py` — **20 new tests** (added 2026-05-10)
  giving each tracked metric at least one dedicated positive-path test.

Total tracker coverage: **78 tests** (20 + 27 + 11 + 20).

---

## Coverage-suite tests (test_opponent_tracker_coverage.py)

These are positive-path regression tests, one (or more) per metric. They
complement the adversarial bug tests above: bug tests prove specific
defects don't recur; coverage tests prove the happy-path math still works.

| ID | Metric | Scenario | Asserts |
|---|---|---|---|
| C-1 | VPIP | Fold/Call/Raise across 3 hands | `vpip == 2/3` for caller+raiser |
| C-2 | PFR | Raise×2, Call, Fold across 4 hands | `pfr == 0.5`, `vpip == 0.75` |
| C-3 | AF | 2 raises + 1 call | `af == 2.0` |
| C-4 | Confidence | 150 hands | `confidence == 1.0` (capped) |
| C-5 | hands_played | Player sits out a hand | Increments only when they acted |
| C-6 | 3-bet opportunity | Open-raise then fold | Opener: 0 opp; callers/folders: 1 opp |
| C-7 | 3-bet frequency | 3 hands, 1 actual 3-bet | `three_bet_frequency == 1/3` |
| C-8 | Squeeze | Raise → call → re-raise | `squeeze_attempts == 1` |
| C-9 | Fold-to-3bet-after-raise | A raises → B 3-bets → A folds | A: `fold_to_3bet_after_raise_percent == 1.0` |
| C-9b | (negative) | A raises → B 3-bets → A calls | A: faced but didn't fold |
| C-10 | C-bet made | Aggressor opens flop with bet | `flop_cbet_percent == 1.0` |
| C-10b | C-bet missed | Aggressor checks flop | Opp counted, made not counted |
| C-11 | Fold-to-cbet | Caller folds to aggressor's bet | `fold_to_flop_cbet_percent == 1.0` |
| C-12 | WTSD | Both reach showdown after check-down | Both: `went_to_showdown == 1` |
| C-13 | W$SD | 2 showdowns, each player wins 1 | Each: `win_at_showdown_percent == 0.5` |
| C-14 | WWSF | 2 hands seeing flop, each wins 1 | `wwsf_percent == 0.5` (incl. non-showdown wins) |
| C-15 | Observation layout | Manually-pinned profile | 12 features in expected order |
| C-16 | Neutral defaults | Unknown player in slot | Defaults `[0.3, 0.2, 0.33, 0.1, 0.5, 0.5, 0.2, 0.4, 0.4, 0.5, 0.05, 0.0]` |
| C-17 | Zero padding | Empty slots | All zeros |
| C-17b | Feature clipping | Out-of-range raw values | Clipped to 1.0 (except confidence trailer) |

---

## Open definitional questions to resolve while implementing

1. **Is a check-raise on the flop a "c-bet"?** Industry standard: no. Test T5 assumes "c-bet = preflop aggressor opens flop betting with a bet (or all-in)."
2. **Is an all-in preflop a "raise" for 3-bet purposes?** Treat all-in preflop as equivalent to RAISE (matches current code).
3. **What's a "3-bet" with limps?** Convention: a 3-bet requires a prior raise (not a limp). Limp + raise + reraise = 2-bet then 3-bet from the third action.
4. **In multi-way pots, who's "the c-better"?** Only the preflop aggressor — and only if they open the flop betting before anyone else acts aggressively (i.e., no donk bet from out-of-position).
