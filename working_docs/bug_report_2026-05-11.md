---
title: Bug Report — Remaining Game-Logic Failures
date: 2026-05-11
status: open
---

# Bug Report — 2026-05-11

After the Agent/Player marriage refactor, 8 tests remain red. This report categorises each failure as one of:

- **PROD** — real production bug in game logic.
- **TEST** — test setup is wrong (production code is correct).
- **DESIGN** — a design-decision conflict where production code and test expectation disagree.

Test count: **307 pass / 8 fail**.

> *Test names use the form `file::Class::method`.*

---

## Headline bugs (highest blast radius)

### **PROD-1** — `env.step()` after terminal re-distributes the pot, creating chips

**Where:** `src/poker_env/texas_holdem_env.py` `_execute_step` (line ~184–265).

**Symptom:** Once `is_hand_complete() == True`, the env never short-circuits. Each subsequent call to `env.step(...)` runs through `execute_action`, sees `done = True`, and invokes `determine_winners()` *again* — re-distributing the same pot to the same winners. `Pot.amount` is only reset on `start_new_hand()`, so the same chips are paid out repeatedly.

**Reproduction (verified):**

```
After hand complete (3-way all-in):
  P0: stack=400  P1: stack=1500  P2: stack=600
extra step 0: P0=800   P1=3000   P2=1200
extra step 1: P0=1200  P1=4500   P2=1800
extra step 2: P0=1600  P1=6000   P2=2400
```

Stacks grow by the full pot every extra step.

**Affected tests:** `test_all_in_calculations.py::test_all_in_side_pots`, `test_bust_and_rebuy.py::test_chip_accounting_with_rake_and_rebuy`, `test_bust_and_rebuy.py::test_detailed_chip_report` (and any random‑action test whose post-hand loop misses a `break`).

**Severity:** HIGH. Silently breaks chip conservation and reward attribution. Any training loop that fails to `env.reset()` immediately on `terminated` will see inflated rewards.

**Proposed fix:** Make `_execute_step` no-op when the hand is already complete:

```python
if self.game_state.is_hand_complete():
    return self._get_observation(), 0.0, True, False, {'action': 'noop', 'hand_complete': True}
```

Or — preferred — gate `determine_winners()` by a `winnings_distributed` flag set after the first call and cleared in `reset()`.

---

### **PROD-2** — Auto-rebuy on `reset()` creates chips with no externally visible accounting

**Where:** `src/poker_env/texas_holdem_env.py:139–143`.

```python
for player in self.game_state.players:
    if player.stack <= 0:
        player.record_buy_in(self.starting_stack)
        player.stack = self.starting_stack
```

**Symptom:** Whenever a busted player sits at the table during `reset()`, the env injects a fresh `starting_stack` worth of chips. `Player.total_buy_in` is updated, but no chip-conservation invariant is exposed. Two tests (`test_chip_accounting_with_rake_and_rebuy`, `test_detailed_chip_report`) sum `player.stack` across players and assert it stays bounded — they fail because the env quietly creates chips.

**This is intentional behaviour for training continuity, but is silent.**

**Severity:** MEDIUM (correctness for training, but chip-accounting tests can't trust raw stack totals).

**Proposed fix:** Either:

- (a) Disable auto-rebuy in non-training contexts (make it a constructor flag, default off).
- (b) Expose `total_chips_in_play() = sum(stacks) + sum(pots) - sum(buy_ins) + initial_chips`. Then tests can assert a real invariant.
- (c) At minimum: leave the auto-rebuy in place but update the two failing tests to account for `total_buy_in` deltas.

Recommendation: **(c) for now, (a) longer-term.** The auto-rebuy is load-bearing for training — don't pull it without a replacement.

---

## Test setup bugs (production code is correct)

These four tests all assume **Player 0 is the first to act preflop** in a 3-player hand. They are wrong. After one button rotation in `start_new_hand()` (button goes `0 → 1`), the layout is:

- Button = P1
- SB = P2 (`current_bet=5`)
- BB = P0 (`current_bet=10`)
- **UTG / first to act = P1**

Every `game.execute_action(2, raise_amount=X)` in these tests is therefore applied to **P1**, not P0. The assertions then check P0's state, which never changed.

### **TEST-1** — `test_all_in_comprehensive.py::TestAllInCurrentBetFix::test_all_in_as_raise_updates_current_bet`

**What the test does:** `game.execute_action(2, raise_amount=995)` then asserts `pot_manager.current_bet == p0.current_bet`.

**What actually happens:** P1 (current player) bets 995, becoming all-in. P0 untouched. Output confirms: `P0: stack=990, current_bet=10, is_all_in=False  pot_manager.current_bet = 995`.

**Severity:** LOW (test bug). **Fix:** rewrite to operate on the actual current player, e.g. `current = game.get_current_player(); game.execute_action(...)` then assert on `current`.

### **TEST-2** — `test_all_in_comprehensive.py::TestAllInCurrentBetFix::test_all_in_as_call_does_not_update_current_bet`

Same root cause. Test advances to the flop then assumes P2 acts after P1. In fact preflop order is P1 → P2 → P0; after flop the order starts at the seat after the button (P2). The test mis-tracks who's on action by 1.

### **TEST-3** — `test_all_in_comprehensive.py::TestAllInCurrentBetFix::test_three_player_all_in_sequence`

Same root cause **plus** a typo: after the second action it asserts `p1.is_all_in` but the actor was P2.

### **TEST-4** — `test_all_in_comprehensive.py::TestAllInCurrentBetFix::test_all_in_short_stack_folds`

Test sets `p0.stack = 3` and calls `game.execute_action(2, raise_amount=p0.stack)`, expecting P0 to fold because `amount < to_call`. But P0 is BB (`to_call=0`), so even if P0 *were* the current player, the logic wouldn't fire. The current player is P1, so P0 isn't touched at all.

**However,** there's a **PROD smell** lurking under this test (see DESIGN-1).

---

## Design conflicts

### **DESIGN-1** — Sub-min-raise all-in: fold or all-in?

**The case:** Player has stack `s`, `to_call = c`, and tries to bet exactly `s` where `0 < s < min_raise` but `s >= c`. They can cover the call but cannot make a legal raise.

**Real poker rules:** Player goes all-in for `s`. The all-in counts as a call (does **not** reopen action — i.e. the original raiser cannot re-raise on this round).

**Current code (`pot_manager.place_bet`):** If `s < c` → fold. If `s >= c` → treat as a raise; `current_bet` is set to `player.current_bet`. The "doesn't reopen action" rule is **not** implemented anywhere — `last_aggressor_idx` gets set regardless.

**Severity:** MEDIUM — affects strategic correctness in multi-way pots. Doesn't violate chip conservation, but allows incorrect re-raises after a sub-min all-in.

**Proposed fix:** Two parts.

1. In `place_bet`, distinguish "all-in raise that reopens action" from "all-in raise that does not" based on whether `actual_bet - to_call >= min_raise`.
2. In `execute_action`, only set `last_aggressor_idx` when the all-in reopens action.

### **DESIGN-2** — Uncalled-bet refund: side pot vs. main-pot trim

**Where:** `pot_manager.calculate_side_pots`.

**The case from `test_hand_3_reproduction`:** Heads-up, P0 goes all-in for 850 more on the river, P1 has matched 150 on the river and refuses to call. State: `P0.total_bet=1000, P1.total_bet=150`.

**Current behaviour:** `calculate_side_pots` produces two pots:

1. Pot of 300 (both eligible) — the matched portion.
2. Pot of 850 (only P0 eligible) — the unmatched portion.

`distribute_pots` then gives the 850 back to P0 as a "win" in pot 2.

**Conventional poker semantics:** The 850 should be **refunded to P0's stack** (not paid as a pot), so the hand only has one contested pot of 300.

**Chip outcome is identical** in both conventions when P0 is in the only eligible group — they get the chips back. **But the test asserts `len(pots) == 1`**, which the current implementation violates.

**Severity:** LOW — UX/semantics only. No chip conservation issue.

**Proposed fix:** In `calculate_side_pots`, if the topmost level has only one eligible player, refund into that player's stack rather than creating a 1-eligible pot. Then update `test_hand_3_reproduction` to pass.

---

## Summary table

| # | Test | Category | Severity | Fix complexity |
|---|------|----------|----------|----------------|
| PROD-1 | `test_all_in_side_pots` | PROD | HIGH | LOW (guard at top of `_execute_step`) |
| PROD-1 | `test_chip_accounting_with_rake_and_rebuy` | PROD + DESIGN | HIGH | LOW (same fix) |
| PROD-1 | `test_detailed_chip_report` | PROD + DESIGN | HIGH | LOW (same fix) |
| PROD-2 | (subset of above) | PROD | MED | MED (config flag) |
| TEST-1 | `test_all_in_as_raise_updates_current_bet` | TEST | LOW | LOW (rewrite test) |
| TEST-2 | `test_all_in_as_call_does_not_update_current_bet` | TEST | LOW | LOW (rewrite test) |
| TEST-3 | `test_three_player_all_in_sequence` | TEST | LOW | LOW (rewrite test) |
| TEST-4 | `test_all_in_short_stack_folds` | TEST + DESIGN | LOW | LOW (rewrite test) |
| DESIGN-1 | (latent — not directly tested) | PROD smell | MED | MED |
| DESIGN-2 | `test_hand_3_reproduction` | DESIGN | LOW | MED |

---

## Do I know how to fix these?

Yes, with confidence on PROD-1 and the four TEST bugs. PROD-2 needs a product decision (auto-rebuy stays / goes / becomes optional). DESIGN-1 and DESIGN-2 are semantically correct under one reading of the rules and incorrect under another — they need *your* call before code changes.

**Recommended order:**

1. Fix PROD-1 (single guard). Re-run failing tests; expect 3 to flip green (`test_all_in_side_pots`, `test_chip_accounting_with_rake_and_rebuy`, `test_detailed_chip_report`).
2. Rewrite the four TEST tests to use the actual current player. Expect 4 more green.
3. Decide on DESIGN-2 (uncalled-bet refund). If "refund into pot 1," implement it in `calculate_side_pots`, then `test_hand_3_reproduction` will pass.
4. (Separately, not in the 8) Implement DESIGN-1 (no-reopen-on-sub-min-all-in) for strategic correctness.

---

## Diagnostic tests added

See `tests/test_env/test_pot_manager_diagnostic.py`. These tests isolate each suspected root cause by calling `place_bet` and `calculate_side_pots` directly with hand-rolled state, rather than driving through `env.step` (which has its own bugs). They will fail in known ways that pin each diagnosis.

---

## PROD-3 — Abandoned side-pot when over-better folds after an all-in (discovered 2026-05-11)

**Where:** `pot_manager.place_bet` (over-bet against all-in) + `pot_manager.distribute_pots` (no refund for unmatched portion).

**Symptom:** When player A is all-in and player B (deeper stack) raises *past* A's all-in amount, the excess chips go into the pot. If B later folds on a subsequent street, those uncalled chips end up in a side pot whose only eligible player has folded — `distribute_pots` then skips that pot (no `hand_ranks` entry for B), and the chips are abandoned.

**Reproduction (verified):** 3-player game, P0 all-in for 500 on flop, P1 raises to 1370 (uncalled portion = 870), P1 folds on turn. After hand: P0 wins 1005 (matched portion + blinds). Pot still holds 870 chips on entry to `start_new_hand()`, which wipes the pot — 870 chips evaporate. Chip-conservation: `stacks + pots < total_buy_in` at the start of every subsequent hand.

**Severity:** MEDIUM. Doesn't crash; just leaks chips. Hard to notice in training where rewards are normalized. Visible only via chip-conservation tests.

**Proposed fix (preferred):** In `place_bet`, when the bettor's stack-after-bet exceeds the max remaining stack of any non-folded opponent, cap the bet at the opponent ceiling and credit the excess back to the bettor immediately. This is the standard "no uncalled bets" poker rule.

**Alternative fix:** In `distribute_pots`, if `eligible_ranks` is empty for a pot, refund proportionally to the contributors of that pot tier.

**Affected tests:** `test_detailed_chip_report` (now uses `total_chips <= total_buy_in` as a weaker invariant — still catches chip *creation* but not chip *loss*).
