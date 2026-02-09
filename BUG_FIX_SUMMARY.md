# Bug Fix Summary: Reward Calculation and Stack Reset

## Date: 2026-02-08

## Priority: CRITICAL ✅ FIXED

These bugs were causing impossible reward values (e.g., -490 BB) that corrupted training signals.

---

## ✅ Bug 1: Mid-Hand Stack Reset (FIXED)

### Problem
`reset_stacks_every_n_timesteps` could fire mid-hand in `step()`, injecting chips during play:
- Stack set to 1000 mid-hand
- `total_bet_this_hand` not reset (still 800)
- Next reward calculation: `starting_stack = 1000 + 800 = 1800`
- Phantom losses up to -180 BB for a single hand (impossible!)

### Fix
**File:** `src/poker_env/texas_holdem_env.py`

Moved stack reset logic from `step()` to `reset()`:
- Resets only happen between hands (in `reset()` method)
- Never during `step()` (mid-hand)
- Preserves chip conservation within each hand

**Lines changed:**
- `reset()`: Added stack reset check before `start_new_hand()`
- `step()`: Removed mid-hand reset block (lines 215-220)

---

## ✅ Bug 2: Fragile Reward Calculation (FIXED)

### Problem
Reward calculation recomputed starting stack on every step:
```python
starting_stack = current_player.stack + current_player.total_bet_this_hand
```
Any modification to `stack` or `total_bet_this_hand` between steps corrupted rewards.

### Fix
**File:** `src/poker_env/texas_holdem_env.py`

Use stored value instead of recomputing:
```python
# BEFORE:
starting_stack = current_player.stack + current_player.total_bet_this_hand

# AFTER:
starting_stack = current_player.starting_stack_this_hand
```

This value is set once per hand in `Player.reset_for_new_hand()` BEFORE blinds are posted.

**Methods updated:**
- `step()` (line 154)
- `step_with_raise()` (line 395)

---

## ✅ Bug 3: Verification (CONFIRMED CORRECT)

### Verified
**File:** `src/poker_env/player.py`

Confirmed `starting_stack_this_hand` is set correctly in `reset_for_new_hand()`:
- Set to `self.stack` BEFORE blinds are posted ✅
- Called by `GameState.start_new_hand()` before blind posting ✅
- Value remains constant throughout the hand ✅

---

## Test Results

### New Tests (All Passing ✅)
**File:** `tests/test_env/test_reward_bugs.py`

| Test | Status | Purpose |
|------|--------|---------|
| `test_reward_never_exceeds_starting_stack` | ✅ PASS | Reward magnitude ≤ 100 BB (for 1000/10 game) |
| `test_stack_reset_only_between_hands` | ✅ PASS | Chip conservation preserved mid-hand |
| `test_starting_stack_this_hand_set_before_blinds` | ✅ PASS | Correct value stored |
| `test_reward_calculation_uses_stored_starting_stack` | ✅ PASS | Uses stored value, not recomputed |
| `test_no_mid_hand_reset_messages` | ✅ PASS | [RESET] only in reset(), never step() |
| `test_multiple_resets_preserve_chip_conservation` | ✅ PASS | Multiple resets don't break conservation |

**All 6 tests passing!**

### Existing Tests
**File:** All test_env tests

- **189 tests passing** (unchanged) ✅
- **13 tests failing** (pre-existing, unrelated to bug fixes)

The failing tests were already broken before these changes and are unrelated to reward/reset logic.

---

## Impact on Training

### Before Fix
```python
agent/min_reward: -490    # Impossible! (should be ≤ -100 for 1000/10 game)
agent/max_reward: 51.5
agent/avg_reward: -0.0504  # Negative despite 83.9% win rate
```

### After Fix (Expected)
```python
agent/min_reward: -100    # Worst case: lose entire starting stack
agent/max_reward: 200     # Best case: win from 2 opponents
agent/avg_reward: Positive # Should align with 80%+ win rate
```

---

## Files Modified

1. ✅ `src/poker_env/texas_holdem_env.py`
   - Moved stack reset to `reset()` method
   - Fixed `starting_stack` calculation in `step()` and `step_with_raise()`

2. ✅ `tests/test_env/test_reward_bugs.py`
   - Added 6 comprehensive tests for the bugs

3. ✅ `src/poker_env/player.py`
   - No changes (verified correct)

---

## Validation Commands

```bash
# Run new bug fix tests
pytest tests/test_env/test_reward_bugs.py -v

# Run all environment tests
pytest tests/test_env/ -v

# Validate in training (run for 50k steps)
python train.py --config configs/default_config.yaml --name validation_run
# Check TensorBoard: agent/min_reward should never go below -100
```

---

## Next Steps for User

1. **Stop current training** (corrupted by the bugs):
   ```bash
   ps aux | grep train.py
   kill <PID>
   ```

2. **Start fresh training** with the fixes:
   ```bash
   cd /Users/mbb/Developer/Personal_Projects/LagBot
   python3 train.py --config configs/deep_architecture_3M.yaml --name deep_arch_3M_fixed
   ```

3. **Monitor in TensorBoard**:
   ```bash
   tensorboard --logdir ./logs/deep_arch_3M_fixed
   ```

   Verify:
   - `agent/min_reward` ≥ -100 (never below)
   - `agent/max_reward` ≤ 200 (never above)
   - `agent/avg_reward` trends positive with high win rate
   - No impossible spikes in reward values

---

## Root Cause Analysis

### Why Did This Happen?

1. **Mid-hand stack resets**: Attempting to be "helpful" by resetting stacks periodically, but doing it at the wrong time (mid-hand instead of between hands).

2. **Recomputed values**: Not trusting stored state, recomputing `starting_stack` on every call led to fragility when any intermediate state changed.

3. **Lack of bounds checking**: No assertions that rewards fall within possible ranges (±100 BB for 1000/10 game).

### Prevention

- Always store immutable hand state (starting_stack) at hand start
- Never modify game state mid-hand (resets, rebuys only between hands)
- Add bounds checks/assertions for critical values
- Test with `reset_stacks_every_n_timesteps` enabled to catch mid-hand issues

---

**Status: ✅ All Critical Bugs Fixed and Tested**
