"""
Native tkinter GUI to play heads-up vs a trained PPO bot.

Launch:
    .venv/bin/python play_app/play.py

Single window, no browser. Felt-green oval table with the bot at top and
you at the bottom. Cards, stacks, bets, pot, and street all render on the
table; action buttons hang off the bottom edge. Model picker in the top
toolbar lists every PPO card in models/registry.json.
"""

from __future__ import annotations

import json
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

from treys import Card

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.opponent_ppo import OpponentPPO  # noqa: E402
from src.poker_env.texas_holdem_env import TexasHoldemEnv  # noqa: E402


REGISTRY_PATH = ROOT / "models" / "registry.json"
HUMAN_SEAT = 0
BOT_SEAT = 1

# ------- visual constants
FELT = "#0b6e3d"
FELT_EDGE = "#063b1f"
RAIL = "#3a1c0a"
TEXT = "#f4f4f4"
CARD_BG = "#fafafa"
CARD_BACK = "#3b3b8a"
RED = "#c0382b"
BLACK = "#1c1c1c"
CHIP = "#e7c84a"
POT_RING = "#f4e07a"

WIN_W = 1000
WIN_H = 720


# ---------------------------------------------------------------- registry

def load_registry_models():
    """Returns sorted list of (card_id, absolute_path) for PPO cards with
    a model file on disk."""
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    out = []
    for aid, card in reg["agents"].items():
        if card.get("kind") != "ppo":
            continue
        p = card.get("path")
        if not p:
            continue
        full = (ROOT / p.lstrip("./")).resolve()
        if full.exists():
            out.append((aid, str(full)))
    out.sort(key=lambda kv: kv[0])
    return out


# ---------------------------------------------------------------- helpers

def card_label(card_int: int) -> tuple[str, str]:
    """Return (rank_text, color) for a treys card int. Treys cards are
    rendered as e.g. 'Ah' / 'Td'. We turn the suit char into a coloured
    symbol so the table looks like a real table."""
    s = Card.int_to_str(card_int)
    rank, suit_char = s[0], s[1]
    suit_symbol = {"h": "♥", "d": "♦", "s": "♠", "c": "♣"}[suit_char]
    color = RED if suit_char in ("h", "d") else BLACK
    return f"{rank}{suit_symbol}", color


def action_label(action: int, env: TexasHoldemEnv) -> str:
    if action == 0:
        return "Fold"
    if action == 1:
        # Show $ to-call so the player knows what calling actually costs.
        gs = env.game_state
        me = gs.players[gs.current_player_idx]
        to_call = max(0, gs.pot_manager.current_bet - me.current_bet)
        return "Check" if to_call == 0 else f"Call ${to_call}"
    all_in_idx = 2 + len(env.raise_bins)
    if action == all_in_idx:
        gs = env.game_state
        me = gs.players[gs.current_player_idx]
        return f"All-in (${me.stack})"
    bin_idx = action - 2
    frac = env.raise_bins[bin_idx]
    pot = env.game_state.pot_manager.get_pot_total()
    chips = int(round(frac * pot))
    return f"Raise ${chips} ({frac:.1f}× pot)"


def pseudo_harmonic_translate(
    amount_chips: int, env: TexasHoldemEnv, rng: random.Random | None = None
) -> tuple[int, str]:
    """Map an off-bin human raise amount to one of the bot's discrete
    actions via randomised pseudo-harmonic mapping (Ganzfried & Sandholm,
    2013). Returns (action_index, explanation).

    The bot was trained on a discrete action set: each raise bin is a
    fraction of the pot, plus all-in. When the human bets an amount that
    falls between adjacent bins A and B, we pick A or B with probability

        P(treat as A) = (B - x)(1 + A) / ((B - A)(1 + x))

    where x is the user's bet expressed as a fraction of the pot. The
    weighting preserves pot odds in expectation so the bot's response
    stays approximately game-theoretically valid even when the human
    bets oddly. Randomising prevents an opponent from exploiting a
    deterministic boundary.

    Below the smallest bin → that bin. Above the largest non-all-in bin
    but below the all-in stack → mapped against all-in expressed as a
    fraction of pot. Above all-in → all-in.
    """
    if rng is None:
        rng = random
    gs = env.game_state
    me = gs.players[gs.current_player_idx]
    pot = max(1, gs.pot_manager.get_pot_total())
    bins = env.raise_bins
    all_in_idx = 2 + len(bins)
    # Translate the typed chip amount into a pot fraction so the math
    # lines up with the (pot-fraction) bin definitions.
    x = amount_chips / pot
    all_in_frac = me.stack / pot

    # Construct the full ladder of legal bet fractions in ascending
    # order: each raise bin, then all-in.
    ladder: list[tuple[int, float]] = [(2 + i, bins[i]) for i in range(len(bins))]
    ladder.append((all_in_idx, all_in_frac))
    ladder.sort(key=lambda kv: kv[1])

    # Clamp below the lowest bin.
    if x <= ladder[0][1]:
        return ladder[0][0], f"≤ smallest bin → {action_label(ladder[0][0], env)}"
    # Clamp at all-in.
    if x >= ladder[-1][1]:
        return ladder[-1][0], f"≥ all-in → {action_label(ladder[-1][0], env)}"

    # Find adjacent bins (A, B) bracketing x.
    for i in range(len(ladder) - 1):
        a_idx, A = ladder[i]
        b_idx, B = ladder[i + 1]
        if A <= x <= B:
            # Randomised pseudo-harmonic mapping (Ganzfried & Sandholm).
            p_a = ((B - x) * (1 + A)) / ((B - A) * (1 + x))
            p_a = min(1.0, max(0.0, p_a))
            chose_a = rng.random() < p_a
            chosen_idx = a_idx if chose_a else b_idx
            chosen_label = action_label(chosen_idx, env)
            return (
                chosen_idx,
                f"{x:.2f}× pot → P(low={A:.2f})={p_a:.2f}, chose {chosen_label}",
            )

    # Fallback — shouldn't happen given the clamps above.
    return ladder[-1][0], "fallback → top of ladder"


# ---------------------------------------------------------------- app

class PokerTableApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LagBot — Play")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg=RAIL)

        # Pre-instantiate every font used on the Canvas. Registering them
        # with the Tk interpreter (as Font objects, not raw tuples) sidesteps
        # the macOS Tk bug where Canvas text drawn before the window is fully
        # realized comes up with the default/system fill color instead of
        # the one we asked for.
        self.font_card = tkfont.Font(family="Helvetica", size=22, weight="bold")
        self.font_seat_name = tkfont.Font(family="Helvetica", size=14, weight="bold")
        self.font_seat_stack = tkfont.Font(family="Helvetica", size=15, weight="bold")
        self.font_pot = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.font_street = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.font_tocall = tkfont.Font(family="Helvetica", size=14, weight="bold")
        self.font_chip = tkfont.Font(family="Helvetica", size=13, weight="bold")
        self.font_result = tkfont.Font(family="Helvetica", size=22, weight="bold")
        self.font_match_over = tkfont.Font(family="Helvetica", size=20, weight="bold")
        self.font_button = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.font_button_italic = tkfont.Font(family="Helvetica", size=12,
                                              weight="bold", slant="italic")

        self.models = load_registry_models()
        if not self.models:
            tk.Label(root, text=f"No PPO models with disk paths in {REGISTRY_PATH}",
                     fg="red", bg=RAIL).pack(pady=40)
            return

        self.bot_cache: dict[str, OpponentPPO] = {}
        self.env: TexasHoldemEnv | None = None
        self.bot: OpponentPPO | None = None
        self.hand_over = False
        self.action_log: list[str] = []
        self.stack_before_hand = 0
        self._action_buttons: list[tk.Widget] = []
        self._game_over = False  # session-level: someone busted with no rebuy

        self._build_toolbar()
        self._build_canvas()
        self._build_action_bar()
        self._build_log()

        # Auto-start match with the latest model.
        latest_id, latest_path = self.models[-1]
        self.model_var.set(latest_id)
        self._start_match(latest_path)

    # ------- layout

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=RAIL)
        bar.pack(side=tk.TOP, fill=tk.X, padx=12, pady=8)
        tk.Label(bar, text="Opponent:", fg=TEXT, bg=RAIL,
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0, 6))
        self.model_var = tk.StringVar(value=self.models[-1][0])
        names = [m[0] for m in self.models]
        ttk.Combobox(bar, textvariable=self.model_var, values=names,
                     state="readonly", width=42).pack(side=tk.LEFT)
        start_btn = tk.Label(bar, text="Start match", bg="#1f6f43", fg=TEXT,
                             font=self.font_button, padx=14, pady=6,
                             cursor="hand2", relief=tk.RAISED, bd=2)
        start_btn.bind("<Button-1>", lambda _e: self._on_start_match())
        start_btn.pack(side=tk.LEFT, padx=8)
        self.status_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.status_var, fg=TEXT, bg=RAIL,
                 font=("Helvetica", 11, "italic")).pack(side=tk.RIGHT)

    def _build_canvas(self):
        self.canvas = tk.Canvas(self.root, bg=RAIL, width=WIN_W, height=480,
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        # Re-render on first map and on every resize so canvas text always
        # gets drawn after the window is fully realized (avoids the macOS
        # Tk bug where text drawn pre-map renders with the wrong fill).
        self.canvas.bind("<Map>", lambda _e: self._safe_render())
        self.canvas.bind("<Configure>", lambda _e: self._safe_render())

    def _safe_render(self):
        if self.env is not None:
            self._render()

    def _build_action_bar(self):
        self.action_frame = tk.Frame(self.root, bg=RAIL)
        self.action_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=10)

    def _build_log(self):
        self.log_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.log_var, fg=TEXT, bg=RAIL,
                 font=("Helvetica", 10), justify=tk.LEFT,
                 anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 6))

    # ------- gameplay control

    def _on_start_match(self):
        mid = self.model_var.get()
        path = dict(self.models)[mid]
        self._start_match(path)

    def _start_match(self, model_path: str):
        env = TexasHoldemEnv(
            num_players=2, starting_stack=1000, small_blind=5, big_blind=10,
            track_opponents=True, learning_agent_id=HUMAN_SEAT,
        )
        env.reset()
        self.env = env
        self.bot = self._get_bot(model_path)
        self.action_log = []
        self.hand_over = False
        self._game_over = False
        self.stack_before_hand = env.game_state.players[HUMAN_SEAT].starting_stack_this_hand
        self.status_var.set(f"vs {Path(model_path).parent.name}")
        self._auto_play_bot()
        self._render()

    def _deal_next_hand(self):
        # Detect session bust (heads-up, no auto-rebuy mid-session).
        env = self.env
        human = env.game_state.players[HUMAN_SEAT]
        bot_p = env.game_state.players[BOT_SEAT]
        # env.reset() will auto-rebuy a busted player. For a "you vs the
        # bot" session feel, declare the match over once either side hits
        # zero rather than silently regenerating chips.
        if human.stack <= 0 or bot_p.stack <= 0:
            self._game_over = True
            self._render()
            return
        env.reset()
        self.action_log = []
        self.hand_over = False
        self.stack_before_hand = env.game_state.players[HUMAN_SEAT].starting_stack_this_hand
        self._auto_play_bot()
        self._render()

    def _get_bot(self, path: str) -> OpponentPPO:
        if path not in self.bot_cache:
            self.bot_cache[path] = OpponentPPO(path, deterministic=False)
        return self.bot_cache[path]

    def _auto_play_bot(self):
        env = self.env
        while (not self.hand_over
               and env.game_state.current_player_idx == BOT_SEAT):
            obs = env._get_observation()
            valid = env.get_valid_actions()
            action = self.bot.select_action(obs, valid)
            label = action_label(action, env)
            _, _, terminated, _, _ = env.step(action)
            self.action_log.append(f"Bot: {label}")
            if terminated:
                self.hand_over = True
                break

    def _human_action(self, action: int):
        env = self.env
        label = action_label(action, env)
        _, _, terminated, _, _ = env.step(action)
        self.action_log.append(f"You: {label}")
        if terminated:
            self.hand_over = True
        else:
            self._auto_play_bot()
        self._render()

    # ------- drawing

    def _render(self):
        c = self.canvas
        c.delete("all")
        w = int(c.winfo_width()) or WIN_W
        h = int(c.winfo_height()) or 480
        cx, cy = w / 2, h / 2

        # Felt oval table.
        c.create_oval(40, 40, w - 40, h - 40, fill=FELT_EDGE, outline="")
        c.create_oval(60, 60, w - 60, h - 60, fill=FELT, outline="")

        env = self.env
        human = env.game_state.players[HUMAN_SEAT]
        bot_p = env.game_state.players[BOT_SEAT]
        pot = env.game_state.pot_manager.get_pot_total()
        to_call = max(0, env.game_state.pot_manager.current_bet - human.current_bet)
        street_names = ["Preflop", "Flop", "Turn", "River", "Showdown"]
        street = street_names[env.game_state.betting_round.value]

        # Bot (top seat). Show only the trailing "...genN" so the plate fits.
        full_name = self.status_var.get()[3:] if self.status_var.get().startswith('vs ') else 'bot'
        short = full_name.split("_")[-1] if "_" in full_name else full_name
        # Bot's hand is only revealed at a real showdown — i.e., when the
        # hand is over AND neither player folded. A fold ends the hand
        # without obligating the winner to show.
        went_to_showdown = (
            self.hand_over and human.is_active and bot_p.is_active
        )
        self._draw_seat(
            cx, 120,
            label=f"BOT ({short})",
            stack=bot_p.stack,
            bet=bot_p.current_bet,
            cards=bot_p.hand if (went_to_showdown and bot_p.hand) else None,
            face_down=not went_to_showdown,
            is_turn=(env.game_state.current_player_idx == BOT_SEAT
                     and not self.hand_over),
            cards_above=True,
            chip_y=170,
        )
        # Human (bottom seat). Cards drawn BELOW plate so they sit in front
        # of the user — and so they never collide with the pot in the middle.
        self._draw_seat(
            cx, h - 130,
            label="YOU",
            stack=human.stack,
            bet=human.current_bet,
            cards=human.hand,
            face_down=False,
            is_turn=(env.game_state.current_player_idx == HUMAN_SEAT
                     and not self.hand_over),
            cards_above=False,
            chip_y=h - 180,
        )

        # Community cards row.
        self._draw_community(cx, cy - 40, env.game_state.community_cards)

        # Pot in middle.
        c.create_oval(cx - 80, cy + 20, cx + 80, cy + 78,
                      fill="#0d4a2a", outline=POT_RING, width=3)
        c.create_text(cx, cy + 49, text=f"POT  ${pot}",
                      fill=POT_RING, font=self.font_pot)

        # Street + to-call indicator.
        c.create_text(cx, 80, text=street.upper(),
                      fill=TEXT, font=self.font_street)
        if to_call > 0 and not self.hand_over:
            c.create_text(cx, h - 70, text=f"To call: ${to_call}",
                          fill=POT_RING, font=self.font_tocall)

        # Result banner.
        if self.hand_over:
            profit = human.stack - self.stack_before_hand
            if profit > 0:
                msg, col = f"YOU WON  +${profit}", "#5be07a"
            elif profit < 0:
                msg, col = f"YOU LOST  -${-profit}", "#ff7575"
            else:
                msg, col = "PUSH", POT_RING
            c.create_text(cx, cy - 110, text=msg, fill=col,
                          font=self.font_result)

        if self._game_over:
            c.create_text(cx, cy, text="MATCH OVER — start a new match",
                          fill="#ff7575", font=self.font_match_over)

        # Action log.
        self.log_var.set("  •  ".join(self.action_log[-6:]))

        # Rebuild action buttons row.
        self._refresh_action_buttons()

    def _draw_seat(self, cx, cy, *, label, stack, bet, cards, face_down,
                   is_turn, cards_above=True, chip_y=None):
        c = self.canvas
        # Player nameplate — wider, two-line so name and stack never collide.
        plate_w = 280
        plate_h = 56
        outline = POT_RING if is_turn else "#1d3a26"
        c.create_rectangle(cx - plate_w / 2, cy - plate_h / 2,
                           cx + plate_w / 2, cy + plate_h / 2,
                           fill="#0d4a2a", outline=outline, width=3)
        # Truncate long model names so the plate stays readable.
        shown = label if len(label) <= 28 else (label[:25] + "...")
        c.create_text(cx, cy - 12, text=shown, fill=TEXT,
                      font=self.font_seat_name)
        c.create_text(cx, cy + 12, text=f"${stack}", fill=POT_RING,
                      font=self.font_seat_stack)

        # Hole cards: above or below the plate depending on seat.
        offset = -75 if cards_above else 75
        card_y = cy + offset
        spacing = 66
        if cards is None:
            for i in range(2):
                x = cx + (i - 0.5) * spacing
                self._draw_card_back(x, card_y)
        else:
            for i, card_int in enumerate(cards):
                x = cx + (i - 0.5) * spacing
                self._draw_card(x, card_y, card_int)

        # Current-bet chip indicator (caller picks y so we don't collide).
        if bet > 0 and chip_y is not None:
            self._draw_chip(cx, chip_y, bet)

    def _draw_card(self, x, y, card_int):
        c = self.canvas
        w, h = 56, 78
        c.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                           fill=CARD_BG, outline="#111", width=2)
        text, color = card_label(card_int)
        c.create_text(x, y, text=text, fill=color, font=self.font_card)

    def _draw_card_back(self, x, y):
        c = self.canvas
        w, h = 56, 78
        c.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                           fill=CARD_BACK, outline="#fff", width=2)
        c.create_text(x, y, text="✦", fill="#fff", font=self.font_card)

    def _draw_community(self, cx, cy, cards):
        c = self.canvas
        slot_w = 66
        n_slots = 5
        start = cx - (n_slots - 1) / 2 * slot_w
        for i in range(n_slots):
            x = start + i * slot_w
            if i < len(cards):
                self._draw_card(x, cy, cards[i])
            else:
                # Empty slot outline.
                c.create_rectangle(x - 28, cy - 39, x + 28, cy + 39,
                                   outline="#1d3a26", dash=(3, 3), width=2)

    def _draw_chip(self, x, y, amount):
        c = self.canvas
        c.create_oval(x - 34, y - 18, x + 34, y + 18,
                      fill=CHIP, outline="#7a5b00", width=2)
        c.create_text(x, y, text=f"${amount}", fill="#3a2700",
                      font=self.font_chip)

    # ------- action buttons

    def _make_button(self, parent, text, bg, command, padx=18):
        """tk.Button on macOS Aqua ignores fg/bg, leaving the label rendered
        in the system default colour (washes out on coloured backgrounds).
        A tk.Label with click bindings respects both colours and looks like
        a button when given a raised relief + cursor."""
        lbl = tk.Label(parent, text=text, bg=bg, fg=TEXT,
                       font=self.font_button, padx=padx, pady=8,
                       cursor="hand2", relief=tk.RAISED, bd=2)
        lbl.bind("<Button-1>", lambda _e: command())
        return lbl

    def _refresh_action_buttons(self):
        for w in self._action_buttons:
            w.destroy()
        self._action_buttons.clear()

        env = self.env
        if self._game_over:
            return

        if self.hand_over:
            btn = self._make_button(self.action_frame, "Deal next hand",
                                    "#1f6f43", self._deal_next_hand, padx=24)
            btn.pack(side=tk.LEFT, padx=6)
            self._action_buttons.append(btn)
            return

        if env.game_state.current_player_idx != HUMAN_SEAT:
            lbl = tk.Label(self.action_frame, text="Bot is thinking…",
                           fg=TEXT, bg=RAIL, font=self.font_button_italic)
            lbl.pack(side=tk.LEFT, padx=6)
            self._action_buttons.append(lbl)
            return

        valid = env.get_valid_actions()
        for action in valid:
            color = "#8b1a1a" if action == 0 else "#1f4f7a" if action == 1 else "#a05a14"
            btn = self._make_button(
                self.action_frame, action_label(action, env), color,
                lambda a=action: self._human_action(a),
            )
            btn.pack(side=tk.LEFT, padx=4)
            self._action_buttons.append(btn)

        # Custom-$ entry — typed amount is translated to one of the bot's
        # discrete actions via randomised pseudo-harmonic mapping. Every
        # widget here is appended to self._action_buttons so the next
        # _refresh_action_buttons() destroys them cleanly (avoids the
        # "walking widgets" bug from before).
        raise_actions = [a for a in valid if a >= 2]
        if raise_actions:
            label = tk.Label(self.action_frame, text="  Custom $:",
                             fg=TEXT, bg=RAIL, font=self.font_button)
            label.pack(side=tk.LEFT)
            self._action_buttons.append(label)

            entry = tk.Entry(self.action_frame, width=8,
                             font=self.font_button)
            entry.pack(side=tk.LEFT, padx=4)
            self._action_buttons.append(entry)

            def submit_custom(_event=None, e=entry):
                try:
                    amt = int(e.get().strip().lstrip("$"))
                except ValueError:
                    return
                if amt <= 0:
                    return
                action, explanation = pseudo_harmonic_translate(amt, env)
                self.action_log.append(f"You typed ${amt} — {explanation}")
                self._human_action(action)

            submit_btn = self._make_button(
                self.action_frame, "Bet ↑", "#a05a14", submit_custom,
                padx=14,
            )
            submit_btn.pack(side=tk.LEFT, padx=4)
            self._action_buttons.append(submit_btn)
            entry.bind("<Return>", submit_custom)


def main():
    root = tk.Tk()
    app = PokerTableApp(root)
    # First render needs the window mapped so winfo_width returns real size.
    root.update_idletasks()
    if app.env is not None:
        app._render()
    # macOS Tk renders Canvas text as default/white until the window gets
    # focus and is fully realized. Force focus + schedule a few delayed
    # re-renders so the post-realization frame has correct text colors.
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.update()
    root.after(50, app._safe_render)
    root.after(300, app._safe_render)
    root.after(800, app._safe_render)
    root.mainloop()


if __name__ == "__main__":
    main()
