"""
Wordle AI Strategy Engine
A mathematically optimized Wordle solver using position-aware letter frequency analysis.

Usage:
    python wordle_ai.py

Input Format:
    After each guess, enter feedback in the format: GUESS: [LETTER][STATUS]...
    - g = Green  (correct letter, correct position)
    - y = Yellow (correct letter, wrong position)
    - w = Gray   (wrong letter / eliminated)

Commands:
    GUESS: <feedback>  — Submit feedback for the last guess
    RESTART             — Reset the game
    HINT                — Show letter frequency analysis
    TOP N               — Show top N candidate words
    EXIT                — Quit
"""

import numpy as np
import random
import requests
from collections import Counter
from dataclasses import dataclass

# ── Configuration ───────────────────────────────────────────────────────────────
WORD_FILE = "words.txt"
MAX_GUESSES = 6

# ── API Client ─────────────────────────────────────────────────────────────────

API_BASE = "https://wordle.votee.dev:8000"

# Map API result strings to internal g/y/w codes
_RESULT_MAP = {"correct": "g", "present": "y", "absent": "w"}


@dataclass
class GuessResult:
    guess: str
    result: str  # "g" / "y" / "w"


class WordleAPI:
    """Thin client for the Wordle FastAPI server."""

    def __init__(self, base_url: str = API_BASE, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _result_to_str(self, response_json: list) -> str:
        """Convert API JSON response to a 5-char g/y/w string."""
        return "".join(_RESULT_MAP[item["result"]] for item in response_json)

    def _get(self, path: str, params: dict) -> list:
        try:
            r = requests.get(self.base_url + path, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Could not connect to API server.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 422:
                detail = e.response.json().get("detail", [])
                if isinstance(detail, list) and detail:
                    msgs = [d.get("msg", str(d)) for d in detail]
                    raise ValueError(f"Validation error: {', '.join(msgs)}")
            raise

    def daily(self, guess: str) -> GuessResult:
        resp = self._get("/daily", {"guess": guess.lower().strip()})
        return GuessResult(guess=guess, result=self._result_to_str(resp))

    def random(self, guess: str, size: int = 5) -> GuessResult:
        resp = self._get("/random", {"guess": guess.lower().strip(), "size": size})
        return GuessResult(guess=guess, result=self._result_to_str(resp))

    def specific(self, word: str, guess: str) -> GuessResult:
        resp = self._get(f"/word/{word.strip().lower()}", {"guess": guess.lower().strip()})
        return GuessResult(guess=guess, result=self._result_to_str(resp))


# ── Game State ───────────────────────────────────────────────────────────────

class WordleGame:
    def __init__(self, word_file: str):
        with open(word_file) as f:
            self.dictionary = [w.strip().lower() for w in f
                               if len(w.strip()) == 5 and w.strip().isalpha()]
        if not self.dictionary:
            raise ValueError("No valid 5-letter words in word list.")
        self.target = random.choice(self.dictionary)
        self.guesses = 0
        self.history = []

        self.green  = {}    # {pos: letter}
        self.yellow = {}    # {letter: set of forbidden positions}
        self.gray   = set() # letters eliminated from the answer
        self.yellow_letters = set()   # letters known to be in the answer but wrong position
        self.confirmed_in_answer = set()  # letters confirmed (green or yellow) anywhere

        print(f"Loaded {len(self.dictionary)} words.")

    # ── Core Solver Logic ──────────────────────────────────────────────────

    def _calc_position_frequencies(self, words: list[str]) -> np.ndarray:
        """Frequency of each letter (a-z) at each position (0-4). Shape: (5, 26)."""
        freq = np.zeros((5, 26), dtype=int)
        for word in words:
            for pos, ch in enumerate(word):
                freq[pos, ord(ch) - 97] += 1
        return freq

    def _score_word(self, word: str, freq: np.ndarray) -> float:
        """
        Score = sum over positions of log(1 + freq_at_pos[letter]).

        Using log prevents overflow from multiplicative compounding.
        Score ≈ higher when letters are common at their respective positions.
        A tiny random uniform(0,1) multiplier is used for tiebreaking.

        Formula: score = Σ_pos log(1 + freq[pos][letter])
        """
        total = 0.0
        max_freq = freq.max(axis=1)
        for pos, ch in enumerate(word):
            idx = ord(ch) - 97
            total += np.log1p(freq[pos, idx])
        # Diversity bonus: penalize repeated letters (each duplicate gets -0.5)
        letter_counts = Counter(word)
        dup_penalty = sum(v - 1 for v in letter_counts.values() if v > 1) * 0.5
        return (total - dup_penalty) * random.uniform(0.999, 1.001)

    def _filter_candidates(self, words: list[str]) -> list[str]:
        """Apply elimination, anchor, and position-exclusion rules."""
        filtered = []
        for word in words:
            # Rule 1: Elimination — skip words containing gray-only letters.
            # A letter is gray-only if it is in self.gray AND NOT in
            # self.confirmed_in_answer (i.e., not also green/yellow elsewhere).
            # Words containing a confirmed-in-answer letter are always kept.
            if any(ch in self.gray and ch not in self.confirmed_in_answer for ch in word):
                continue

            # Rule 2: Anchor — must have green letters in exact positions
            if not all(self.green.get(p, word[p]) == word[p] for p in self.green):
                continue

            # Rule 3: Position Exclusion — must NOT have yellow letters
            # in the specific positions where they were guessed
            yellow_violation = False
            for letter, forbidden in self.yellow.items():
                if letter in word:
                    for p in forbidden:
                        if word[p] == letter:
                            yellow_violation = True
                            break
                if yellow_violation:
                    break
            if yellow_violation:
                continue

            # Rule 4: Must contain all known yellow letters
            if not all(ch in word for ch in self.yellow_letters):
                continue

            # Rule 5: Gray-only letters must not appear more times in the candidate
            # than the count allowed by confirmed (green/yellow) occurrences.
            # Since gray-only letters are never confirmed, max_allowed = 0 always.
            for letter in self.gray:
                green_count = sum(1 for p, ch in self.green.items() if ch == letter)
                yellow_count = sum(1 for p in self.yellow.get(letter, set()))
                max_allowed = green_count + yellow_count
                if max_allowed > 0 and word.count(letter) > max_allowed:
                    continue

            filtered.append(word)
        return filtered

    def _parse_feedback(self, feedback_str: str, guess: str) -> dict:
        """Parse a feedback string like 'gwyyg' into categorized letters."""
        result = {'green': {}, 'yellow': {}, 'gray': set()}
        feedback_str = feedback_str.strip().lower()

        if len(feedback_str) != 5:
            raise ValueError(f"Feedback must be exactly 5 characters, got {len(feedback_str)}.")

        for pos, (ch, status) in enumerate(zip(guess, feedback_str)):
            if status == 'g':
                result['green'][pos] = ch
            elif status == 'y':
                result['yellow'].setdefault(ch, set()).add(pos)
            elif status == 'w':
                result['gray'].add(ch)
            else:
                raise ValueError(f"Invalid status '{status}' at position {pos}. Use g/y/w.")

        return result

    def apply_feedback(self, guess: str, feedback_str: str):
        """Update letter knowledge from a guess and its feedback."""
        parsed = self._parse_feedback(feedback_str, guess)

        confirmed_before = set(self.confirmed_in_answer)

        greens_this = set(parsed['green'].values())
        yellows_this = set(parsed['yellow'].keys())
        confirmed_this = greens_this | yellows_this
        truly_gray = set(parsed['gray']) - confirmed_this - confirmed_before
        self.gray.update(truly_gray)

        for pos, letter in parsed['green'].items():
            self.green[pos] = letter
            self.confirmed_in_answer.add(letter)

        for letter, positions in parsed['yellow'].items():
            self.yellow.setdefault(letter, set()).update(positions)
            self.yellow_letters.add(letter)
            self.confirmed_in_answer.add(letter)

    def get_best_guess(self, candidates: list[str]) -> str:
        """Return the best next guess from the candidate pool."""
        if not candidates:
            raise ValueError("No candidate words remain!")
        freq = self._calc_position_frequencies(candidates)
        scored = [(self._score_word(w, freq), w) for w in candidates]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def get_top_guesses(self, candidates: list[str], n: int = 10) -> list[tuple[str, float]]:
        """Return the top N guesses with scores."""
        if not candidates:
            return []
        freq = self._calc_position_frequencies(candidates)
        scored = [(self._score_word(w, freq), w) for w in candidates]
        scored.sort(key=lambda x: -x[0])
        return [(w, round(s, 3)) for s, w in scored[:n]]

    def show_hint(self, candidates: list[str]):
        """Print letter frequency analysis."""
        if not candidates:
            print("No candidates to analyze.")
            return

        freq = self._calc_position_frequencies(candidates)
        total = len(candidates)
        max_freq = freq.max(axis=1)

        print(f"\n{'='*56}")
        print(f"  LETTER FREQUENCY ANALYSIS  ({total} candidates)")
        print(f"{'='*56}")
        print(f"  {'Pos 1':>8}  {'Pos 2':>8}  {'Pos 3':>8}  {'Pos 4':>8}  {'Pos 5':>8}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        for rank in range(1, 6):
            print(f"  #{rank}   ", end="")
            for pos in range(5):
                idx = freq[pos].argmax()
                letter = chr(idx + 97)
                pct = freq[pos, idx] / total * 100
                print(f"{letter} ({pct:5.1f}%)", end="  ")
                freq[pos, idx] = -1
            print()

        print(f"\n  Most informative positions: ", end="")
        entropy = [max_freq[p] / total for p in range(5)]
        best_pos = sorted(range(5), key=lambda p: entropy[p])[:3]
        print([f"#{p+1}" for p in best_pos])
        print(f"{'='*56}\n")

    # ── Gameplay ───────────────────────────────────────────────────────────

    def run(self):
        print()
        print("=" * 54)
        print("  WORDLE AI STRATEGY ENGINE")
        print("=" * 54)
        print("  Rules:")
        print("    Enter your guess word, then the feedback (g/y/w)")
        print("    - g = GREEN  (correct letter, correct position)")
        print("    - y = YELLOW (correct letter, WRONG position)")
        print("    - w = GRAY   (wrong letter / eliminated)")
        print()
        print("  Commands: HINT | TOP N | RESTART | EXIT")
        print("=" * 54)

        candidates = self.dictionary.copy()
        freq = self._calc_position_frequencies(candidates)
        opening_guesses = self.get_top_guesses(candidates, 5)
        print(f"\n  Opening recommendation: {opening_guesses[0][0].upper()}")
        print(f"  Top 5 openings:")
        for i, (word, score) in enumerate(opening_guesses):
            print(f"    {i+1}. {word.upper():8s}  (score: {score:.3f})")

        while self.guesses < MAX_GUESSES:
            print(f"\n[{self.guesses + 1}/{MAX_GUESSES}] Enter guess: ", end="")
            guess = input().strip().lower()

            if guess.upper() == "EXIT":
                print(f"\nThe secret word was: {self.target.upper()}")
                print("Goodbye!")
                break

            if guess.upper() == "RESTART":
                self.__init__(WORD_FILE)
                self.run()
                return

            if guess.upper() == "NEW":
                print(f"  Re-analyzing from scratch...")
                candidates = self.dictionary.copy()
                self.green = {}
                self.yellow = {}
                self.gray = set()
                self.yellow_letters = set()
                self.history = []
                self.guesses = 0
                freq = self._calc_position_frequencies(candidates)
                opening = self.get_best_guess(candidates)
                print(f"  Fresh opening: {opening.upper()}")
                continue

            if guess not in self.dictionary:
                print(f"  Warning: '{guess}' not in dictionary. Proceeding anyway.")

            self.guesses += 1
            print(f"  Your guess: {guess.upper()}")
            print(f"  Feedback (g/y/w): ", end="")
            feedback = input().strip()

            try:
                self.apply_feedback(guess, feedback)
            except ValueError as e:
                print(f"  Error: {e}")
                self.guesses -= 1
                continue

            self.history.append((guess, feedback))

            if all(feedback[i] == 'g' for i in range(5)):
                print(f"\n  *** CORRECT! Solved in {self.guesses} guess(es)! ***")
                print(f"  The word was: {guess.upper()}")
                break

            candidates = self._filter_candidates(candidates)
            if not candidates:
                print("\n  ERROR: No candidate words remain. Check your feedback.")
                break

            print(f"  {len(candidates)} candidates remaining.")

            top = self.get_top_guesses(candidates, 5)
            print(f"  Top 5 recommendations:")
            for i, (word, score) in enumerate(top):
                print(f"    {i+1}. {word.upper():8s}  (score: {score:.3f})")

        if self.guesses >= MAX_GUESSES:
            print(f"\n  Out of guesses! The word was: {self.target.upper()}")


# ── Standalone Solver ─────────────────────────────────────────────────────────

class WordleSolver:
    """
    Pure solver mode — recommend the best next guess given board state.
    Usage:
        solver = WordleSolver("words.txt")
        solver.feedback("crane", "wwygg")
        best = solver.get_best_guess()
    """

    def __init__(self, word_file: str):
        with open(word_file) as f:
            self.dictionary = [w.strip().lower() for w in f
                               if len(w.strip()) == 5 and w.strip().isalpha()]
        self.green  = {}
        self.yellow = {}
        self.gray   = set()
        self.yellow_letters = set()
        self.confirmed_in_answer = set()
        self.history = []

    def feedback(self, guess: str, result: str):
        """Apply guess feedback. result is 5-char string of g/y/w."""
        guess = guess.lower().strip()
        result = result.lower().strip()
        if len(guess) != 5 or len(result) != 5:
            raise ValueError("Guess and result must both be 5 characters.")
        self.history.append((guess, result))

        confirmed_before = set(self.confirmed_in_answer)

        greens_in_this = {}
        yellows_in_this = {}

        for pos, (ch, status) in enumerate(zip(guess, result)):
            if status == 'g':
                greens_in_this[pos] = ch
                self.green[pos] = ch
                self.confirmed_in_answer.add(ch)
            elif status == 'y':
                yellows_in_this.setdefault(ch, set()).add(pos)
                self.yellow.setdefault(ch, set()).add(pos)
                self.yellow_letters.add(ch)
                self.confirmed_in_answer.add(ch)
            elif status == 'w':
                pass  # W status is ambiguous; don't gray-mark yet

        # After processing the full guess, only gray-mark letters that appeared
        # ONLY as W (never as G/Y) in THIS guess AND were not confirmed before.
        confirmed_this = set(greens_in_this.values()) | set(yellows_in_this.keys())
        for pos, (ch, status) in enumerate(zip(guess, result)):
            if status == 'w' and ch not in confirmed_this and ch not in confirmed_before:
                self.gray.add(ch)

    def get_candidates(self):
        filtered = []
        for word in self.dictionary:
            if any(ch in self.gray and ch not in self.confirmed_in_answer for ch in word):
                continue
            if not all(self.green.get(p, word[p]) == word[p] for p in self.green):
                continue
            yellow_violation = False
            for letter, forbidden in self.yellow.items():
                if letter in word:
                    for p in forbidden:
                        if word[p] == letter:
                            yellow_violation = True
                            break
                if yellow_violation:
                    break
            if yellow_violation:
                continue
            if not all(ch in word for ch in self.yellow_letters):
                continue
            for letter in self.gray:
                green_count = sum(1 for p, ch in self.green.items() if ch == letter)
                yellow_count = sum(1 for p in self.yellow.get(letter, set()))
                max_allowed = green_count + yellow_count
                if max_allowed > 0 and word.count(letter) > max_allowed:
                    break
            else:
                filtered.append(word)
        return filtered

    def _calc_frequencies(self, words):
        freq = np.zeros((5, 26), dtype=int)
        for word in words:
            for pos, ch in enumerate(word):
                freq[pos, ord(ch) - 97] += 1
        return freq

    def _score_word(self, word, freq):
        total = 0.0
        for pos, ch in enumerate(word):
            idx = ord(ch) - 97
            total += np.log1p(freq[pos, idx])
        letter_counts = Counter(word)
        dup_penalty = sum(v - 1 for v in letter_counts.values() if v > 1) * 0.5
        return (total - dup_penalty) * random.uniform(0.999, 1.001)

    def get_best_guess(self):
        candidates = self.get_candidates()
        if not candidates:
            return None, 0.0, 0
        freq = self._calc_frequencies(candidates)
        scored = sorted([(self._score_word(w, freq), w) for w in candidates],
                        key=lambda x: -x[0])
        s, w = scored[0]
        return w, round(s, 3), len(candidates)

    def get_top(self, n=10):
        candidates = self.get_candidates()
        if not candidates:
            return []
        freq = self._calc_frequencies(candidates)
        scored = sorted([(self._score_word(w, freq), w) for w in candidates],
                        key=lambda x: -x[0])
        return [(w, round(s, 3)) for s, w in scored[:n]]


# ── API Game (auto-solver against remote server) ──────────────────────────────

class WordleAPIGame:
    """
    Fully automated Wordle solver that plays against the remote API.
    Handles Daily / Random / Specific-Word modes and auto-solves using
    the built-in solver with intelligent timing.
    """

    def __init__(self, word_file: str):
        self.solver = WordleSolver(word_file)
        self.api = WordleAPI()
        self.word_file = word_file
        self.guesses = 0

    def run_daily(self):
        print("\n" + "=" * 54)
        print("  WORDLE API — DAILY CHALLENGE")
        print("=" * 54)
        self._run(mode="daily")

    def run_random(self, size: int = 5):
        print("\n" + "=" * 54)
        print(f"  WORDLE API — RANDOM PUZZLE  (size={size})")
        print("=" * 54)
        self._run(mode="random", size=size)

    def run_specific(self, word: str):
        print("\n" + "=" * 54)
        print(f"  WORDLE API — SPECIFIC WORD: {word.upper()}")
        print("=" * 54)
        self._run(mode="specific", target_word=word)

    def _run(self, mode: str, size: int = 5, target_word: str = None):
        print("\n  Solving automatically...")
        candidates = self.solver.get_candidates()
        opening = self.solver.get_best_guess()[0]
        print(f"  First guess: {opening.upper()}")
        print()

        while self.guesses < MAX_GUESSES:
            self.guesses += 1
            guess = self.solver.get_best_guess()[0]

            try:
                if mode == "daily":
                    result = self.api.daily(guess)
                elif mode == "random":
                    result = self.api.random(guess, size=size)
                else:
                    result = self.api.specific(target_word, guess)

                print(f"  [{self.guesses}/{MAX_GUESSES}] {guess.upper()}  →  {result.result.upper()}")

            except (TimeoutError, ConnectionError, ValueError) as e:
                print(f"  API error: {e}")
                print("  Falling back to local game — you can enter feedback manually.")
                self._run_local_fallback()
                return

            self.solver.feedback(guess, result.result)

            if result.result == "ggggg":
                print(f"\n  *** SOLVED in {self.guesses} guess(es)! ***")
                return

            candidates = self.solver.get_candidates()
            if not candidates:
                print("\n  No candidates remain — something went wrong.")
                return

            print(f"  {len(candidates)} candidates remaining.")

        print(f"\n  Out of guesses! The puzzle was not solved.")

    def _run_local_fallback(self):
        """When the API is unavailable, fall back to a local solver session."""
        print("\n  [FALLBACK MODE — manual feedback]")
        print("  Enter: GUESS RESULT  (e.g. CRANE WWYGG)")
        print("  Type EXIT to quit.\n")
        solver = WordleSolver(self.word_file)
        while True:
            print("> ", end="")
            line = input().strip()
            if line.upper() == "EXIT":
                break
            parts = line.split()
            if len(parts) != 2:
                print("  Usage: CRANE WWYGG")
                continue
            guess, result = parts
            try:
                solver.feedback(guess, result)
            except ValueError as e:
                print(f"  Error: {e}")
                continue
            word, score, n = solver.get_best_guess()
            if word is None:
                print("  No candidates match that feedback.")
            else:
                print(f"  [{n} candidates] Best: {word.upper()}  (score: {score:.3f})")


# ── Entry Point ───────────────────────────────────────────────────────────────

def _print_banner():
    print()
    print("=" * 54)
    print("  WORDLE AI STRATEGY ENGINE")
    print("=" * 54)


def _print_mode_menu():
    print("  MODE SELECTION")
    print("  [1] Local Game     — Play against a random local word")
    print("  [2] Pure Solver   — Enter guesses + feedback manually")
    print("  [3] Daily Puzzle  — Play today's challenge via API")
    print("  [4] Random Puzzle — Play a random puzzle via API")
    print("  [5] Specific Word — Target a chosen word via API")
    print("=" * 54)


if __name__ == "__main__":
    _print_banner()
    _print_mode_menu()
    print("  Choice (1-5): ", end="")
    choice = input().strip()

    if choice == "2":
        _print_banner()
        solver = WordleSolver(WORD_FILE)
        print("\n  Solver ready. Enter: GUESS RESULT")
        print("  Example: CRANE WWYGG  (5-char result: g/y/w)")
        print("  Type EXIT to quit.\n")
        while True:
            print("> ", end="")
            line = input().strip()
            if line.upper() == "EXIT":
                break
            parts = line.split()
            if len(parts) != 2:
                print("  Usage: CRANE WWYGG")
                continue
            guess, result = parts
            try:
                solver.feedback(guess, result)
            except ValueError as e:
                print(f"  Error: {e}")
                continue
            word, score, n = solver.get_best_guess()
            if word is None:
                print("  No candidates match that feedback.")
            else:
                print(f"  [{n} candidates] Best: {word.upper()}  (score: {score:.3f})")

    elif choice == "3":
        api_game = WordleAPIGame(WORD_FILE)
        api_game.run_daily()

    elif choice == "4":
        print("\n  Word size (default 5): ", end="")
        size_input = input().strip()
        size = int(size_input) if size_input.isdigit() else 5
        api_game = WordleAPIGame(WORD_FILE)
        api_game.run_random(size=size)

    elif choice == "5":
        print("\n  Enter target word: ", end="")
        word = input().strip().lower()
        if len(word) < 3:
            print("  Word must be at least 3 letters.")
        else:
            api_game = WordleAPIGame(WORD_FILE)
            api_game.run_specific(word)

    else:
        game = WordleGame(WORD_FILE)
        game.run()
