#!/usr/bin/env python3
"""
Calma AI — Phoneme Learning Session Runner

Usage:
    python calma.py                    # Start/continue a session
    python calma.py status             # Show learner progress
    python calma.py reset              # Reset learner state
    python calma.py generate [N]       # Generate N exercises (default 5) and print as JSON
    python calma.py exercise           # Generate and run a single exercise interactively

Requires ANTHROPIC_API_KEY environment variable (or .env file).
"""

import sys
import os
import json
import time
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip("'\"")
                if _key not in os.environ:
                    os.environ[_key] = _val

from exercise_generator import (
    load_curriculum,
    generate_exercise,
    generate_batch,
    pick_target_sounds,
    get_stage,
    get_all_sounds_up_to,
)

LEARNER_STATE_PATH = "./learner_state.json"


def default_learner_state() -> dict:
    return {
        "current_stage": 1,
        "mastered_sounds": [],
        "accuracy": {},
        "attempts": {},
        "error_streaks": {},
        "exercises_completed": 0,
        "sessions": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_session": None,
    }


def load_learner(path: str = LEARNER_STATE_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default_learner_state()


def save_learner(learner: dict, path: str = LEARNER_STATE_PATH):
    with open(path, "w") as f:
        json.dump(learner, f, indent=2)


def update_accuracy(learner: dict, phoneme: str, correct: bool):
    """Update running accuracy for a phoneme using exponential moving average."""
    alpha = 0.3  # weight for new observation
    current = learner["accuracy"].get(phoneme, 0.5)
    new_val = 1.0 if correct else 0.0
    learner["accuracy"][phoneme] = current * (1 - alpha) + new_val * alpha

    # Track attempts
    learner["attempts"][phoneme] = learner["attempts"].get(phoneme, 0) + 1


def update_error_streak(learner: dict, pair_key: str, correct: bool):
    """Track consecutive errors for confusion pairs."""
    if correct:
        learner["error_streaks"][pair_key] = 0
    else:
        learner["error_streaks"][pair_key] = learner["error_streaks"].get(pair_key, 0) + 1


def check_mastery(learner: dict, curriculum: dict) -> list[str]:
    """Check if any sounds should be marked as mastered. Returns newly mastered sounds."""
    threshold = curriculum["progression"]["discrimination_pass"]
    min_attempts = 5
    newly_mastered = []

    stage = get_stage(curriculum, learner["current_stage"])
    if not stage or "sounds" not in stage:
        return []

    for sound in stage["sounds"]:
        phoneme = sound["phoneme"]
        if phoneme in learner["mastered_sounds"]:
            continue
        acc = learner["accuracy"].get(phoneme, 0.0)
        attempts = learner["attempts"].get(phoneme, 0)
        if acc >= threshold and attempts >= min_attempts:
            learner["mastered_sounds"].append(phoneme)
            newly_mastered.append(phoneme)

    return newly_mastered


def check_stage_advancement(learner: dict, curriculum: dict) -> bool:
    """Check if learner should advance to the next stage. Returns True if advanced."""
    stage = get_stage(curriculum, learner["current_stage"])
    if not stage:
        return False

    if "sounds" not in stage:
        return False

    stage_phonemes = [s["phoneme"] for s in stage["sounds"]]
    all_mastered = all(p in learner["mastered_sounds"] for p in stage_phonemes)

    if all_mastered:
        next_stage = learner["current_stage"] + 1
        max_stage = max(s["id"] for s in curriculum["stages"])
        if next_stage <= max_stage:
            learner["current_stage"] = next_stage
            return True

    return False


# --- Interactive exercise runner ---

def run_listen_identify(exercise: dict) -> tuple[bool, list[str]]:
    content = exercise["content"]
    print(f"\n  {exercise['prompt']}")
    choices = content["choices"]
    for i, c in enumerate(choices):
        print(f"    {i + 1}) \"{c['sound']}\"")

    answer = input("\n  Your answer (number): ").strip()
    try:
        idx = int(answer) - 1
        correct = choices[idx].get("correct", False)
        touched_phonemes = [content["target"]]
    except (ValueError, IndexError):
        correct = False
        touched_phonemes = [content["target"]]

    return correct, touched_phonemes


def run_same_different(exercise: dict) -> tuple[bool, list[str]]:
    content = exercise["content"]
    trials = content["trials"]
    print(f"\n  {exercise['prompt']}")

    correct_count = 0
    touched = set()
    for i, trial in enumerate(trials):
        print(f"\n    Trial {i + 1}: \"{trial['sound_a']}\" and \"{trial['sound_b']}\"")
        answer = input("    Same or Different? (s/d): ").strip().lower()
        expected = trial["answer"].lower()
        is_correct = (answer.startswith("s") and expected == "same") or \
                     (answer.startswith("d") and expected == "different")
        if is_correct:
            correct_count += 1
            print("    Correct!")
        else:
            print(f"    The answer was: {trial['answer']}")
        touched.add(trial.get("phoneme_a", ""))
        touched.add(trial.get("phoneme_b", ""))

    total_correct = correct_count >= len(trials) * 0.75
    touched.discard("")
    return total_correct, list(touched)


def run_phoneme_label(exercise: dict) -> tuple[bool, list[str]]:
    content = exercise["content"]
    print(f"\n  {exercise['prompt']}")
    print(f"  Sound played: \"{content['sound_played']}\"")
    choices = content["choices"]
    for i, c in enumerate(choices):
        print(f"    {i + 1}) {c['label']}")

    answer = input("\n  Your answer (number): ").strip()
    try:
        idx = int(answer) - 1
        correct = choices[idx].get("correct", False)
        touched = [c["label"] for c in choices if c.get("correct")]
    except (ValueError, IndexError):
        correct = False
        touched = [c["label"] for c in choices if c.get("correct")]

    return correct, touched


def run_letter_match(exercise: dict) -> tuple[bool, list[str]]:
    content = exercise["content"]
    print(f"\n  {exercise['prompt']}")
    print(f"  Sound: \"{content['sound_played']}\" ({content['phoneme']})")
    choices = content["choices"]
    for i, c in enumerate(choices):
        print(f"    {i + 1}) {c['letter']}")

    answer = input("\n  Your answer (number): ").strip()
    try:
        idx = int(answer) - 1
        correct = choices[idx].get("correct", False)
    except (ValueError, IndexError):
        correct = False

    return correct, [content["phoneme"]]


def run_category_sort(exercise: dict) -> tuple[bool, list[str]]:
    content = exercise["content"]
    categories = content["categories"]
    items = content["items"]
    print(f"\n  {exercise['prompt']}")
    cat_labels = " | ".join(f"{i + 1}) {c}" for i, c in enumerate(categories))
    print(f"  Categories: {cat_labels}")

    correct_count = 0
    touched = set()
    for item in items:
        print(f"\n    Sound: \"{item['sound']}\" ({item['phoneme']})")
        answer = input(f"    Which category? (1-{len(categories)}): ").strip()
        try:
            idx = int(answer) - 1
            is_correct = categories[idx] == item["category"]
        except (ValueError, IndexError):
            is_correct = False

        if is_correct:
            correct_count += 1
            print("    Correct!")
        else:
            print(f"    It's: {item['category']}")
        touched.add(item["phoneme"])

    return correct_count >= len(items) * 0.75, list(touched)


def run_odd_one_out(exercise: dict) -> tuple[bool, list[str]]:
    content = exercise["content"]
    trials = content["trials"]
    print(f"\n  {exercise['prompt']}")

    correct_count = 0
    touched = set()
    for i, trial in enumerate(trials):
        sounds = trial["sounds"]
        print(f"\n    Trial {i + 1}: {' '.join(sounds)}")
        answer = input(f"    Which is different? (1-{len(sounds)}): ").strip()
        try:
            idx = int(answer) - 1
            is_correct = sounds[idx] == trial["odd"]
        except (ValueError, IndexError):
            is_correct = False

        if is_correct:
            correct_count += 1
            print("    Correct!")
        else:
            print(f"    The odd one was: {trial['odd']} ({trial.get('reason', '')})")
        for s in sounds:
            touched.add(s)

    return correct_count >= len(trials) * 0.6, list(touched)


def run_cvc_blend(exercise: dict) -> tuple[bool, list[str]]:
    content = exercise["content"]
    words = content["words"]
    print(f"\n  {exercise['prompt']}")

    correct_count = 0
    touched = set()
    for w in words:
        phonemes_str = " — ".join(w["phonemes"])
        print(f"\n    Sounds: {phonemes_str}")
        if "hint" in w:
            print(f"    Hint: {w['hint']}")
        answer = input("    What word? ").strip().lower()
        if answer == w["word"].lower():
            correct_count += 1
            print("    Correct!")
        else:
            print(f"    The word is: {w['word']}")
        for p in w["phonemes"]:
            touched.add(p)

    return correct_count >= len(words) * 0.6, list(touched)


def run_generic(exercise: dict) -> tuple[bool, list[str]]:
    """Fallback for exercise types without dedicated runners."""
    print(f"\n  {exercise['prompt']}")
    print(f"\n  Exercise content:")
    print(json.dumps(exercise["content"], indent=4))
    answer = input("\n  Did you complete this correctly? (y/n): ").strip().lower()
    return answer.startswith("y"), []


EXERCISE_RUNNERS = {
    "listen_identify": run_listen_identify,
    "same_different": run_same_different,
    "phoneme_label": run_phoneme_label,
    "letter_match": run_letter_match,
    "category_sort": run_category_sort,
    "odd_one_out": run_odd_one_out,
    "cvc_blend": run_cvc_blend,
    "segmenting": run_cvc_blend,  # same interaction pattern
    "cluster_blend": run_cvc_blend,
}


def run_exercise(exercise_data: dict, learner: dict, curriculum: dict):
    """Run a single exercise interactively and update learner state."""
    exercise = exercise_data["exercise"]
    target = exercise_data["target"]

    ex_type = exercise["exercise_type"]
    print(f"\n{'=' * 50}")
    print(f"  {exercise.get('title', ex_type)}")
    print(f"  Type: {ex_type}")
    if target.get("review"):
        print("  (Spaced review)")
    print(f"{'=' * 50}")

    runner = EXERCISE_RUNNERS.get(ex_type, run_generic)
    correct, touched_phonemes = runner(exercise)

    # Show teaching moment
    if correct:
        print(f"\n  {exercise.get('teaching_moment_correct', 'Well done!')}")
    else:
        fallback = "Let's try again next time."
        print(f"\n  {exercise.get('teaching_moment_incorrect', fallback)}")

    # Update learner state
    for phoneme in touched_phonemes:
        update_accuracy(learner, phoneme, correct)

    # Update error streaks for confusion pairs
    if len(touched_phonemes) == 2:
        pair_key = "|".join(sorted(touched_phonemes))
        update_error_streak(learner, pair_key, correct)

    learner["exercises_completed"] += 1

    # Check mastery
    newly_mastered = check_mastery(learner, curriculum)
    if newly_mastered:
        print(f"\n  *** Sound{'s' if len(newly_mastered) > 1 else ''} mastered: {', '.join(newly_mastered)} ***")

    # Check stage advancement
    if check_stage_advancement(learner, curriculum):
        stage = get_stage(curriculum, learner["current_stage"])
        print(f"\n  *** STAGE UP! Welcome to Stage {learner['current_stage']}: {stage['name']} ***")

    return correct


def show_status(learner: dict, curriculum: dict):
    """Display learner progress."""
    print(f"\n{'=' * 50}")
    print("  Calma AI — Learner Progress")
    print(f"{'=' * 50}")

    stage = get_stage(curriculum, learner["current_stage"])
    stage_name = stage["name"] if stage else "Unknown"
    print(f"\n  Current Stage: {learner['current_stage']} — {stage_name}")
    print(f"  Exercises Completed: {learner['exercises_completed']}")
    print(f"  Sounds Mastered: {len(learner['mastered_sounds'])}")

    if learner["mastered_sounds"]:
        print(f"    {', '.join(learner['mastered_sounds'])}")

    if learner["accuracy"]:
        print(f"\n  Accuracy by Sound:")
        # Sort by accuracy (lowest first — these need work)
        sorted_acc = sorted(learner["accuracy"].items(), key=lambda x: x[1])
        for phoneme, acc in sorted_acc:
            attempts = learner["attempts"].get(phoneme, 0)
            mastered = "✓" if phoneme in learner["mastered_sounds"] else " "
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"    [{mastered}] {phoneme:6s} {bar} {acc:.0%} ({attempts} attempts)")

    # Show current stage sounds and progress
    if stage and "sounds" in stage:
        print(f"\n  Stage {learner['current_stage']} Sounds:")
        for sound in stage["sounds"]:
            p = sound["phoneme"]
            status = "mastered" if p in learner["mastered_sounds"] else "learning"
            acc = learner["accuracy"].get(p, 0.0)
            print(f"    {p:6s} {sound['letter']:3s} — {status} ({acc:.0%})")

    # Show error streaks if any
    active_streaks = {k: v for k, v in learner.get("error_streaks", {}).items() if v >= 2}
    if active_streaks:
        print(f"\n  Needs Extra Practice:")
        for pair, count in sorted(active_streaks.items(), key=lambda x: -x[1]):
            print(f"    {pair}: {count} consecutive errors")

    print()


def run_session(learner: dict, curriculum: dict, exercise_count: int = 5):
    """Run an interactive learning session."""
    print(f"\n{'=' * 50}")
    print("  Welcome to Calma AI")
    print(f"{'=' * 50}")

    stage = get_stage(curriculum, learner["current_stage"])
    print(f"\n  Stage {learner['current_stage']}: {stage['name']}" if stage else "")
    print(f"  Sounds mastered: {len(learner['mastered_sounds'])}")
    print(f"  Today's session: {exercise_count} exercises\n")

    session_correct = 0
    session_total = 0

    for i in range(exercise_count):
        print(f"\n  --- Exercise {i + 1}/{exercise_count} ---")

        # Show what we're targeting (for transparency)
        target = pick_target_sounds(learner, curriculum)
        print(f"  Focus: {', '.join(target['target_phonemes'][:3])}")
        print(f"  ({target['reason']})")

        try:
            result = generate_exercise(curriculum, learner)
        except Exception as e:
            print(f"\n  Error generating exercise: {e}")
            continue

        if not result["exercise"]:
            print("  Could not generate exercise — skipping")
            continue

        correct = run_exercise(result, learner, curriculum)
        session_total += 1
        if correct:
            session_correct += 1

        save_learner(learner)

    # Session summary
    print(f"\n{'=' * 50}")
    print(f"  Session Complete!")
    print(f"  Score: {session_correct}/{session_total}")
    if session_total > 0:
        pct = session_correct / session_total
        if pct >= 0.8:
            print("  Excellent work!")
        elif pct >= 0.6:
            print("  Good progress — keep practising!")
        else:
            print("  Every attempt helps you learn. You'll get there!")
    print(f"{'=' * 50}\n")

    learner["last_session"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    learner["sessions"].append({
        "date": learner["last_session"],
        "exercises": session_total,
        "correct": session_correct,
    })
    save_learner(learner)


def main():
    curriculum = load_curriculum()
    learner = load_learner()

    if len(sys.argv) < 2:
        run_session(learner, curriculum)
        return

    command = sys.argv[1]

    if command == "status":
        show_status(learner, curriculum)

    elif command == "reset":
        confirm = input("Reset all progress? (yes/no): ").strip().lower()
        if confirm == "yes":
            save_learner(default_learner_state())
            print("Learner state reset.")
        else:
            print("Cancelled.")

    elif command == "generate":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(f"Generating {count} exercises...")
        results = generate_batch(curriculum, learner, count)
        for i, r in enumerate(results):
            print(f"\n--- Exercise {i + 1} ({r['target']['exercise_type']}) ---")
            print(f"Target: {', '.join(r['target']['target_phonemes'])}")
            print(f"Reason: {r['target']['reason']}")
            print(json.dumps(r["exercise"], indent=2))

    elif command == "exercise":
        print("Generating exercise...")
        result = generate_exercise(curriculum, learner)
        if result["exercise"]:
            run_exercise(result, learner, curriculum)
            save_learner(learner)
        else:
            print("Could not generate exercise.")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
