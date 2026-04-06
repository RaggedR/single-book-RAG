#!/usr/bin/env python3
"""
Calma AI — Web server for dynamic phoneme exercises.

Serves the curriculum page and provides API endpoints
for AI-generated exercises.

Usage:
    python server.py          # Start on port 5001
    python server.py 8080     # Start on port 8080
"""

import sys
import os
import json
import time
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val

from flask import Flask, jsonify, request, send_from_directory
from exercise_generator import (
    load_curriculum,
    generate_exercise,
    pick_target_sounds,
    get_stage,
)

app = Flask(__name__, static_folder=".", static_url_path="")

LEARNER_STATE_PATH = "./learner_state.json"
curriculum = load_curriculum()


def load_learner() -> dict:
    if os.path.exists(LEARNER_STATE_PATH):
        with open(LEARNER_STATE_PATH) as f:
            return json.load(f)
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


def save_learner(learner: dict):
    with open(LEARNER_STATE_PATH, "w") as f:
        json.dump(learner, f, indent=2)


@app.route("/")
def index():
    return send_from_directory(".", "calma-curriculum.html")


@app.route("/api/curriculum")
def get_curriculum():
    return jsonify(curriculum)


@app.route("/api/status")
def get_status():
    learner = load_learner()
    stage = get_stage(curriculum, learner["current_stage"])
    return jsonify({
        "learner": learner,
        "stage_name": stage["name"] if stage else "Unknown",
    })


@app.route("/api/exercise", methods=["POST"])
def api_generate_exercise():
    learner = load_learner()
    try:
        result = generate_exercise(curriculum, learner)
        if result["exercise"]:
            return jsonify(result)
        return jsonify({"error": "Could not generate exercise"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/respond", methods=["POST"])
def api_respond():
    """Record a learner's response to an exercise."""
    learner = load_learner()
    data = request.json

    phonemes = data.get("phonemes", [])
    correct = data.get("correct", False)
    alpha = 0.3

    for phoneme in phonemes:
        current = learner["accuracy"].get(phoneme, 0.5)
        new_val = 1.0 if correct else 0.0
        learner["accuracy"][phoneme] = current * (1 - alpha) + new_val * alpha
        learner["attempts"][phoneme] = learner["attempts"].get(phoneme, 0) + 1

    # Error streaks for pairs
    if len(phonemes) == 2:
        pair_key = "|".join(sorted(phonemes))
        if correct:
            learner["error_streaks"][pair_key] = 0
        else:
            learner["error_streaks"][pair_key] = learner["error_streaks"].get(pair_key, 0) + 1

    learner["exercises_completed"] += 1

    # Check mastery
    threshold = curriculum["progression"]["discrimination_pass"]
    newly_mastered = []
    stage = get_stage(curriculum, learner["current_stage"])
    if stage and "sounds" in stage:
        for sound in stage["sounds"]:
            p = sound["phoneme"]
            if p in learner["mastered_sounds"]:
                continue
            acc = learner["accuracy"].get(p, 0.0)
            attempts = learner["attempts"].get(p, 0)
            if acc >= threshold and attempts >= 5:
                learner["mastered_sounds"].append(p)
                newly_mastered.append(p)

    # Check stage advancement
    advanced = False
    if stage and "sounds" in stage:
        stage_phonemes = [s["phoneme"] for s in stage["sounds"]]
        if all(p in learner["mastered_sounds"] for p in stage_phonemes):
            next_stage = learner["current_stage"] + 1
            max_stage = max(s["id"] for s in curriculum["stages"])
            if next_stage <= max_stage:
                learner["current_stage"] = next_stage
                advanced = True

    save_learner(learner)

    return jsonify({
        "newly_mastered": newly_mastered,
        "advanced": advanced,
        "current_stage": learner["current_stage"],
        "accuracy": learner["accuracy"],
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    learner = {
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
    save_learner(learner)
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    print(f"Calma AI server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
