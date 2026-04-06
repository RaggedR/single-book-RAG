"""
Calma AI Exercise Generator

Generates phoneme exercises on the fly using Claude API,
based on curriculum data and learner state.
"""

import json
import random
import anthropic


EXERCISE_SCHEMA = {
    "type": "object",
    "properties": {
        "exercise_type": {"type": "string"},
        "title": {"type": "string"},
        "prompt": {"type": "string"},
        "content": {
            "type": "object",
            "description": "Exercise-specific content (trials, choices, words, etc.)"
        },
        "teaching_moment_correct": {"type": "string"},
        "teaching_moment_incorrect": {"type": "string"},
        "pass_criteria": {"type": "string"}
    },
    "required": ["exercise_type", "title", "prompt", "content"]
}


def load_curriculum(path: str = "curriculum.json") -> dict:
    with open(path) as f:
        return json.load(f)


def get_stage(curriculum: dict, stage_id: int) -> dict | None:
    for stage in curriculum["stages"]:
        if stage["id"] == stage_id:
            return stage
    return None


def get_all_sounds_up_to(curriculum: dict, stage_id: int) -> list[dict]:
    """Get all sounds from stage 1 up to and including the given stage."""
    sounds = []
    for stage in curriculum["stages"]:
        if stage["id"] <= stage_id and "sounds" in stage:
            sounds.extend(stage["sounds"])
    return sounds


def pick_target_sounds(learner: dict, curriculum: dict) -> dict:
    """Decide what to practice based on learner state.

    Returns a dict with:
      - target_phonemes: list of phonemes to focus on
      - exercise_type: suggested exercise type
      - reason: why this was chosen (for debugging / logging)
      - review: whether this is spaced review of earlier material
    """
    stage_id = learner.get("current_stage", 1)
    stage = get_stage(curriculum, stage_id)
    if not stage:
        return {"target_phonemes": [], "exercise_type": None, "reason": "invalid stage"}

    accuracy = learner.get("accuracy", {})
    mastered = learner.get("mastered_sounds", [])
    review_ratio = curriculum["progression"]["spaced_review_ratio"]
    remediation_trigger = curriculum["progression"]["remediation_trigger"]

    # Check for remediation: any confusion pair with consecutive errors
    streaks = learner.get("error_streaks", {})
    for pair_key, count in streaks.items():
        if count >= remediation_trigger:
            phonemes = pair_key.split("|")
            return {
                "target_phonemes": phonemes,
                "exercise_type": "same_different",
                "reason": f"remediation: {count} consecutive errors on {pair_key}",
                "review": False
            }

    # Spaced review of earlier stages
    if random.random() < review_ratio and stage_id > 1:
        review_sounds = get_all_sounds_up_to(curriculum, stage_id - 1)
        if review_sounds:
            sample = random.sample(review_sounds, min(3, len(review_sounds)))
            return {
                "target_phonemes": [s["phoneme"] for s in sample],
                "exercise_type": random.choice(["listen_identify", "phoneme_label", "letter_match"]),
                "reason": "spaced review of earlier stages",
                "review": True
            }

    # Normal progression through current stage
    if "sounds" not in stage:
        # Stage 6 (blending) — no new sounds
        return {
            "target_phonemes": [s["phoneme"] for s in get_all_sounds_up_to(curriculum, 5)],
            "exercise_type": random.choice(stage.get("exercise_sequence", ["cvc_blend"])),
            "reason": "blending stage",
            "review": False
        }

    # Find sounds in current stage not yet mastered
    stage_phonemes = [s["phoneme"] for s in stage["sounds"]]
    unmastered = [p for p in stage_phonemes if p not in mastered]

    if not unmastered:
        # All sounds in this stage mastered — use exercise sequence for review
        exercise_seq = stage.get("exercise_sequence", [])
        session_idx = learner.get("exercises_completed", 0)
        ex_type = exercise_seq[session_idx % len(exercise_seq)] if exercise_seq else "sound_bingo"
        return {
            "target_phonemes": stage_phonemes,
            "exercise_type": ex_type,
            "reason": "stage review — all sounds mastered, cycling exercises",
            "review": False
        }

    # Focus on first unmastered sound(s)
    focus = unmastered[:2]

    # Pick exercise type based on accuracy
    focus_accuracy = [accuracy.get(p, 0.0) for p in focus]
    avg_acc = sum(focus_accuracy) / len(focus_accuracy) if focus_accuracy else 0.0

    if avg_acc < 0.3:
        ex_type = "listen_identify"
    elif avg_acc < 0.5:
        ex_type = "same_different"
    elif avg_acc < 0.7:
        ex_type = "phoneme_label"
    else:
        ex_type = "letter_match"

    # Use confusion pairs if available
    confusion = stage.get("confusion_pairs", [])
    for pair in confusion:
        if any(p in focus for p in pair):
            other = [p for p in pair if p not in focus]
            focus.extend(other)
            break

    return {
        "target_phonemes": list(set(focus)),
        "exercise_type": ex_type,
        "reason": f"focus on unmastered sounds, avg accuracy {avg_acc:.0%}",
        "review": False
    }


def build_exercise_prompt(
    exercise_type: str,
    target_phonemes: list[str],
    stage: dict,
    curriculum: dict,
    learner: dict
) -> str:
    """Build the prompt for Claude to generate an exercise."""

    # Gather sound details for context
    all_sounds = get_all_sounds_up_to(curriculum, learner.get("current_stage", 1))
    sound_details = []
    for s in all_sounds:
        if s["phoneme"] in target_phonemes:
            sound_details.append(
                f"  {s['phoneme']} — letter: {s['letter']}, "
                f"example: \"{s['example']}\", "
                f"articulation: {s['articulation']}"
            )

    # Find the exercise type definition
    ex_def = None
    for et in curriculum["exercise_types"]:
        if et["id"] == exercise_type:
            ex_def = et
            break

    ex_description = ex_def["description"] if ex_def else exercise_type
    ex_name = ex_def["name"] if ex_def else exercise_type

    # Available distractors (sounds the learner knows but aren't the target)
    mastered = learner.get("mastered_sounds", [])
    all_phonemes = [s["phoneme"] for s in all_sounds]
    distractors = [p for p in all_phonemes if p not in target_phonemes]

    # Blending stage gets a different prompt
    if exercise_type in ("cvc_blend", "segmenting", "cluster_blend", "sentence_read"):
        return _build_blending_prompt(exercise_type, curriculum, learner)

    return f"""Generate a phoneme exercise for a literacy learning app called Calma AI.

EXERCISE TYPE: {ex_name}
DESCRIPTION: {ex_description}

TARGET SOUNDS:
{chr(10).join(sound_details)}

AVAILABLE DISTRACTOR SOUNDS: {', '.join(distractors[:8])}

LEARNER CONTEXT:
- Current stage: {learner.get('current_stage', 1)} ({stage['name'] if stage else 'unknown'})
- Sounds already mastered: {', '.join(mastered) if mastered else 'none yet'}

REQUIREMENTS:
- Generate exactly ONE exercise as a JSON object
- The exercise must focus on the target sounds listed above
- Use simple, encouraging, age-appropriate language
- Include a "teaching_moment_correct" (what to say when they get it right)
- Include a "teaching_moment_incorrect" (gentle guidance when wrong)
- Include "pass_criteria" describing what counts as passing

EXERCISE-SPECIFIC FORMAT:
{_exercise_format_instructions(exercise_type)}

Return ONLY valid JSON, no markdown fences, no explanation."""


def _build_blending_prompt(exercise_type: str, curriculum: dict, learner: dict) -> str:
    """Build prompt for blending/reading exercises."""
    stage6 = get_stage(curriculum, 6)
    mastered = learner.get("mastered_sounds", [])

    if exercise_type == "cvc_blend":
        words = stage6.get("cvc_words", [])
        # Filter to words using only mastered sounds
        usable = [w for w in words if all(p in mastered for p in w["phonemes"])]
        sample = random.sample(usable, min(6, len(usable))) if usable else words[:6]
        word_list = json.dumps(sample, indent=2)
        return f"""Generate a CVC blending exercise for Calma AI.

The learner has mastered these sounds: {', '.join(mastered)}

Here are CVC words they can decode (use 4-6 of these):
{word_list}

Generate a JSON exercise where the learner hears individual phonemes and blends them into a word.

FORMAT:
{{
  "exercise_type": "cvc_blend",
  "title": "Sound-by-Sound Blending",
  "prompt": "encouraging instruction to the learner",
  "content": {{
    "words": [
      {{"word": "mat", "phonemes": ["/m/", "/æ/", "/t/"], "hint": "something you stand on"}}
    ]
  }},
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "..."
}}

Return ONLY valid JSON."""

    if exercise_type == "segmenting":
        words = stage6.get("cvc_words", [])
        usable = [w for w in words if all(p in mastered for p in w["phonemes"])]
        sample = random.sample(usable, min(4, len(usable))) if usable else words[:4]
        word_list = json.dumps(sample, indent=2)
        return f"""Generate a segmenting exercise for Calma AI (reverse of blending — word → individual sounds).

Learner's mastered sounds: {', '.join(mastered)}

Usable words:
{word_list}

FORMAT:
{{
  "exercise_type": "segmenting",
  "title": "Break It Apart",
  "prompt": "encouraging instruction",
  "content": {{
    "words": [
      {{"word": "map", "phonemes": ["/m/", "/æ/", "/p/"]}}
    ]
  }},
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "..."
}}

Return ONLY valid JSON."""

    if exercise_type == "cluster_blend":
        words = stage6.get("cluster_words", [])
        usable = [w for w in words if all(p in mastered for p in w["phonemes"])]
        sample = random.sample(usable, min(4, len(usable))) if usable else words[:4]
        word_list = json.dumps(sample, indent=2)
        return f"""Generate a consonant cluster blending exercise for Calma AI (CCVC/CVCC words).

Learner's mastered sounds: {', '.join(mastered)}

Usable cluster words:
{word_list}

FORMAT:
{{
  "exercise_type": "cluster_blend",
  "title": "Tricky Blends",
  "prompt": "encouraging instruction",
  "content": {{
    "words": [
      {{"word": "stop", "phonemes": ["/s/", "/t/", "/ɒ/", "/p/"], "hint": "what a red light means"}}
    ]
  }},
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "..."
}}

Return ONLY valid JSON."""

    if exercise_type == "sentence_read":
        sentences = stage6.get("decodable_sentences", [])
        sample = random.sample(sentences, min(3, len(sentences)))
        return f"""Generate a decodable sentence reading exercise for Calma AI.

Learner's mastered sounds: {', '.join(mastered)}

Use these decodable sentences (or generate similar ones using ONLY the mastered sounds):
{json.dumps(sample, indent=2)}

FORMAT:
{{
  "exercise_type": "sentence_read",
  "title": "Read the Sentence",
  "prompt": "encouraging instruction",
  "content": {{
    "sentences": [
      {{"text": "The fat cat sat on a mat.", "focus_words": ["fat", "cat", "sat", "mat"]}}
    ]
  }},
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "..."
}}

Return ONLY valid JSON."""

    return ""


def _exercise_format_instructions(exercise_type: str) -> str:
    """Return JSON format instructions specific to each exercise type."""
    formats = {
        "listen_identify": """{
  "exercise_type": "listen_identify",
  "title": "short descriptive title",
  "prompt": "instruction to the learner",
  "content": {
    "target": "/m/",
    "target_example": "mmm",
    "choices": [
      {"sound": "mmm", "phoneme": "/m/", "correct": true},
      {"sound": "sss", "phoneme": "/s/", "correct": false},
      {"sound": "fff", "phoneme": "/f/", "correct": false}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "1/1 correct"
}""",
        "same_different": """{
  "exercise_type": "same_different",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "trials": [
      {"sound_a": "fff", "phoneme_a": "/f/", "sound_b": "fff", "phoneme_b": "/f/", "answer": "same"},
      {"sound_a": "sss", "phoneme_a": "/s/", "sound_b": "fff", "phoneme_b": "/f/", "answer": "different"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "3/4 correct"
}""",
        "listen_tap": """{
  "exercise_type": "listen_tap",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "target": "/s/",
    "sequence": [
      {"sound": "mmm", "phoneme": "/m/", "should_tap": false},
      {"sound": "sss", "phoneme": "/s/", "should_tap": true}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "identify all targets with ≤1 false positive"
}""",
        "odd_one_out": """{
  "exercise_type": "odd_one_out",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "trials": [
      {"sounds": ["/p/", "/p/", "/b/"], "odd": "/b/", "reason": "voiced vs voiceless"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "2/3 correct"
}""",
        "voicing_pair": """{
  "exercise_type": "voicing_pair",
  "title": "short title",
  "prompt": "instruction with tactile cue (hand on throat)",
  "content": {
    "pair": ["/p/", "/b/"],
    "voiced": "/b/",
    "voiceless": "/p/",
    "tactile_cue": "description of what to feel",
    "trials": [
      {"sound": "puh", "phoneme": "/p/", "answer": "/p/"},
      {"sound": "buh", "phoneme": "/b/", "answer": "/b/"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "5/6 correct"
}""",
        "category_sort": """{
  "exercise_type": "category_sort",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "categories": ["Continuous", "Stop"],
    "items": [
      {"sound": "mmm", "phoneme": "/m/", "category": "Continuous"},
      {"sound": "puh", "phoneme": "/p/", "category": "Stop"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "5/6 correct"
}""",
        "phoneme_label": """{
  "exercise_type": "phoneme_label",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "sound_played": "sss",
    "choices": [
      {"label": "/m/", "correct": false},
      {"label": "/s/", "correct": true},
      {"label": "/f/", "correct": false}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "1/1 correct"
}""",
        "letter_match": """{
  "exercise_type": "letter_match",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "sound_played": "sss",
    "phoneme": "/s/",
    "choices": [
      {"letter": "m", "correct": false},
      {"letter": "s", "correct": true},
      {"letter": "f", "correct": false}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "1/1 correct"
}""",
        "rapid_match": """{
  "exercise_type": "rapid_match",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "available_letters": ["p", "b", "t", "d", "k", "g"],
    "trials": [
      {"sound": "puh", "phoneme": "/p/", "correct_letter": "p"},
      {"sound": "buh", "phoneme": "/b/", "correct_letter": "b"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "10/12 correct"
}""",
        "sound_bingo": """{
  "exercise_type": "sound_bingo",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "grid": ["/m/", "/n/", "/s/", "/f/", "/h/"],
    "play_order": ["/s/", "/m/", "/h/", "/f/", "/n/"]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "4/5 correct"
}""",
        "say_record": """{
  "exercise_type": "say_record",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "sounds_to_produce": [
      {"phoneme": "/m/", "example": "mmm", "duration_seconds": 3, "articulation_tip": "press your lips together"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "attempt all sounds"
}""",
        "minimal_pairs": """{
  "exercise_type": "minimal_pairs",
  "title": "short title",
  "prompt": "instruction",
  "content": {
    "trials": [
      {"word_a": "cat", "word_b": "cut", "answer": "different", "difference": "/æ/ vs /ʌ/"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "3/4 correct"
}""",
        "magic_e": """{
  "exercise_type": "magic_e",
  "title": "short title",
  "prompt": "instruction about the magic-e rule",
  "content": {
    "pairs": [
      {"short_word": "cap", "long_word": "cape", "short_vowel": "/æ/", "long_vowel": "/eɪ/"}
    ]
  },
  "teaching_moment_correct": "...",
  "teaching_moment_incorrect": "...",
  "pass_criteria": "identify the vowel change in 3/4 pairs"
}"""
    }
    return formats.get(exercise_type, '{"exercise_type": "' + exercise_type + '", ...}')


def generate_exercise(
    curriculum: dict,
    learner: dict,
    api_key: str | None = None
) -> dict:
    """Generate a single exercise using Claude API.

    Returns a dict with:
      - exercise: the generated exercise JSON
      - target: the targeting info (what sounds, why)
    """
    import os
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Determine what to practice
    target = pick_target_sounds(learner, curriculum)
    if not target["target_phonemes"]:
        return {"exercise": None, "target": target}

    stage = get_stage(curriculum, learner.get("current_stage", 1))

    # Build prompt
    prompt = build_exercise_prompt(
        target["exercise_type"],
        target["target_phonemes"],
        stage,
        curriculum,
        learner
    )

    # Call Claude
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    exercise = json.loads(raw)

    return {
        "exercise": exercise,
        "target": target
    }


def generate_batch(
    curriculum: dict,
    learner: dict,
    count: int = 5,
    api_key: str | None = None
) -> list[dict]:
    """Generate multiple exercises for a session."""
    exercises = []
    for _ in range(count):
        result = generate_exercise(curriculum, learner, api_key)
        if result["exercise"]:
            exercises.append(result)
    return exercises
