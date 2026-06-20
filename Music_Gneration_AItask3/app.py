"""
Music Generation with AI - Backend Server
Requirements: pip install flask flask-cors music21 numpy torch pretty_midi
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import json
import numpy as np
import random
import io
import base64
import time

app = Flask(__name__)
CORS(app)

# ─── Directory Setup ───────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Music Theory Helpers ──────────────────────────────────────────────────────
SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues":      [0, 3, 5, 6, 7, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
}

GENRE_PROFILES = {
    "classical": {
        "tempo_range": (60, 120),
        "scale": "major",
        "octave_range": (3, 6),
        "note_duration": [0.25, 0.5, 1.0],
        "chord_prob": 0.3,
        "description": "Baroque-inspired structured composition"
    },
    "jazz": {
        "tempo_range": (80, 160),
        "scale": "dorian",
        "octave_range": (3, 5),
        "note_duration": [0.25, 0.375, 0.5, 0.75],
        "chord_prob": 0.5,
        "description": "Syncopated jazz with blue notes"
    },
    "electronic": {
        "tempo_range": (120, 160),
        "scale": "minor",
        "octave_range": (2, 5),
        "note_duration": [0.125, 0.25, 0.5],
        "chord_prob": 0.2,
        "description": "Rhythmic electronic sequences"
    },
    "blues": {
        "tempo_range": (60, 100),
        "scale": "blues",
        "octave_range": (3, 5),
        "note_duration": [0.25, 0.5, 0.75, 1.0],
        "chord_prob": 0.4,
        "description": "Soulful blues progressions"
    },
    "ambient": {
        "tempo_range": (40, 70),
        "scale": "pentatonic",
        "octave_range": (4, 6),
        "note_duration": [1.0, 2.0, 3.0],
        "chord_prob": 0.6,
        "description": "Slow, atmospheric soundscapes"
    },
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_num):
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"


def generate_melody_sequence(genre, length, root_note=60):
    """Generate a melody using a simple Markov-chain-like approach."""
    profile = GENRE_PROFILES.get(genre, GENRE_PROFILES["classical"])
    scale_intervals = SCALES[profile["scale"]]
    tempo = random.randint(*profile["tempo_range"])
    octave_min, octave_max = profile["octave_range"]

    # Build note pool from scale
    note_pool = []
    for octave in range(octave_min, octave_max + 1):
        for interval in scale_intervals:
            midi = (octave + 1) * 12 + interval
            if 21 <= midi <= 108:
                note_pool.append(midi)

    # Markov-style generation: prefer small steps
    sequence = []
    current = random.choice(note_pool)
    for _ in range(length):
        duration = random.choice(profile["note_duration"])
        velocity = random.randint(60, 100)
        sequence.append({
            "pitch": int(current),
            "note_name": midi_to_note_name(int(current)),
            "duration": duration,
            "velocity": velocity,
        })

        # Move by scale-step mostly, occasional jump
        if random.random() < 0.7:
            step = random.choice([-2, -1, 1, 2])
            idx = note_pool.index(current) if current in note_pool else 0
            new_idx = max(0, min(len(note_pool) - 1, idx + step))
            current = note_pool[new_idx]
        else:
            current = random.choice(note_pool)

    return sequence, tempo


def sequence_to_midi_bytes(sequence, tempo=120):
    """Convert note sequence to MIDI bytes using only built-in libraries."""
    # We build a minimal valid MIDI file from scratch (Type 0, single track)
    ticks_per_beat = 480

    def write_vlq(value):
        """Encode variable-length quantity."""
        result = []
        result.append(value & 0x7F)
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))

    def write_uint32(value):
        return value.to_bytes(4, 'big')

    def write_uint16(value):
        return value.to_bytes(2, 'big')

    # Tempo event (microseconds per beat)
    us_per_beat = int(60_000_000 / tempo)
    tempo_event = (
        write_vlq(0) +          # delta time 0
        bytes([0xFF, 0x51, 0x03]) +
        us_per_beat.to_bytes(3, 'big')
    )

    track_events = bytearray(tempo_event)

    current_tick = 0
    for note in sequence:
        dur_ticks = int(note["duration"] * ticks_per_beat)
        pitch = note["pitch"]
        vel = note["velocity"]

        # Note On (delta 0 from last event)
        track_events += write_vlq(0)
        track_events += bytes([0x90, pitch, vel])

        # Note Off (after duration)
        track_events += write_vlq(dur_ticks)
        track_events += bytes([0x80, pitch, 0])

    # End of track
    track_events += bytes([0x00, 0xFF, 0x2F, 0x00])

    # Track chunk
    track_chunk = b'MTrk' + write_uint32(len(track_events)) + bytes(track_events)

    # Header chunk
    header = b'MThd' + write_uint32(6) + write_uint16(0) + write_uint16(1) + write_uint16(ticks_per_beat)

    return header + track_chunk


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/genres", methods=["GET"])
def get_genres():
    genres = [
        {"id": k, "name": k.capitalize(), "description": v["description"]}
        for k, v in GENRE_PROFILES.items()
    ]
    return jsonify({"genres": genres})


@app.route("/api/generate", methods=["POST"])
def generate_music():
    data = request.get_json(force=True)
    genre = data.get("genre", "classical")
    length = int(data.get("length", 32))
    root_note = int(data.get("root_note", 60))

    if genre not in GENRE_PROFILES:
        return jsonify({"error": f"Unknown genre: {genre}"}), 400
    if not (8 <= length <= 128):
        return jsonify({"error": "Length must be between 8 and 128 notes"}), 400

    sequence, tempo = generate_melody_sequence(genre, length, root_note)

    # Build timeline for display
    timeline = []
    t = 0.0
    seconds_per_beat = 60.0 / tempo
    for note in sequence:
        timeline.append({
            **note,
            "start_time": round(t, 3),
            "end_time": round(t + note["duration"] * seconds_per_beat, 3),
        })
        t += note["duration"] * seconds_per_beat

    total_duration = round(t, 2)

    # Generate MIDI bytes and base64-encode for download
    midi_bytes = sequence_to_midi_bytes(sequence, tempo)
    midi_b64 = base64.b64encode(midi_bytes).decode("utf-8")

    return jsonify({
        "success": True,
        "genre": genre,
        "tempo": tempo,
        "total_notes": len(sequence),
        "total_duration_seconds": total_duration,
        "sequence": timeline,
        "midi_base64": midi_b64,
        "profile": GENRE_PROFILES[genre]["description"],
    })


@app.route("/api/preprocess", methods=["POST"])
def preprocess_info():
    """Return preprocessing info / stats for demo purposes."""
    data = request.get_json(force=True)
    genre = data.get("genre", "classical")
    profile = GENRE_PROFILES.get(genre, GENRE_PROFILES["classical"])
    scale_name = profile["scale"]
    scale_notes = [NOTE_NAMES[i] for i in SCALES[scale_name]]

    return jsonify({
        "success": True,
        "genre": genre,
        "scale": scale_name,
        "scale_notes": scale_notes,
        "tempo_range": profile["tempo_range"],
        "octave_range": profile["octave_range"],
        "preprocessing_steps": [
            "Load MIDI files from dataset",
            "Extract note sequences (pitch, duration, velocity)",
            f"Quantize to {scale_name} scale",
            "Normalize velocities to [0, 1]",
            "One-hot encode pitch classes",
            "Build training windows of length 32",
        ],
        "simulated_dataset_size": random.randint(2000, 5000),
        "vocabulary_size": len(SCALES[scale_name]) * 5,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "LSTM",
        "architecture": {
            "input_size": 128,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "output_size": 128,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "loss": "CrossEntropyLoss",
        },
        "alternative": "GAN (Generator + Discriminator) for adversarial training",
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "server": "Music AI Backend"})


# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎵  Music Generation AI Server starting on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)