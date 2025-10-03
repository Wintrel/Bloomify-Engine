import os
import json
import random
import librosa  # The new library for audio analysis
import numpy as np

def generate_smart_beatmap(audio_path, difficulty=1, lanes=4):
    """
    Generates a beatmap by analyzing the audio file for musical onsets.

    Args:
        audio_path (str): Path to the audio file (.mp3, .wav, etc.).
        difficulty (float): A value from 0.1 (easiest) to 1.0 (hardest).
                          This controls how many of the detected sound onsets
                          are turned into notes.
        lanes (int): The number of lanes for the beatmap.
    """
    print("Loading audio file...")
    # 1. Load the audio file using librosa.
    # 'y' is the audio time series, 'sr' is the sampling rate.
    y, sr = librosa.load(audio_path)

    # --- AUTOMATIC METADATA DETECTION ---

    print("Analyzing song structure...")
    # 2. Automatically get song info.
    title = os.path.splitext(os.path.basename(audio_path))[0]
    song_length_ms = librosa.get_duration(y=y, sr=sr) * 1000

    # 3. Automatically detect BPM and the timing of beats.
    bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(bpm, np.ndarray):
        bpm = np.mean(bpm)

    beat_times_ms = librosa.frames_to_time(beat_frames, sr=sr) * 1000
    offset_ms = beat_times_ms[0] if len(beat_times_ms) > 0 else 0

    print(f"Detected BPM: {bpm:.2f}")

    # --- SMART NOTE GENERATION ---

    print("Detecting musical onsets for note placement...")
    # 4. Detect sound onsets (the "smart" part).
    # These are the timestamps where distinct musical events begin.
    # We use 'backtrack=True' to place the onset at the start of the sound.
    onset_times_sec = librosa.onset.onset_detect(
        y=y, sr=sr, units='time', backtrack=True
    )

    # 5. Filter onsets based on their musical intensity and the difficulty setting.
    if len(onset_times_sec) > 0:
        # Calculate the energy (RMS) of the audio at each onset frame.
        onset_frames = librosa.time_to_frames(onset_times_sec, sr=sr)
        onset_strengths = librosa.feature.rms(y=y)[0][onset_frames]

        # Determine the minimum strength an onset needs to become a note.
        # A higher difficulty means a lower threshold, so more notes are created.
        strength_threshold = np.percentile(onset_strengths, (1 - difficulty) * 100)

        # Keep only the onsets that are strong enough.
        strong_onset_times_sec = [
            t for t, s in zip(onset_times_sec, onset_strengths) if s >= strength_threshold
        ]
    else:
        strong_onset_times_sec = []

    print(f"Total onsets detected: {len(onset_times_sec)}")
    print(f"Notes to be placed (after difficulty filtering): {len(strong_onset_times_sec)}")

    # 6. Generate note objects with a smarter placement pattern.
    notes = []
    last_lane = -1  # Initialize with an invalid lane

    for t_sec in strong_onset_times_sec:
        time_ms = int(t_sec * 1000)

        # Simple smart pattern: Avoid placing a note in the same lane as the last one.
        # This prevents awkward, rapid repetitions on a single key.
        possible_lanes = list(range(lanes))
        if last_lane in possible_lanes and len(possible_lanes) > 1:
            possible_lanes.remove(last_lane)

        chosen_lane = random.choice(possible_lanes)

        notes.append({"time": time_ms, "lane": chosen_lane})
        last_lane = chosen_lane

    # --- FINAL BEATMAP ASSEMBLY ---

    beatmap = {
        "title": title,
        "artist": "Unknown Artist",
        "mapper": "SmartGenerator",
        "audio_path": os.path.basename(audio_path),
        "image_path": "art.png",
        "bpm": round(bpm, 2),
        "offset_ms": round(offset_ms, 2),
        "song_length_ms": int(song_length_ms),
        "notes": notes
    }
    return beatmap


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    # Search for common audio formats, not just mp3
    audio_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".mp3", ".wav", ".ogg", ".flac"))
    ]

    if not audio_files:
        print("No audio file found. Drop a song in this folder and re-run.")
    else:
        audio_file = audio_files[0]
        audio_path = os.path.join(folder, audio_file)
        print(f"Found audio file: {audio_file}")

        # --- CONFIGURE YOUR BEATMAP ---
        beatmap = generate_smart_beatmap(
            audio_path=audio_path,
            difficulty=1,  # Try values from 0.1 (easy) to 1.0 (hard)
            lanes=4
        )

        out_name = os.path.splitext(audio_file)[0] + "_smart_autogen.json"
        out_path = os.path.join(folder, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(beatmap, f, indent=4)

        print(f"\nSuccessfully generated smart beatmap: {out_name}")
