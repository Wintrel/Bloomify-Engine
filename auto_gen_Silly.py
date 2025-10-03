import os
import json
import random
import librosa
import numpy as np


# noinspection D
def generate_configurable_beatmap(audio_path, config):
    """
    Generates a beatmap by analyzing the audio file for musical onsets,
    with configurable settings for difficulty and note placement mode.

    Args:
        audio_path (str): Path to the audio file (.mp3, .wav, etc.).
        config (dict): A dictionary containing beatmap generation settings.
                       Keys include:
                       - difficulty (float): from 0.0 to 1.0. Lower is easier.
                       - lanes (int): The number of lanes for the beatmap.
                       - mode (str): 'chaotic' for all onsets, 'smart' to filter.
    """
    difficulty = config.get('difficulty', 1.0)
    lanes = config.get('lanes', 4)
    mode = config.get('mode', 'chaotic')

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

    # --- NOTE GENERATION BASED ON CONFIGURATION ---

    print(f"Detecting musical onsets with '{mode}' mode...")
    # 4. Detect all sound onsets. We use 'backtrack=True' to place the onset
    #    at the start of the sound.
    onset_times_sec = librosa.onset.onset_detect(
        y=y, sr=sr, units='time', backtrack=True
    )

    # 5. Apply filtering based on the 'mode' setting.
    if mode == 'smart' and len(onset_times_sec) > 0:
        # Calculate the energy (RMS) of the audio at each onset frame.
        onset_frames = librosa.time_to_frames(onset_times_sec, sr=sr)
        onset_strengths = librosa.feature.rms(y=y)[0][onset_frames]

        # Determine the minimum strength an onset needs to become a note.
        strength_threshold = np.percentile(onset_strengths, (1 - difficulty) * 100)

        # Keep only the onsets that are strong enough.
        notes_to_place = [
            t for t, s in zip(onset_times_sec, onset_strengths) if s >= strength_threshold
        ]
    else:  # 'chaotic' mode
        notes_to_place = onset_times_sec

    print(f"Total onsets detected: {len(onset_times_sec)}")
    print(f"Notes to be placed: {len(notes_to_place)}")

    # 6. Generate note objects with a smart placement pattern.
    notes = []
    last_lane = -1  # Initialize with an invalid lane

    for t_sec in notes_to_place:
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
        "artist": "BLOOMIFY Engine",
        "mapper": f"ConfigurableGenerator ({mode})",
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
    # Search for common audio formats
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

        # --- CONFIGURE YOUR BEATMAP SETTINGS HERE ---
        # Change these values to customize the output beatmap.
        CONFIG = {
            # 'chaotic' mode places a note on every detected onset.
            # 'smart' mode filters onsets based on the 'difficulty' setting.
            "mode": "chaotic",

            # A value from 0.0 (easiest, fewest notes) to 1.0 (hardest, most notes).
            "difficulty": 1.0,

            # The number of lanes for your beatmap.
            "lanes": 4,
        }
        print("\nUsing the following configuration:")
        print(json.dumps(CONFIG, indent=4))
        print("-" * 30)

        beatmap = generate_configurable_beatmap(
            audio_path=audio_path,
            config=CONFIG
        )

        out_name = os.path.splitext(audio_file)[0] + "_configurable_autogen.json"
        out_path = os.path.join(folder, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(beatmap, f, indent=4)

        print(f"\nSuccessfully generated configurable beatmap: {out_name}")
