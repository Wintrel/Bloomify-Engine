import os
import json
import random
import librosa
import numpy as np


# noinspection D
def generate_stream_map(
        audio_path,
        lanes=4,
        stream_subdivision=4,
        min_stream_duration_beats=4.0,
        chord_chance=0.25
):
    """
    Generates a beatmap with playable, patterned streams similar to osu! maps.

    Args:
        audio_path (str): Path to the audio file.
        lanes (int): Number of lanes for the beatmap.
        stream_subdivision (int): The speed of the streams. 4 = 1/4 notes (standard),
                                  6 = 1/6 notes (triplets), 8 = 1/8 notes (very fast).
        min_stream_duration_beats (float): How many beats of sustained energy are
                                           required to start a stream section.
        chord_chance (float): Chance to add a chord on a strong downbeat.
    """
    print("Loading audio for stream map generation...")
    y, sr = librosa.load(audio_path)
    title = os.path.splitext(os.path.basename(audio_path))[0]

    # --- 1. Foundational Rhythm Analysis ---
    print("Analyzing BPM and beat grid...")
    bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)  # Start guess for stability
    if isinstance(bpm, np.ndarray):
        bpm = np.mean(bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"Detected BPM: {bpm:.2f}")

    # --- 2. Identify Potential Stream Sections ---
    # We find sections with sustained high energy to place streams in.
    print("Identifying high-energy sections for streams...")
    rms = librosa.feature.rms(y=y)[0]
    energy_threshold = np.percentile(rms, 60)  # Top 40% of energy is considered "high"
    is_beat_high_energy = []
    for t in beat_times:
        frame_index = librosa.time_to_frames(t, sr=sr)
        # Check energy at the beat time
        if frame_index < len(rms):
            is_beat_high_energy.append(rms[frame_index] > energy_threshold)
        else:
            is_beat_high_energy.append(False)

    stream_sections = []
    current_stream_start = -1
    for i, is_high in enumerate(is_beat_high_energy):
        if is_high and current_stream_start == -1:
            current_stream_start = i  # Start of a potential stream
        elif not is_high and current_stream_start != -1:
            # End of a stream. Check if it was long enough.
            if i - current_stream_start >= min_stream_duration_beats:
                stream_sections.append((current_stream_start, i))
            current_stream_start = -1
    # Check for a stream that goes to the end of the song
    if current_stream_start != -1 and len(beat_times) - current_stream_start >= min_stream_duration_beats:
        stream_sections.append((current_stream_start, len(beat_times)))

    print(f"Found {len(stream_sections)} potential stream sections.")

    # --- 3. Quantized Note Generation ---
    # Generate a clean, quantized list of timestamps for our notes.
    note_timestamps = []
    beat_flags = []  # To mark which notes are on the main beat

    # Get onsets to place notes in non-stream sections
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    onset_idx = 0

    current_beat = 0
    while current_beat < len(beat_times) - 1:
        in_stream = any(start <= current_beat < end for start, end in stream_sections)

        if in_stream:
            # Create a perfect subdivision stream
            start_time = beat_times[current_beat]
            end_time = beat_times[current_beat + 1]
            for i in range(stream_subdivision):
                note_timestamps.append(start_time + i * (end_time - start_time) / stream_subdivision)
                beat_flags.append(i == 0)  # Mark the downbeat
            current_beat += 1
        else:
            # Place a single note based on the strongest onset in the beat interval
            start_time = beat_times[current_beat]
            end_time = beat_times[current_beat + 1]

            # Find onsets within this beat
            onsets_in_beat = [t for t in onset_times if start_time <= t < end_time]
            if onsets_in_beat:
                note_timestamps.append(onsets_in_beat[0])  # Add the first onset
            else:
                note_timestamps.append(start_time)  # Default to the downbeat if no onsets
            beat_flags.append(True)
            current_beat += 1

    # --- 4. Pattern-Based Note Placement ---
    print("Placing notes using stream patterns...")
    notes = []

    # Define some playable stream patterns
    patterns = {
        "zig_zag": [0, 2, 1, 3, 2, 0, 3, 1],
        "staircase": [0, 1, 2, 3, 2, 1, 0, 1],
        "in_out": [0, 3, 1, 2, 1, 3, 0, 2]
    }
    current_pattern = random.choice(list(patterns.values()))
    pattern_index = 0

    for i, t_sec in enumerate(note_timestamps):
        time_ms = int(t_sec * 1000)
        is_downbeat = beat_flags[i]

        # Place the main note using the pattern
        lane1 = current_pattern[pattern_index % len(current_pattern)]
        notes.append({"time": time_ms, "lane": lane1})
        pattern_index += 1

        # Add a chord only on strong downbeats
        if is_downbeat and random.random() < chord_chance:
            possible_lanes = [l for l in range(lanes) if l != lane1]
            if possible_lanes:
                lane2 = random.choice(possible_lanes)
                notes.append({"time": time_ms, "lane": lane2})

    print(f"Generated a playable stream map with {len(notes)} notes.")

    # Assemble beatmap
    beatmap = {
        "title": title, "artist": "Unknown Artist", "mapper": "StreamGenerator",
        "audio_path": os.path.basename(audio_path), "image_path": "art.png",
        "bpm": round(bpm, 2), "notes": notes
    }
    return beatmap


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    audio_files = [f for f in os.listdir(folder) if f.lower().endswith((".mp3", ".wav", ".ogg"))]

    if not audio_files:
        print("No audio file found in this folder.")
    else:
        audio_file = audio_files[0]
        audio_path = os.path.join(folder, audio_file)
        print(f"Found audio file: {audio_file}")

        # --- CONFIGURE YOUR STREAM MAP ---
        # For a song like "Lionheart", you'd want fast streams (1/4 or 1/6)
        # and a fairly long stream duration threshold.
        beatmap = generate_stream_map(
            audio_path=audio_path,
            lanes=4,
            stream_subdivision=6,  # 4 is great for most songs. Try 6 for triplets.
            min_stream_duration_beats=8.0,  # Requires 8 solid beats of energy to start a stream.
            chord_chance=0.20  # Lower chance to keep streams clean.
        )

        out_name = os.path.splitext(audio_file)[0] + "_stream_autogen.json"
        out_path = os.path.join(folder, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(beatmap, f, indent=4)

        print(f"\nSuccessfully generated stream map: {out_name}")