#!/usr/bin/env python3
"""
Prepare the Speech Accent Archive dataset for YourTTS fine-tuning.

Steps:
  1. Read speakers_all.csv → filter rows with audio files present
  2. Convert MP3 → WAV (16kHz mono) using torchaudio
  3. Write metadata.csv: wav_filename|transcript|speaker_name
  4. Print stats

The output metadata uses the LJSpeech-style pipe-delimited format:
  filename_no_ext|transcript|transcript

Usage:
    conda activate ml_env
    cd /home/nibiru/Documents/sem6project/Speech2/pipeline_v2
    python prepare_archive.py
"""

import os
import csv
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = pathlib.Path(__file__).resolve().parent
ARCHIVE_DIR = BASE / "data" / "raw" / "archive"
RECORDINGS_DIR = ARCHIVE_DIR / "recordings" / "recordings"
WAV_DIR = ARCHIVE_DIR / "wav_16k"
CSV_FILE = ARCHIVE_DIR / "speakers_all.csv"
PASSAGE_FILE = ARCHIVE_DIR / "reading-passage.txt"
METADATA_FILE = ARCHIVE_DIR / "metadata.csv"

# Read the standard passage
TRANSCRIPT = open(PASSAGE_FILE).read().strip()


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", "16000", "-ac", "1",
        "-sample_fmt", "s16",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0


def main():
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    # Parse speakers_all.csv
    print("Reading speakers_all.csv...")
    speakers = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing files
            missing = row.get("file_missing?", "").strip().upper()
            if missing == "TRUE":
                continue
            speakers.append(row)

    print(f"  {len(speakers)} speakers with audio files")

    # Build filename → speaker mapping
    # CSV 'filename' column contains just the language prefix (e.g. "english")
    # Actual MP3 filename is: {filename}{speakerid}.mp3
    # But some entries use a different pattern. Let's check what MP3s actually exist.
    existing_mp3s = {f.stem: f for f in RECORDINGS_DIR.glob("*.mp3")}
    print(f"  {len(existing_mp3s)} MP3 files on disk")

    # Match speakers to their MP3 files
    matched = []
    for spk in speakers:
        filename_base = spk.get("filename", "").strip()
        speaker_id = spk.get("speakerid", "").strip()
        native_lang = spk.get("native_language", "").strip().lower().replace(" ", "_")

        # Try: {filename}{speakerid}
        key = f"{filename_base}{speaker_id}"
        if key in existing_mp3s:
            matched.append({
                "mp3_stem": key,
                "mp3_path": existing_mp3s[key],
                "speaker_name": f"ACCENT_{native_lang}_{speaker_id}",
                "native_language": native_lang,
                "speaker_id": speaker_id,
            })
        else:
            # Try just the filename field as-is
            if filename_base in existing_mp3s:
                matched.append({
                    "mp3_stem": filename_base,
                    "mp3_path": existing_mp3s[filename_base],
                    "speaker_name": f"ACCENT_{native_lang}_{speaker_id}",
                    "native_language": native_lang,
                    "speaker_id": speaker_id,
                })

    print(f"  {len(matched)} speakers matched to MP3 files")

    # Deduplicate by mp3_stem (some speakers may map to same file)
    seen = set()
    unique_matched = []
    for m in matched:
        if m["mp3_stem"] not in seen:
            seen.add(m["mp3_stem"])
            unique_matched.append(m)
    matched = unique_matched
    print(f"  {len(matched)} unique audio files to convert")

    # Convert MP3 → WAV (parallel with ffmpeg)
    print(f"\nConverting MP3 → WAV (16kHz mono) into {WAV_DIR}...")
    converted = 0
    skipped = 0
    failed = 0

    def convert_one(item):
        wav_path = WAV_DIR / f"{item['mp3_stem']}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            return "skipped", item
        success = convert_mp3_to_wav(item["mp3_path"], wav_path)
        return ("ok" if success else "fail"), item

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(convert_one, m): m for m in matched}
        for i, future in enumerate(as_completed(futures)):
            status, item = future.result()
            if status == "ok":
                converted += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                if failed <= 5:
                    print(f"  ⚠ Failed: {item['mp3_stem']}")
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}/{len(matched)} "
                      f"(converted={converted}, skipped={skipped}, failed={failed})")

    print(f"\n  Converted: {converted}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Failed: {failed}")

    # Write metadata.csv
    # Format: wav_filename_no_ext|transcript|speaker_name
    # This will be read by our custom formatter
    print(f"\nWriting metadata to {METADATA_FILE}...")
    valid_entries = []
    for item in matched:
        wav_path = WAV_DIR / f"{item['mp3_stem']}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            valid_entries.append(item)

    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["wav_stem", "transcript", "speaker_name"])
        for item in valid_entries:
            writer.writerow([item["mp3_stem"], TRANSCRIPT, item["speaker_name"]])

    # Stats
    unique_speakers = set(item["speaker_name"] for item in valid_entries)
    unique_langs = set(item["native_language"] for item in valid_entries)
    print(f"\n{'='*60}")
    print(f"Archive Dataset Prepared")
    print(f"{'='*60}")
    print(f"  Audio files:      {len(valid_entries)}")
    print(f"  Unique speakers:  {len(unique_speakers)}")
    print(f"  Native languages: {len(unique_langs)}")
    print(f"  WAV directory:    {WAV_DIR}")
    print(f"  Metadata file:    {METADATA_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
