#!/usr/bin/env python3
"""
Compute 512-dim d-vectors for all training audio using YourTTS's own speaker encoder.

This MUST use the same speaker encoder (model_se.pth + config_se.json) that
YourTTS was trained with — the d-vector dimensions and representation space
must match exactly.

The output JSON uses keys in the format: {dataset_name}#{relative_path_without_ext}
which matches Coqui TTS's `add_extra_keys()` naming convention used during training.

Datasets processed:
  - LibriTTS dev-clean (~40 speakers, ~5,700 clips)
  - LibriTTS test-clean (~39 speakers, ~4,800 clips)
  - Speech Accent Archive (~2,133 speakers, 1 clip each)
  Total: ~2,212 speakers, ~12,700 embeddings

Usage:
    conda activate ml_env
    cd /home/nibiru/Documents/sem6project/Speech2/pipeline_v2
    python prepare_archive.py   # Run first to convert MP3→WAV + build metadata
    python compute_dvectors.py
"""

import os
import csv
import json
import pathlib
from glob import glob

# Paths
BASE = pathlib.Path(__file__).resolve().parent
YOURTTS_DIR = pathlib.Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--your_tts"
SE_MODEL = str(YOURTTS_DIR / "model_se.pth")
SE_CONFIG = str(YOURTTS_DIR / "config_se.json")

# Dataset roots — these MUST match the BaseDatasetConfig(path=...) in finetune_yourtts.py
LIBRITTS_DC_ROOT = BASE / "data" / "raw" / "libritts_clean100" / "LibriTTS" / "dev-clean"
LIBRITTS_TC_ROOT = BASE / "data" / "raw" / "libritts_clean100" / "LibriTTS" / "test-clean"

# Speech Accent Archive (converted WAVs + metadata)
ARCHIVE_WAV_DIR = BASE / "data" / "raw" / "archive" / "wav_16k"
ARCHIVE_METADATA = BASE / "data" / "raw" / "archive" / "metadata.csv"

# Output — use .json extension (Coqui's load_file() dispatches by extension;
# .pth triggers torch.load() which fails on JSON content)
OUTPUT_DIR = BASE / "data" / "dvectors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DVECTORS_FILE = OUTPUT_DIR / "speakers.json"


def main():
    import torch
    from TTS.tts.utils.speakers import SpeakerManager

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # Initialize speaker encoder
    encoder = SpeakerManager(
        encoder_model_path=SE_MODEL,
        encoder_config_path=SE_CONFIG,
        use_cuda=use_cuda,
    )

    speaker_mapping = {}

    # --- LibriTTS dev-clean + test-clean (~79 speakers) ---
    for split_name, split_root, dataset_name in [
        ("dev-clean", LIBRITTS_DC_ROOT, "libritts_devclean"),
        ("test-clean", LIBRITTS_TC_ROOT, "libritts_testclean"),
    ]:
        if not split_root.exists():
            print(f"⚠ LibriTTS {split_name} not found at {split_root}")
            continue

        print(f"\nProcessing LibriTTS {split_name} from {split_root}...")
        wav_files = sorted(glob(str(split_root / "**" / "*.wav"), recursive=True))
        print(f"  Found {len(wav_files)} WAV files")

        for i, wav_path in enumerate(wav_files):
            try:
                # Compute embedding
                embedd = encoder.compute_embedding_from_clip(wav_path)

                # Convert to list for JSON serialization
                if hasattr(embedd, 'tolist'):
                    embedd = embedd.tolist()
                elif hasattr(embedd, 'numpy'):
                    embedd = embedd.numpy().tolist()
                # Flatten if nested
                if isinstance(embedd, list) and len(embedd) == 1 and isinstance(embedd[0], list):
                    embedd = embedd[0]

                # Key format: {dataset_name}#{relative_path_without_extension}
                # This MUST match what add_extra_keys() produces in Coqui TTS
                rel_path = os.path.relpath(wav_path, str(split_root))
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                key = f"{dataset_name}#{rel_path_no_ext}"

                # Extract speaker name from path: .../speaker_id/chapter_id/file.wav
                parts = pathlib.Path(wav_path).parts
                speaker_name = f"LTTS_{parts[-3]}"

                speaker_mapping[key] = {
                    "name": speaker_name,
                    "embedding": embedd,
                }

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i+1}/{len(wav_files)} files...")
            except Exception as e:
                if i < 5:
                    print(f"  ⚠ Failed on {wav_path}: {e}")
                continue

        count = sum(1 for k in speaker_mapping if k.startswith(dataset_name))
        print(f"  ✓ LibriTTS {split_name}: {count} embeddings")

    # --- Speech Accent Archive (~2,133 speakers, 1 clip each) ---
    if ARCHIVE_WAV_DIR.exists() and ARCHIVE_METADATA.exists():
        print(f"\nProcessing Speech Accent Archive from {ARCHIVE_WAV_DIR}...")
        dataset_name = "accent_archive"

        # Read metadata to get speaker names
        stem_to_speaker = {}
        with open(ARCHIVE_METADATA, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                stem_to_speaker[row["wav_stem"]] = row["speaker_name"]

        wav_files = sorted(glob(str(ARCHIVE_WAV_DIR / "*.wav")))
        print(f"  Found {len(wav_files)} WAV files, {len(stem_to_speaker)} metadata entries")

        archive_ok = 0
        archive_fail = 0
        for i, wav_path in enumerate(wav_files):
            try:
                embedd = encoder.compute_embedding_from_clip(wav_path)

                if hasattr(embedd, 'tolist'):
                    embedd = embedd.tolist()
                elif hasattr(embedd, 'numpy'):
                    embedd = embedd.numpy().tolist()
                if isinstance(embedd, list) and len(embedd) == 1 and isinstance(embedd[0], list):
                    embedd = embedd[0]

                # Key: accent_archive#{wav_stem}
                stem = pathlib.Path(wav_path).stem
                key = f"{dataset_name}#{stem}"

                speaker_name = stem_to_speaker.get(stem, f"ACCENT_unknown_{stem}")

                speaker_mapping[key] = {
                    "name": speaker_name,
                    "embedding": embedd,
                }
                archive_ok += 1

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i+1}/{len(wav_files)} files...")
            except Exception as e:
                archive_fail += 1
                if archive_fail <= 5:
                    print(f"  ⚠ Failed on {wav_path}: {e}")
                continue

        print(f"  ✓ Accent Archive: {archive_ok} embeddings ({archive_fail} failed)")
    else:
        print(f"\n⚠ Accent Archive not found. Run prepare_archive.py first.")
        print(f"  Expected WAV dir: {ARCHIVE_WAV_DIR}")
        print(f"  Expected metadata: {ARCHIVE_METADATA}")

    # Summary
    if speaker_mapping:
        # Count unique speakers
        unique_speakers = set(v["name"] for v in speaker_mapping.values())
        print(f"\n{'='*60}")
        print(f"Total embeddings: {len(speaker_mapping)}")
        print(f"Unique speakers:  {len(unique_speakers)}")
        print(f"{'='*60}")

        for spk in sorted(unique_speakers)[:10]:
            count = sum(1 for v in speaker_mapping.values() if v["name"] == spk)
            print(f"  {spk}: {count} clips")
        if len(unique_speakers) > 10:
            print(f"  ... and {len(unique_speakers) - 10} more")

        # Verify embedding dimensions
        sample_key = next(iter(speaker_mapping))
        emb_dim = len(speaker_mapping[sample_key]["embedding"])
        print(f"\nEmbedding dimension: {emb_dim}")
        assert emb_dim == 512, f"Expected 512-dim d-vectors, got {emb_dim}!"

        # Save as JSON (Coqui TTS format)
        print(f"\nSaving to {DVECTORS_FILE}...")
        with open(str(DVECTORS_FILE), 'w') as f:
            json.dump(speaker_mapping, f)

        file_size = os.path.getsize(str(DVECTORS_FILE)) / (1024 * 1024)
        print(f"  File size: {file_size:.1f} MB")
        print("\n✓ D-vectors computed successfully!")
        return str(DVECTORS_FILE)
    else:
        print("⚠ No embeddings computed!")
        return None


if __name__ == "__main__":
    main()
