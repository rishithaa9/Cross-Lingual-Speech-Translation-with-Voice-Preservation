#!/usr/bin/env python3
"""
Fine-tune YourTTS on LibriTTS + Speech Accent Archive (~2,212 speakers) for
improved zero-shot voice cloning.

The pretrained YourTTS was only trained on 6 speakers (common_voice PT & EN).
Fine-tuning on 2,212 diverse speakers (79 LibriTTS + 2,133 Accent Archive)
dramatically improves the model's ability to generalize to unseen speaker
d-vectors at inference time.

Datasets:
  - LibriTTS dev-clean:     40 speakers, ~5,700 clips, diverse English text
  - LibriTTS test-clean:    39 speakers, ~4,800 clips, diverse English text
  - Speech Accent Archive: 2,133 speakers, 1 clip each, same English passage
                           214 native-language accents, 200 languages

Strategy:
  - Restore from ORIGINAL pretrained YourTTS (clean slate, no overfitting)
  - Use precomputed 512-dim d-vectors from YourTTS's own speaker encoder
  - Lower learning rate (5e-5) for stable fine-tuning
  - Mixed precision (fp16) to fit in 8GB VRAM
  - Speaker encoder as loss (maintains discriminability)

Usage:
    conda activate ml_env
    cd /home/nibiru/Documents/sem6project/Speech2/pipeline_v2
    # Step 1: Prepare archive dataset (MP3→WAV + metadata)
    python prepare_archive.py
    # Step 2: Compute d-vectors for all datasets
    python compute_dvectors.py
    # Step 3: Fine-tune
    python finetune_yourtts.py
"""

import os
import sys
import csv
import json
import pathlib

# Paths
BASE = pathlib.Path(__file__).resolve().parent
YOURTTS_DIR = pathlib.Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--your_tts"
CHECKPOINT_DIR = BASE / "checkpoints" / "yourtts_finetuned"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained model files — restore from ORIGINAL pretrained (not previous fine-tuned)
# We use a staging directory to provide our d-vectors as speakers.json alongside
# the pretrained model, because Coqui loads speakers.json from the same dir as
# the model checkpoint. The original pretrained speakers.json has only 6 speakers.
PRETRAINED_MODEL_ORIG = str(YOURTTS_DIR / "model_file.pth")
PRETRAINED_CONFIG = str(YOURTTS_DIR / "config.json")
SE_MODEL = str(YOURTTS_DIR / "model_se.pth")
SE_CONFIG = str(YOURTTS_DIR / "config_se.json")
LANGUAGE_IDS = str(YOURTTS_DIR / "language_ids.json")

# Staging directory for restore (symlinked model + our speakers.json)
RESTORE_DIR = BASE / "checkpoints" / "yourtts_restore"

# Precomputed d-vectors (combined LibriTTS + Accent Archive)
DVECTORS_FILE = str(BASE / "data" / "dvectors" / "speakers.json")

# Dataset roots
LIBRITTS_DC_ROOT = str(BASE / "data" / "raw" / "libritts_clean100" / "LibriTTS" / "dev-clean")
LIBRITTS_TC_ROOT = str(BASE / "data" / "raw" / "libritts_clean100" / "LibriTTS" / "test-clean")

# Speech Accent Archive
ARCHIVE_WAV_DIR = str(BASE / "data" / "raw" / "archive" / "wav_16k")
ARCHIVE_METADATA = str(BASE / "data" / "raw" / "archive" / "metadata.csv")


def accent_archive_formatter(root_path, meta_file, ignored_speakers=None):
    """Custom Coqui TTS formatter for the Speech Accent Archive.

    Reads metadata.csv (pipe-delimited) with columns: wav_stem|transcript|speaker_name.
    Each speaker has exactly one recording of the same English passage.
    """
    items = []
    with open(meta_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            speaker_name = row["speaker_name"]
            if ignored_speakers and speaker_name in ignored_speakers:
                continue
            wav_path = os.path.join(root_path, f"{row['wav_stem']}.wav")
            if os.path.exists(wav_path):
                items.append({
                    "text": row["transcript"],
                    "audio_file": wav_path,
                    "speaker_name": speaker_name,
                    "root_path": root_path,
                })
    return items


def main():
    import torch
    # Reduce CUDA memory fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from trainer import Trainer, TrainerArgs
    from TTS.config import load_config
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits

    # Register our custom formatter so Coqui TTS can find it by name
    # _get_formatter_by_name() looks up on TTS.tts.datasets (the __init__ module),
    # NOT TTS.tts.datasets.formatters — must register on both to be safe.
    import TTS.tts.datasets as datasets_module
    import TTS.tts.datasets.formatters as formatters_module
    setattr(datasets_module, "accent_archive", accent_archive_formatter)
    setattr(formatters_module, "accent_archive", accent_archive_formatter)

    # ── Load pretrained config (preserves EXACT architecture) ──
    print("Loading pretrained YourTTS config...")
    config = load_config(PRETRAINED_CONFIG)

    # ── Set up staging directory for model restore ──
    # Coqui loads speakers.json from the same dir as the checkpoint.
    # We symlink the pretrained model + copy our d-vector file as speakers.json
    # so the trainer picks up our 2,200+ speaker embeddings instead of the
    # pretrained's 6-speaker file.
    import shutil
    RESTORE_DIR.mkdir(parents=True, exist_ok=True)
    staged_model = RESTORE_DIR / "model_file.pth"
    staged_speakers = RESTORE_DIR / "speakers.pth"  # Coqui looks for speakers.pth too
    staged_speakers_json = RESTORE_DIR / "speakers.json"

    if not staged_model.exists():
        os.symlink(PRETRAINED_MODEL_ORIG, str(staged_model))
        print(f"  Symlinked pretrained model → {staged_model}")

    # Copy our d-vectors as the speakers file in the restore dir
    shutil.copy2(DVECTORS_FILE, str(staged_speakers_json))
    # Also copy as .pth in case Coqui looks for that extension
    shutil.copy2(DVECTORS_FILE, str(staged_speakers))
    print(f"  Copied d-vectors to staging dir ({staged_speakers_json})")

    PRETRAINED_MODEL = str(staged_model)

    # ── Dataset configs ──
    datasets = []

    if os.path.isdir(LIBRITTS_DC_ROOT):
        datasets.append(
            BaseDatasetConfig(
                formatter="libri_tts",
                dataset_name="libritts_devclean",
                path=LIBRITTS_DC_ROOT,
                language="en",
            )
        )
        print(f"✓ LibriTTS dev-clean: {LIBRITTS_DC_ROOT}")

    if os.path.isdir(LIBRITTS_TC_ROOT):
        datasets.append(
            BaseDatasetConfig(
                formatter="libri_tts",
                dataset_name="libritts_testclean",
                path=LIBRITTS_TC_ROOT,
                language="en",
            )
        )
        print(f"✓ LibriTTS test-clean: {LIBRITTS_TC_ROOT}")

    # Speech Accent Archive (~2,133 speakers, 1 clip each)
    if os.path.isdir(ARCHIVE_WAV_DIR) and os.path.isfile(ARCHIVE_METADATA):
        datasets.append(
            BaseDatasetConfig(
                formatter="accent_archive",
                dataset_name="accent_archive",
                meta_file_train=ARCHIVE_METADATA,
                path=ARCHIVE_WAV_DIR,
                language="en",
            )
        )
        print(f"✓ Accent Archive: {ARCHIVE_WAV_DIR}")
    else:
        print(f"⚠ Accent Archive not found — run prepare_archive.py first")

    if not datasets:
        print("ERROR: No datasets found!")
        sys.exit(1)

    # ── Override training hyperparameters ──
    config.datasets = datasets
    config.output_path = str(CHECKPOINT_DIR)

    # Training params (conservative for fine-tuning)
    config.batch_size = 1               # RTX 4060 8GB VRAM — OOM at batch_size=2
    config.eval_batch_size = 1
    config.num_loader_workers = 4
    config.num_eval_loader_workers = 2
    config.run_eval = True
    config.eval_split_size = 0.005      # Small eval set for speed
    config.epochs = 1000                # Will stop by step limit

    # Learning rate (lower than original for fine-tuning stability)
    config.lr_gen = 5e-5
    config.lr_disc = 5e-5
    config.optimizer = "AdamW"
    config.optimizer_params = {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01}
    config.lr_scheduler_gen = "ExponentialLR"
    config.lr_scheduler_gen_params = {"gamma": 0.999875, "last_epoch": -1}
    config.lr_scheduler_disc = "ExponentialLR"
    config.lr_scheduler_disc_params = {"gamma": 0.999875, "last_epoch": -1}
    config.scheduler_after_epoch = True

    # Mixed precision
    config.mixed_precision = True

    # Gradient clipping
    config.grad_clip = [5.0, 5.0]

    # D-vector file (list format to match pretrained config)
    # Must override BOTH config and model_args since model_args takes priority
    config.d_vector_file = [DVECTORS_FILE]
    config.model_args.d_vector_file = [DVECTORS_FILE]

    # Language IDs file
    config.language_ids_file = LANGUAGE_IDS

    # Speaker encoder loss (maintains speaker discriminability during training)
    config.model_args.use_speaker_encoder_as_loss = True
    config.model_args.speaker_encoder_config_path = SE_CONFIG
    config.model_args.speaker_encoder_model_path = SE_MODEL

    # Audio length constraints (VRAM management — tight on RTX 4060 8GB)
    config.min_audio_len = 16000         # 1s minimum at 16kHz
    config.max_audio_len = 16000 * 7     # 7s maximum (10s → OOM at batch_size=1)

    # Logging and saving
    config.print_step = 100
    config.print_eval = True
    config.save_step = 1000
    config.save_best_after = 500
    config.save_checkpoints = True
    config.save_all_best = True
    config.save_n_checkpoints = 2

    # Test sentences: [text, speaker_name, style_wav, language_name]
    config.test_sentences = [
        ["This is a test of voice cloning quality.", None, None, "en"],
        ["The quick brown fox jumps over the lazy dog.", None, None, "en"],
        ["Hello, my name is Sarah and I love reading books.", None, None, "en"],
    ]

    # ── Build model from config ──
    print("\nInitializing model from config...")
    model = Vits.init_from_config(config)

    # ── Load training samples ──
    # NOTE: We disable Coqui's built-in eval_split because split_dataset() has
    # an infinite loop when multi-speaker datasets have exactly 1 clip per
    # speaker (all Archive speakers). Instead we manually take eval samples
    # from LibriTTS data only.
    print("Loading training samples (no auto-split)...")
    all_samples, _ = load_tts_samples(
        config.datasets,
        eval_split=False,
    )

    # Manual eval split: take ~0.5% of LibriTTS samples (speakers with >1 clip)
    import random
    random.seed(42)
    random.shuffle(all_samples)
    # Filter by audio_unique_name prefix (load_tts_samples sets "{dataset_name}#..." key)
    libritts_samples = [s for s in all_samples if s.get("audio_unique_name", "").startswith("libritts")]
    archive_samples = [s for s in all_samples if s.get("audio_unique_name", "").startswith("accent_archive")]
    print(f"  Total samples: {len(all_samples)} (LibriTTS: {len(libritts_samples)}, Archive: {len(archive_samples)})")
    eval_count = max(10, int(len(libritts_samples) * 0.005))
    eval_samples = libritts_samples[:eval_count]
    train_samples = libritts_samples[eval_count:] + archive_samples
    random.shuffle(train_samples)
    print(f"  Train: {len(train_samples)} ({len(libritts_samples)-eval_count} LibriTTS + {len(archive_samples)} Archive)")
    print(f"  Eval:  {len(eval_samples)} (LibriTTS only)")

    # ── Verify d-vectors file exists (skip full JSON parse — 138 MB is too slow) ──
    print("\nVerifying d-vector file...")
    dvec_path = pathlib.Path(DVECTORS_FILE)
    if not dvec_path.exists():
        print(f"  ERROR: D-vector file not found: {DVECTORS_FILE}")
        sys.exit(1)
    dvec_size_mb = dvec_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ D-vector file exists: {dvec_size_mb:.1f} MB")
    # Quick key count by scanning for top-level keys (much faster than json.load)
    import subprocess
    key_count = int(subprocess.check_output(
        ["grep", "-c", '"name":', DVECTORS_FILE], text=True
    ).strip())
    print(f"  ✓ Approximate entries: {key_count} (expected ~12,706)")
    # Show a few training sample keys for reference
    print(f"  Training sample keys (first 3): {[s.get('audio_unique_name', '') for s in train_samples[:3]]}")

    # ── Trainer ──
    trainer_args = TrainerArgs(
        restore_path=PRETRAINED_MODEL,
        grad_accum_steps=1,              # VITS dual-optimizer doesn't support grad_accum > 1
    )

    print(f"\n{'='*60}")
    print(f"YourTTS Fine-tuning")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Eval samples:  {len(eval_samples)}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  LR:            {config.lr_gen}")
    print(f"  Mixed prec:    {config.mixed_precision}")
    print(f"  D-vectors:     {DVECTORS_FILE}")
    print(f"  Speakers:      ~2,212 (LibriTTS + Accent Archive)")
    print(f"  Restore from:  {PRETRAINED_MODEL}")
    print(f"  Output:        {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")

    # ── Start training ──
    trainer = Trainer(
        trainer_args,
        config,
        str(CHECKPOINT_DIR),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
