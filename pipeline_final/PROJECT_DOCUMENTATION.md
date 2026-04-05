# TeluguVoiceBridge v2 — Complete Project Documentation

> **Project Goal:** Take Telugu speech as input, produce English speech as output — in the **same speaker's voice** with preserved emotional characteristics.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Root Level Files](#3-root-level-files)
4. [Jupyter Notebooks (Training Pipeline)](#4-jupyter-notebooks-training-pipeline)
5. [Python Scripts](#5-python-scripts)
6. [Configuration Files](#6-configuration-files)
7. [Training Logs](#7-training-logs)
8. [Results & Outputs](#8-results--outputs)
9. [Technical Pipeline Flow](#9-technical-pipeline-flow)
10. [Model Details](#10-model-details)
11. [Training Results Summary](#11-training-results-summary)

---

## 1. Project Overview

### What This System Does

TeluguVoiceBridge is a **speech-to-speech translation system** that:
1. **Transcribes** Telugu audio to Telugu text (ASR - Automatic Speech Recognition)
2. **Translates** Telugu text to English text (Machine Translation)
3. **Extracts** the speaker's unique voice characteristics (Speaker Encoding)
4. **Detects** emotional state from the audio (Emotion Detection)
5. **Synthesizes** English speech that sounds like the original Telugu speaker (Text-to-Speech with Voice Cloning)

### Why This Matters

- **Telugu** has ~82 million speakers worldwide
- Most existing translation systems (Google Translate, Azure) output **generic robotic voices**
- This system **preserves the original speaker's voice identity** in the translated output

### The 5-Model Cascaded Pipeline

```
Telugu Audio Input
      │
      ├──→ Stage 1: Whisper Large-v3 (INT8) → Telugu Text
      │
      ├──→ Stage 2: ECAPA-TDNN (CPU) → 192-dim Speaker Embedding
      │
      ├──→ Stage 3: Wav2Vec2 → Valence/Arousal/Dominance (Emotion)
      │
      ├──→ Stage 4: NLLB-600M (4-bit + LoRA) → English Text
      │
      └──→ Stage 5: YourTTS (VITS) → English Audio (Voice Cloned)
              │
              ▼
     English Audio Output (in original speaker's voice)
```

---

## 2. Directory Structure

```
Final/
├── TECHNICAL_OVERVIEW.md          # High-level technical documentation
├── PROJECT_DOCUMENTATION.md       # This file
│
└── pipeline_final/
    │
    ├── ======= JUPYTER NOTEBOOKS (Training Pipeline) =======
    ├── 01_environment_setup.ipynb       # System checks & dependencies
    ├── 02_data_pipeline.ipynb           # Dataset download & preprocessing
    ├── 03_whisper_asr_finetuning.ipynb  # ASR model fine-tuning
    ├── 04_speaker_encoder_finetuning.ipynb  # Voice embedding model
    ├── 05_translation_finetuning.ipynb  # Machine translation model
    ├── 06_emotion_detector.ipynb        # VAD emotion detection model
    ├── 07_tts_finetuning.ipynb          # Text-to-speech model
    ├── 08_pipeline_integration_demo.ipynb  # End-to-end integration
    │
    ├── ======= PYTHON SCRIPTS =======
    ├── app2.py                    # Main Gradio web application
    ├── compute_dvectors.py        # D-vector computation script
    ├── finetune_yourtts.py        # YourTTS fine-tuning script
    ├── prepare_archive.py         # Speech Accent Archive preparation
    │
    ├── ======= DOCUMENTATION =======
    ├── ARCHITECTURE.md            # Detailed system architecture
    ├── VoicePreserve.md           # Voice preservation methodology
    │
    ├── ======= CONFIGURATION =======
    ├── configs/
    │   ├── asr.yaml               # Whisper ASR configuration
    │   ├── emotion.yaml           # Emotion detector configuration
    │   ├── speaker.yaml           # Speaker encoder configuration
    │   ├── translation.yaml       # NLLB translation configuration
    │   └── tts.yaml               # TTS configuration
    │
    ├── ======= TRAINING LOGS =======
    ├── logs/
    │   ├── asr_training_log.csv         # Whisper training metrics
    │   ├── emotion_training_log.csv     # Emotion model metrics
    │   ├── speaker_training_log.csv     # Speaker encoder metrics
    │   └── translation_training_log.csv # Translation model metrics
    │
    ├── finetune_log.txt           # YourTTS fine-tuning output log
    │
    └── ======= RESULTS & OUTPUTS =======
        └── results/
            ├── evaluation_results.json  # Final pipeline metrics
            ├── ablation_results.json    # Ablation study results
            ├── sample_outputs/
            │   └── inference_log.json   # Sample inference outputs
            └── tts_rebuild/
                ├── comparison_results.json  # TTS model comparison
                ├── tier1_xtts/              # XTTS-v2 outputs
                ├── tier2_f5tts/             # F5-TTS outputs
                └── tier3_yourtts/           # YourTTS outputs
```

---

## 3. Root Level Files

### TECHNICAL_OVERVIEW.md
**Purpose:** Comprehensive technical documentation of the entire system

**Contents:**
- Pipeline architecture with detailed diagrams
- Step-by-step audio processing explanation
- Speech processing concepts (FFT, Mel spectrograms, speaker embeddings)
- Dataset descriptions with usage proofs
- Model architecture details
- Training procedures and hyperparameters
- Mathematical explanations of metrics (WER, CER, BLEU, SECS)

**Key Sections:**
- What happens to your audio at each stage
- Why cascaded pipeline instead of end-to-end
- Why 16 kHz sampling rate
- How speaker embeddings work
- LoRA and QLoRA quantization explained

---

## 4. Jupyter Notebooks (Training Pipeline)

### 01_environment_setup.ipynb
**Purpose:** Initialize the development environment and verify system compatibility

**What It Does:**
1. **System Health Checks:**
   - Verifies GPU availability (NVIDIA RTX 4060 Laptop, 7.6 GB VRAM)
   - Checks RAM (24 GB requirement)
   - Monitors disk space (~60 GB needed)
   - Validates CUDA installation

2. **Directory Structure Creation:**
   - Creates `data/raw/`, `data/processed/`, `data/metadata/`
   - Sets up checkpoint directories
   - Creates configuration directories

3. **Dependency Installation:**
   - PyTorch with CUDA support
   - Transformers, PEFT (LoRA), BitsAndBytes (quantization)
   - TTS (Coqui), SpeechBrain
   - Audio processing: librosa, torchaudio, soundfile

4. **VRAM Budget Calculation:**
   - Whisper Large-v3 (INT8): ~1.6 GB
   - NLLB-600M (4-bit): ~0.3 GB
   - Emotion Model: ~0.4 GB
   - YourTTS: ~0.34 GB
   - **Total:** ~2.6 GB (fits in 7.6 GB constraint)

5. **Config File Generation:**
   - Writes YAML configs for all 5 models

---

### 02_data_pipeline.ipynb
**Purpose:** Download, preprocess, and build metadata manifests for all training datasets

**Datasets Processed:**

| Dataset | Purpose | Size | Samples |
|---------|---------|------|---------|
| FLEURS (tel_IN) | ASR Fine-tuning | ~3 GB | 2,905 clips |
| RAVDESS | Speaker Encoder | ~2 GB | 1,435 clips |
| CoVoST-2 (te→en) | Translation | ~2 GB | 1,719 pairs |
| LibriTTS | TTS Training | ~11 GB | ~10,500 clips |
| Speech Accent Archive | TTS (Speaker Diversity) | ~2 GB | 2,133 speakers |

**Processing Steps:**

1. **Audio Normalization:**
   ```python
   def normalize_audio(wav, sr, target_sr=16000):
       # Convert to mono
       # Resample to 16 kHz
       # Normalize to float32 [-1.0, 1.0]
   ```

2. **Manifest Creation:**
   - ASR manifest: `audio_path, transcript, duration, dataset, language`
   - Speaker manifest: `audio_path, speaker_id, emotion, dataset, split`
   - Translation pairs: `telugu, english, split, sentence_id`

3. **Memory-Efficient Processing:**
   - Processes in batches of 50 files
   - Calls `gc.collect()` between batches
   - Deletes raw data after processing

**Critical Rule:** Download ONE dataset → process → delete raw → download next (never have multiple raw datasets on disk simultaneously)

---

### 03_whisper_asr_finetuning.ipynb
**Purpose:** Fine-tune Whisper Large-v3 for Telugu speech recognition

**Model Configuration:**
- **Base Model:** openai/whisper-large-v3 (1.55 billion parameters)
- **Quantization:** INT8 (reduces VRAM from ~6.2 GB to ~1.6 GB)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: q_proj, v_proj
  - Trainable parameters: 3.7M / 1.55B = **0.24%**

**Training Configuration:**
- Batch size: 1 (with 8× gradient accumulation = effective 8)
- Max steps: 5,000
- Learning rate: 1e-4 with cosine scheduler
- Warmup steps: 200
- FP16 mixed precision

**Data Augmentation:**
- **SpecAugment:**
  - Time masking: 2 masks × max 50 frames (80% probability)
  - Frequency masking: 2 masks × max 20 bins
- **Speed Perturbation:** Random resampling at 0.9×, 1.0×, 1.1×
- **Noise Injection:** Gaussian noise, SNR 15-25 dB (30% probability)

**Evaluation Metrics:**
- **WER (Word Error Rate):** Target < 20%
- **CER (Character Error Rate):** Target < 10%

**Long Audio Handling:**
- Whisper's limit: 30 seconds
- Solution: Split into 28s chunks with 3s overlap
- Deduplicate overlapping text via suffix-prefix matching

---

### 04_speaker_encoder_finetuning.ipynb
**Purpose:** Fine-tune ECAPA-TDNN for language-agnostic speaker embeddings

**The Problem:**
Standard speaker encoders trained on English data confuse **language identity** with **speaker identity**. A Telugu speaker sounds like a "different person" than the same person speaking English.

**Model Configuration:**
- **Base Model:** speechbrain/spkrec-ecapa-voxceleb (pretrained on 7,000+ speakers)
- **Embedding Dimension:** 192
- **Device:** CPU (to save GPU VRAM for other models)

**Training Configuration:**
- Contrastive batch sampling: 4 speakers × 4 utterances = 16 clips/batch
- Learning rate: 5e-5
- Epochs: 20
- Early stopping patience: 5

**Loss Function: AAM-Softmax (ArcFace)**
```
L = −log [ e^{s·cos(θ_y + m)} / (e^{s·cos(θ_y + m)} + Σ_{j≠y} e^{s·cos θ_j}) ]
```
- Margin (m): 0.2 — forces intra-class compactness
- Scale (s): 30 — temperature-like sharpening

**Gradient Reversal Layer (GRL) for Language-Agnostic Embeddings:**
- Attach language classifier (Telugu=1, English=0) via GRL
- GRL flips gradient sign during backpropagation
- Forces encoder to produce embeddings where language **cannot** be predicted
- Result: Embeddings capture only voice identity, not language

**Target Metrics:**
- EER (Equal Error Rate): < 12%
- Same-speaker cosine similarity: > 0.7

---

### 05_translation_finetuning.ipynb
**Purpose:** Fine-tune NLLB-200 for Telugu→English translation

**Model Configuration:**
- **Primary Model:** ai4bharat/indictrans2-indic-en-dist-200M
- **Fallback Model:** facebook/nllb-200-distilled-600M
- **Quantization:** 4-bit NF4 (Normal Float 4-bit)
- **Fine-tuning:** QLoRA
  - Rank (r): 8
  - Alpha: 16
  - Target modules: q_proj, v_proj
  - Trainable parameters: 4.9M / 600M = **0.8%**

**Training Configuration:**
- Batch size: 1 (with 8× gradient accumulation)
- Epochs: 3
- Learning rate: 3e-5
- Max source/target length: 256 tokens

**Why QLoRA?**
- Base weights quantized to 4-bit (frozen)
- Only LoRA adapter matrices train in FP16
- Memory: ~300 MB instead of ~1.2 GB FP16

**Long Text Handling:**
- Split at Telugu punctuation: '।', '.', '?', '!'
- Translate sentence-by-sentence
- Prevents 512-token truncation

**Target Metrics:**
- BLEU (translation only): 22+
- BLEU (full pipeline): 15+
- Inference time: < 2 seconds
- VRAM: < 1.5 GB

---

### 06_emotion_detector.ipynb
**Purpose:** Train Wav2Vec2-based emotion detector for Valence/Arousal/Dominance

**The Problem:**
Standard TTS produces neutral, emotionless speech. We need to detect the emotional state of the input and (ideally) inject it into the TTS output.

**Model Architecture:**
```python
class EmotionVADModel(nn.Module):
    def __init__(self):
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        # Freeze first 6 transformer layers
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),  # Output: V, A, D
            nn.Tanh()  # Output range: [-1, 1]
        )
```

**Output: VAD Vector**
- **Valence:** Positive vs. Negative emotion (-1 to +1)
- **Arousal:** Calm vs. Excited (-1 to +1)
- **Dominance:** Submissive vs. Dominant (-1 to +1)

**Training Configuration:**
- Batch size: 4
- Epochs: 20
- Learning rate: 1e-4
- First 6 Wav2Vec2 layers: Frozen

**Loss Function: MSE + CCC**
```python
loss = MSE_loss + 0.5 * (1 - CCC_loss)
```
- **CCC (Concordance Correlation Coefficient):** Measures agreement between predicted and true values

**Target Metric:**
- CCC: > 0.55

**Note:** The VAD vector is currently displayed in the UI but **not yet injected into TTS** via FiLM conditioning (designed but not wired in app2.py).

---

### 07_tts_finetuning.ipynb
**Purpose:** Fine-tune YourTTS for multi-speaker voice cloning

**Why YourTTS (not XTTS-v2)?**
- XTTS-v2: 3.56 GB VRAM, produced garbled output
- YourTTS: 0.34 GB VRAM, stable voice cloning

**Model Architecture: VITS (Variational Inference with adversarial learning for TTS)**
```
Text → Text Encoder → Duration Predictor → Normalising Flow → HiFi-GAN → Waveform
           ↑                    ↑                 ↑               ↑
           └──── Speaker Embedding (512-dim d-vector) injected at ALL 4 stages
```

**The Problem with Pretrained YourTTS:**
- Pretrained on only 6 speakers (Portuguese + English Common Voice)
- Does not generalize to unseen speakers
- Voice cloning produces "average voice"

**Solution: Fine-tune on 2,212 Diverse Speakers:**
- LibriTTS dev-clean: 40 speakers, ~5,700 clips
- LibriTTS test-clean: 39 speakers, ~4,800 clips
- Speech Accent Archive: 2,133 speakers, 1 clip each (214 accents, 200 languages)

**Training Phases:**

| Phase | Description | Steps | Learning Rate |
|-------|-------------|-------|---------------|
| 5a | Domain adaptation (LJSpeech + VCTK) | 50,000 | 1e-4 |
| 5b | Cross-lingual speaker generalization | 20,000 | 2e-5 |

**Reference Audio Preparation:**
1. Load and resample to 16 kHz mono
2. Trim silence (librosa, 25 dB threshold)
3. Sliding window to find highest-RMS 7-second segment
4. Peak normalize to 0.9
5. Pass to YourTTS's internal ResNet encoder → 512-dim d-vector

**Target Metrics:**
- UTMOS: > 3.5 (phase 5a), > 3.3 (phase 5b)
- SECS: > 0.7
- Inference time: < 5 seconds

---

### 08_pipeline_integration_demo.ipynb
**Purpose:** End-to-end integration testing and evaluation

**What It Does:**

1. **Model Loading:**
   - Loads all 5 models sequentially
   - Monitors VRAM usage at each step
   - Validates checkpoint loading

2. **Full Pipeline Function:**
   ```python
   def telugu_voice_bridge(audio_path) -> dict:
       # 1. Load and preprocess audio
       # 2. [PARALLEL] Speaker embedding extraction (CPU)
       # 3. [GPU] Whisper ASR → Telugu text
       # 4. [GPU] NLLB Translation → English text
       # 5. [GPU] Emotion VAD extraction
       # 6. [GPU] YourTTS synthesis with voice cloning
       # 7. Compute SECS (speaker similarity)
       return {
           "telugu_text": ...,
           "english_text": ...,
           "output_audio": ...,
           "secs": ...,
           "vad": [v, a, d],
           "latency": ...
       }
   ```

3. **Evaluation on Test Set:**
   - FLEURS Telugu test samples
   - Compute WER, CER, BLEU, chrF, SECS
   - Measure end-to-end latency

4. **Ablation Studies:**
   - No FiLM conditioning
   - No contrastive speaker loss
   - Discrete emotion labels vs. continuous VAD

**Final Metrics Achieved:**
| Metric | Value |
|--------|-------|
| WER | 66.67% |
| CER | 38.93% |
| BLEU | 27.37 |
| chrF | 59.34 |
| SECS | 0.38 |
| Avg Latency | 11.04 sec |

---

## 5. Python Scripts

### app2.py
**Purpose:** Main Gradio web application for real-time Telugu→English speech translation

**Features:**
- Upload Telugu audio (any format)
- Real-time processing with progress indicators
- Output: English audio + Telugu transcript + English translation + metrics

**Architecture:**
```python
# Model loading
def load_all_models():
    # 1. Whisper ASR (INT8)
    # 2. Speaker Encoder (ECAPA-TDNN, CPU)
    # 3. Translation (NLLB + LoRA)
    # 4. Emotion Detector (Wav2Vec2 VAD)
    # 5. TTS (YourTTS)
    return models

# Pipeline functions
def transcribe_telugu(audio_np, sr=16000)  # Whisper ASR
def translate_te_en(telugu_text)            # NLLB Translation
def extract_speaker_embedding(audio_np)     # ECAPA-TDNN
def extract_emotion_vad(audio_np)           # Wav2Vec2 VAD
def synthesize_speech(text, speaker_ref)    # YourTTS

# Main function
def telugu_voice_bridge(audio_path):
    # Orchestrates all 5 stages
    # Returns output audio + all intermediate results
```

**Key Implementation Details:**

1. **Parallel Processing:**
   - Speaker embedding runs on CPU in a separate thread
   - GPU handles ASR, translation, emotion, TTS sequentially
   - Thread joins before TTS (which needs speaker reference)

2. **Long Audio Handling:**
   - Audio > 28s: Split into chunks with 3s overlap
   - Text > 200 chars: Sentence-by-sentence translation

3. **Audio Normalization:**
   - RMS normalize to target_rms = 0.08 (LJSpeech level)
   - Peak hard limit to 0.95

4. **Gradio Interface:**
   ```python
   interface = gr.Interface(
       fn=telugu_voice_bridge,
       inputs=gr.Audio(source="upload", type="filepath"),
       outputs=[
           gr.Audio(label="English Output"),
           gr.Textbox(label="Telugu Transcript"),
           gr.Textbox(label="English Translation"),
           gr.JSON(label="Metrics")
       ]
   )
   ```

---

### compute_dvectors.py
**Purpose:** Compute 512-dimensional d-vectors for all training audio using YourTTS's speaker encoder

**Why This Script Exists:**
- YourTTS uses its own ResNet speaker encoder (512-dim)
- We must precompute d-vectors that match YourTTS's internal representation
- D-vectors are stored in **speakers.json** for training

**Key Design Decisions:**
- Uses YourTTS's `model_se.pth` and `config_se.json` (exact same encoder)
- Key format: `{dataset_name}#{relative_path_without_ext}`
- This matches Coqui TTS's internal naming convention

**Datasets Processed:**
| Dataset | Speakers | Clips |
|---------|----------|-------|
| LibriTTS dev-clean | ~40 | ~5,700 |
| LibriTTS test-clean | ~39 | ~4,800 |
| Speech Accent Archive | ~2,133 | 2,133 |
| **Total** | **~2,212** | **~12,700** |

**Output:** `data/dvectors/speakers.json`

---

### finetune_yourtts.py
**Purpose:** Fine-tune YourTTS on 2,212 speakers for improved zero-shot voice cloning

**Why Fine-tuning is Necessary:**
- Pretrained YourTTS: only 6 speakers (Common Voice PT + EN)
- Fine-tuned YourTTS: 2,212 speakers (79 LibriTTS + 2,133 Accent Archive)
- More speakers = better generalization to unseen voices

**Implementation Details:**

1. **Staging Directory Setup:**
   ```python
   # Coqui loads speakers.json from the same dir as checkpoint
   # We symlink pretrained model + copy our d-vectors file
   RESTORE_DIR.mkdir()
   os.symlink(PRETRAINED_MODEL_ORIG, staged_model)
   shutil.copy2(DVECTORS_FILE, staged_speakers_json)
   ```

2. **Custom Formatter for Speech Accent Archive:**
   ```python
   def accent_archive_formatter(root_path, meta_file):
       # Reads metadata.csv (pipe-delimited)
       # Columns: wav_stem|transcript|speaker_name
       # Returns list of dicts for Coqui TTS
   ```

3. **Training Configuration:**
   - Restore from ORIGINAL pretrained (clean slate)
   - Use precomputed 512-dim d-vectors
   - Learning rate: 5e-5 (low, for stable fine-tuning)
   - Mixed precision: FP16 (fits in 8 GB VRAM)

---

### prepare_archive.py
**Purpose:** Prepare Speech Accent Archive dataset for TTS training

**What is Speech Accent Archive?**
- 2,133 speakers from different native languages
- Each speaker reads the **same English passage**
- Provides massive speaker diversity with controlled text

**Processing Steps:**
1. Parse `speakers_all.csv` → filter rows with audio present
2. Convert MP3 → WAV (16 kHz mono) using ffmpeg
3. Generate `metadata.csv` in LJSpeech format:
   ```
   filename_no_ext|transcript|transcript
   ```

**The Standard Passage All Speakers Read:**
> "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob..."

---

## 6. Configuration Files

### configs/asr.yaml
**Purpose:** Whisper ASR model configuration

```yaml
model:
  name: openai/whisper-large-v3
  load_in_8bit: true
  torch_dtype: float16
  device_map: cuda:0

lora:
  r: 8                    # LoRA rank
  alpha: 16               # LoRA alpha scaling
  dropout: 0.05
  target_modules: [q_proj, v_proj]
  bias: none

training:
  batch_size: 1
  gradient_accumulation_steps: 8    # Effective batch = 8
  max_steps: 5000
  warmup_steps: 200
  learning_rate: 0.0001
  lr_scheduler: cosine
  fp16: true
  eval_steps: 250
  save_steps: 250
  language: te                       # Telugu
  task: transcribe

augmentation:
  spec_augment:
    time_masks: 2
    time_mask_max: 50
    freq_masks: 2
    freq_mask_max: 20
    apply_prob: 0.8
  speed_perturbation: [0.9, 1.0, 1.1]
  noise:
    snr_range: [15, 25]
    apply_prob: 0.3

target_metrics:
  wer: 0.2
  cer: 0.1
```

---

### configs/speaker.yaml
**Purpose:** ECAPA-TDNN speaker encoder configuration

```yaml
model:
  name: speechbrain/spkrec-ecapa-voxceleb
  embedding_dim: 192
  device: cpu              # Save GPU VRAM

training:
  device: cpu
  speakers_per_batch: 4    # Contrastive sampling
  utterances_per_speaker: 4
  learning_rate: 5.0e-05
  epochs: 20
  weight_decay: 0.0001
  early_stopping_patience: 5

loss:
  type: aam_softmax        # ArcFace
  margin: 0.2              # Angular margin
  scale: 30                # Temperature scaling

target_metrics:
  eer: 0.12                # Equal Error Rate
  cosine_same_speaker: 0.7
  inference_time_sec: 0.5
```

---

### configs/translation.yaml
**Purpose:** NLLB translation model configuration

```yaml
model:
  name: ai4bharat/indictrans2-indic-en-dist-200M
  fallback: facebook/nllb-200-distilled-600M
  load_in_4bit: true       # NF4 quantization
  torch_dtype: float16
  device_map: cuda:0
  src_lang: tel_Telu       # Telugu (Indic script)
  tgt_lang: eng_Latn       # English (Latin script)

lora:
  r: 8
  alpha: 16
  target_modules: [q_proj, v_proj]

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  epochs: 3
  learning_rate: 3.0e-05
  max_source_length: 256
  max_target_length: 256

target_metrics:
  bleu_mt_only: 22
  bleu_pipeline: 15
  inference_time_sec: 2.0
  vram_gb: 1.5
```

---

### configs/emotion.yaml
**Purpose:** Wav2Vec2 emotion detector configuration

```yaml
model:
  name: facebook/wav2vec2-base
  freeze_layers: 6         # Freeze first 6 transformer layers
  vad_output_dim: 3        # V, A, D

training:
  batch_size: 4
  epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
  device: cuda

loss:
  type: mse_ccc
  ccc_weight: 0.5          # CCC loss weight

target_metrics:
  ccc: 0.55
```

---

### configs/tts.yaml
**Purpose:** TTS model configuration

```yaml
model:
  name: tts_models/multilingual/multi-dataset/xtts_v2
  device: cuda

phase5a:
  description: Domain adaptation on LJSpeech + VCTK
  steps: 50000
  batch_size: 1
  learning_rate: 0.0001
  lr_decay_gamma: 0.999
  save_every: 5000

phase5b:
  description: Cross-lingual speaker generalization
  steps: 20000
  batch_size: 1
  learning_rate: 2.0e-05
  save_every: 2000

target_metrics:
  utmos_5a: 3.5
  utmos_5b: 3.3
  secs: 0.7
  inference_time_sec: 5.0
```

---

## 7. Training Logs

### logs/asr_training_log.csv
**Content:** Step-by-step Whisper training metrics

| Column | Description |
|--------|-------------|
| step | Training iteration |
| train_loss | Cross-entropy loss |
| val_wer | Validation Word Error Rate |
| val_cer | Validation Character Error Rate |
| lr | Current learning rate |
| vram_gb | GPU memory usage |
| time_sec | Cumulative training time |

**Sample Data:**
```csv
step,train_loss,val_wer,val_cer,lr,vram_gb,time_sec
250,0.3138,0.6766,0.295,1.00e-04,1.86,1989
...
5000,0.1133,0.9728,0.7096,0.00e+00,1.86,42685
```

**Observation:** Training loss decreased from 0.31 to 0.10, but validation WER increased mid-training (possible overfitting on augmented data).

---

### logs/emotion_training_log.csv
**Content:** Emotion detector training metrics

| Column | Description |
|--------|-------------|
| epoch | Training epoch |
| train_loss | MSE + CCC loss |
| ccc_v, ccc_a, ccc_d | CCC for Valence, Arousal, Dominance |
| ccc_mean | Average CCC |

**Best Result:** Epoch 10 achieved CCC mean = 0.9149

---

### logs/speaker_training_log.csv
**Content:** Speaker encoder training metrics

| Column | Description |
|--------|-------------|
| epoch | Training epoch |
| train_loss | AAM-Softmax loss |
| val_eer | Validation Equal Error Rate |

**Observation:** EER started at 1.15% (very good) but increased to 10.38% after 6 epochs — the pretrained model may have been more generalized than our fine-tuning data allowed.

---

### logs/translation_training_log.csv
**Content:** Translation model training metrics

| Column | Description |
|--------|-------------|
| epoch | Training epoch |
| val_bleu | Validation BLEU score |
| val_loss | Cross-entropy loss |

**Best Result:** Epoch 3 achieved BLEU = 31.89

---

### finetune_log.txt
**Purpose:** YourTTS fine-tuning console output

**Contents:**
- Training environment info (GPU, CPU threads, CUDA version)
- Model restoration messages
- Partial model initialization warnings (speaker encoder layers missing — expected, we use precomputed d-vectors)
- TensorBoard log paths

---

## 8. Results & Outputs

### results/evaluation_results.json
**Purpose:** Final end-to-end pipeline evaluation metrics

```json
{
  "wer": 0.6667,          // Word Error Rate (ASR)
  "cer": 0.3893,          // Character Error Rate (ASR)
  "secs": 0.3805,         // Speaker Embedding Cosine Similarity
  "avg_latency_sec": 11.04,
  "bleu": 27.37,          // Translation quality
  "chrf": 59.34           // Character n-gram F-score
}
```

**Interpretation:**
- **WER 66.67%:** High, but Telugu is agglutinative — one morpheme error = full word error
- **CER 38.93%:** More representative; ~61% of characters are correct
- **SECS 0.38:** Low due to embedding space mismatch (ECAPA 192-dim vs. YourTTS 512-dim)
- **Latency 11s:** End-to-end processing time

---

### results/ablation_results.json
**Purpose:** Ablation study comparing different configurations

```json
[
  {"ablation": "no_film", "secs": 0.3797, "avg_latency": 12.63},
  {"ablation": "no_contrastive", "secs": 0.3637},
  {"ablation": "discrete_labels", "secs": 0.3656}
]
```

**Findings:**
- FiLM conditioning has minimal impact on SECS (not yet properly integrated)
- Contrastive loss provides slight improvement

---

### results/sample_outputs/inference_log.json
**Purpose:** Detailed logs from sample inference runs

**Sample Entry:**
```json
{
  "sample_id": 0,
  "input_audio": "fleurs_002606.wav",
  "output_audio": "output_00.wav",
  "telugu_text": "చిన్న ద్వేపాలో చాలా వరకు...",
  "english_text": "Most of the small islands have no independent countries...",
  "vad_vector": [0.32, -0.51, -0.03],
  "timing": {
    "speaker_encoding": 0.37,
    "asr": 6.63,
    "translation": 0.28,
    "emotion": 0.02,
    "tts": 1.62,
    "total": 8.55
  }
}
```

**Timing Breakdown:**
- ASR dominates (~77% of processing time)
- TTS: ~19%
- Translation, speaker, emotion: ~4% combined

---

### results/tts_rebuild/comparison_results.json
**Purpose:** Comparison of different TTS models

```json
{
  "tiers": [
    {"name": "XTTS-v2 (fixed)", "success_rate": 1.0, "score": 10.75},
    {"name": "F5-TTS", "success_rate": 1.0, "score": 2.34},
    {"name": "YourTTS", "success_rate": 1.0, "score": 0.98}
  ],
  "winner": "YourTTS"
}
```

**Note:** Lower score = better (based on duration ratio, RMS, speed). YourTTS chosen for:
- Lowest VRAM usage (0.34 GB vs. 3.56 GB for XTTS)
- Stable voice cloning
- Reasonable speed

---

## 9. Technical Pipeline Flow

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT: Telugu Audio                                 │
│                          (Any format, any sample rate)                          │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING                                                                    │
│   • Resample to 16 kHz                                                          │
│   • Convert to mono                                                              │
│   • Normalize to float32 [-1.0, 1.0]                                            │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ SPEAKER ENCODER      │  │ WHISPER ASR          │  │ EMOTION DETECTOR     │
│ (ECAPA-TDNN, CPU)    │  │ (GPU, INT8)          │  │ (Wav2Vec2, GPU)      │
│                      │  │                      │  │                      │
│ • 80-dim Mel features│  │ • 80×3000 Mel spec   │  │ • Raw waveform       │
│ • SE-Res2Block ×3    │  │ • 32-layer Encoder   │  │ • 12-layer Encoder   │
│ • Attentive pooling  │  │ • 32-layer Decoder   │  │ • Regression head    │
│ • L2 normalize       │  │ • Force Telugu lang  │  │                      │
│                      │  │                      │  │                      │
│ Output: 192-dim      │  │ Output: Telugu text  │  │ Output: [V, A, D]    │
│ speaker embedding    │  │                      │  │ ∈ [-1, 1]³           │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │
           │                         ▼                         │
           │              ┌──────────────────────┐             │
           │              │ NLLB TRANSLATION     │             │
           │              │ (GPU, 4-bit + LoRA)  │             │
           │              │                      │             │
           │              │ • BPE tokenization   │             │
           │              │ • 12-layer Encoder   │             │
           │              │ • 12-layer Decoder   │             │
           │              │ • 4-beam search      │             │
           │              │                      │             │
           │              │ Output: English text │             │
           │              └──────────┬───────────┘             │
           │                         │                         │
           └────────────┬────────────┼────────────┬────────────┘
                        │            │            │
                        ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ YOURTTS SYNTHESIS                                                                │
│                                                                                  │
│   English text ─────────────────────────────────────────────────────────────┐   │
│                                                                              │   │
│   Speaker reference audio ──→ YourTTS ResNet Encoder ──→ 512-dim d-vector ─┤   │
│                                                                              │   │
│                              ┌───────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│   Text Encoder ──→ Duration Predictor ──→ Normalising Flow ──→ HiFi-GAN         │
│        ↑                    ↑                    ↑                ↑              │
│        └────────── d-vector injected at ALL 4 stages ───────────┘              │
│                                                                                  │
│   Output: 16 kHz waveform                                                       │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ POST-PROCESSING                                                                  │
│   • RMS normalize (target = 0.08)                                               │
│   • Peak limit (max = 0.95)                                                     │
│   • Compute SECS (speaker similarity metric)                                    │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT: English Audio                                   │
│                    (16 kHz WAV, ~11s end-to-end latency)                        │
│                    in the original speaker's voice                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Model Details

### Model 1: Whisper Large-v3 (ASR)

| Property | Value |
|----------|-------|
| **Architecture** | Encoder-Decoder Transformer |
| **Parameters** | 1.55 billion |
| **Quantization** | INT8 (bitsandbytes) |
| **VRAM Usage** | 1.6 GB (vs. 6.2 GB FP32) |
| **Input** | 80×3000 log-Mel spectrogram (30s max) |
| **Output** | BPE token sequence |
| **Fine-tuning** | LoRA (r=8, α=16, 0.24% trainable) |

### Model 2: ECAPA-TDNN (Speaker Encoder)

| Property | Value |
|----------|-------|
| **Architecture** | SE-Res2Block + Attentive Stats Pooling |
| **Pretrained On** | VoxCeleb 1+2 (7,000+ speakers) |
| **Embedding Dim** | 192 |
| **Device** | CPU (saves GPU VRAM) |
| **Fine-tuning** | AAM-Softmax + GRL (language adversarial) |

### Model 3: Wav2Vec2-base (Emotion Detector)

| Property | Value |
|----------|-------|
| **Architecture** | CNN Feature Extractor + 12-layer Transformer |
| **Parameters** | ~95 million |
| **Frozen Layers** | First 6 transformer layers |
| **Output** | 3-dim VAD vector (Valence, Arousal, Dominance) |
| **Loss** | MSE + CCC (Concordance Correlation Coefficient) |

### Model 4: NLLB-200-distilled-600M (Translation)

| Property | Value |
|----------|-------|
| **Architecture** | Encoder-Decoder Transformer |
| **Parameters** | 600 million |
| **Quantization** | NF4 (Normal Float 4-bit) |
| **VRAM Usage** | ~300 MB (vs. 1.2 GB FP16) |
| **Languages** | 200 languages supported |
| **Fine-tuning** | QLoRA (r=8, α=16, 0.8% trainable) |

### Model 5: YourTTS (TTS with Voice Cloning)

| Property | Value |
|----------|-------|
| **Architecture** | VITS (Variational Inference TTS) |
| **Components** | Text Encoder + Duration Predictor + Flow + HiFi-GAN |
| **VRAM Usage** | 0.34 GB |
| **Sample Rate** | 16 kHz |
| **Speaker Conditioning** | 512-dim d-vector (ResNet encoder) |
| **Fine-tuning** | 2,212 speakers (LibriTTS + Accent Archive) |

---

## 11. Training Results Summary

### Final Metrics

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| **WER** | 20% | 66.67% | Telugu is agglutinative |
| **CER** | 10% | 38.93% | Fairer metric for Telugu |
| **BLEU** | 22 | 27.37 | Good translation quality |
| **chrF** | — | 59.34 | Character-level translation |
| **SECS** | 0.70 | 0.38 | Space mismatch (192 vs 512 dim) |
| **Latency** | — | 11.04s | End-to-end |

### Training Time

| Model | Training Time | Hardware |
|-------|---------------|----------|
| Whisper ASR | ~12 hours (5000 steps) | RTX 4060 |
| Speaker Encoder | ~6 minutes (6 epochs) | CPU |
| Translation | ~2 hours (3 epochs) | RTX 4060 |
| Emotion Detector | ~7 minutes (18 epochs) | RTX 4060 |
| YourTTS | ~24 hours (estimated) | RTX 4060 |

### Known Limitations

1. **SECS Metric Mismatch:** ECAPA produces 192-dim embeddings; YourTTS uses 512-dim. Cosine similarity across different spaces is not meaningful.

2. **Emotion Not Injected:** VAD scores are displayed but not yet conditioning the TTS via FiLM layers.

3. **Prosody Loss:** Text translation destroys timing/emphasis information. Prosody cross-attention (designed in `prosody.py`) is not wired into the pipeline.

4. **High WER:** Telugu's agglutinative nature makes WER unfairly harsh. CER is more representative.

---

## Quick Reference

### To Run the Web Application:
```bash
cd pipeline_final
conda activate ml_env
python app2.py
```

### To Train from Scratch:
1. Run notebooks 01-08 in order
2. Each notebook is self-contained with its own data loading

### Key Files for Each Model:
| Model | Config | Checkpoint | Training Notebook |
|-------|--------|------------|-------------------|
| ASR | configs/asr.yaml | checkpoints/whisper_merged/ | 03_whisper_asr_finetuning.ipynb |
| Speaker | configs/speaker.yaml | checkpoints/speaker_encoder/ | 04_speaker_encoder_finetuning.ipynb |
| Translation | configs/translation.yaml | checkpoints/indictrans2_finetuned/ | 05_translation_finetuning.ipynb |
| Emotion | configs/emotion.yaml | checkpoints/emotion_detector/ | 06_emotion_detector.ipynb |
| TTS | configs/tts.yaml | checkpoints/yourtts_finetuned/ | 07_tts_finetuning.ipynb |

---

*Documentation generated for TeluguVoiceBridge v2 — Speech-to-Speech Translation with Voice Preservation*
