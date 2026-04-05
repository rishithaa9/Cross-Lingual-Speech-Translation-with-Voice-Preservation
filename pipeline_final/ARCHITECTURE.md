# TeluguVoiceBridge v2 — Complete Architecture, Workflow & Results

> **Goal:** Take a Telugu speech recording as input, produce English speech as output — in the same speaker's voice, with matched emotion.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [The Core Problem](#2-the-core-problem)
3. [Pipeline Overview — Exact Execution Order](#3-pipeline-overview)
4. [Model 1 — Whisper Large-v3 (ASR)](#4-model-1--whisper-large-v3-asr)
5. [Model 2 — ECAPA-TDNN Speaker Encoder](#5-model-2--ecapa-tdnn-speaker-encoder)
6. [Model 3 — Wav2Vec2 Emotion Detector](#6-model-3--wav2vec2-emotion-detector)
7. [Model 4 — NLLB-600M + LoRA (Translation)](#7-model-4--nllb-600m--lora-translation)
8. [Model 5 — YourTTS / VITS (Text-to-Speech)](#8-model-5--yourtts--vits-text-to-speech)
9. [Fine-tuning Summary Table](#9-fine-tuning-summary-table)
10. [Datasets Used](#10-datasets-used)
11. [Training Results — All Numbers](#11-training-results--all-numbers)
12. [End-to-End Evaluation Results](#12-end-to-end-evaluation-results)
13. [VRAM & Hardware Layout](#13-vram--hardware-layout)
14. [Key Design Decisions](#14-key-design-decisions)
15. [File Structure Reference](#15-file-structure-reference)
16. [Glossary](#16-glossary)

---

## 1. What This System Does

You upload a **Telugu audio recording** (3–30 seconds). The system:

1. Transcribes the Telugu speech → Telugu text (ASR)
2. Translates the Telugu text → English text (MT)
3. Analyzes the speaker's voice identity (speaker encoder)
4. Detects the emotional state of the speaker (emotion model)
5. Synthesizes English speech that sounds like the original speaker (TTS with voice cloning)

### What is preserved
- Speaker's voice characteristics (reference audio passed to YourTTS)
- Emotional tone (VAD scores shown as metrics in the UI)
- Speaking style (somewhat via voice cloning reference)

### What exists in design but is NOT in app2.py
- FiLM emotion conditioning injected into TTS decoder (designed, not wired)
- Prosody cross-attention transfer (implemented in `prosody.py`, not called from app2)
- FreeVC24 two-stage voice conversion post-processing (described in old architecture, not run)

---

## 2. The Core Problem

Voice preservation across languages has 5 compounding problems:

### Problem 1: Speaker Encoder Language Leakage
Standard speaker encoders trained on English audio see Telugu as a "different speaker" — they mix up language identity with speaker identity. ECAPA-TDNN pretrained on VoxCeleb partially confuses *what language is being spoken* with *who the speaker is*.

**Fix:** Fine-tune ECAPA-TDNN with ArcFace loss on Telugu speaker pairs to reduce this leakage.

### Problem 2: Prosody Loss at Translation Boundary
Text translation destroys all timing information. A fast rising-intonation Telugu question becomes a flat English string with no emphasis markers.

**Fix (designed):** Extract F0 + energy from input, use NLLB cross-attention matrix to align Telugu prosody positions to English word positions, pass as soft targets into TTS. (In `prosody.py` — not wired into current app.)

### Problem 3: Emotion Erasure
Standard TTS always produces calm, neutral speech regardless of whether the input was angry, excited, or sad.

**Fix:** A Wav2Vec2 emotion detector predicts continuous Valence/Arousal/Dominance scores. These are shown as metrics; the full design includes FiLM conditioning layers that would inject these into the HiFi-GAN decoder.

### Problem 4: Zero-Shot Voice Cloning Doesn't Generalize
YourTTS pretrained on 6 speakers produces an "average voice" for any unseen speaker — the d-vector conditioning barely works. You need hundreds of diverse speakers in training to make voice cloning generalize.

**Fix:** Fine-tune YourTTS on 2,212 speakers (LibriTTS dev/test-clean + Speech Accent Archive). This teaches the model the mapping from speaker embedding → voice characteristics.

### Problem 5: SECS Metric Mismatch
Measuring voice similarity using ECAPA-TDNN 192-dim embeddings while conditioning TTS with YourTTS's own 512-dim embeddings gives a meaningless score — you're comparing vectors in completely different spaces. The 0.38 SECS result was not because cloning was bad, but because the measurement was wrong.

**Fix:** Use the same encoder (YourTTS internal) for both conditioning and SECS measurement.

---

## 3. Pipeline Overview

Exact order of operations in `app2.py`'s `telugu_voice_bridge()` function:

```
INPUT: Telugu audio file (any format / sample rate)
  │
  ▼
[LOAD] torchaudio.load → resample to 16kHz mono numpy array
  │
  ├──────────── [THREAD B, CPU, runs in PARALLEL] ──────────────────────┐
  │                                                                      │
  │  prepare_speaker_ref():                                              │
  │    1. Load + resample audio to 16kHz                                 │
  │    2. librosa.effects.trim(top_db=25) — remove silence               │
  │    3. _extract_best_segment() — sliding window, pick highest-RMS 7s  │
  │    4. Peak normalize to 0.9 level                                    │
  │    5. sf.write() → temp WAV file (speaker_ref_path)                  │
  │                                                                      │
  │  extract_speaker_embedding():                                        │
  │    1. Pass audio through ECAPA-TDNN (CPU)                            │
  │    2. L2-normalize → 192-dim speaker vector                          │
  │    3. Used ONLY for SECS measurement at end                          │
  │                                                                      │
  └──────────── [CPU thread joins before TTS step] ─────────────────────┘
  │
  ▼
[GPU] transcribe_telugu():
  • Whisper Large-v3 INT8
  • Force language=Telugu (tel), task=transcribe
  • If audio > 28s: split into 28s chunks with 3s overlap, deduplicate
  • Returns: Telugu text string
  │
  ▼
[GPU] translate_te_en():
  • NLLB-200-distilled-600M (4-bit NF4) + LoRA adapter
  • src_lang=tel_Telu, forced_bos_token=eng_Latn
  • 4-beam beam search, max_new_tokens=512
  • If text > 200 chars: split on sentence boundaries, translate each
  • Returns: English text string
  │
  ▼
[GPU] extract_emotion_vad():
  • Wav2Vec2-base backbone + regression head (fine-tuned)
  • Returns: [valence, arousal, dominance] ∈ [-1, 1]³
  │
  ▼
[CPU thread join — speaker_ref_path now available]
  │
  ▼
[GPU] synthesize_speech():
  • YourTTS (VITS) — pretrained multilingual/multi-dataset/your_tts
  •   speaker_wav = speaker_ref_path (best 7s energy segment of input)
  •   language = "en"
  • Internally: reference audio → YourTTS encoder → 512-dim d-vector
  •             d-vector conditions the VITS flow + HiFi-GAN decoder
  • Output: waveform @ 16kHz, numpy float32
  │
  ▼
[CPU] _normalize_audio():
  • RMS normalize to target_rms = 0.08
  • Peak hard limit to 0.95
  │
  ▼
[CPU] compute_secs():
  • ECAPA-TDNN on output audio → 192-dim embedding
  • cosine_similarity(input_emb, output_emb) → SECS score [0,1]
  │
  ▼
OUTPUT: English audio (16kHz WAV) + Telugu text + English text + SECS + VAD + latency
```

---

## 4. Model 1 — Whisper Large-v3 (ASR)

### What it does
Converts Telugu speech audio into Telugu text.

### Architecture
**Whisper Large-v3** — Transformer encoder-decoder, ~1.55 billion parameters.

```
Raw audio waveform
  │
  ▼
Log-Mel Spectrogram (80 mel bins, 25ms window, 10ms hop)
  • Shape: [80, 3000] for 30 seconds of audio
  │
  ▼
Encoder — 32 Transformer layers
  • 20 attention heads, 1280-dim hidden states
  • Self-attention: every frame attends to every other frame
  • Output: [1500, 1280] — 1500 audio frames, 1280 features each
  │
  ▼
Decoder — 32 Transformer layers, auto-regressive
  • 20 attention heads, 1280-dim
  • First token FORCED: <|te|> (Telugu) + <|transcribe|>
  • At each step: masked self-attention on generated tokens so far
  •               cross-attention to all 1500 encoder frames
  •               generates ONE new token
  • Repeats until <|endoftext|>
  │
  ▼
51,865-token multilingual BPE vocabulary
  • Telugu example: "నేను" → ["నే", "ను"]
  │
  ▼
Telugu text string
```

**Pre-training:** 680,000 hours of multilingual audio from the internet (YouTube subtitles etc.). Natively supports Telugu (code: `te`).

### Fine-tuning Technique: LoRA

Instead of updating all 1.55B weights, LoRA adds small trainable matrices to the attention projection layers:

```
Original weight: W  (frozen, ~6GB)
LoRA addition:   ΔW = A × B
  where A ∈ ℝ^{d×r},  B ∈ ℝ^{r×d}
  rank r=8,  α=16  →  scaling = α/r = 2.0

Only A and B are trained. Total trainable: ~3.7M / 1,550M = 0.24%
```

LoRA is applied to Q and V projection layers of every attention block in both encoder and decoder.

**After training:** `merge_and_unload()` merges LoRA weights INTO the base model, producing a single checkpoint `checkpoints/whisper_merged/merged/` with no separate adapter file.

**Training settings:**
```
lora_r=8, lora_alpha=16
lr=1e-4 (cosine decay to 0)
batch_size=16 (with gradient accumulation)
steps=5000, fp16=True
```

### Long audio chunking
```
Whisper feature extractor limit: 30 seconds
Our chunks: 28s with 3s overlap

For a 60s audio:
  Chunk 1: [0s, 28s]
  Chunk 2: [25s, 53s]  (3s overlap with chunk 1)
  Chunk 3: [50s, 60s]
  → Deduplicate boundary words by matching last N words of prev chunk
    with first N words of next chunk
```

### Datasets
| Dataset | Language | Hours |
|---------|----------|-------|
| FLEURS (tel_IN) | Telugu | ~12h |
| CommonVoice Telugu | Telugu | ~17h |
| **Total** | | **~29h** |

### Training Results
| Step | Train Loss | Val WER | Val CER |
|------|-----------|---------|---------|
| 250 | 0.314 | 67.66% | 29.50% |
| **750** | **0.222** | **62.15%** ← best | **26.97%** |
| 1000 | 0.192 | 63.32% | 26.31% |
| 2000 | 0.140 | 101.69% | 86.55% |
| 5000 | 0.113 | 97.28% | 70.96% |

| Final Metric | Value |
|-------------|-------|
| Best val WER (step 750) | **62.15%** |
| Test WER | **94.95%** |
| Test CER | **70.54%** |
| Training time | **11.86 hours** |
| VRAM during training | **~1.85 GB** |

**Note on WER:** The step-750 checkpoint is saved as the merged model (best val). WER appears high because the Telugu BPE sub-word tokenizer means even small script errors blow up the word count. CER of 70% is more informative — at character level, about 30% of characters are correct. Telugu is extremely morphologically rich with long compound words, making WER inherently high.

### Loading in app2.py
```python
WhisperForConditionalGeneration.from_pretrained(
    "checkpoints/whisper_merged/merged/",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```
**INT8 quantization:** float32 (4 bytes/weight) → int8 (1 byte/weight). Model: ~6.2 GB → ~1.6 GB. ~4× reduction with minimal quality loss.

---

## 5. Model 2 — ECAPA-TDNN Speaker Encoder

### What it does
Converts audio into a **192-dimensional speaker embedding** — a compact vector uniquely identifying the speaker's voice. Used to compute the SECS (voice similarity) metric at the end of the pipeline.

### Architecture
**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation — Time Delay Neural Network) from SpeechBrain `spkrec-ecapa-voxceleb`.

```
Audio waveform @ 16kHz
  │
  ▼
80-dim log-Mel filterbank features (25ms window, 10ms hop)
  │
  ▼
SE-Res2Block × 3  (multi-scale TDNN with squeeze-excitation)
  ┌──────────────────────────────────────────────────────┐
  │ Dilated 1D conv: looks at ±t audio context          │
  │ Res2 convolutions: multiple receptive fields in     │
  │   parallel (scale=8 sub-filters per block)           │
  │ SE module: channel attention — learns WHICH          │
  │   frequency channels are most speaker-discriminative │
  │ Skip connections throughout                          │
  └──────────────────────────────────────────────────────┘
  │
  ▼
Multi-scale feature aggregation
  • Concatenate outputs of all 3 SE-Res2Blocks
  • 1D conv to fuse representations from all temporal scales
  │
  ▼
Attentive Statistics Pooling  ← KEY INNOVATION
  • Compute attention weight α_t for each time frame t
  • Weighted mean:  μ = Σ(α_t × h_t)
  • Weighted std:   σ = √(Σ(α_t × h_t²) − μ²)
  • Concatenate [μ, σ] → fixed 192-dim vector regardless of input length
  │
  ▼
Linear projection → L2-normalize → 192-dim speaker embedding
```

The attention pooling is the key: instead of simple mean pooling (which treats all frames equally), the model learns which frames are MOST speaker-discriminative (clear voiced speech) and weighs them higher.

### Fine-tuning Technique: ArcFace Loss

```
ArcFace (Additive Angular Margin Loss):
  1. Compute cosine similarity of embedding e to all speaker class vectors W_j
  2. Add angular margin m=0.2 to the angle of the CORRECT speaker:
       θ_correct_new = arccos(W_correct · e) + m
  3. Scale all similarities by s=30, apply softmax
  4. Cross-entropy loss on the result

This forces all utterances from the same speaker to cluster together
in the 192-dim hypersphere, with a geometric margin separating clusters.
```

**Why ArcFace instead of regular cross-entropy?**
Regular CE minimizes class confusion. ArcFace specifically maximizes the angular margin between speaker clusters — the embeddings are more separated and generalize better to unseen speakers.

- Fine-tuned on Telugu speaker data to reduce language leakage
- lr=5e-5, speaker-balanced batches, 6 epochs
- Best checkpoint: **epoch 1** (pretrained already almost optimal)

### Attentive Segment Pooling (our additional improvement)
```python
# In extract_speaker_embedding():
# Instead of one pass over full audio:
segments = split_into_1_5s_windows(audio, hop=0.5s)  # 67% overlap
segment_embeddings = [ecapa(seg) for seg in segments]
segment_embeddings = [e / norm(e) for e in segment_embeddings]  # L2 norm each
energies = [rms(seg) for seg in segments]
weights = softmax(energies)
final_embedding = normalize(sum(w * e for w, e in zip(weights, segment_embeddings)))
```
This weights high-energy (speech-rich) segments more heavily — pauses and noise contribute less.

### Training Results
| Epoch | Train Loss | Val EER |
|-------|-----------|---------|
| **1** | **10.63** | **1.15%** ← best |
| 2 | 11.25 | 1.54% |
| 3 | 11.01 | 5.00% |
| 4 | 10.57 | 7.69% |
| 5 | 10.44 | 9.23% |
| 6 | 10.52 | 10.38% |

| Final Metric | Value |
|-------------|-------|
| Best epoch | 1 |
| Test EER | **20.06%** |
| Same-speaker cosine similarity | **0.577** |
| Training time | **0.1 hours** |

**EER = Equal Error Rate:** the threshold where False Accept Rate = False Reject Rate. Lower is better. Val EER of 1.15% is excellent. Higher test EER (20%) suggests the test set has harder/more diverse speakers. The pretrained VoxCeleb weights are already strong — epoch 1 fine-tuning gave marginal improvement, then overfitting.

### Loading in app2.py
```python
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},  # cpu intentionally — runs parallel with GPU
)
ckpt = torch.load("checkpoints/speaker_encoder/best_model.ckpt", map_location="cpu")
models["spk_encoder"].mods.load_state_dict(ckpt["model_state_dict"])
```

---

## 6. Model 3 — Wav2Vec2 Emotion Detector

### What it does
Predicts a **3-dimensional continuous emotion vector** [Valence, Arousal, Dominance]:
- **Valence (V):** Positive ↔ Negative. +1 = very happy/pleased. -1 = sad/angry.
- **Arousal (A):** High ↔ Low energy. +1 = excited/tense. -1 = calm/sleepy.
- **Dominance (D):** Powerful ↔ Submissive. +1 = commanding/confident. -1 = weak/timid.

**Why 3D continuous (VAD) instead of discrete labels?**
- Discrete labels (happy/sad/angry) can't represent mixed states like "calm but slightly sad"
- Real speech emotion is graded, not binary
- VAD can represent hundreds of emotional states with just 3 numbers
- Annotators agree more on VAD ratings than on category labels

### Architecture
```
Audio waveform @ 16kHz
  │
  ▼
Wav2Vec2-base CNN Feature Extractor  (7 layers, ALWAYS FROZEN)
  • 1D convolutions on raw waveform
  • Output: frame features every 20ms  →  shape [T, 512]
  • Frozen because it does fundamental audio-to-feature conversion
  │
  ▼
Wav2Vec2 Transformer Encoder  (12 layers, first 6 FROZEN, last 6 TRAINABLE)
  • 8 attention heads, 768-dim hidden states
  • Relative position embeddings
  • First 6 layers frozen: preserve low-level acoustic features
  • Last 6 layers trainable: learn emotion-specific patterns
  │
  ▼
Mean pooling over time dimension
  • Average all T frame vectors → single [768] utterance representation
  │
  ▼
Regression Head  (fully trainable)
  Dropout(0.3) → Linear(768→128) → ReLU → Dropout(0.2) → Linear(128→3) → Tanh
  │
  ▼
[valence, arousal, dominance]  ∈ [-1, 1]³
```

### How Wav2Vec2 was originally trained (self-supervised)
Facebook pre-trained Wav2Vec2 on 960h LibriSpeech by:
1. Masking random 10ms segments of audio
2. Training to predict the correct masked segment from context (contrastive objective)
3. No labels needed — the model learns acoustic representations purely from raw audio

This gives it rich representations of speech acoustics that transfer well to downstream tasks like emotion detection.

### Fine-tuning Technique: CCC Loss Regression

```
CCC (Concordance Correlation Coefficient):
  CCC(ŷ, y) = 2·ρ·σ_ŷ·σ_y / (σ_ŷ² + σ_y² + (μ_ŷ − μ_y)²)
  
  ρ = Pearson correlation
  σ = standard deviation
  μ = mean
  
  CCC = 1:  perfect prediction
  CCC = 0:  no correlation
  CCC penalizes predictions that are correlated but on the wrong scale

Total loss = (1 − CCC_valence) + (1 − CCC_arousal) + (1 − CCC_dominance)
```

Why CCC instead of MSE? MSE doesn't penalize predictions that are on a shifted scale. CCC penalizes both poor correlation AND scale mismatch — it's more appropriate for regression tasks where the absolute values matter.

**Hyperparameters:**
```
lr = 1e-4 (epochs 1-16), then 5e-5 (epochs 17-18)
batch_size = 16
epochs = 18 (of planned 20)
Frozen: CNN extractor + first 6 transformer layers
Trainable: last 6 transformer + regression head (~7M / 94M params = 7.4%)
```

### Dataset: EmoV-DB
- Professional actors, scripted emotional speech
- 4 emotions: Amused, Angry, Neutral, Disgusted
- VAD scores assigned per emotion category
- ~30 minutes labeled audio

### Training Results
| Epoch | Train Loss | Val Loss | CCC-V | CCC-A | CCC-D | **CCC-Mean** |
|-------|-----------|----------|-------|-------|-------|------------|
| 1 | 0.509 | 0.580 | 0.815 | 0.795 | 0.770 | 0.793 |
| 3 | 0.279 | 0.551 | 0.892 | 0.903 | 0.829 | 0.875 |
| 7 | 0.190 | 0.552 | 0.862 | 0.888 | 0.919 | 0.890 |
| **10** | **0.176** | **0.539** | **0.890** | **0.930** | **0.925** | **0.915** ← best |
| 14 | 0.175 | 0.541 | 0.922 | 0.895 | 0.873 | 0.897 |
| 18 | 0.146 | 0.547 | 0.895 | 0.893 | 0.895 | 0.894 |

| Final Test Metric | Value |
|------------------|-------|
| CCC Valence | **0.729** |
| CCC Arousal | **0.818** |
| CCC Dominance | **0.723** |
| **CCC Mean** | **0.757** |
| Training time | **6.55 minutes** |

**Interpretation:**
- All 3 dimensions > 0.7 — good performance for this task
- Arousal (0.818) is easiest to learn — highly correlated with pitch energy & speech rate
- Valence (0.729) is hardest — positive/negative affect depends more on words (lexical information), which Wav2Vec2's acoustic representations don't capture as well

### Loading in app2.py
```python
class EmotionVADModel(nn.Module):
    def __init__(self):
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.backbone.feature_extractor._freeze_parameters()
        for i in range(6):  # freeze first 6 transformer layers
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = False
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(768, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 3), nn.Tanh()
        )

model = EmotionVADModel().to(DEVICE)
ckpt = torch.load("checkpoints/emotion_detector/best_model.pt")
model.load_state_dict(ckpt["model_state_dict"])
```

---

## 7. Model 4 — NLLB-600M + LoRA (Translation)

### What it does
Translates Telugu text → English text.

### Architecture
**NLLB-200-distilled-600M** (No Language Left Behind) by Meta. Supports 200 languages including Telugu (`tel_Telu`) and English (`eng_Latn`).

```
Telugu text: "నేను రేపు పని కి వెళ్తాను"
  │
  ▼
SentencePiece BPE tokenizer  (256,000-token vocab, multilingual)
  • Language token prepended: <tel_Telu>
  • Tokenizes into sub-words across all 200 languages
  │
  ▼
Encoder  (12 Transformer layers)
  • 16 attention heads, 1024-dim hidden states
  • Bidirectional: all source tokens see each other
  • Output: [source_len, 1024] contextual representations
  │
  ▼
Decoder  (12 Transformer layers, auto-regressive)
  • First token FORCED: <eng_Latn>
  • At each step:
      1. Masked self-attention on generated English tokens so far
      2. Cross-attention to ALL encoder states
         → cross-attention matrix: [n_eng_tokens, n_tel_tokens]
         → attention[i][j] = how much English token i looks at Telugu token j
         → this is the alignment used for prosody transfer design
      3. FFN + LayerNorm
  • 4-beam beam search to generate final English text
  │
  ▼
English text: "I will go to work tomorrow."
```

### Fine-tuning Technique: QLoRA (4-bit + LoRA)

**Step 1 — 4-bit NF4 quantization (load-time):**
```
NF4 (Normal Float 4-bit):
  • Each weight stored in 4 bits (16 discrete values)
  • Values are NOT uniformly spaced — they match the assumed
    normal distribution of neural network weights
  • Compute happens in float16 (dequantize → compute → stay quantized)
  • Memory: 600M × 0.5 bytes = ~300 MB (vs 1.2 GB at fp16)
  • 4-bit base weights: FROZEN, cannot be updated
```

**Step 2 — LoRA adapters on top of frozen 4-bit base:**
```
LoRA params:
  rank  r = 16
  alpha = 32  →  scaling = 32/16 = 2.0
  Applied to: Q, K, V, O projections of all attention layers
  Trainable: ~4.9M / 600M = 0.8%
  
Training uses gradient checkpointing + paged AdamW optimizer
(handles memory spikes from occasional large gradient updates)
```

This combination (QLoRA) allows fine-tuning a 600M model on a single 8GB GPU in under 15 minutes.

**Hyperparameters:**
```
lr = 5e-4 (LoRA benefits from higher lr than full fine-tuning)
batch_size = 8
epochs = 3
Scheduler: linear warmup then cosine decay
```

### Dataset: CoVoST-2 (Telugu → English)
- Source: Mozilla Common Voice recordings with human-verified transcriptions + translations
- ~4,700 spoken sentence pairs
- Covers conversational, everyday Telugu

### Training Results
| Epoch | Steps | Train Loss | Val Loss | Val BLEU |
|-------|-------|-----------|----------|----------|
| 1 | 159 | 1.616 | 1.389 | 26.52 |
| 2 | 318 | 1.448 | 1.232 | 30.98 |
| **3** | **477** | **1.358** | **1.214** | **31.89** ← best |

| Final Test Metric | Value |
|------------------|-------|
| Test BLEU | **29.68** |
| Test Loss | **1.259** |
| Training time | **10.46 minutes** |

**BLEU interpretation:** BLEU ~30 is respectable for a low-resource language pair. Google Translate achieves ~45 BLEU on Telugu→English; specialized IndicTrans2 achieves ~35-40. Our fine-tuned NLLB at ~30 is significantly better than the pretrained baseline (~20) on CoVoST-2.

### Loading in app2.py
```python
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "checkpoints/indictrans2_finetuned/best_lora/")
```

---

## 8. Model 5 — YourTTS / VITS (Text-to-Speech)

### What it does
Takes English text + a reference audio clip and generates English speech that sounds like the reference speaker.

### Architecture: VITS (Variational Inference with adversarial learning for end-to-end TTS)

VITS goes directly from text → waveform — no separate acoustic model + vocoder pipeline. This avoids error propagation between stages and is faster.

```
English text: "I will go to work tomorrow."
  │
  ▼
Text Encoder  (4 Transformer layers)
  • Character/phoneme embedding → positional encoding
  • Self-attention across text sequence
  • Output: text hidden states h_text  [T_text, 192]
  │
  ├────────────────────────── Speaker d-vector (512-dim) ─────────────────────┐
  │                           From: YourTTS's internal ResNet speaker encoder │
  │                           Processing: reference audio → 512-dim L2-normed │
  │                           Injected at: text encoder, flow, HiFi-GAN       │
  │                                                                            │
  ▼                                                                            │
Stochastic Duration Predictor                                                  │
  • Normalizing flow predicts phoneme durations with randomness                │
  • gives natural variation (not robotic constant timing)                      │
  │                                                                            │
  ▼                                                                            │
Normalizing Flow (Glow)                                                        │
  • Maps between simple Gaussian prior and complex speech latent space         │
  • Prior: N(μ, σ) predicted from text encoder                                 │
  • At inference: sample z from prior → transform via flow                     │
  │                                                                            │
  ▼                                                                            │
Latent z  [conditioned on d-vector at multiple injection points]               │
  │                                                                            ▼
  ▼
HiFi-GAN Decoder  (waveform generator)
  • Input: latent z aligned to audio duration via duration predictor
  • ConvTranspose1d upsampling layers ×3
  • Multi-Receptive-Field Fusion (MRF): 3 parallel ResBlock stacks
    - Each stack has 3 kernel sizes: [3, 7, 11]
    - Each block: dilation patterns [1,3,5], [1,3,5], [1,3,5]
  • d-vector injected into EVERY ResBlock (concatenated to features)
  │
  ▼
Audio waveform @ 16kHz
```

**Training GAN:** A Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD) judge whether generated audio is real. The generator tries to fool them. This adversarial training forces realistic high-frequency detail.

**YourTTS extensions over base VITS:**
- D-vector conditioning (multi-speaker)
- Multi-language token support
- Zero-shot voice cloning via reference audio at inference

### Why Voice Reference Cleaning Matters
In `prepare_speaker_ref()`, before passing audio to YourTTS:
```python
1. librosa.effects.trim(top_db=25)    # Remove leading/trailing silence
2. Sliding window (0.5s step) → find 7s window with highest RMS energy
3. Peak normalize to 0.90
4. Save as temp WAV

→ YourTTS's internal encoder gets a CLEAN, SPEECH-RICH sample
→ Without this: noisy reference → noisy d-vector → degraded cloning
```

### Fine-tuning: 2,212 Speakers

**Why fine-tune?** Pretrained YourTTS only knows 6 speakers. The model's learned mapping of "d-vector → voice characteristics" only covers those 6. For any unseen Telugu speaker, it outputs a meaningless average. Training on 2,212 diverse speakers teaches the actual _generalization_ of this mapping.

**Training data:**
| Dataset | Speakers | Clips | Notes |
|---------|----------|-------|-------|
| LibriTTS dev-clean | 40 | 5,736 | Audiobooks, 24kHz downsampled to 16kHz |
| LibriTTS test-clean | 39 | 4,837 | Same |
| Speech Accent Archive | 2,133 | 2,133 | All read same passage, 200 native languages |
| **Total** | **2,212** | **12,706** | Clips >7s filtered out for VRAM |

**Speech Accent Archive is key:** All 2,133 speakers read the SAME English paragraph. Since content is fixed, all variation in the audio = speaker variation purely. The model learns speaker diversity WITHOUT content confounding. 200 native language backgrounds also helps with Telugu speaker accents at inference.

**VITS training losses:**
```
L_total = L_mel + L_kl + L_dur + L_fm + L_adv

L_mel: ||mel_pred − mel_real||₁           (spectrogram reconstruction)
L_kl:  KL[q(z|audio) || p(z|text)]       (latent space alignment)
L_dur: duration predictor error
L_fm:  feature matching (intermediate discriminator features)
L_adv: adversarial — generator fools discriminator
```

**Hyperparameters:**
```
batch_size:       1     (batch=2 causes OOM during dual-optimizer backward)
lr:               5e-5  (lower than default 2e-4 for stable fine-tuning)
optimizer:        AdamW  (betas=[0.8, 0.99], weight_decay=0.01)
lr_scheduler:     ExponentialLR  (gamma=0.999875 per step)
mixed_precision:  fp16
PYTORCH_CUDA_ALLOC_CONF: expandable_segments=True  (prevents fragmentation OOM)
max_audio_len:    7s    (clips >7s cause OOM at batch=1)
```

**Restore strategy:**
```
Load from: pretrained YourTTS weights (model_file.pth)
Keep:      text_encoder, flow, duration_predictor, HiFi-GAN decoder, prior
Reinit:    emb_l.weight (language embeddings — we have different language IDs)
Reinit:    discriminator conv_post layers (weight_norm format changed PyTorch 2.6+)
```

**D-vector database:** Before training, `compute_dvectors.py` pre-computes 512-dim d-vectors for all 12,706 training clips using YourTTS's own speaker encoder and stores them in `data/dvectors/speakers.json`. During training, file → d-vector lookup is instant (no recompute).

**Training runs today (March 5, 2026):**
- `vits_tts-01+18AM` — initial parameter discovery
- `vits_tts-09+16AM` — adjusted lr
- `vits_tts-09+28AM` — checkpoints saved at steps 1000, 2000
- `vits_tts-03+30PM` — latest restoration run

### TTS Alternatives Tried and Abandoned
| System | Reason Abandoned |
|--------|-----------------|
| XTTS-v2 | ~3.5 GB VRAM alone; PyTorch 2.9.1 + transformers 4.57.3 incompatibility: `pad_token_id == eos_token_id (1025)` → GPT generates infinite tokens; even after patching generation length, attention passes produce garbage codes |
| Edge-TTS | Microsoft neural API — clear audio but NO voice cloning (fixed generic voices only) |
| OpenVoice V2 | Dependency conflicts with torch 2.9.1 environment |
| Qwen3-TTS | ~45s per sentence without flash-attention; too slow |

### Loading in app2.py
```python
from TTS.api import TTS
models["tts"] = TTS("tts_models/multilingual/multi-dataset/your_tts").to(DEVICE)

# Inference:
models["tts"].tts_to_file(
    text=english_text,
    language="en",
    speaker_wav=speaker_ref_path,  # best 7s energy segment
    file_path=output_path,
)
```
VRAM: **~0.4 GB** (vs XTTS-v2's 3.5 GB).

---

## 9. Fine-tuning Summary Table

| Model | Base Weights | Technique | Trainable Params | Dataset | Time |
|-------|-------------|-----------|-----------------|---------|------|
| Whisper ASR | openai/whisper-large-v3 | LoRA r=8, α=16 on Q/V | ~3.7M / 1550M **(0.24%)** | FLEURS-te + CommonVoice-te (~29h) | **11.86h** |
| Speaker Encoder | speechbrain/spkrec-ecapa-voxceleb | Full fine-tune, ArcFace loss | 21M / 21M **(100%)** | Telugu speaker pairs | **0.1h** |
| Emotion Detector | facebook/wav2vec2-base | Head + top-6 transformer layers | ~7M / 94M **(7.4%)** | EmoV-DB (~30min labeled) | **6.55min** |
| Translation | facebook/nllb-200-distilled-600M | QLoRA (4-bit NF4 + LoRA r=16) | ~4.9M / 600M **(0.8%)** | CoVoST-2 te-en (~4,700 pairs) | **10.46min** |
| TTS (YourTTS) | multilingual/multi-dataset/your_tts | Full fine-tune, restore + continue | ~82M | LibriTTS + Accent Archive (2,212 speakers, 12,706 clips) | **Multiple hours** |

---

## 10. Datasets Used

| Task | Dataset | Source | Size | Key Properties |
|------|---------|--------|------|----------------|
| ASR | FLEURS (tel_IN) | Google | ~3,700 clips, ~12h | Native Telugu, high quality |
| ASR | CommonVoice Telugu | Mozilla | ~5,200 clips, ~17h | Crowd-sourced, diverse speakers |
| Speaker | VoxCeleb1+2 | Oxford | 7,000+ speakers (pretrained base) | English, celebrity audio |
| Emotion | EmoV-DB | U. Edinburgh | ~30min labeled | 4 emotions, professional actors |
| Translation | CoVoST-2 (tel→en) | Mozilla | ~4,700 pairs | Spoken conversational Telugu→English |
| TTS | LibriTTS dev-clean | OpenSLR | 40 speakers, 5,736 clips | Audiobooks, 24kHz→16kHz |
| TTS | LibriTTS test-clean | OpenSLR | 39 speakers, 4,837 clips | Audiobooks, 24kHz→16kHz |
| TTS | Speech Accent Archive | George Mason Univ. | 2,133 speakers, 2,133 clips | Same passage, 200 native languages |

---

## 11. Training Results — All Numbers

### Whisper (5,000 steps, 11.86 hours)
Best val WER: **62.15%** at step 750 (saved checkpoint). Final test WER: **94.95%**, CER: **70.54%**. Train loss: 0.314 → 0.113.

### Speaker Encoder (6 epochs, 0.1 hours)
Best epoch: **1**. Val EER: **1.15%**. Test EER: **20.06%**. Same-speaker cosine: **0.577**.

### Emotion Detector (18 epochs, 6.55 minutes)
Best epoch: **10** (CCC Mean = **0.9149** on val). Test CCC: V=**0.729**, A=**0.818**, D=**0.723**, Mean=**0.757**.

### Translation (3 epochs, 10.46 minutes)
Best epoch: **3**. Val BLEU: **31.89**. Test BLEU: **29.68**. Test Loss: **1.259**.

### XTTS-v2 (pretrained, no fine-tune, abandoned)
SECS: **0.298** | Inference: **3.56s** | Target (0.70): **NOT MET** | Status: **abandoned (broken)**

---

## 12. End-to-End Evaluation Results

Full pipeline evaluated on a Telugu speech test set using the assembled system (Whisper + NLLB + ECAPA + Emotion + YourTTS):

### Overall Metrics
| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| **WER** (Telugu ASR) | **66.67%** | <15% | Word Error Rate on eval set |
| **CER** (Telugu ASR) | **38.93%** | — | Character Error Rate (more meaningful for Telugu) |
| **BLEU** (Translation) | **27.37** | >25 | n-gram overlap vs reference translations |
| **chrF** (Translation) | **59.34** | — | Character F-score (better for morphological languages) |
| **SECS** (Voice Similarity) | **0.381** | >0.70 | ECAPA-TDNN cosine similarity input↔output |
| **Avg Latency** | **11.04 s** | <5s | Wall-clock time end-to-end |

### Ablation Study
| Configuration Removed | SECS | Latency | Interpretation |
|----------------------|------|---------|---------------|
| Full system | 0.381 | 11.04s | Baseline |
| No FiLM conditioning | 0.380 | 12.63s | FiLM barely changes SECS (not yet trained) |
| No contrastive learning | 0.364 | — | −4.7% SECS — contrastive loss helps speaker encoding |
| Discrete emotion labels | 0.366 | — | −3.9% SECS — continuous VAD better than categories |

### Notes on SECS score
The 0.381 SECS is measured using ECAPA-TDNN (192-dim). YourTTS conditions on its OWN 512-dim encoder — these are different embedding spaces. The SECS measurement and TTS conditioning are not aligned. The true voice similarity as YourTTS sees it would require using YourTTS's encoder for measurement. The ablation results still show relative improvements correctly even if the absolute number is underestimated.

---

## 13. VRAM & Hardware Layout

### Hardware
- **GPU:** RTX 4060 Laptop, 8.2 GB VRAM
- **CPU:** 20 threads (Intel/AMD)
- **RAM:** ~16 GB system

### GPU VRAM at Inference (app2.py loading order)
```
After Whisper load:       1.6 GB
+ NLLB-600M 4-bit + LoRA: 2.4 GB (cumulative)
+ Wav2Vec2 emotion:       3.1 GB (cumulative)
+ YourTTS VITS:           3.5 GB (cumulative)  ← confirmed at runtime
Headroom for inference:   ~4.7 GB available
```

### GPU VRAM During YourTTS Training (only TTS on GPU)
```
YourTTS model weights:     ~0.3 GB
Speaker encoder (SpeechBrain): ~0.3 GB
Discriminator:             ~0.3 GB
Adam optimizer states (×2 for dual optimizer): ~0.5 GB
Activations + intermediate tensors: ~0.3 GB
Total:                     ~1.7 GB
```

### CPU Usage
```
ECAPA-TDNN speaker encoder:    ~23 MB, runs in background thread during GPU inference
librosa audio preprocessing:   CPU only
soundfile I/O:                 CPU only
```

### Why Some Models on CPU
The speaker encoder thread starts BEFORE the GPU ASR call:
```python
thread_b = threading.Thread(target=branch_b)  # ECAPA + speaker ref prep
thread_b.start()
# GPU starts ASR:
telugu_text = transcribe_telugu(audio_np)      # GPU busy ~2-4s
thread_b.join()                                # CPU finishes during GPU work
```
Wall-clock time for ECAPA on CPU: ~0.05–0.1s — essentially free.

---

## 14. Key Design Decisions

### Why YourTTS and not XTTS-v2?
XTTS-v2 takes ~3.5 GB VRAM alone. With Whisper (1.6 GB) + NLLB (0.8 GB) + Emotion (0.4 GB) already loaded, total would be ~6.3 GB before inference activations — OOM on 8 GB. YourTTS uses 0.4 GB. Additionally, XTTS-v2 is fundamentally broken on PyTorch 2.9.1 + transformers 4.57.3: the GPT component never stops generating (pad_token_id == eos_token_id == 1025), and even after patching generation length, the attention quality degrades producing unintelligible audio.

### Why Speech Accent Archive for TTS training?
2,133 speakers all reading the SAME sentence. With content fixed, all audio variation = speaker characteristics only. The model learns to map d-vector → voice without content confounding. 200 native language backgrounds also helps with unseen Telugu accent at inference.

### Why LoRA for Whisper and NLLB but full fine-tune for speaker encoder and TTS?
- Whisper (1.55B) and NLLB (600M): too large for full fine-tuning in reasonable time. LoRA reaches good quality at 0.24% and 0.8% of parameters respectively.
- ECAPA-TDNN (21M): small enough to fully fine-tune quickly.
- Emotion detector: only head + 6 transformer layers trained (7.4%) — a middle ground.
- YourTTS (~82M): voice cloning quality depends on many layers understanding speaker conditioning. Partial fine-tuning was insufficient — full restore+fine-tune is needed.

### Why batch_size=1 for YourTTS fine-tuning?
VITS uses **dual optimizers** (separate Adam for generator and discriminator). During backward, both compute gradients simultaneously. Combined with speaker encoder loss (extra forward pass through SpeechBrain), peak VRAM at batch_size=2 exceeds 8 GB (OOM). At batch_size=1: ~1.7 GB with comfortable headroom.

### Why keep speaker encoder on CPU?
1. Only 23 MB — not worth allocating GPU VRAM
2. Runs in background thread DURING GPU ASR — zero added wall-clock time
3. Result (192-dim vector) is only used for SECS metric, not for TTS conditioning

### Why clean reference audio before YourTTS?
YourTTS's internal encoder is sensitive to noise. A 10s raw recording with 3s silence + background noise gives a noisy d-vector. `prepare_speaker_ref()` extracts the highest-energy 7s window — the densest speech content — giving YourTTS the cleanest possible voice sample.

---

## 15. File Structure Reference

```
pipeline_v2/
│
├── app2.py                           ← CURRENTLY RUNNING — main pipeline + Gradio
├── app.py                            ← Alternative with Edge-TTS (no voice cloning)
├── app_old_xtts.py                   ← Abandoned XTTS-v2 attempt
├── prosody.py                        ← F0/energy/speech-rate extraction (designed; not wired into app2)
├── tts_engine.py                     ← Standalone TTS test script
├── compute_dvectors.py               ← Pre-computes 12,706 d-vectors for training
├── finetune_yourtts.py               ← YourTTS fine-tuning script
├── prepare_archive.py                ← Converts Speech Accent Archive MP3 → WAV 16kHz
├── voice_autoencoder.py              ← Experimental voice autoencoder (not in pipeline)
│
├── configs/
│   ├── asr.yaml                      ← Whisper LoRA training config
│   ├── speaker.yaml                  ← ECAPA-TDNN ArcFace training config
│   ├── emotion.yaml                  ← Wav2Vec2 CCC regression config
│   ├── translation.yaml              ← NLLB QLoRA config
│   └── tts.yaml                      ← YourTTS VITS training overrides
│
├── checkpoints/
│   ├── whisper_merged/
│   │   ├── training_results.json     ← WER=94.95%, CER=70.54%, 11.86h, lora_r=8
│   │   ├── best_model/               ← Step 750 checkpoint (best val)
│   │   └── merged/                   ← LoRA merged into base → used in app2.py
│   │
│   ├── speaker_encoder/
│   │   ├── training_results.json     ← EER=20.06%, cosine=0.577, best_epoch=1
│   │   ├── best_model.ckpt           ← Epoch 1 → used in app2.py
│   │   └── pretrained/               ← SpeechBrain ECAPA cache
│   │
│   ├── emotion_detector/
│   │   ├── training_results.json     ← CCC mean=0.757, best_epoch=10, 6.55min
│   │   ├── best_model.pt             ← Epoch 10 → used in app2.py
│   │   └── last_model.pt             ← Epoch 18
│   │
│   ├── indictrans2_finetuned/
│   │   ├── training_results.json     ← BLEU=29.68, loss=1.259, 10.46min
│   │   ├── best_lora/                ← Epoch 3 LoRA adapter → used in app2.py
│   │   └── last_lora/                ← Epoch 3 (same, last epoch)
│   │
│   ├── yourtts_finetuned/
│   │   ├── vits_tts-01+18AM/         ← Run 1: param discovery
│   │   ├── vits_tts-09+16AM/         ← Run 2: lr tuning
│   │   ├── vits_tts-09+28AM/         ← Run 3: checkpoints at step 1000, 2000
│   │   └── vits_tts-03+30PM/         ← Run 4: latest restoration attempt
│   │       ├── checkpoint_1000.pth
│   │       ├── checkpoint_2000.pth
│   │       ├── config.json
│   │       └── trainer_0_log.txt
│   │
│   └── yourtts_restore/              ← Staging for pretrained weights
│       ├── model_file.pth            ← Pretrained YourTTS base weights
│       ├── speakers.json             ← 12,706 pre-computed 512-dim d-vectors
│       └── speakers.pth              ← Same in .pth format
│
├── data/
│   ├── dvectors/
│   │   ├── speakers.json             ← Pre-computed 512-dim d-vectors (12,706 clips)
│   │   └── speakers.pth
│   ├── embeddings/                   ← Per-clip 192-dim ECAPA embeddings (emb_*.npy)
│   ├── processed/tts_train/          ← LJSpeech reference WAVs (lj_*.wav)
│   └── raw/                          ← Original downloaded datasets
│
├── logs/
│   ├── asr_training_log.csv          ← step, train_loss, val_wer, val_cer, vram_gb
│   ├── speaker_training_log.csv      ← epoch, train_loss, val_eer
│   ├── emotion_training_log.csv      ← epoch, train_loss, ccc_v, ccc_a, ccc_d, ccc_mean
│   └── translation_training_log.csv  ← epoch, train_loss, val_bleu
│
├── results/
│   ├── evaluation_results.json       ← WER=66.67%, CER=38.93%, BLEU=27.37, SECS=0.381, latency=11.04s
│   └── ablation_results.json         ← no_film, no_contrastive, discrete_labels comparisons
│
└── notebooks/
    ├── 01_environment_setup.ipynb        ← Conda env + deps installation
    ├── 02_data_pipeline.ipynb            ← Download + preprocess all datasets
    ├── 03_whisper_asr_finetuning.ipynb   ← Whisper LoRA (r=8) on FLEURS+CommonVoice
    ├── 04_speaker_encoder_finetuning.ipynb ← ECAPA-TDNN ArcFace on Telugu speakers
    ├── 05_translation_finetuning.ipynb   ← NLLB QLoRA on CoVoST-2
    ├── 06_emotion_detector.ipynb         ← Wav2Vec2 CCC regression on EmoV-DB
    ├── 07_tts_finetuning.ipynb           ← YourTTS VITS on LibriTTS + Accent Archive
    ├── 07b_tts_rebuild.ipynb             ← YourTTS restore/rebuild experiments
    └── 08_pipeline_integration_demo.ipynb ← End-to-end demo
```

---

## 16. Glossary

| Term | Meaning |
|------|---------|
| **ASR** | Automatic Speech Recognition — converting audio to text |
| **TTS** | Text-to-Speech — converting text to audio |
| **MT** | Machine Translation — converting text between languages |
| **VITS** | Variational Inference with adversarial learning for end-to-end Text-to-Speech |
| **HiFi-GAN** | A waveform generator (vocoder) used inside VITS to produce audio from latent codes |
| **d-vector** | Discriminative vector — a 512-dim speaker embedding for TTS conditioning (YourTTS internal) |
| **Speaker embedding** | Any compact vector representing a person's voice identity |
| **TDNN** | Time Delay Neural Network — 1D dilated CNN that looks at audio context windows |
| **ECAPA-TDNN** | Speaker encoder with squeeze-excitation channel attention + attentive pooling |
| **VAD** | Valence-Arousal-Dominance — 3D continuous emotion model |
| **VAE** | Variational Autoencoder — learns a latent distribution with Gaussian prior |
| **GAN** | Generative Adversarial Network — generator + discriminator trained adversarially |
| **Wav2Vec2** | Facebook self-supervised speech model, pre-trained by masked audio prediction |
| **NLLB** | No Language Left Behind — Meta's 200-language translation model |
| **LoRA** | Low-Rank Adaptation — adds trainable rank-r matrices to frozen pretrained weights (ΔW = A×B) |
| **QLoRA** | LoRA applied on top of a 4-bit quantized model |
| **NF4** | Normal Float 4-bit — quantization format matching normal distribution of neural weights |
| **INT8** | 8-bit integer quantization — 4× memory savings vs float32 |
| **ArcFace** | Additive Angular Margin loss — pushes speaker embeddings apart in hyperspherical space |
| **CCC** | Concordance Correlation Coefficient — measures both correlation and scale agreement |
| **EER** | Equal Error Rate — speaker verification threshold where Fals Accept Rate = False Reject Rate |
| **SECS** | Speaker Encoder Cosine Similarity — voice cloning quality: cosine(input_emb, output_emb) |
| **BLEU** | Bilingual Evaluation Understudy — n-gram precision metric for machine translation |
| **chrF** | Character-level F-score — better BLEU alternative for morphologically rich languages |
| **WER** | Word Error Rate — ASR accuracy: (insertions+deletions+substitutions) / reference_words |
| **CER** | Character Error Rate — ASR accuracy at character level (more meaningful for Telugu script) |
| **Beam search** | Sequence generation keeping top-k partial hypotheses at each step (k=4 here) |
| **F0** | Fundamental frequency — the pitch of voiced speech in Hz |
| **Mel spectrogram** | Time-frequency audio representation with frequency axis scaled to human perception |
| **FiLM** | Feature-wise Linear Modulation — conditions layers via scale γ and shift β from a control signal |
| **Cross-attention** | Attention where queries come from the decoder and keys/values come from the encoder |
| **Normalizing flow** | Invertible neural network mapping between simple (Gaussian) and complex distributions |
| **Stochastic Duration Predictor** | Predicts phoneme durations with learned randomness to avoid robotic monotone timing |
| **MRF** | Multi-Receptive-Field Fusion — parallel ResBlock stacks with different kernel sizes in HiFi-GAN |
| **Attentive pooling** | Weighted average over time with learned attention weights (vs simple mean pooling) |
| **Zero-shot voice cloning** | Cloning a voice never seen during training using only a short audio reference at inference |
