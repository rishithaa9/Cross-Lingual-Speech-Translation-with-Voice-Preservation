#!/usr/bin/env python3
"""
TeluguVoiceBridge v2 — Web Application
Telugu → English Speech-to-Speech Translation

Pipeline:
  1. ASR (Whisper Large-v3 INT8) — Telugu speech → Telugu text
  2. Translation (NLLB-600M 4-bit + LoRA) — Telugu text → English text
  3. Speaker Encoder (ECAPA-TDNN on CPU) — extract speaker embedding
  4. Emotion Detector (Wav2Vec2 VAD) — extract valence/arousal/dominance
  5. TTS (YourTTS VITS) — English text → English speech (voice cloned from input)

Voice Preservation:
  The input Telugu audio is passed as speaker_wav to YourTTS, so the output
  English speech preserves the vocal characteristics of the original speaker.

Usage:
    cd /home/nibiru/Documents/sem6project/Speech2/pipeline_v2
    conda activate ml_env
    python app.py
"""

import os, gc, pathlib, time, threading, tempfile, math
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import librosa
import gradio as gr

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
BASE = pathlib.Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEMO_DIR = BASE / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Default LJSpeech reference audio pool (8-10s clips, pre-selected for quality)
DEFAULT_REF_PATHS = [
    str(BASE / "data" / "processed" / "tts_train" / "lj_002726.wav"),
    str(BASE / "data" / "processed" / "tts_train" / "lj_002595.wav"),
    str(BASE / "data" / "processed" / "tts_train" / "lj_004021.wav"),
    str(BASE / "data" / "processed" / "tts_train" / "lj_012147.wav"),
    str(BASE / "data" / "processed" / "tts_train" / "lj_003734.wav"),
]
DEFAULT_REF = DEFAULT_REF_PATHS[0]

# Audio normalization constants
TARGET_RMS = 0.08        # LJSpeech-level RMS for consistent output volume
PEAK_LIMIT = 0.95        # Hard peak limit to prevent clipping
REF_BEST_DUR = 7.0       # Ideal reference clip duration (seconds)
REF_MIN_DUR = 3.0        # Minimum reference clip duration
REF_MAX_DUR = 10.0       # Maximum reference clip duration

# Whisper chunking constants
WHISPER_MAX_SEC = 28     # Max seconds per chunk (< 30s Whisper limit)
WHISPER_OVERLAP_SEC = 3  # Overlap between chunks for continuity

print(f"Device: {DEVICE}")
print(f"Base: {BASE}")


# ══════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════

def load_all_models():
    """Load all 5 pipeline models. Returns a dict of models."""
    models = {}

    torch.cuda.empty_cache()
    gc.collect()

    # --- 1. Whisper ASR (INT8) ---
    print("Loading Whisper ASR...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers import BitsAndBytesConfig

    WHISPER_DIR = BASE / "checkpoints" / "whisper_merged" / "merged"
    if WHISPER_DIR.exists() and (WHISPER_DIR / "model.safetensors").exists():
        models["whisper"] = WhisperForConditionalGeneration.from_pretrained(
            str(WHISPER_DIR),
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        models["whisper_proc"] = WhisperProcessor.from_pretrained(str(WHISPER_DIR))
    else:
        models["whisper"] = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        models["whisper_proc"] = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    models["whisper"].eval()
    print(f"  ✓ Whisper loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # --- 2. Speaker Encoder (CPU — no GPU needed) ---
    print("Loading speaker encoder...")
    from speechbrain.inference.speaker import EncoderClassifier

    models["spk_encoder"] = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(BASE / "checkpoints" / "speaker_encoder" / "pretrained"),
        run_opts={"device": "cpu"},
    )
    best_spk = BASE / "checkpoints" / "speaker_encoder" / "best_model.ckpt"
    if best_spk.exists():
        ckpt = torch.load(best_spk, map_location="cpu", weights_only=False)
        models["spk_encoder"].mods.load_state_dict(ckpt["model_state_dict"])
        print("  ✓ Fine-tuned speaker weights loaded.")

    # --- 3. Translation (NLLB + LoRA) ---
    print("Loading NLLB translation...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    LORA_DIR = BASE / "checkpoints" / "indictrans2_finetuned" / "best_lora"

    models["trans"] = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
    )
    models["trans_tok"] = AutoTokenizer.from_pretrained(BASE_MODEL)

    if LORA_DIR.exists() and (LORA_DIR / "adapter_model.safetensors").exists():
        from peft import PeftModel
        models["trans"] = PeftModel.from_pretrained(models["trans"], str(LORA_DIR))
        print("  ✓ LoRA adapter loaded.")
    models["trans"].eval()
    print(f"  ✓ Translation loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # --- 4. Emotion Detector (Wav2Vec2 VAD) ---
    print("Loading emotion detector...")
    from transformers import Wav2Vec2Model

    class EmotionVADModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.backbone.feature_extractor._freeze_parameters()
            for i in range(6):
                for param in self.backbone.encoder.layers[i].parameters():
                    param.requires_grad = False
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3),
                nn.Tanh(),
            )

        def forward(self, input_values, attention_mask=None):
            outputs = self.backbone(input_values, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
            return self.head(pooled)

    models["emotion"] = EmotionVADModel().to(DEVICE)
    emo_ckpt = BASE / "checkpoints" / "emotion_detector" / "best_model.pt"
    if emo_ckpt.exists():
        ckpt = torch.load(emo_ckpt, map_location=DEVICE, weights_only=False)
        models["emotion"].load_state_dict(ckpt["model_state_dict"])
        print("  ✓ Fine-tuned emotion weights loaded.")
    models["emotion"].eval()
    print(f"  ✓ Emotion loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # --- 5. TTS (YourTTS — VITS-based, voice cloning) ---
    # YourTTS: 0.34 GB VRAM, 16kHz output, ~0.45s/sentence
    # Replaces broken XTTS-v2 (was 3.56 GB, garbled output)
    print("Loading YourTTS...")
    from TTS.api import TTS

    models["tts"] = TTS("tts_models/multilingual/multi-dataset/your_tts").to(DEVICE)
    models["tts_sr"] = 16000  # YourTTS native sample rate
    print(f"  ✓ YourTTS loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\n{'='*50}")
    print(f"All models loaded! Total VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"{'='*50}\n")

    return models


# ══════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════

import unicodedata, re


def normalize_telugu_text(text: str) -> str:
    """Normalize Telugu text: NFC normalization + whitespace cleanup."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def postprocess_english(text: str) -> str:
    """Capitalize first letter and ensure ending punctuation."""
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
    return text


def _transcribe_chunk(audio_np, sr=16000):
    """Transcribe a single audio chunk (must be <= 30s)."""
    proc = M["whisper_proc"]
    inputs = proc(audio_np, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    forced_decoder_ids = proc.get_decoder_prompt_ids(language="te", task="transcribe")

    with torch.no_grad(), torch.amp.autocast("cuda"):
        generated = M["whisper"].generate(
            input_features.half(),
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=444,  # 448 minus 4 decoder start tokens
        )

    text = proc.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip()


def transcribe_telugu(audio_np, sr=16000):
    """Whisper ASR: Telugu speech → Telugu text.

    Handles audio up to ~60s by splitting into overlapping chunks.
    Each chunk is <= 28s (within Whisper's 30s feature-extractor limit).
    Overlap of 3s ensures no words are lost at chunk boundaries.
    """
    total_samples = len(audio_np)
    total_sec = total_samples / sr

    # Short audio: single pass (no chunking needed)
    if total_sec <= WHISPER_MAX_SEC:
        text = _transcribe_chunk(audio_np, sr)
        return normalize_telugu_text(text)

    # Long audio: chunk with overlap
    chunk_samples = int(WHISPER_MAX_SEC * sr)
    overlap_samples = int(WHISPER_OVERLAP_SEC * sr)
    step = chunk_samples - overlap_samples

    transcripts = []
    offset = 0
    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk = audio_np[offset:end]

        # Skip very short trailing chunks (< 1s)
        if len(chunk) < sr:
            break

        text = _transcribe_chunk(chunk, sr)
        transcripts.append(text)
        offset += step

    # Deduplicate overlapping text: simple suffix-prefix matching
    if len(transcripts) <= 1:
        combined = transcripts[0] if transcripts else ""
    else:
        combined = transcripts[0]
        for t in transcripts[1:]:
            # Find overlap: check if end of combined matches start of t
            words_prev = combined.split()
            words_next = t.split()
            best_overlap = 0
            # Check last N words of previous against first N words of next
            max_check = min(len(words_prev), len(words_next), 10)
            for n in range(max_check, 0, -1):
                if words_prev[-n:] == words_next[:n]:
                    best_overlap = n
                    break
            # Append non-overlapping portion
            if best_overlap > 0:
                combined += " " + " ".join(words_next[best_overlap:])
            else:
                combined += " " + t

    return normalize_telugu_text(combined)


def translate_te_en(telugu_text):
    """NLLB: Telugu → English.

    For long text, splits into sentence-level chunks to avoid
    tokenizer truncation at max_length.
    """
    tok = M["trans_tok"]
    tok.src_lang = "tel_Telu"
    eng_tok = tok.convert_tokens_to_ids("eng_Latn")

    # Split long text into sentences (Telugu uses '।' and '.' as delimiters)
    sentences = re.split(r'(?<=[।.!?])\s+', telugu_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # If short enough, translate in one shot
    if len(sentences) <= 1 or len(telugu_text) < 200:
        inputs = tok(telugu_text, return_tensors="pt", max_length=512,
                     truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            generated = M["trans"].generate(
                **inputs, forced_bos_token_id=eng_tok,
                max_new_tokens=512, num_beams=4,
            )
        text = tok.decode(generated[0], skip_special_tokens=True)
        return postprocess_english(text)

    # Translate in sentence batches to avoid truncation
    translated_parts = []
    for sent in sentences:
        inputs = tok(sent, return_tensors="pt", max_length=512,
                     truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            generated = M["trans"].generate(
                **inputs, forced_bos_token_id=eng_tok,
                max_new_tokens=512, num_beams=4,
            )
        part = tok.decode(generated[0], skip_special_tokens=True)
        translated_parts.append(part.strip())

    return postprocess_english(" ".join(translated_parts))


def extract_speaker_embedding(audio_np, sr=16000):
    """ECAPA-TDNN: extract normalized speaker embedding (CPU)."""
    wav_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
    if sr != 16000:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
    with torch.no_grad():
        emb = M["spk_encoder"].encode_batch(wav_tensor).squeeze()
        emb = emb / emb.norm()
    return emb.numpy()


def extract_emotion_vad(audio_np, sr=16000):
    """Wav2Vec2 VAD: extract Valence, Arousal, Dominance vector."""
    wav_tensor = torch.from_numpy(audio_np).unsqueeze(0).float().to(DEVICE)
    if sr != 16000:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
    with torch.no_grad():
        vad = M["emotion"](wav_tensor)
    return vad.cpu().numpy().squeeze()


def _extract_best_segment(audio_np, sr, min_dur=None, max_dur=None, target_dur=None):
    """Extract the highest-energy segment of target_dur seconds from audio.

    This gives YourTTS the cleanest, most representative speech segment
    for voice cloning instead of passing potentially noisy full audio.
    """
    min_dur = min_dur or REF_MIN_DUR
    max_dur = max_dur or REF_MAX_DUR
    target_dur = target_dur or REF_BEST_DUR

    total_dur = len(audio_np) / sr

    # If audio is already short enough, return as-is
    if total_dur <= max_dur:
        return audio_np

    # Compute RMS energy in sliding windows to find best segment
    window_samples = int(target_dur * sr)
    step_samples = int(0.5 * sr)  # 0.5s step

    best_start = 0
    best_rms = 0.0

    for start in range(0, len(audio_np) - window_samples, step_samples):
        segment = audio_np[start : start + window_samples]
        rms = np.sqrt(np.mean(segment ** 2))
        if rms > best_rms:
            best_rms = rms
            best_start = start

    return audio_np[best_start : best_start + window_samples]


def _normalize_audio(audio_np, target_rms=None, peak_limit=None):
    """RMS-normalize audio to target level with peak limiting.

    Ensures consistent output volume regardless of input level.
    """
    target_rms = target_rms or TARGET_RMS
    peak_limit = peak_limit or PEAK_LIMIT

    # Current RMS
    current_rms = np.sqrt(np.mean(audio_np ** 2))
    if current_rms < 1e-8:
        return audio_np  # silence, don't amplify noise

    # Scale to target RMS
    gain = target_rms / current_rms
    audio_np = audio_np * gain

    # Peak limiting: soft clip to prevent distortion
    peak = np.max(np.abs(audio_np))
    if peak > peak_limit:
        audio_np = audio_np * (peak_limit / peak)

    return audio_np.astype(np.float32)


def prepare_speaker_ref(audio_path, target_sr=16000):
    """Prepare a clean, normalized reference for YourTTS voice cloning.

    Steps:
      1. Load and resample to 16kHz mono
      2. Trim silence (librosa trim at 25dB)
      3. Extract best 5-10s segment (highest energy)
      4. Peak-normalize to consistent level
      5. Save as temporary WAV

    This produces a much cleaner reference than raw input, improving
    voice cloning consistency and reducing loudness variance.
    """
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    audio_np = wav.squeeze().numpy()

    # Trim silence from ends
    audio_np, _ = librosa.effects.trim(audio_np, top_db=25)

    # Extract best segment if too long
    audio_np = _extract_best_segment(audio_np, target_sr)

    # Peak-normalize the reference (important for YourTTS consistency)
    peak = np.max(np.abs(audio_np))
    if peak > 1e-8:
        audio_np = audio_np * (0.9 / peak)  # normalize to 90% peak

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(DEMO_DIR))
    sf.write(tmp.name, audio_np.astype(np.float32), target_sr)
    return tmp.name


def synthesize_speech(english_text, speaker_wav=None):
    """YourTTS: English text → speech with voice cloning.

    Args:
        english_text: Text to synthesize in English.
        speaker_wav: Path to reference audio for voice cloning.
                     Uses the input Telugu speaker's audio to preserve voice.
                     Falls back to LJSpeech reference if None.
    Returns:
        (wav_numpy, sample_rate) tuple

    Output is RMS-normalized to TARGET_RMS (~0.08) for consistent volume.
    """
    ref = speaker_wav or DEFAULT_REF

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(DEMO_DIR)) as tmp:
        tmp_path = tmp.name

    try:
        with torch.no_grad():
            M["tts"].tts_to_file(
                text=english_text,
                language="en",
                speaker_wav=ref,
                file_path=tmp_path,
            )
        wav, sr = librosa.load(tmp_path, sr=None)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Normalize output volume for consistency
    wav = _normalize_audio(wav, target_rms=TARGET_RMS, peak_limit=PEAK_LIMIT)

    return wav.astype(np.float32), int(sr)


def compute_secs(emb_input, output_wav, output_sr=16000):
    """Speaker Embedding Cosine Similarity (SECS).

    Measures how similar the output voice is to the input voice.
    Higher = better voice preservation (>0.7 is good).
    """
    emb_out = extract_speaker_embedding(output_wav, sr=output_sr)
    cos_sim = np.dot(emb_input, emb_out) / (
        np.linalg.norm(emb_input) * np.linalg.norm(emb_out) + 1e-8
    )
    return round(float(cos_sim), 4)


# ══════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════

def telugu_voice_bridge(audio_path, target_sr=16000):
    """Full Telugu → English speech-to-speech translation pipeline.

    Steps:
      1. Load & resample input audio to 16kHz mono
      2. [Parallel] Speaker embedding on CPU (branch B)
      3. [GPU] ASR: Telugu speech → Telugu text
      4. [GPU] Translation: Telugu text → English text
      5. [GPU] Emotion VAD: extract valence/arousal/dominance
      6. [GPU] TTS: English text → English speech (voice cloned from input)
    """
    t_total = time.time()
    timing = {}

    # Load audio
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    audio_np = wav.mean(dim=0).numpy()  # mono

    # Prepare a clean 16kHz WAV for YourTTS voice cloning
    speaker_ref_path = prepare_speaker_ref(audio_path, target_sr)

    # Branch B (CPU): Speaker embedding — runs in parallel
    results = {}

    def branch_b():
        t = time.time()
        results["speaker_embedding"] = extract_speaker_embedding(audio_np, target_sr)
        timing["speaker_encoding"] = time.time() - t

    thread_b = threading.Thread(target=branch_b)
    thread_b.start()

    # Branch A (GPU): ASR → Translation
    t = time.time()
    telugu_text = transcribe_telugu(audio_np, target_sr)
    timing["asr"] = time.time() - t

    t = time.time()
    english_text = translate_te_en(telugu_text)
    timing["translation"] = time.time() - t

    # Branch C (GPU): Emotion
    t = time.time()
    vad_vector = extract_emotion_vad(audio_np, target_sr)
    timing["emotion"] = time.time() - t

    thread_b.join()

    # TTS with voice cloning from the input speaker
    t = time.time()
    output_wav, output_sr = synthesize_speech(
        english_text, speaker_wav=speaker_ref_path
    )
    timing["tts"] = time.time() - t
    timing["total"] = time.time() - t_total

    # Cleanup the temp speaker reference
    if os.path.exists(speaker_ref_path):
        os.unlink(speaker_ref_path)

    return {
        "telugu_text": telugu_text,
        "english_text": english_text,
        "speaker_embedding": results["speaker_embedding"],
        "vad_vector": vad_vector.tolist(),
        "output_wav": output_wav,
        "output_sr": output_sr,
        "timing": timing,
    }


# ══════════════════════════════════════════════════════════
# GRADIO INTERFACE
# ══════════════════════════════════════════════════════════

def gradio_translate(audio_file):
    """Process uploaded Telugu audio → English audio."""
    if audio_file is None:
        return None, "", "", "", ""

    try:
        result = telugu_voice_bridge(audio_file)

        # Save output
        out_path = str(DEMO_DIR / "output.wav")
        sf.write(out_path, result["output_wav"], result["output_sr"])

        # SECS (speaker similarity)
        secs = compute_secs(
            result["speaker_embedding"],
            result["output_wav"],
            result["output_sr"],
        )

        # Timing breakdown
        t = result["timing"]
        latency = (
            f"ASR: {t.get('asr',0):.2f}s | "
            f"Translation: {t.get('translation',0):.2f}s | "
            f"Speaker: {t.get('speaker_encoding',0):.2f}s | "
            f"Emotion: {t.get('emotion',0):.2f}s | "
            f"TTS: {t.get('tts',0):.2f}s | "
            f"Total: {t.get('total',0):.2f}s"
        )

        # VAD display
        vad = result["vad_vector"]
        emotion_str = f"V={vad[0]:.2f}  A={vad[1]:.2f}  D={vad[2]:.2f}"

        return (
            out_path,
            result["telugu_text"],
            result["english_text"],
            f"SECS: {secs:.4f} | Emotion: {emotion_str}",
            latency,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None, f"Error: {str(e)}", "", "", ""


def build_app():
    """Build and return the Gradio app."""
    with gr.Blocks(
        title="TeluguVoiceBridge v2",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🎙️ TeluguVoiceBridge v2
            ### Telugu → English Speech-to-Speech Translation

            Upload a Telugu audio file and get an English translation **spoken in the same voice**.

            **Pipeline:** ASR (Whisper) → Translation (NLLB + LoRA) → Voice Cloning (YourTTS)
            **Speaker:** ECAPA-TDNN embedding | **Emotion:** Wav2Vec2 VAD
            **Voice Preservation:** Your input voice is cloned into the English output
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    type="filepath",
                    label="🎤 Upload Telugu Audio",
                    sources=["upload", "microphone"],
                )
                translate_btn = gr.Button(
                    "🔄 Translate", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="🔊 English Output (Voice Cloned)", type="filepath"
                )

        with gr.Row():
            with gr.Column():
                telugu_text = gr.Textbox(label="Telugu Transcript (ASR)", lines=3)
            with gr.Column():
                english_text = gr.Textbox(label="English Translation", lines=3)

        with gr.Row():
            metrics = gr.Textbox(label="Speaker Similarity & Emotion")
            latency = gr.Textbox(label="Latency Breakdown")

        translate_btn.click(
            fn=gradio_translate,
            inputs=[audio_input],
            outputs=[audio_output, telugu_text, english_text, metrics, latency],
        )

        gr.Markdown(
            """
            ---
            **Notes:**
            - First inference may be slower (GPU warmup)
            - Best results with clear Telugu speech, 3-15 seconds
            - SECS measures voice similarity (higher = more similar, >0.7 is good)
            - Emotion VAD: Valence, Arousal, Dominance (range -1 to 1)
            - Voice cloning preserves your vocal characteristics in the English output
            """
        )

    return app


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading models... (this takes ~30s)")
    M = load_all_models()

    print("Building Gradio app...")
    app = build_app()

    print("\nLaunching at http://localhost:7860")
    app.queue(max_size=2)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        max_threads=1,
        show_error=True,
    )
