![Chatterbox Turbo Image](./Chatterbox-Turbo.jpg)


# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_turbo_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/chatterbox-turbo-demo)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

*Made with ♥️ by* <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

**Chatterbox** is a family of three state-of-the-art, open-source text-to-speech models by Resemble AI.

We are excited to introduce **Chatterbox-Turbo**, our most efficient model yet. Built on a streamlined 350M parameter architecture, **Turbo** delivers high-quality speech with less compute and VRAM than our previous models. We have also distilled the speech-token-to-mel decoder, previously a bottleneck, reducing generation from 10 steps to just **one**, while retaining high-fidelity audio output.

**Paralinguistic tags** are now native to the Turbo model, allowing you to use `[cough]`, `[laugh]`, `[chuckle]`, and more to add distinct realism. While Turbo was built primarily for low-latency voice agents, it excels at narration and creative workflows.

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

<img width="1200" height="600" alt="Podonos Turbo Eval" src="https://storage.googleapis.com/chatterbox-demo-samples/turbo/podonos_turbo.png" />

### ⚡ Model Zoo

Choose the right model for your application.

| Model                                                                                                           | Size | Languages | Key Features                                            | Best For                                     | 🤗                                                                  | Examples |
|:----------------------------------------------------------------------------------------------------------------| :--- | :--- |:--------------------------------------------------------|:---------------------------------------------|:--------------------------------------------------------------------------| :--- |
| **Chatterbox-Turbo**                                                                                            | **350M** | **English** | Paralinguistic Tags (`[laugh]`), Lower Compute and VRAM | Zero-shot voice agents,  Production          | [Demo](https://huggingface.co/spaces/ResembleAI/chatterbox-turbo-demo)        | [Listen](https://resemble-ai.github.io/chatterbox_turbo_demopage/) |
| Chatterbox-Multilingual [(Language list)](#supported-languages)                                                 | 500M | 23+ | Zero-shot cloning, Multiple Languages                   | Global applications, Localization            | [Demo](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS) | [Listen](https://resemble-ai.github.io/chatterbox_demopage/) |
| Chatterbox [(Tips and Tricks)](#original-chatterbox-tips)                                                       | 500M | English | CFG & Exaggeration tuning                               | General zero-shot TTS with creative controls | [Demo](https://huggingface.co/spaces/ResembleAI/Chatterbox)              | [Listen](https://resemble-ai.github.io/chatterbox_demopage/) |

---

## Indic Language LoRA — 8 Indian Languages

![Atoms of AI](./atoms2.png)

This fork adds **8 Indian languages** to Chatterbox-Multilingual via LoRA fine-tuning. No phoneme engineering, no G2P — just grapheme-level adaptation on 1.4% of the model parameters.

**LoRA weights:** [reenigne314/chatterbox-indic-lora](https://huggingface.co/reenigne314/chatterbox-indic-lora)

### Indic LoRA Results

| Language | Script | CER (LoRA) | Status |
|----------|--------|:----------:|--------|
| Hindi | Devanagari | **0.1058** | Stable |
| Kannada | Kannada | **0.1434** | Trained |
| Tamil | Tamil | **0.1608** | Trained |
| Marathi | Devanagari | **0.1976** | Trained |
| Gujarati | Gujarati | **0.2377** | Trained |
| Bengali | Bengali | **0.2450** | Trained |
| Telugu | Telugu | **0.2853** | Trained |
| Malayalam | Malayalam | 0.8593 | Experimental |
| English | Latin | Preserved | Base model (frozen) |

*CER measured via Whisper large-v3 ASR on 100 held-out samples per language.*

> **Hindi baseline (no LoRA):** CER 0.2897 on 10 samples. After LoRA training across all 8 languages: **CER 0.1058** on 100 samples — a significant improvement with zero catastrophic forgetting.

### Indic LoRA Inference

```python
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# One-line load: base model + LoRA + tokenizer + speaker
model = ChatterboxMultilingualTTS.from_indic_lora(device="cuda", speaker="te_female")

# Telugu
wav = model.generate("నమస్కారం, మీరు ఎలా ఉన్నారు?", language_id="te")
sf.write("telugu.wav", wav.squeeze(0).cpu().numpy(), model.sr)

# Hindi
from chatterbox.mtl_tts import Conditionals
model.conds = Conditionals.load("hi_male.pt").to("cuda")
wav = model.generate("नमस्ते, आप कैसे हैं?", language_id="hi")
sf.write("hindi.wav", wav.squeeze(0).cpu().numpy(), model.sr)

# Kannada
model.conds = Conditionals.load("kn_female.pt").to("cuda")
wav = model.generate("ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?", language_id="kn")
sf.write("kannada.wav", wav.squeeze(0).cpu().numpy(), model.sr)
```

Available speakers: `{hi,te,kn,bn,ta,ml,mr,gu}_{female,male}.pt` (16 total)

---

## Installation

Install PyTorch **first** for your GPU, then install this package:

```bash
# Step 1: PyTorch (pick your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # RTX 50-series / Blackwell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124  # RTX 30/40-series / Ampere/Ada

# Step 2: Install chatterbox
pip install git+https://github.com/reenigne314/chatterbox-indic.git
```

> **Why not `pip install chatterbox-tts`?** The upstream package pins `torch==2.6.0` which breaks on newer GPUs (RTX 5060 Ti, 5070, 5090). This fork removes the torch pin so your pre-installed CUDA-matched torch is preserved.

Alternatively, install from source:
```bash
git clone https://github.com/reenigne314/chatterbox-indic.git
cd chatterbox-indic
pip install -e .
```

## Usage

##### Chatterbox-Turbo

```python
import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with Paralinguistic Tags
text = "Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat about the billing issue?"

# Generate audio (requires a reference clip for voice cloning)
wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")

sf.write("test-turbo.wav", wav.squeeze(0).cpu().numpy(), model.sr)
```

##### Chatterbox and Chatterbox-Multilingual

```python
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
sf.write("test-english.wav", wav.squeeze(0).cpu().numpy(), model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav_french = multilingual_model.generate(french_text, language_id="fr")
sf.write("test-french.wav", wav_french.squeeze(0).cpu().numpy(), multilingual_model.sr)

chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
sf.write("test-chinese.wav", wav_chinese.squeeze(0).cpu().numpy(), multilingual_model.sr)

# Voice cloning with a reference audio
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
sf.write("test-cloned.wav", wav.squeeze(0).cpu().numpy(), model.sr)
```

##### Indic Languages (LoRA)

```python
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Loads base model + LoRA + extended tokenizer + speaker — all in one call
model = ChatterboxMultilingualTTS.from_indic_lora(device="cuda", speaker="te_female")

# Generate Telugu
wav = model.generate("నమస్కారం, మీరు ఎలా ఉన్నారు?", language_id="te")
sf.write("test-telugu.wav", wav.squeeze(0).cpu().numpy(), model.sr)

# Generate Bengali
from chatterbox.mtl_tts import Conditionals
model.conds = Conditionals.load("bn_male.pt").to("cuda")
wav = model.generate("নমস্কার, আপনি কেমন আছেন?", language_id="bn")
sf.write("test-bengali.wav", wav.squeeze(0).cpu().numpy(), model.sr)
```

See `example_tts.py` and `example_vc.py` for more examples.

## Supported Languages

**Base Chatterbox-Multilingual (23):**
Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

**Added via Indic LoRA (+7 new):**
Telugu (te) • Kannada (kn) • Bengali (bn) • Tamil (ta) • Malayalam (ml) • Marathi (mr) • Gujarati (gu)

## Original Chatterbox Tips
- **General Use (TTS and Voice Agents):**
  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.


## Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.


## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```


## Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

## Evaluation
Chatterbox Turbo was evaluated using Podonos, a platform for reproducible subjective speech evaluation.

We compared Chatterbox Turbo to competitive TTS systems using Podonos' standardized evaluation suite, focusing on overall preference, naturalness, and expressiveness.

Evaluation reports:
- [Chatterbox Turbo vs ElevenLabs Turbo v2.5](https://podonos.com/resembleai/chatterbox-turbo-vs-elevenlabs-turbo)
- [Chatterbox Turbo vs Cartesia Sonic 3](https://podonos.com/resembleai/chatterbox-turbo-vs-cartesia-sonic3)
- [Chatterbox Turbo vs VibeVoice 7B](https://podonos.com/resembleai/chatterbox-turbo-vs-vibevoice7b)

These evaluations were conducted under identical conditions and are publicly accessible via Podonos.

## Acknowledgements
- [Podonos](https://podonos.com) — for supporting reproducible subjective speech evaluation
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)
- [SPRINGLab / IIT Madras](https://huggingface.co/SPRINGLab) — IndicTTS dataset
- [ai4bharat](https://ai4bharat.iitm.ac.in/) — Rasa dataset

## Citation
If you find this model useful, please consider citing.
```
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```

```
@misc{chatterbox_indic_lora_2025,
  author       = {Bharadwaj Kommanamanchi},
  title        = {Chatterbox Indic LoRA — Indian Language TTS via Grapheme-Level Fine-Tuning},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/reenigne314/chatterbox-indic-lora}},
  note         = {LoRA adapters for Chatterbox-Multilingual}
}
```

## Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.

