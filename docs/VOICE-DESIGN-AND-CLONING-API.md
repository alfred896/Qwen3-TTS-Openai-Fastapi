# Voice Design and Cloning API Guide

## Overview

Qaudio TTS engine supports three voice modes through a unified API:

1. **Preset Voices (CustomVoice)**: Pre-configured voices (Vivian, Ryan, Sophia, etc.)
2. **Voice Cloning**: Clone voices from reference audio
3. **Voice Design**: Create voices via natural language instructions

All features are accessible through the same `/v1/audio/speech` endpoint or dedicated endpoints.

---

## Model Selection

Models are loaded at startup via the `TTS_MODEL_ID` environment variable. **Only one model can be active at a time.**

```bash
# CustomVoice (default) - preset voices
TTS_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" python main.py

# Base - supports voice cloning and design
TTS_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base" python main.py

# VoiceDesign - optimized for design (limited preset voices)
TTS_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" python main.py
```

### Capability Matrix

| Feature | CustomVoice | Base | VoiceDesign |
|---------|:-----------:|:----:|:-----------:|
| Preset Voices | ✓ | ✓ | ✓ |
| Voice Cloning | ✗ | ✓ | ✗ |
| Voice Design | ✗ | ✓ | ✓ |

---

## Voice Design API

### Create and Save Voice Design

**Save a voice design profile for reuse:**

```python
import requests

# 1. Create and save design profile (one-time setup)
design_profile = {
    "design_name": "Professional Female",
    "instruct": "Clear female voice, professional and warm tone",
    "language": "English"
}

# Save via Python API or manual profile creation
# (See Voice Profile Management section)
```

### Synthesize with Saved Design

**Use saved design in speech endpoint:**

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Welcome to Qaudio voice design demo.",
    "voice": "design:Professional Female",
    "response_format": "wav",
    "language": "English"
  }' \
  -o output.wav
```

### Direct Voice Design Synthesis

**Synthesize directly with design instructions:**

```bash
curl -X POST http://localhost:8880/v1/audio/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a voice design test.",
    "language": "English",
    "instruct": "Clear female voice, professional and warm",
    "response_format": "wav",
    "speed": 1.0
  }' \
  -o output.wav
```

### Design Examples by Language

#### English
```json
{
  "instruct": "Clear female voice, professional and warm tone"
}
```

```json
{
  "instruct": "Deep male voice, calm and authoritative"
}
```

```json
{
  "instruct": "Young female voice, cheerful and energetic"
}
```

#### Chinese (Mandarin)
```json
{
  "instruct": "清晰女声，温和的语气"
}
```

```json
{
  "instruct": "深沉男声，权威稳重"
}
```

```json
{
  "instruct": "年轻女声，活泼开朗"
}
```

#### Italian
```json
{
  "instruct": "Voce femminile chiara, tono caldo e professionale"
}
```

---

## Voice Cloning API

### Create Voice Clone Profile

**Upload reference audio and create profile:**

```bash
curl -X POST http://localhost:8880/v1/audio/voice-clone \
  -F "file=@reference_voice.wav" \
  -F "voice_name=Alice Custom" \
  -F "ref_text=Hello, this is my voice reference." \
  -F "language=English"
```

**Response:**
```json
{
  "status": "success",
  "profile": {
    "name": "Alice Custom",
    "profile_id": "alice_custom_a1b2c3d4",
    "ref_audio_filename": "reference.wav",
    "ref_text": "Hello, this is my voice reference.",
    "language": "English",
    "task_type": "VoiceCloning",
    "created_at": "2026-03-09T12:34:56.789123"
  },
  "usage": "Use voice_name='Alice Custom' in TTS requests"
}
```

### Synthesize with Cloned Voice

**Use saved clone in speech endpoint:**

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is my cloned voice speaking new text.",
    "voice": "clone:Alice Custom",
    "response_format": "wav"
  }' \
  -o output.wav
```

### Voice Cloning Modes

#### ICL Mode (In-Context Learning)
- **Requirements:** Reference audio + transcript
- **Quality:** Highest fidelity, best for accent/emotion preservation
- **Usage:** Provide `ref_text` when uploading

```bash
curl -X POST http://localhost:8880/v1/audio/voice-clone \
  -F "file=@reference.wav" \
  -F "voice_name=My Voice" \
  -F "ref_text=This is the exact text spoken in the reference audio" \
  -F "language=English"
```

#### X-Vector Mode
- **Requirements:** Reference audio only (no transcript)
- **Quality:** Good, works when transcript unavailable
- **Usage:** Omit or leave empty `ref_text`

```bash
curl -X POST http://localhost:8880/v1/audio/voice-clone \
  -F "file=@reference.wav" \
  -F "voice_name=Quick Clone" \
  -F "language=English"
```

---

## Voice Profile Management

### Python API

```python
from api.services.voice_profile_manager import VoiceProfileManager

manager = VoiceProfileManager()

# Create voice design profile
design = manager.create_voice_design_profile(
    design_name="Cheerful Voice",
    instruct="Young, cheerful, energetic female voice",
    language="English"
)

# Create voice clone profile
clone = manager.create_voice_clone_profile(
    voice_name="Custom Speaker",
    reference_audio_path="ref_audio.wav",
    reference_text="Optional transcript",
    language="English"
)

# List all profiles
all_profiles = manager.list_voice_profiles()

# List specific types
designs = manager.list_voice_design_profiles()
clones = manager.list_voice_clone_profiles()

# Retrieve single profile
profile = manager.get_voice_profile("Custom Speaker")

# Delete profile
manager.delete_profile("Custom Speaker")
```

### Storage Structure

Profiles are stored in `./voice_library/profiles/{profile_id}/`:

```
voice_library/
├── profiles/
│   ├── cheerful_voice_a1b2c3d4/
│   │   └── meta.json                 # Profile metadata
│   └── custom_speaker_e5f6g7h8/
│       ├── meta.json                 # Profile metadata
│       ├── reference.wav             # Reference audio
│       └── x_vector.pkl              # Optional speaker embedding
```

### Profile Metadata Format

**Voice Design Profile:**
```json
{
  "name": "Cheerful Voice",
  "profile_id": "cheerful_voice_a1b2c3d4",
  "instruct": "Young, cheerful, energetic female voice",
  "language": "English",
  "task_type": "VoiceDesign",
  "created_at": "2026-03-09T12:34:56.789123"
}
```

**Voice Clone Profile:**
```json
{
  "name": "Custom Speaker",
  "profile_id": "custom_speaker_e5f6g7h8",
  "ref_audio_filename": "reference.wav",
  "ref_text": "Optional transcript",
  "x_vector_only_mode": false,
  "use_icl_mode": true,
  "language": "English",
  "task_type": "VoiceCloning",
  "created_at": "2026-03-09T12:34:56.789123"
}
```

---

## Configuration

### Environment Variables

```bash
# Voice design and management
TTS_VOICE_DESIGN_ENABLED=true              # Enable voice design (default: true)
VOICE_LIBRARY_DIR=./voice_library          # Profile storage (default: ./voice_library)
VOICE_PKL_DIR=./voice_library/pkl_profiles # Pkl embeddings (default: as above)

# Model selection
TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base # Which model to load
TTS_DEVICE=auto                             # GPU/CPU (default: auto-detect)
TTS_DTYPE=auto                              # Precision (default: bfloat16 GPU, float32 CPU)
```

### Configuration File

Edit `api/config.py` to customize defaults:

```python
TTS_VOICE_DESIGN_ENABLED = True
VOICE_LIBRARY_DIR = "./voice_library"
VOICE_PKL_DIR = "./voice_library/pkl_profiles"
```

---

## Performance Characteristics

### Voice Design
- **Latency:** 0.5–1.5 seconds per sentence
- **VRAM:** ~8–10 GB for 1.7B model
- **Languages:** English, Chinese, Italian, German, French, Spanish, Portuguese, Russian, Japanese, Korean

### Voice Cloning
- **ICL Mode Latency:** 1.0–2.0 seconds (higher quality)
- **X-Vector Mode Latency:** 0.5–1.5 seconds (faster, slightly lower quality)
- **VRAM:** ~10–12 GB for Base model
- **Languages:** Same as Voice Design

---

## Troubleshooting

### "Voice design not supported" Error
**Cause:** VoiceDesign model is not loaded.
**Solution:** Start server with correct model:
```bash
TTS_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" python main.py
```

### "Voice cloning not supported" Error
**Cause:** CustomVoice model loaded (doesn't support cloning).
**Solution:** Load Base model:
```bash
TTS_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base" python main.py
```

### "Design profile not found" Error
**Cause:** Profile name doesn't match.
**Solution:** Check available profiles:
```python
manager = VoiceProfileManager()
designs = manager.list_voice_design_profiles()
print([d['name'] for d in designs])
```

### "Reference audio file not found" Error
**Cause:** Audio file path invalid.
**Solution:** Provide absolute or correct relative path:
```bash
curl -F "file=@/absolute/path/to/ref_audio.wav" ...
```

---

## API Reference

### POST /v1/audio/speech
Standard speech synthesis with voice library support.

**Parameters:**
- `input` (string, required): Text to synthesize
- `voice` (string): Voice name
  - Preset: `"Vivian"`, `"Ryan"`, etc.
  - Cloned: `"clone:ProfileName"`
  - Designed: `"design:ProfileName"`
- `response_format` (string): `"mp3"` (default), `"wav"`, `"pcm"`, etc.
- `speed` (float): 0.25–4.0 (default: 1.0)
- `language` (string): Language code

### POST /v1/audio/voice-design
Direct voice design synthesis without profile.

**Parameters:**
- `text` (string, required): Text to synthesize
- `language` (string): Language code (default: `"English"`)
- `instruct` (string, required): Voice design instruction
- `response_format` (string): `"mp3"` (default), `"wav"`, etc.
- `speed` (float): 0.25–4.0 (default: 1.0)

### POST /v1/audio/voice-clone
Create and save voice clone profile.

**Parameters:**
- `file` (file, required): Reference audio file
- `voice_name` (string, required): Name for the profile
- `ref_text` (string, optional): Transcript (for ICL mode)
- `language` (string): Language code (default: `"English"`)

---

## Examples

### Python Client

```python
import requests

base_url = "http://localhost:8880"

# 1. Synthesize with voice design
response = requests.post(
    f"{base_url}/v1/audio/voice-design",
    json={
        "text": "Hello, this is a voice design demo.",
        "language": "English",
        "instruct": "Clear female voice, professional",
        "response_format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# 2. Create voice clone
with open("reference.wav", "rb") as f:
    response = requests.post(
        f"{base_url}/v1/audio/voice-clone",
        files={"file": f},
        data={
            "voice_name": "My Voice",
            "ref_text": "This is my voice.",
            "language": "English"
        }
    )

print(response.json())

# 3. Use cloned voice in speech
response = requests.post(
    f"{base_url}/v1/audio/speech",
    json={
        "input": "New text with my cloned voice.",
        "voice": "clone:My Voice",
        "response_format": "wav"
    }
)

with open("cloned_output.wav", "wb") as f:
    f.write(response.content)
```

---

## License

This API documentation is part of the Qaudio TTS project.
See LICENSE for details.
