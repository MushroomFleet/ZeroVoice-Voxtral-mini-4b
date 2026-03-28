<!-- TINS Specification v1.0 -->
<!-- ZS:COMPLEXITY:MEDIUM -->
<!-- ZS:PRIORITY:HIGH -->
<!-- ZS:PLATFORM:WEB -->
<!-- ZS:LANGUAGE:PYTHON -->

# ZeroVoice Voxtral Frontend V2

## Description

A standalone Gradio 5.50 web application for the ZeroVoice Voxtral TTS system. This is a **new, independent frontend** (`zerovoice_frontend.py`) that replaces the original vllm-omni demo without modifying it. It provides two modes of operation in a tabbed interface: **Preset Voices** (the original 20 voices with language/voice cascading dropdowns) and **Voice Explorer** (3D coordinate sliders that procedurally generate infinite blended voices via the ZeroVoice engine).

The frontend communicates with the existing vLLM server on `localhost:8000` using the same `POST /v1/audio/speech` endpoint. ZeroVoice coordinates are passed as the voice name (e.g., `"zv_50_30_10"`), and the server computes the SLERP blend on-the-fly.

**Target audience:** Developers and researchers exploring the ZeroVoice procedural voice space interactively.

**Dependencies:** `gradio==5.50`, `httpx`, `soundfile`, `numpy`, and the `zerovoice.py` module (for client-side recipe display only -- all actual blending happens server-side).

---

## Functionality

### Core Features

- **Tab 1 — Preset Voices:** Original TTS functionality with 20 preset voices organized by language. Cascading Language/Voice dropdowns identical to V1 behavior.
- **Tab 2 — Voice Explorer:** Three integer sliders (X, Y, Z) that map to a ZeroVoice coordinate. Displays the voice recipe (which 3 base voices are blended, at what weights) and generates speech. Includes navigator buttons to step through adjacent coordinates.
- **Shared text input:** A single text prompt area shared between both tabs, so switching tabs preserves the text.
- **Audio player:** Autoplay with download button. Shows generation time.
- **Voice recipe panel:** JSON display showing `voice_a`, `voice_b`, `voice_c`, `t_ab`, `t_abc` derived from the current coordinates.
- **World seed control:** Optional numeric input to change the world seed (default 42), which reshuffles the entire voice space mapping.
- **Coordinate bookmarks:** A history list of recently generated coordinates with one-click recall.

### UI Layout

```
+=====================================================================+
|  ZeroVoice Voxtral TTS                                    [Seed: 42]|
+=====================================================================+
| [ Preset Voices ]  [ Voice Explorer ]                               |
+---------------------------------------------------------------------+
|                                                                     |
|  Text:                                                              |
|  +---------------------------------------------------------------+  |
|  | Enter the text you want to synthesize...                      |  |
|  |                                                               |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
+=====================================================================+

=== TAB: Preset Voices ===============================================

|  Language: [English v]     Voice: [neutral_male v]                  |
|                                                                     |
|  [Clear]  [Generate]                                                |
|                                                                     |
|  Audio: [ --------- audio player --------- ] [download]            |

=== TAB: Voice Explorer ==============================================

|  +--- Coordinate Controls ---+  +--- Voice Recipe ---------------+ |
|  |                           |  |                                 | |
|  | X (Blend):                |  | Coordinate: zv_50_30_10        | |
|  | [===========O====] 50    |  | Voice A: hi_male               | |
|  |                           |  | Voice B: ar_male               | |
|  | Y (Timbre):               |  | Voice C: hi_male               | |
|  | [======O==========] 30   |  | t_AB: 0.7535                   | |
|  |                           |  | t_ABC: 0.2221                  | |
|  | Z (Family):               |  | Family: asian_arabic           | |
|  | [==O===============] 10  |  |                                 | |
|  |                           |  +---------------------------------+ |
|  +---------------------------+                                      |
|                                                                     |
|  Navigate: [X-] [X+]  [Y-] [Y+]  [Z-] [Z+]  Step: [10]          |
|                                                                     |
|  [Generate]                                                         |
|                                                                     |
|  Audio: [ --------- audio player --------- ] [download]            |
|                                                                     |
|  +--- Recent Coordinates ------------------------------------+     |
|  | zv_50_30_10 (hi_male + ar_male)  [Recall]                |     |
|  | zv_51_30_10 (hi_female + ar_male) [Recall]               |     |
|  | zv_50_30_150 (fr_male + it_female) [Recall]              |     |
|  +------------------------------------------------------------+    |
+=====================================================================+
```

### User Flows

**Flow 1: Generate with Preset Voice**
```
[1] User selects "Preset Voices" tab
[2] Selects Language -> Voice dropdown updates
[3] Types text in the shared text area
[4] Clicks "Generate"
[5] Audio plays automatically, download available
```

**Flow 2: Explore Voice Space**
```
[1] User selects "Voice Explorer" tab
[2] Adjusts X, Y, Z sliders (recipe panel updates live)
[3] Types text (or uses existing text from shared area)
[4] Clicks "Generate"
[5] Audio plays. Coordinate added to recent history.
[6] User clicks [X+] navigator button
[7] X slider increments by step size, recipe updates
[8] User clicks "Generate" again to hear adjacent voice
```

**Flow 3: Recall a Bookmark**
```
[1] User sees previous coordinate in "Recent Coordinates"
[2] Clicks [Recall] next to it
[3] Sliders update to that coordinate, recipe refreshes
[4] User can modify sliders or generate immediately
```

### Edge Cases

| Scenario | Behavior |
|---|---|
| Empty text prompt | "Generate" button disabled (grayed out) |
| Server not running | Show `gr.Error("Server unavailable at {base_url}")` on generate |
| Very large coordinates (99999, 99999, 99999) | Valid. Server computes on-the-fly. |
| Negative coordinates | Valid. Sliders allow range -500 to 500 for X/Y. |
| World seed changed | Recipe panel updates immediately. History entries retain their original seed. |
| Tab switch | Shared text preserved. Audio output clears. |
| Server returns error for zv_ voice | Display error message from server response body |
| Generate clicked while previous is loading | Button disabled during generation (re-enabled on complete) |

---

## Technical Implementation

### Architecture

```
zerovoice_frontend.py (standalone Gradio app)
        |
        |-- imports zerovoice.py (for voice_recipe, voice_name, VOICE_FAMILIES, parse_voice_name)
        |-- imports text_preprocess.py (sanitize_tts_input_text_for_demo)
        |
        |-- HTTP requests to vLLM server:
        |     GET  http://localhost:8000/health
        |     GET  http://localhost:8000/v1/audio/voices
        |     POST http://localhost:8000/v1/audio/speech
        |
        +-- Gradio 5.50 UI (served on 0.0.0.0:7860)
```

### File: `zerovoice_frontend.py`

**Location:** `K:\voxtral-mini-4b\zerovoice_frontend.py`

**Also copied to WSL2 at:** `~/voxtral-tts/zerovoice_frontend.py`

**Single-file application.** No additional templates or assets required.

### Dependencies

```
gradio==5.50.0    # Already installed with vllm-omni
httpx             # Already installed
soundfile         # Already installed
numpy             # Already installed (via torch)
```

Plus local modules (already in site-packages):
- `zerovoice.py` — `voice_recipe()`, `voice_name()`, `VOICE_FAMILIES`, `ALL_VOICES`
- `text_preprocess.py` — `sanitize_tts_input_text_for_demo()`

### CLI Arguments

```python
parser.add_argument("--host", type=str, default="localhost",
                    help="vLLM server hostname")
parser.add_argument("--port", type=int, default=8000,
                    help="vLLM server port")
parser.add_argument("--model", type=str,
                    default="/mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603",
                    help="Model path (must match server)")
parser.add_argument("--world-seed", type=int, default=42,
                    help="Default world seed for ZeroVoice")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Directory to persist generated audio files")
```

### Component Specification

#### Shared Components (outside tabs)

| Component | Type | Properties |
|---|---|---|
| `title` | `gr.Markdown` | `"## ZeroVoice Voxtral TTS"` |
| `world_seed` | `gr.Number` | `value=42, label="World Seed", precision=0, minimum=0, maximum=999999` |
| `text_prompt` | `gr.Textbox` | `label="Text", placeholder="Enter text to synthesize...", lines=3` |

#### Tab 1: Preset Voices

| Component | Type | Properties |
|---|---|---|
| `language_dropdown` | `gr.Dropdown` | `choices=languages, label="Language", value="English"` |
| `preset_voice` | `gr.Dropdown` | `choices=english_voices, label="Voice", value="neutral_male"` |
| `preset_generate_btn` | `gr.Button` | `"Generate", interactive=False` |
| `preset_clear_btn` | `gr.Button` | `"Clear"` |
| `preset_audio` | `gr.Audio` | `label="Generated Audio", autoplay=True, interactive=False, type="filepath"` |

**Callbacks:**
- `language_dropdown.change` -> update `preset_voice` choices (cascading)
- `text_prompt.change` -> enable/disable `preset_generate_btn`
- `preset_generate_btn.click` -> call `run_inference(preset_voice, text_prompt)` -> set `preset_audio`
- `preset_clear_btn.click` -> reset all to defaults

#### Tab 2: Voice Explorer

| Component | Type | Properties |
|---|---|---|
| `coord_x` | `gr.Slider` | `minimum=-500, maximum=500, step=1, value=50, label="X (Blend)"` |
| `coord_y` | `gr.Slider` | `minimum=-500, maximum=500, step=1, value=50, label="Y (Timbre)"` |
| `coord_z` | `gr.Slider` | `minimum=0, maximum=300, step=1, value=50, label="Z (Family)"` |
| `nav_step` | `gr.Number` | `value=10, label="Step", precision=0, minimum=1, maximum=100` |
| `nav_xm` | `gr.Button` | `"X-", size="sm"` |
| `nav_xp` | `gr.Button` | `"X+", size="sm"` |
| `nav_ym` | `gr.Button` | `"Y-", size="sm"` |
| `nav_yp` | `gr.Button` | `"Y+", size="sm"` |
| `nav_zm` | `gr.Button` | `"Z-", size="sm"` |
| `nav_zp` | `gr.Button` | `"Z+", size="sm"` |
| `recipe_display` | `gr.JSON` | `label="Voice Recipe"` |
| `coord_label` | `gr.Textbox` | `label="Coordinate", interactive=False, show_copy_button=True` |
| `explorer_generate_btn` | `gr.Button` | `"Generate", interactive=False` |
| `explorer_audio` | `gr.Audio` | `label="Generated Audio", autoplay=True, interactive=False, type="filepath"` |
| `history_display` | `gr.Dataframe` | `headers=["Coordinate", "Voice A", "Voice B", "t_AB"], label="Recent", interactive=False, row_count=5` |

**Callbacks:**

Recipe update (fires on any slider or seed change):
```python
# All three sliders + world_seed trigger recipe refresh
for component in [coord_x, coord_y, coord_z, world_seed]:
    component.change(
        fn=update_recipe,
        inputs=[coord_x, coord_y, coord_z, world_seed],
        outputs=[recipe_display, coord_label],
    )
```

Navigator buttons:
```python
def make_nav(axis: str, direction: int):
    """Return callback that increments/decrements one slider."""
    def nav(x, y, z, step):
        if axis == "x": x += direction * int(step)
        elif axis == "y": y += direction * int(step)
        elif axis == "z": z = max(0, z + direction * int(step))
        return x, y, z
    return nav

nav_xp.click(fn=make_nav("x", +1), inputs=[coord_x, coord_y, coord_z, nav_step],
             outputs=[coord_x, coord_y, coord_z])
# ... same pattern for xm, yp, ym, zp, zm
```

Generate:
```python
explorer_generate_btn.click(
    fn=run_zerovoice_inference,
    inputs=[coord_x, coord_y, coord_z, world_seed, text_prompt],
    outputs=[explorer_audio, history_display],
)
```

### Key Functions

#### `update_recipe(x, y, z, seed)`

```python
def update_recipe(x: int, y: int, z: int, seed: int) -> tuple[dict, str]:
    """Compute and return the voice recipe for display. No server call needed."""
    from zerovoice import voice_recipe, voice_name
    recipe = voice_recipe(int(x), int(y), int(z), int(seed))
    coord = voice_name(int(x), int(y), int(z), int(seed))
    return recipe, coord
```

This runs **entirely client-side** (no HTTP request). The `zerovoice.py` module is imported locally and `voice_recipe()` uses only hashing and noise -- no torch, no model access needed. The recipe updates instantly as sliders move.

#### `run_preset_inference(voice, text, base_url, model)`

```python
def run_preset_inference(voice: str, text: str) -> tuple[int, np.ndarray]:
    """Generate TTS audio using a preset voice name."""
    text = sanitize_tts_input_text_for_demo(text.strip())
    response = httpx.post(f"{base_url}/audio/speech", json={
        "input": text,
        "model": model,
        "response_format": "wav",
        "voice": voice,
    }, timeout=120.0)
    response.raise_for_status()
    audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    return sr, audio_array
```

#### `run_zerovoice_inference(x, y, z, seed, text)`

```python
def run_zerovoice_inference(
    x: int, y: int, z: int, seed: int, text: str
) -> tuple[tuple[int, np.ndarray], list[list]]:
    """Generate TTS audio using a ZeroVoice coordinate.

    Returns:
        audio: (sample_rate, audio_array) tuple for gr.Audio
        history_row: updated history dataframe rows
    """
    from zerovoice import voice_name, voice_recipe
    coord = voice_name(int(x), int(y), int(z), int(seed))
    recipe = voice_recipe(int(x), int(y), int(z), int(seed))

    text = sanitize_tts_input_text_for_demo(text.strip())
    response = httpx.post(f"{base_url}/audio/speech", json={
        "input": text,
        "model": model,
        "response_format": "wav",
        "voice": coord,
    }, timeout=120.0)
    response.raise_for_status()
    audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")

    # Append to history (kept in closure state, max 10 entries)
    history.insert(0, [coord, recipe["voice_a"], recipe["voice_b"], f"{recipe['t_ab']:.3f}"])
    if len(history) > 10:
        history.pop()

    return (sr, audio_array), history
```

#### `fetch_preset_voices(base_url)`

```python
def fetch_preset_voices(base_url: str) -> tuple[list[str], dict[str, list[str]]]:
    """Fetch voices from the server and organize by language.

    Returns (sorted_languages, {language: [voice_names]})
    """
    # Same logic as V1: GET /v1/audio/voices, organize_voices_by_language()
```

### Voice Organization (reused from V1)

```python
LANGUAGE_PREFIXES = {
    "ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "nl": "Dutch", "pt": "Portuguese", "hi": "Hindi",
}
# Voices without a prefix -> "English"
# English voices sorted with neutral_male first
# Language list sorted: English first, then alphabetical
```

### State Management

| State | Scope | Storage |
|---|---|---|
| Current text | Shared across tabs | `text_prompt` component value |
| Current coordinate | Voice Explorer tab | `coord_x`, `coord_y`, `coord_z` slider values |
| World seed | Global | `world_seed` component value |
| Generation history | Voice Explorer tab | Python list in closure (max 10 entries) |
| Preset voice selection | Preset tab | `language_dropdown`, `preset_voice` values |
| Audio output | Per-tab | Separate `gr.Audio` per tab |

### Error Handling

| Error Source | Detection | User-Facing Message |
|---|---|---|
| Empty text | `text.strip() == ""` | Button stays disabled (preventive) |
| Server down | `httpx.ConnectError` | `gr.Error("Cannot connect to server at {base_url}")` |
| Server 400 | `response.status_code == 400` | `gr.Error(response.json()["error"]["message"])` |
| Server 500 | `response.status_code >= 500` | `gr.Error("Server error. Check server logs.")` |
| Text sanitization failure | `Exception` in `sanitize_tts_input_text_for_demo` | `gr.Error(f"Text preprocessing failed: {e}")` |
| Timeout | `httpx.TimeoutException` | `gr.Error("Generation timed out (120s). Try shorter text.")` |

---

## Implementation Steps

### Step 1: Create `zerovoice_frontend.py`

Write the complete Gradio application as a single Python file at `K:\voxtral-mini-4b\zerovoice_frontend.py`.

The file structure:

```python
"""ZeroVoice Voxtral TTS — Frontend V2"""
import argparse
import io
import logging

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

# --- Constants & Voice Organization ---
LANGUAGE_PREFIXES = { ... }
_DEFAULT_VOICES = [ ... ]

def organize_voices_by_language(voices): ...
def sanitize_text(text): ...      # import from text_preprocess or inline
def wait_for_server(base_url): ... # block until server healthy

# --- TTS Inference ---
def run_inference(voice_name, text, base_url, model): ...

# --- Recipe (client-side, no server) ---
def update_recipe(x, y, z, seed): ...

# --- Navigator helpers ---
def make_nav(axis, direction): ...

# --- Main app ---
def main(host, port, model, world_seed, output_dir):
    base_url = f"http://{host}:{port}/v1"
    languages, language_voices = fetch_preset_voices(base_url)
    history = []

    with gr.Blocks(title="ZeroVoice Voxtral TTS", fill_height=True) as demo:
        # ... build UI as specified ...
        pass

    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... add args ...
    args = parser.parse_args()
    main(args.host, args.port, args.model, args.world_seed, args.output_dir)
```

### Step 2: Copy to WSL2

```bash
cp /mnt/k/voxtral-mini-4b/zerovoice_frontend.py ~/voxtral-tts/
```

The `text_preprocess.py` module needs to be accessible. Either copy from the vllm-omni clone:

```bash
cp ~/vllm-omni/examples/online_serving/voxtral_tts/text_preprocess.py \
   ~/voxtral-tts/
```

Or inline the sanitization function directly in `zerovoice_frontend.py`.

### Step 3: Launch

```bash
cd ~/voxtral-tts
source .venv/bin/activate
python zerovoice_frontend.py --host localhost --port 8000 \
  --model /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603
```

The server must already be running (`start_server.sh`). The frontend connects on startup, fetches the voice list, and serves the Gradio UI on port 7860.

### Step 4: Verify

1. Open `http://localhost:7860` in browser
2. **Preset tab:** Select "English" / "neutral_male", type "Hello world", click Generate. Audio should play.
3. **Explorer tab:** Set X=50, Y=30, Z=10. Recipe should show `hi_male + ar_male`. Click Generate. Audio should play.
4. Click [X+] button. X slider moves to 60. Recipe updates. Generate again -- voice should be similar but slightly different.
5. Change Z from 10 to 150. Recipe should shift to European voices. Generate -- accent should change.
6. Change World Seed from 42 to 99. Entire recipe should change.

---

## Testing Scenarios

| Test | Action | Expected Result |
|---|---|---|
| Preset voice works | Select neutral_male, type "Hello", Generate | Valid audio plays |
| ZeroVoice works | Set (50,30,10), type "Hello", Generate | Valid audio plays, recipe shows blend |
| Recipe updates live | Move X slider | Recipe JSON updates without clicking Generate |
| Navigator buttons | Click X+ with step=10 | X slider moves from 50 to 60, recipe updates |
| Z-axis family shift | Change Z from 10 to 150 | Recipe voices shift from Asian/Arabic to European |
| History populates | Generate 3 different coordinates | History table shows 3 rows |
| Recall from history | Click Recall on a history entry | Sliders update to that coordinate |
| Shared text | Type in Preset tab, switch to Explorer tab | Text preserved |
| Empty text prevention | Clear text field | Generate button disabled on both tabs |
| Server error | Stop server, click Generate | Error message displayed, no crash |
| World seed change | Change seed from 42 to 99 | Recipe completely changes |
| Negative coordinates | Set X=-100 | Valid recipe, valid audio |
| Large coordinates | Set X=500, Y=500, Z=300 | Valid recipe, valid audio |

---

## Performance Goals

| Metric | Target |
|---|---|
| Recipe update on slider change | < 5ms (client-side hash only) |
| Preset voice TTS | Same latency as V1 (~1-3s for short text) |
| ZeroVoice TTS (first for a coordinate) | < 5s (includes ~15ms SLERP + model inference) |
| ZeroVoice TTS (cached coordinate) | Same as preset (~1-3s) |
| Frontend load time | < 3s after server health check |
| History recall | Instant (slider update only) |

---

## File Manifest

```
K:\voxtral-mini-4b\
+-- zerovoice_frontend.py              # NEW: V2 frontend (this plan's output)
+-- zerovoice.py                       # Existing: coordinate engine
+-- slerp_voices.py                    # Existing: SLERP math
+-- ZeroVoiceVoxtral-FrontendV2-plan.md # This file

WSL2:
~/voxtral-tts/
+-- zerovoice_frontend.py              # Copy of V2 frontend
+-- text_preprocess.py                 # Copied from vllm-omni examples
+-- .venv/lib/python3.12/site-packages/
    +-- zerovoice.py                   # Already installed
    +-- slerp_voices.py                # Already installed
    +-- gradio/                        # 5.50.0, already installed
```
