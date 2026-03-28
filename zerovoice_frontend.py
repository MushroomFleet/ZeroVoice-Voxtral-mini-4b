"""ZeroVoice Voxtral TTS — Frontend V2

Standalone Gradio 5.50 web application with two tabs:
  - Preset Voices: original 20 voices with language/voice cascading dropdowns
  - Voice Explorer: 3D coordinate sliders for procedural ZeroVoice blending

Requires:
  - vLLM server running with ZeroVoice patches (start_server.sh)
  - zerovoice.py and slerp_voices.py in Python path
  - text_preprocess.py in the same directory or Python path

Launch:
  python zerovoice_frontend.py --host localhost --port 8000 \
    --model /mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603
"""

import argparse
import io
import logging
import time
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Voice organization (same logic as V1)
# ---------------------------------------------------------------------------

LANGUAGE_PREFIXES = {
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "hi": "Hindi",
}

_DEFAULT_VOICES = [
    "casual_female", "casual_male", "cheerful_female", "neutral_female", "neutral_male",
    "ar_male", "de_female", "de_male", "es_female", "es_male",
    "fr_female", "fr_male", "hi_female", "hi_male",
    "it_female", "it_male", "nl_female", "nl_male", "pt_female", "pt_male",
]


def organize_voices_by_language(voices: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    language_voices: dict[str, list[str]] = {}
    for voice in voices:
        found = None
        for prefix, lang_name in LANGUAGE_PREFIXES.items():
            if voice.lower().startswith(f"{prefix}_"):
                found = lang_name
                break
        lang = found or "English"
        language_voices.setdefault(lang, []).append(voice)
    for lang in language_voices:
        if lang == "English":
            language_voices[lang].sort(key=lambda v: (0 if v == "neutral_male" else 1, v))
        else:
            language_voices[lang].sort()
    sorted_langs = sorted(language_voices.keys(), key=lambda x: (0 if x == "English" else 1, x.lower()))
    return sorted_langs, language_voices


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def sanitize_text(text: str) -> str:
    """Sanitize text for TTS. Tries to import the full preprocessor, falls back to basic cleanup."""
    try:
        from text_preprocess import sanitize_tts_input_text_for_demo
        return sanitize_tts_input_text_for_demo(text)
    except ImportError:
        text = text.strip()
        if not text:
            raise ValueError("Text is empty")
        if text[-1] not in ".!?":
            text += "."
        return text


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------

def wait_for_server(base_url: str, timeout: float = 120.0) -> bool:
    health_url = base_url.replace("/v1", "") + "/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(health_url, timeout=5.0)
            if r.status_code == 200:
                logger.info("Server is available")
                return True
        except Exception:
            pass
        logger.info("Waiting for server...")
        time.sleep(3.0)
    return False


def fetch_preset_voices(base_url: str) -> tuple[list[str], dict[str, list[str]]]:
    try:
        r = httpx.get(f"{base_url}/audio/voices", timeout=10.0)
        if r.status_code == 200:
            voices = r.json().get("voices", [])
            if voices:
                return organize_voices_by_language(sorted(voices))
    except Exception as e:
        logger.warning(f"Failed to fetch voices: {e}")
    return organize_voices_by_language(_DEFAULT_VOICES)


def run_inference(voice: str, text: str, base_url: str, model: str) -> tuple[int, np.ndarray]:
    text = sanitize_text(text.strip())
    r = httpx.post(
        f"{base_url}/audio/speech",
        json={"input": text, "model": model, "response_format": "wav", "voice": voice},
        timeout=120.0,
    )
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:200])
        except Exception:
            msg = r.text[:200]
        raise gr.Error(f"Server error: {msg}")
    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return sr, audio


# ---------------------------------------------------------------------------
# ZeroVoice recipe (client-side, no server call)
# ---------------------------------------------------------------------------

def get_recipe(x: int, y: int, z: int, seed: int) -> dict:
    from zerovoice import voice_recipe
    return voice_recipe(int(x), int(y), int(z), int(seed))


def get_voice_name(x: int, y: int, z: int, seed: int) -> str:
    from zerovoice import voice_name
    return voice_name(int(x), int(y), int(z), int(seed))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main(host: str, port: int, model: str, default_seed: int, output_dir: str | None) -> None:
    base_url = f"http://{host}:{port}/v1"
    logger.info(f"Connecting to server at {base_url}")

    if not wait_for_server(base_url):
        logger.warning("Server not available, starting with fallback voices")

    languages, language_voices = fetch_preset_voices(base_url)
    logger.info(f"Loaded {sum(len(v) for v in language_voices.values())} voices across {len(languages)} languages")

    # Mutable history state
    history_data: list[list[str]] = []

    # --- Callbacks ---

    def update_preset_dropdown(language: str) -> gr.Dropdown:
        voices = language_voices.get(language, [])
        return gr.Dropdown(choices=voices, value=voices[0] if voices else None, interactive=True)

    def toggle_btn(text: str) -> gr.update:
        return gr.update(interactive=bool(text.strip()))

    def preset_generate(voice: str, text: str):
        if not text.strip():
            raise gr.Error("Please enter text to synthesize.")
        t0 = time.time()
        sr, audio = run_inference(voice, text, base_url, model)
        elapsed = time.time() - t0
        return (sr, audio), f"Generated in {elapsed:.1f}s"

    def preset_clear():
        default_lang = "English" if "English" in languages else languages[0]
        voices = language_voices.get(default_lang, [])
        return (
            default_lang,
            voices[0] if voices else None,
            "",
            None,
            gr.update(interactive=False),
            "",
        )

    def update_recipe(x, y, z, seed):
        recipe = get_recipe(x, y, z, seed)
        coord = get_voice_name(x, y, z, seed)
        return recipe, coord

    def make_nav(axis: str, direction: int):
        def nav(x, y, z, step):
            x, y, z, step = int(x), int(y), int(z), int(step)
            if axis == "x":
                x += direction * step
            elif axis == "y":
                y += direction * step
            elif axis == "z":
                z = max(0, z + direction * step)
            return x, y, z
        return nav

    def explorer_generate(x, y, z, seed, text):
        if not text.strip():
            raise gr.Error("Please enter text to synthesize.")
        x, y, z, seed = int(x), int(y), int(z), int(seed)
        coord = get_voice_name(x, y, z, seed)
        recipe = get_recipe(x, y, z, seed)
        t0 = time.time()
        sr, audio = run_inference(coord, text, base_url, model)
        elapsed = time.time() - t0
        history_data.insert(0, [
            coord,
            recipe["voice_a"],
            recipe["voice_b"],
            f"{recipe['t']:.3f}",
            f"{elapsed:.1f}s",
        ])
        if len(history_data) > 10:
            history_data.pop()
        return (sr, audio), history_data, f"Generated in {elapsed:.1f}s"

    def recall_history(evt: gr.SelectData):
        if evt.index is None or evt.index[0] >= len(history_data):
            return gr.update(), gr.update(), gr.update()
        row = history_data[evt.index[0]]
        coord_str = row[0]  # e.g. "zv_50_30_10"
        from zerovoice import parse_voice_name
        x, y, z, seed = parse_voice_name(coord_str)
        return x, y, z

    # --- Build UI ---

    with gr.Blocks(title="ZeroVoice Voxtral TTS", fill_height=True, theme=gr.themes.Soft()) as demo:
        gr.Markdown("## ZeroVoice Voxtral TTS")

        with gr.Row():
            with gr.Column(scale=5):
                text_prompt = gr.Textbox(
                    label="Text",
                    placeholder="Enter the text you want to synthesize...",
                    lines=3,
                )
            with gr.Column(scale=1, min_width=120):
                world_seed = gr.Number(
                    value=default_seed, label="World Seed",
                    precision=0, minimum=0, maximum=999999,
                )

        with gr.Tabs():
            # ==================== Tab 1: Preset Voices ====================
            with gr.TabItem("Preset Voices"):
                with gr.Row():
                    with gr.Column():
                        default_lang = "English" if "English" in languages else languages[0]
                        language_dropdown = gr.Dropdown(
                            choices=languages, label="Language", value=default_lang,
                        )
                        default_voices = language_voices.get(default_lang, [])
                        preset_voice = gr.Dropdown(
                            choices=default_voices, label="Voice",
                            value=default_voices[0] if default_voices else None,
                        )
                        with gr.Row():
                            preset_clear_btn = gr.Button("Clear")
                            preset_gen_btn = gr.Button("Generate", variant="primary", interactive=False)
                        preset_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

                    with gr.Column():
                        preset_audio = gr.Audio(
                            label="Generated Audio",
                            show_download_button=True, interactive=False,
                            autoplay=True, type="numpy",
                        )

                # Preset wiring
                language_dropdown.change(
                    fn=update_preset_dropdown,
                    inputs=[language_dropdown],
                    outputs=[preset_voice],
                )
                text_prompt.change(fn=toggle_btn, inputs=[text_prompt], outputs=[preset_gen_btn])
                preset_gen_btn.click(
                    fn=preset_generate,
                    inputs=[preset_voice, text_prompt],
                    outputs=[preset_audio, preset_status],
                )
                preset_clear_btn.click(
                    fn=preset_clear,
                    outputs=[language_dropdown, preset_voice, text_prompt, preset_audio, preset_gen_btn, preset_status],
                )

            # ==================== Tab 2: Voice Explorer ====================
            with gr.TabItem("Voice Explorer"):
                with gr.Row():
                    # Left: controls
                    with gr.Column(scale=1):
                        gr.Markdown("### Coordinate Controls")
                        coord_x = gr.Slider(
                            minimum=-500, maximum=500, step=1, value=50,
                            label="X (Blend)",
                        )
                        coord_y = gr.Slider(
                            minimum=-500, maximum=500, step=1, value=50,
                            label="Y (Timbre)",
                        )
                        coord_z = gr.Slider(
                            minimum=0, maximum=300, step=1, value=50,
                            label="Z (Family: 0=English, 100=European, 200=Asian/Arabic)",
                        )

                        gr.Markdown("### Navigator")
                        nav_step = gr.Number(value=10, label="Step Size", precision=0, minimum=1, maximum=100)
                        with gr.Row():
                            nav_xm = gr.Button("X-", size="sm")
                            nav_xp = gr.Button("X+", size="sm")
                            nav_ym = gr.Button("Y-", size="sm")
                            nav_yp = gr.Button("Y+", size="sm")
                            nav_zm = gr.Button("Z-", size="sm")
                            nav_zp = gr.Button("Z+", size="sm")

                        with gr.Row():
                            explorer_gen_btn = gr.Button("Generate", variant="primary", interactive=False)
                        explorer_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

                    # Right: recipe + audio
                    with gr.Column(scale=1):
                        coord_label = gr.Textbox(
                            label="Coordinate", interactive=False,
                            show_copy_button=True, max_lines=1,
                        )
                        recipe_display = gr.JSON(label="Voice Recipe")
                        explorer_audio = gr.Audio(
                            label="Generated Audio",
                            show_download_button=True, interactive=False,
                            autoplay=True, type="numpy",
                        )

                # History below
                gr.Markdown("### Recent Coordinates")
                history_table = gr.Dataframe(
                    headers=["Coordinate", "Voice A", "Voice B", "t_AB", "Time"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=(5, "fixed"),
                )

                # Explorer wiring: recipe updates on slider/seed change
                slider_inputs = [coord_x, coord_y, coord_z, world_seed]
                recipe_outputs = [recipe_display, coord_label]

                for component in slider_inputs:
                    component.change(
                        fn=update_recipe,
                        inputs=slider_inputs,
                        outputs=recipe_outputs,
                    )

                # Enable generate button when text present
                text_prompt.change(fn=toggle_btn, inputs=[text_prompt], outputs=[explorer_gen_btn])

                # Navigator buttons
                nav_outputs = [coord_x, coord_y, coord_z]
                nav_inputs = [coord_x, coord_y, coord_z, nav_step]
                nav_xm.click(fn=make_nav("x", -1), inputs=nav_inputs, outputs=nav_outputs)
                nav_xp.click(fn=make_nav("x", +1), inputs=nav_inputs, outputs=nav_outputs)
                nav_ym.click(fn=make_nav("y", -1), inputs=nav_inputs, outputs=nav_outputs)
                nav_yp.click(fn=make_nav("y", +1), inputs=nav_inputs, outputs=nav_outputs)
                nav_zm.click(fn=make_nav("z", -1), inputs=nav_inputs, outputs=nav_outputs)
                nav_zp.click(fn=make_nav("z", +1), inputs=nav_inputs, outputs=nav_outputs)

                # Generate
                explorer_gen_btn.click(
                    fn=explorer_generate,
                    inputs=[coord_x, coord_y, coord_z, world_seed, text_prompt],
                    outputs=[explorer_audio, history_table, explorer_status],
                )

                # Recall from history
                history_table.select(
                    fn=recall_history,
                    outputs=[coord_x, coord_y, coord_z],
                )

        # Initialize recipe display on load
        demo.load(
            fn=update_recipe,
            inputs=[coord_x, coord_y, coord_z, world_seed],
            outputs=recipe_outputs,
        )

    demo.launch(server_name="0.0.0.0", share=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroVoice Voxtral TTS — Frontend V2")
    parser.add_argument("--host", type=str, default="localhost", help="vLLM server hostname")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument(
        "--model", type=str,
        default="/mnt/k/voxtral-mini-4b/Voxtral-4B-TTS-2603",
        help="Model path (must match server)",
    )
    parser.add_argument("--world-seed", type=int, default=42, help="Default world seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save audio")
    args = parser.parse_args()
    main(args.host, args.port, args.model, args.world_seed, args.output_dir)
