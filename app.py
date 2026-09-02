import os
import random
import tempfile

# Force PyTorch SDPA to use the math backend.
try:
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend

    _original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def _safe_sdpa(query, key, value, *args, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return _original_sdpa(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = _safe_sdpa
    print("✅ SDPA patched: using MATH backend")
except Exception as e:
    print("⚠️ SDPA patch failed:", e)

import gradio as gr
import torch
import torchaudio

from inference import FireRedAudioInference


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "pretrained_models/FireRedAudio"
VAE_PATH = "pretrained_models/RedAE_decoder/model.pt"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
QUANTIZATION = "int4"

print("=" * 60)
print("FireRedAudio")
print("=" * 60)
print(f"Device:       {DEVICE}")
print(f"Quantization: {QUANTIZATION}")
print(f"Model:        {MODEL_PATH}")
print("=" * 60)


# ============================================================
# Load model
# ============================================================

engine = None


def load_engine():
    global engine

    if engine is not None:
        return engine

    print("Loading FireRedAudio...")

    engine = FireRedAudioInference(
        model_path=MODEL_PATH,
        vae_decoder_path=VAE_PATH,
        device=DEVICE,
        quantization=QUANTIZATION,
    )

    print("FireRedAudio loaded successfully!")

    return engine


# ============================================================
# Helpers
# ============================================================

def check_audio(audio):
    if audio is None:
        raise gr.Error("Please provide an audio file.")

    return audio


def audio_to_path(audio):
    """
    Gradio audio input can be a filepath.
    Return the path directly when possible.
    """
    if isinstance(audio, str):
        return audio

    if isinstance(audio, tuple):
        sample_rate, data = audio

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        waveform = torch.tensor(data)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)

        torchaudio.save(
            path,
            waveform.float(),
            sample_rate,
        )

        return path

    raise gr.Error("Unsupported audio input format.")


# ============================================================
# ASR
# ============================================================

def run_asr(audio):
    audio = check_audio(audio)
    audio_path = audio_to_path(audio)

    model = load_engine()

    result = model.understand(
        audio_path,
        "Transcribe the speech to text.",
        task="asr",
    )

    return result.answer


# ============================================================
# Audio Understanding
# ============================================================

def run_understanding(audio, prompt, thinking):
    audio = check_audio(audio)

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a question or instruction.")

    audio_path = audio_to_path(audio)

    model = load_engine()

    result = model.understand(
        audio_path,
        prompt,
        task="understand",
        enable_thinking=thinking,
        max_new_tokens=4096,
    )

    answer = result.answer

    reasoning = result.reasoning

    if reasoning:
        return answer, reasoning

    return answer, "Thinking disabled or no reasoning returned."


# ============================================================
# Zero-Shot TTS / Voice Cloning
# ============================================================

def run_tts(prompt_audio, prompt_text, target_text, language):
    prompt_audio = check_audio(prompt_audio)

    if not prompt_text.strip():
        raise gr.Error("Please enter the transcript of the reference audio.")

    if not target_text.strip():
        raise gr.Error("Please enter the text you want to generate.")

    prompt_path = audio_to_path(prompt_audio)

    model = load_engine()

    result = model.tts(
        prompt_text=prompt_text,
        prompt_audio=prompt_path,
        target_text=target_text,
        language=language,
    )

    output_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ).name

    torchaudio.save(
        output_path,
        result.audio.cpu(),
        sample_rate=24000,
    )

    return output_path


# ============================================================
# Semantic Editing
# ============================================================

def run_semantic_edit(audio, instruction):
    audio = check_audio(audio)

    if not instruction.strip():
        raise gr.Error("Please enter an editing instruction.")

    audio_path = audio_to_path(audio)

    model = load_engine()

    result = model.edit(
        audio_path,
        instruction,
        edit_type="semantic",
    )

    output_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ).name

    torchaudio.save(
        output_path,
        result.audio.cpu(),
        sample_rate=24000,
    )

    return result.text, output_path


# ============================================================
# Acoustic Editing
# ============================================================

def run_acoustic_edit(audio, edit_mode, value):
    audio = check_audio(audio)

    audio_path = audio_to_path(audio)

    model = load_engine()

    if edit_mode == "Pitch":
        instruction = f"shift the pitch by {int(value)} steps"

    elif edit_mode == "Speed":
        instruction = f"adjust the speed to {value:.1f}"

    elif edit_mode == "Volume":
        instruction = f"adjust the volume to {value:.1f}"

    else:
        raise gr.Error("Unknown editing mode.")

    result = model.edit(
        audio_path,
        instruction,
        edit_type="acoustic",
    )

    output_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ).name

    torchaudio.save(
        output_path,
        result.audio.cpu(),
        sample_rate=24000,
    )

    return output_path


# ============================================================
# Voice Design
# ============================================================

def run_voice_design(instruction, text):
    if not instruction.strip():
        raise gr.Error("Please describe the desired voice.")

    if not text.strip():
        raise gr.Error("Please enter the text to speak.")

    model = load_engine()

    result = model.voice_design(
        instruction=instruction,
        text=text,
    )

    output_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ).name

    torchaudio.save(
        output_path,
        result.audio.cpu(),
        sample_rate=24000,
    )

    return output_path


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="FireRedAudio",
) as demo:

    gr.Markdown(
        """
# 🔥 FireRedAudio

**One model to listen, understand, reason, speak, and edit.**

FireRedAudio provides:

- 🎤 Speech Recognition
- 🧠 Audio Understanding
- 🔊 Zero-Shot TTS / Voice Cloning
- ✏️ Semantic Speech Editing
- 🎚️ Acoustic Speech Editing
- 🎙️ Voice Design

Running with **NF4 INT4 quantization** for lower VRAM usage.
"""
    )

    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    with gr.Tab("🎤 ASR"):

        gr.Markdown(
            "Upload an audio file and FireRedAudio will transcribe it."
        )

        asr_audio = gr.Audio(
            type="filepath",
            label="Audio",
        )

        asr_button = gr.Button(
            "🎤 Transcribe",
            variant="primary",
        )

        asr_output = gr.Textbox(
            label="Transcription",
            lines=8,
        )

        asr_button.click(
            run_asr,
            inputs=asr_audio,
            outputs=asr_output,
        )

    # --------------------------------------------------------
    # Understanding
    # --------------------------------------------------------

    with gr.Tab("🧠 Understanding"):

        understand_audio = gr.Audio(
            type="filepath",
            label="Audio",
        )

        understand_prompt = gr.Textbox(
            label="Question / Prompt",
            placeholder="What is being discussed in this audio?",
            lines=4,
        )

        thinking = gr.Checkbox(
            label="Enable Thinking",
            value=False,
        )

        understand_button = gr.Button(
            "🧠 Analyze Audio",
            variant="primary",
        )

        understand_answer = gr.Textbox(
            label="Answer",
            lines=6,
        )

        understand_reasoning = gr.Textbox(
            label="Reasoning",
            lines=10,
        )

        understand_button.click(
            run_understanding,
            inputs=[
                understand_audio,
                understand_prompt,
                thinking,
            ],
            outputs=[
                understand_answer,
                understand_reasoning,
            ],
        )

    # --------------------------------------------------------
    # TTS / Voice Cloning
    # --------------------------------------------------------

    with gr.Tab("🔊 Voice Cloning / TTS"):

        gr.Markdown(
            """
### Zero-Shot Voice Cloning

Upload a short reference recording, provide its transcript,
then enter the text you want FireRedAudio to speak.
"""
        )

        tts_prompt_audio = gr.Audio(
            type="filepath",
            label="Reference Voice",
        )

        tts_prompt_text = gr.Textbox(
            label="Reference Audio Transcript",
            placeholder="Enter exactly what is spoken in the reference audio.",
            lines=4,
        )

        tts_target_text = gr.Textbox(
            label="Text to Generate",
            placeholder="Enter the text you want the cloned voice to speak.",
            lines=6,
        )

        tts_language = gr.Dropdown(
            choices=[
                "zh",
                "en",
            ],
            value="en",
            label="Language",
        )

        tts_button = gr.Button(
            "🔊 Generate Speech",
            variant="primary",
        )

        tts_output = gr.Audio(
            label="Generated Audio",
            type="filepath",
        )

        tts_button.click(
            run_tts,
            inputs=[
                tts_prompt_audio,
                tts_prompt_text,
                tts_target_text,
                tts_language,
            ],
            outputs=tts_output,
        )

    # --------------------------------------------------------
    # Semantic Editing
    # --------------------------------------------------------

    with gr.Tab("✏️ Semantic Editing"):

        gr.Markdown(
            """
Use natural language to modify what was said.

Examples:

- `delete 'hello'`
- `replace 'Monday' with 'Friday'`
- `insert 'very' before happy`
"""
        )

        semantic_audio = gr.Audio(
            type="filepath",
            label="Original Audio",
        )

        semantic_instruction = gr.Textbox(
            label="Editing Instruction",
            placeholder="delete 'some words'",
            lines=4,
        )

        semantic_button = gr.Button(
            "✏️ Edit Speech",
            variant="primary",
        )

        semantic_text = gr.Textbox(
            label="Resulting Text",
            lines=5,
        )

        semantic_output = gr.Audio(
            label="Edited Audio",
            type="filepath",
        )

        semantic_button.click(
            run_semantic_edit,
            inputs=[
                semantic_audio,
                semantic_instruction,
            ],
            outputs=[
                semantic_text,
                semantic_output,
            ],
        )

    # --------------------------------------------------------
    # Acoustic Editing
    # --------------------------------------------------------

    with gr.Tab("🎚️ Acoustic Editing"):

        gr.Markdown(
            """
Change the acoustic properties of an existing recording.

**Pitch:** -6 to +6 steps  
**Speed:** 0.5x to 2.0x  
**Volume:** 0.3x to 2.0x
"""
        )

        acoustic_audio = gr.Audio(
            type="filepath",
            label="Original Audio",
        )

        acoustic_mode = gr.Radio(
            choices=[
                "Pitch",
                "Speed",
                "Volume",
            ],
            value="Pitch",
            label="Edit Type",
        )

        acoustic_value = gr.Slider(
            minimum=-6,
            maximum=6,
            step=1,
            value=1,
            label="Value",
        )

        acoustic_button = gr.Button(
            "🎚️ Apply Acoustic Edit",
            variant="primary",
        )

        acoustic_output = gr.Audio(
            label="Edited Audio",
            type="filepath",
        )

        def update_acoustic_slider(mode):
            if mode == "Pitch":
                return gr.update(
                    minimum=-6,
                    maximum=6,
                    step=1,
                    value=1,
                    label="Pitch Steps",
                )

            if mode == "Speed":
                return gr.update(
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Speed",
                )

            return gr.update(
                minimum=0.3,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Volume",
            )

        acoustic_mode.change(
            update_acoustic_slider,
            inputs=acoustic_mode,
            outputs=acoustic_value,
        )

        acoustic_button.click(
            run_acoustic_edit,
            inputs=[
                acoustic_audio,
                acoustic_mode,
                acoustic_value,
            ],
            outputs=acoustic_output,
        )

    # --------------------------------------------------------
    # Voice Design
    # --------------------------------------------------------

    with gr.Tab("🎙️ Voice Design"):

        gr.Markdown(
            """
Describe the voice you want and FireRedAudio will generate speech
using that description.

Example:

> A young female voice with a bright, clear tone, slightly fast
> speaking speed and an energetic, friendly personality.
"""
        )

        voice_instruction = gr.Textbox(
            label="Voice Description",
            placeholder="Describe the voice...",
            lines=6,
        )

        voice_text = gr.Textbox(
            label="Text",
            placeholder="Enter the text to speak...",
            lines=6,
        )

        voice_button = gr.Button(
            "🎙️ Generate Voice",
            variant="primary",
        )

        voice_output = gr.Audio(
            label="Generated Audio",
            type="filepath",
        )

        voice_button.click(
            run_voice_design,
            inputs=[
                voice_instruction,
                voice_text,
            ],
            outputs=voice_output,
        )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    gr.Markdown(
        """
---

### ⚙️ Configuration

**Device:** `{device}`  
**Quantization:** `{quantization}`  
**Model:** `{model}`

> Voice cloning should only be used with appropriate authorization
> and for legitimate purposes.
""".format(
            device=DEVICE,
            quantization=QUANTIZATION,
            model=MODEL_PATH,
        )
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    print("Starting FireRedAudio Web UI...")

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
