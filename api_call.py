import time
from pathlib import Path
from urllib.parse import unquote

import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 600


def print_elapsed(name: str, started_at: float) -> None:
    print(f"{name} time: {time.perf_counter() - started_at:.2f}s")


def save_wav(response: requests.Response, output_path: str) -> None:
    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code} for {response.url}: {response.text}"
        )
    Path(output_path).write_bytes(response.content)
    print(f"saved: {output_path}")


# ---- 1) Speech recognition (ASR) -----------------------------------------
started_at = time.perf_counter()
with open("assets/examples/asr_zh_fleurs.wav", "rb") as audio:
    response = requests.post(
        f"{BASE_URL}/v1/asr",
        files={"audio": audio},
        timeout=TIMEOUT,
    )
response.raise_for_status()
print("ASR:", response.json()["text"])
print_elapsed("ASR", started_at)


# ---- 2) Audio understanding (with optional chain-of-thought) --------------
started_at = time.perf_counter()
with open("assets/examples/two_speakers.wav", "rb") as audio:
    response = requests.post(
        f"{BASE_URL}/v1/understand",
        files={"audio": audio},
        data={
            "prompt": "这个音频中有几个说话人",
            "enable_thinking": "true",
            "max_new_tokens": "10240",
        },
        timeout=TIMEOUT,
    )
response.raise_for_status()
understanding = response.json()
print("Understanding:", understanding["answer"])
if understanding.get("reasoning"):
    print("Reasoning:", understanding["reasoning"])
print_elapsed("Understanding", started_at)

started_at = time.perf_counter()
with open("assets/examples/assets_mmau_test.wav", "rb") as audio:
    response = requests.post(
        f"{BASE_URL}/v1/understand",
        files={"audio": audio},
        data={
            "prompt": "What illness did Second speaker's friend suffer from?\n(A) Progressive arthritis (B) Progressive cancer (C) Acute pneumonia (D) Chronic heart disease",
            "enable_thinking": "true",
            "max_new_tokens": "10240",
        },
        timeout=TIMEOUT,
    )
response.raise_for_status()
understanding = response.json()
print("Understanding:", understanding["answer"])
if understanding.get("reasoning"):
    print("Reasoning:", understanding["reasoning"])
print_elapsed("Understanding", started_at)

# ---- 3) Zero-shot TTS (ICL voice cloning) ---------------------------------
started_at = time.perf_counter()
with open("assets/examples/tts_zh_prompt.wav", "rb") as prompt_audio:
    response = requests.post(
        f"{BASE_URL}/v1/tts",
        files={"prompt_audio": prompt_audio},
        data={
            "prompt_text": "同时，他强调微调要科学有序。",
            "target_text": "你好，这是 FireRedAudio。",
            "language": "zh",
        },
        timeout=TIMEOUT,
    )
save_wav(response, "api_tts.wav")
print_elapsed("TTS", started_at)


# ---- 4) Semantic speech editing ------------------------------------------
started_at = time.perf_counter()
with open("assets/examples/edit_semantic_zh_ref.wav", "rb") as audio:
    response = requests.post(
        f"{BASE_URL}/v1/edit",
        files={"audio": audio},
        data={
            "instruction": "delete '比普通的茶叶要'",
            "edit_type": "semantic",
        },
        timeout=TIMEOUT,
    )
print("Edited text:", unquote(response.headers.get("X-FireRedAudio-Text", "")))
save_wav(response, "api_edit_semantic.wav")
print_elapsed("Semantic editing", started_at)


# ---- 5) Acoustic speech editing ------------------------------------------
started_at = time.perf_counter()
with open("assets/examples/edit_acoustic_zh_ref.wav", "rb") as audio:
    response = requests.post(
        f"{BASE_URL}/v1/edit",
        files={"audio": audio},
        data={
            "instruction": "shift the pitch by 3 steps",
            "edit_type": "acoustic",
        },
        timeout=TIMEOUT,
    )
save_wav(response, "api_edit_acoustic.wav")
print_elapsed("Acoustic editing", started_at)


# ---- 6) Voice design ------------------------------------------------------
started_at = time.perf_counter()
response = requests.post(
    f"{BASE_URL}/v1/voice-design",
    data={
        "instruction": "女性高音区的清亮音色，年轻，语速适中。",
        "text": "你好，这是一个音色设计测试。",
    },
    timeout=TIMEOUT,
)
save_wav(response, "api_voice_design.wav")
print_elapsed("Voice design", started_at)