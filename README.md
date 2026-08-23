<h1>FireRedAudio</h1>
<div align="center">
    <p>
    </p>
    <p>
    Official PyTorch code for <br>
    <b><em>FireRedAudio: A General-Purpose Audio Language Model with Decoupled Continuous Representations for Understanding and Generation</em></b>
    </p>
    <p>
    </p>
    <a href="#"><img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="technical report"></a>
    <a href="https://fireredteam.github.io/demos/fireredaudio/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/FireRedTeam/FireRedAudio"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="HF-model"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
</div>

## Overview

> **One model to listen, understand, reason, speak, and edit.**

**FireRedAudio** is a general-purpose audio language model built on a shared **9B-parameter LLM** with **decoupled continuous representations**: an Audio Encoder handles understanding, while a RedAE pathway handles generation. A single model supports **ASR, audio understanding, zero-shot TTS, instruct TTS, semantic/acoustic speech editing, and accurate temporal grounding over recordings up to one hour long**.

<div align="center">
  <img src="assets/fireredaudio_logo.png" alt="FireRedAudio Logo" width="90%">
  <br>
</div>

## Highlights ✨

* 🧩 **Purpose-built representations, one shared backbone** — The Audio Encoder pathway serves understanding, while the RedAE-Patch pathway serves speech generation. Their representations remain decoupled but share the same language and reasoning backbone. To the best of our knowledge, this is the first publicly disclosed design of its kind in a unified audio-language model.
* 📊 **One model, a full audio stack** — FireRedAudio spans ASR, broad and fine-grained audio understanding, zero-shot TTS, Instruct TTS, and free-form speech editing, achieving competitive or leading results across MMAU, MMSU, Seed-TTS-Eval, InstructTTSEval, and Ming-Freeform-Audio-Edit.
* 🎙️ **Create and edit speech with natural language** — Clone a voice from a reference clip, design a voice from a description, or edit what was said and how it sounds through one continuous-latent generation pathway.
* ⏱️ **Go from minutes to hour-long recordings** — Understand recordings up to one hour with precise time-to-content alignment. Organize audio into timestamped structures, produce grounded summaries, retrieve content by time (or time by content), and reason over evidence distributed across the recording.

## News

* [2026.08.21] We release the **FireRedAudio** code and model.

## Contents

* [Quick Start](#quick-start-)
* [Performance](#performance)
* [Limitations](#limitations)
* [Usage Disclaimer](#usage-disclaimer)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)
* [License](#license)

## Quick Start 🚀

### Installation

Requires **Python 3.10** and [uv](https://docs.astral.sh/uv/). System prerequisites: a
**CUDA toolkit** (GPU inference + compiling the causal-conv1d / flash-attn kernels) and
**ffmpeg** (audio decoding via torchaudio/torchcodec).

Wheels target **CUDA 12.8 (`cu128`)** by default. For a different CUDA version, change the
`pytorch-cu128` index URL in [`pyproject.toml`](pyproject.toml) (e.g. `cu126` / `cu129`) and
re-run `uv sync`.

```sh
uv sync --extra accel --extra accel-build --group tools   # + compile causal-conv1d, flash-attn
```

### Quantized inference

Supported linear layers can be loaded with optional NF4 4-bit quantization:

```sh
uv sync --extra quantization
uv run inference.py --task asr --model pretrained_models/FireRedAudio \
  --audio assets/examples/asr_zh_fleurs.wav --quantization int4
```

`int4` uses bitsandbytes NF4 and requires a supported CUDA GPU. Validate ASR and
understanding quality before using quantization for TTS or speech editing.

### Export a local int4 model

To materialize the quantized weights as local safetensors shards, run the
export script on a CUDA machine:

```sh
uv sync --extra quantization
uv run python export_int4_model.py \
  --model pretrained_models/FireRedAudio \
  --output pretrained_models/FireRedAudio-int4 \
  --device cuda:0
```

The output directory contains the NF4 model shards, quantization configuration,
tokenizer, and processor files. It can be used directly for inference:

```sh
uv run inference.py --task asr --model pretrained_models/FireRedAudio-int4 \
  --audio assets/examples/asr_zh_fleurs.wav --quantization int4
```

The output directory must be empty or absent. The export requires enough GPU
memory to load the original model before saving the quantized shards.

### Model Download

Download the pretrained model from Hugging Face with the `hf` CLI:

```sh
hf download FireRedTeam/FireRedAudio --local-dir pretrained_models/
```

### Python API

```python
import torch
import torchaudio
from inference import FireRedAudioInference

# Init the model. Understanding tasks need only --model; generation tasks
# additionally need the RedAE decoder weights.
quantization = "int4"  # use None for BF16 inference
engine = FireRedAudioInference(
    model_path="pretrained_models/FireRedAudio",
    vae_decoder_path="pretrained_models/RedAE_decoder/model.pt",   # only for tts / edit / voice_design
    device="cuda:0",
    quantization=quantization,
)

# ---- 1) Speech recognition (ASR) -----------------------------------------
res = engine.understand("assets/examples/asr_zh_fleurs.wav", "Transcribe speech to text.", task="asr")
print(res.answer)

# ---- 2) Audio understanding (with optional chain-of-thought) --------------
res = engine.understand(
    "assets/examples/assets_mmau_test.wav",
    "What illness did Second speaker's friend suffer from?\n(A) Progressive arthritis (B) Progressive cancer (C) Acute pneumonia (D) Chronic heart disease",
    task="understand", enable_thinking=True, max_new_tokens=10240,
)
print("CoT:")
print(res.reasoning)   # CoT reasoning, or None
print("Answer")
print(res.answer)

# ---- 3) Zero-shot TTS (ICL voice cloning) ---------------------------------
res = engine.tts(
    prompt_text="同时，他强调微调要科学有序。",
    prompt_audio="assets/examples/tts_zh_prompt.wav",
    target_text="安徽淮南秦师傅发现，停在小区的爱车右前驾驶窗玻璃被砸。",
    language="zh",
)
torchaudio.save("tts.wav", res.audio.cpu(), sample_rate=24000)

# ---- 4) Speech editing -----------------------------------------------------
# semantic: rewrite / substitute / insert / delete content. The model first writes
#          <|sot|>{rewritten text}<|eot|> then renders the audio.
res = engine.edit("assets/examples/edit_semantic_zh_ref.wav", "delete '比普通的茶叶要'", edit_type="semantic")
print(res.text)
torchaudio.save("edit_semantic.wav", res.audio.cpu(), sample_rate=24000)

# acoustic: change pitch / speed / volume. The instruction must follow the exact
#           templates below (the model is trained on these, not free-form phrasing):
#   pitch   ->  "shift the pitch by N step(s)"       N in {-6, ..., -1, 1, ..., +6}
#   speed   ->  "adjust the speed to X"              X in [0.5, 2.0], step 0.1
#   volume  ->  "adjust the volume to X"             X in [0.3, 2.0], step 0.1
res = engine.edit("assets/examples/edit_acoustic_zh_ref.wav", "shift the pitch by 3 steps", edit_type="acoustic")
torchaudio.save("edit_acoustic.wav", res.audio.cpu(), sample_rate=24000)

# ---- 5) Voice design (synthesis from a timbre description) -----------------
res = engine.voice_design(
    instruction="以女性高音区的清亮音色,表现出青年阶段的特质,音量略强,语速适中稍快,语调带有解释意味和急切的情感流露,确保语音流畅自然。",
    text="是我请他来的，可他什么也不知道，他来只是想打听一下，你们厂是不是有旧锅炉？",
)
torchaudio.save("voice_design.wav", res.audio.cpu(), sample_rate=24000)
```

### HTTP API

API 服务基于 FastAPI。启动前请确认已经安装项目依赖，并且服务所在机器
可以使用 CUDA GPU。模型只在服务启动时加载一次，后续请求会复用同一个模型实例。

#### 启动服务

安装依赖：

```sh
uv sync --extra quantization
```

直接启动即可使用默认配置：

```sh
uv run api_server.py
```

默认配置如下：

| 参数         | 默认值                                       |
| ------------ | -------------------------------------------- |
| 模型目录     | `pretrained_models/FireRedAudio`           |
| RedAE 解码器 | `pretrained_models/RedAE_decoder/model.pt` |
| 设备         | `cuda:0`                                   |
| 量化         | `int4`                                     |
| 地址         | `127.0.0.1`                                |
| 端口         | `8000`                                     |

也可以覆盖默认配置：

```sh
uv run api_server.py --model /path/to/FireRedAudio \
  --vae-decoder /path/to/model.pt --device cuda:1 \
  --quantization int4 --host 0.0.0.0 --port 8000
```

`--model`、`--vae-decoder`、`--quantization`、`--device`、`--host` 和 `--port`
都可以通过命令行覆盖。当前量化选项只有 `int4`；不使用量化时传入
`--quantization` 以外的方式不可用，需要修改启动配置或直接使用 Python API 的
`quantization=None`。

#### 健康检查

```sh
curl http://127.0.0.1:8000/health
```

正常返回：

```json
{"status":"ok"}
```

如果模型还没有初始化，会返回 `{"status":"starting"}`。

#### ASR 语音识别

请求格式为 `multipart/form-data`，字段名必须是 `audio`：

```sh
curl -X POST http://127.0.0.1:8000/v1/asr \
  -F "audio=@assets/examples/asr_zh_fleurs.wav"
```

返回 JSON：

```json
{"text":"识别出的文字"}
```

#### 音频理解

`prompt` 是必填字段。`enable_thinking` 和 `max_new_tokens` 是可选字段，
适合需要模型先进行推理再回答的场景：

```sh
curl -X POST http://127.0.0.1:8000/v1/understand \
  -F "audio=@assets/examples/two_speakers.wav" \
  -F "prompt=这个音频中有几个说话人" \
  -F "enable_thinking=true" \
  -F "max_new_tokens=1024"
```

返回 JSON：

```json
{
  "answer": "音频中有两个说话人。",
  "reasoning": "模型的思考过程"
}
```

不启用思考时，`reasoning` 通常为 `null`。`enable_thinking` 只适用于
`/v1/understand`，不适用于 `/v1/asr`、`/v1/tts` 或编辑接口。

#### Zero-Shot TTS

需要上传参考音频，并提供参考音频文本和目标文本。接口直接返回 WAV 文件：

```sh
curl -X POST http://127.0.0.1:8000/v1/tts \
  -F "prompt_audio=@assets/examples/tts_zh_prompt.wav" \
  -F "prompt_text=同时，他强调微调要科学有序。" \
  -F "target_text=你好，这是 FireRedAudio。" \
  -F "language=zh" -o output.wav
```

字段说明：

| 字段             | 必填 | 说明                          |
| ---------------- | ---- | ----------------------------- |
| `prompt_audio` | 是   | 参考音频文件                  |
| `prompt_text`  | 是   | 参考音频对应的文本            |
| `target_text`  | 是   | 需要合成的文本                |
| `language`     | 否   | `zh` 或 `en`，默认 `zh` |

#### 语音编辑

语义编辑用于删除、插入或替换内容：

```sh
curl -X POST http://127.0.0.1:8000/v1/edit \
  -F "audio=@assets/examples/edit_semantic_zh_ref.wav" \
  -F "instruction=delete '比普通的茶叶要'" \
  -F "edit_type=semantic" -o edit_semantic.wav
```

声学编辑用于调整音高、速度或音量：

```sh
curl -X POST http://127.0.0.1:8000/v1/edit \
  -F "audio=@assets/examples/edit_acoustic_zh_ref.wav" \
  -F "instruction=shift the pitch by 3 steps" \
  -F "edit_type=acoustic" -o edit_acoustic.wav
```

`edit_type` 可选 `semantic` 或 `acoustic`，默认是 `semantic`。编辑接口返回
WAV 文件；语义编辑生成的文本会以 URL 编码形式放在
`X-FireRedAudio-Text` 响应头中，客户端应先 URL 解码再显示中文。

#### 音色设计

根据音色描述生成语音：

```sh
curl -X POST http://127.0.0.1:8000/v1/voice-design \
  -F "instruction=女性高音区的清亮音色，年轻，语速适中。" \
  -F "text=你好，这是一个音色设计测试。" \
  -o voice_design.wav
```

#### Python 客户端

安装客户端依赖：

```sh
pip install requests
```

下面的脚本需要先启动 `api_server.py`，会依次调用所有接口，并将生成的音频
保存到当前目录。`enable_thinking` 和 `max_new_tokens` 通过表单字段传递。

```python
import os
import time
from urllib.parse import unquote

import requests


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 600
HEADERS = {}
if api_key := os.getenv("FIREREDAUDIO_API_KEY"):
  HEADERS["X-API-Key"] = api_key


def request_json(path, *, files=None, data=None):
  response = requests.post(
    f"{BASE_URL}{path}", headers=HEADERS, files=files, data=data, timeout=TIMEOUT
  )
  if not response.ok:
    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
  return response.json()


def request_wav(path, output_path, *, files=None, data=None):
  response = requests.post(
    f"{BASE_URL}{path}", headers=HEADERS, files=files, data=data, timeout=TIMEOUT
  )
  if not response.ok:
    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
  with open(output_path, "wb") as output:
    output.write(response.content)
  return response


def print_time(name, started_at):
  print(f"{name} time: {time.perf_counter() - started_at:.2f}s")


health = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=30)
health.raise_for_status()
print("Health:", health.json())

started_at = time.perf_counter()
with open("assets/examples/asr_zh_fleurs.wav", "rb") as audio:
  result = request_json("/v1/asr", files={"audio": audio})
print("ASR:", result["text"])
print_time("ASR", started_at)

started_at = time.perf_counter()
with open("assets/examples/two_speakers.wav", "rb") as audio:
  result = request_json(
    "/v1/understand",
    files={"audio": audio},
    data={
      "prompt": "这个音频中有几个说话人",
      "enable_thinking": "true",
      "max_new_tokens": "1024",
    },
  )
print("Understanding:", result["answer"])
print("Reasoning:", result.get("reasoning"))
print_time("Understanding", started_at)

started_at = time.perf_counter()
with open("assets/examples/tts_zh_prompt.wav", "rb") as prompt_audio:
  request_wav(
    "/v1/tts",
    "api_tts.wav",
    files={"prompt_audio": prompt_audio},
    data={
      "prompt_text": "同时，他强调微调要科学有序。",
      "target_text": "你好，这是 FireRedAudio。",
      "language": "zh",
    },
  )
print("Saved: api_tts.wav")
print_time("TTS", started_at)

started_at = time.perf_counter()
with open("assets/examples/edit_semantic_zh_ref.wav", "rb") as audio:
  response = request_wav(
    "/v1/edit",
    "api_edit_semantic.wav",
    files={"audio": audio},
    data={"instruction": "delete '比普通的茶叶要'", "edit_type": "semantic"},
  )
print("Edited text:", unquote(response.headers.get("X-FireRedAudio-Text", "")))
print("Saved: api_edit_semantic.wav")
print_time("Semantic editing", started_at)

started_at = time.perf_counter()
with open("assets/examples/edit_acoustic_zh_ref.wav", "rb") as audio:
  request_wav(
    "/v1/edit",
    "api_edit_acoustic.wav",
    files={"audio": audio},
    data={
      "instruction": "shift the pitch by 3 steps",
      "edit_type": "acoustic",
    },
  )
print("Saved: api_edit_acoustic.wav")
print_time("Acoustic editing", started_at)

started_at = time.perf_counter()
request_wav(
  "/v1/voice-design",
  "api_voice_design.wav",
  data={
    "instruction": "女性高音区的清亮音色，年轻，语速适中。",
    "text": "你好，这是一个音色设计测试。",
  },
)
print("Saved: api_voice_design.wav")
print_time("Voice design", started_at)
```

仓库中的 `api_call.py` 还提供了六个功能的完整调用示例，并会统计每次请求的耗时：

```sh
python api_call.py
```

#### API Key 鉴权

设置环境变量后启动服务：

```sh
set FIREREDAUDIO_API_KEY=your-secret-key
uv run api_server.py
```

Linux/macOS 使用：

```sh
export FIREREDAUDIO_API_KEY=your-secret-key
uv run api_server.py
```

客户端请求增加 `X-API-Key` 请求头：

```sh
curl -H "X-API-Key: your-secret-key" http://127.0.0.1:8000/health
```

#### 接口汇总

| 方法 | 路径                 | 输入                                      | 返回      |
| ---- | -------------------- | ----------------------------------------- | --------- |
| GET  | `/health`          | 无                                        | JSON 状态 |
| POST | `/v1/asr`          | `audio` 文件                            | JSON 文本 |
| POST | `/v1/understand`   | `audio`、`prompt` 等表单字段          | JSON 答案 |
| POST | `/v1/tts`          | 参考音频和文本                            | WAV       |
| POST | `/v1/edit`         | `audio`、`instruction`、`edit_type` | WAV       |
| POST | `/v1/voice-design` | `instruction`、`text`                 | WAV       |

服务端推理使用单模型锁，同一时刻只处理一个推理请求。模型加载失败、缺少
VAE 解码器或 GPU 显存不足时，服务端会返回 HTTP 500；客户端应检查
`response.status_code` 和响应正文。

### Command Line Inference

```sh
# speech recognition
uv run inference.py --task asr --model pretrained_models/FireRedAudio --audio assets/examples/asr_zh_fleurs.wav

# audio understanding and QA; several --audio for e.g. speaker verification,
# --enable-thinking to let the model reason first
uv run inference.py --task understand --model pretrained_models/FireRedAudio --audio assets/examples/assets_mmau_test.wav \
    --prompt "What illness did Second speaker's friend suffer from?\n(A) Progressive arthritis (B) Progressive cancer (C) Acute pneumonia (D) Chronic heart disease" --enable-thinking --max-new-tokens 4096


# ICL voice cloning from a reference audio and its transcript
uv run inference.py --task tts --model pretrained_models/FireRedAudio --vae-decoder pretrained_models/RedAE_decoder/model.pt \
    --prompt-audio assets/examples/tts_zh_prompt.wav --prompt-text "同时，他强调微调要科学有序。" \
    --target-text "安徽淮南秦师傅发现，停在小区的爱车右前驾驶窗玻璃被砸。" --language zh --output tts.wav

# speech editing; semantic rewrites content, acoustic changes pitch / speed / volume.
uv run inference.py --task edit --model pretrained_models/FireRedAudio --vae-decoder pretrained_models/RedAE_decoder/model.pt \
    --audio assets/examples/edit_semantic_zh_ref.wav --instruction "delete '比普通的茶叶要'" --edit-type semantic \
    --output edit_semantic.wav

# acoustic: instructions must use the exact trained templates, e.g.
#   "shift the pitch by N step(s)"  in  -6..6 steps        (pitch)
#   "adjust the speed to X"         in  [0.5, 2.0], step .1 (speed)
#   "adjust the volume to X"        in  [0.3, 2.0], step .1 (volume)
uv run inference.py --task edit --model pretrained_models/FireRedAudio --vae-decoder pretrained_models/RedAE_decoder/model.pt \
    --audio assets/examples/edit_acoustic_zh_ref.wav --instruction "shift the pitch by 3 steps" --edit-type acoustic \
    --output edit_acoustic.wav

# voice design
uv run inference.py --task voice_design --model pretrained_models/FireRedAudio --vae-decoder pretrained_models/RedAE_decoder/model.pt \
    --instruction "以女性高音区的清亮音色,表现出青年阶段的特质,音量略强,语速适中稍快,语调带有解释意味和急切的情感流露,确保语音流畅自然。" \
    --text "是我请他来的，可他什么也不知道，他来只是想打听一下，你们厂是不是有旧锅炉？" --output voice_design.wav
```

## Performance

### Audio Understanding

<div align="center">

* *Results marked with * are obtained from our own evaluation.*

### ASR

<div align="center">

* *Results marked with * are obtained from our own evaluation.*

### Zero-Shot TTS

<div align="center">

### Instruct TTS

<div align="center">

* *These results are obtained from our own evaluation.*

### Semantic Editing

<div align="center">

### Acoustic Editing

<div align="center">

## Limitations

* **Everything but ASR is Chinese/English only.** Speech generation (`tts` / `edit` / `voice_design`) and audio understanding are limited to Chinese and English — `tts` selects the language via `--language zh` / `en`. ASR is the only task that supports more languages.
* **Zero-shot TTS is not deterministic by default.** The flow-matching decoder samples random noise, so output varies run to run; pass a fixed seed (`set_seed(...)` in the API, `--seed` on the CLI) for reproducibility, and note that quality can differ across seeds.
* **Long-form input is supported up to about one hour.** Beyond that the model is untested and time-to-content alignment may degrade.

## Usage Disclaimer

* The project incorporates zero-shot voice cloning functionality; Please note that this capability is intended **solely for academic research purposes**.
* **DO NOT** use this model for **ANY illegal activities**❗️❗️
* The developers assume no liability for any misuse of this model.
* If you identify any instances of **abuse**, **misuse**, or **fraudulent** activities related to this project, **please report them to our team immediately.**

## Citation

```bib
@article{fireredaudio,
  title   = {FireRedAudio: A General-Purpose Audio Language Model with Decoupled Continuous Representations for Understanding and Generation},
  author  = {FireRed Team},
  journal = {arXiv preprint},
  year    = {2026},
}
```

## Acknowledgements

* [Qwen3.5](https://github.com/QwenLM/Qwen3.8) for the language model foundation
* [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) for the Audio Encoder initialization
* [x-transformers](https://github.com/lucidrains/x-transformers) for RotaryEmbedding
* [vocos](https://github.com/gemelo-ai/vocos/tree/main) for ISTFT implementation

## License

Released under the [Apache-2.0](LICENSE) license.
