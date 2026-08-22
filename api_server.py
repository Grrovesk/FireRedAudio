"""HTTP API server for FireRedAudio inference.

Start with:
    python api_server.py

Override defaults when needed:
    python api_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import io
import os
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from inference import FireRedAudioInference

app = FastAPI(title="FireRedAudio API", version="1.0.0")
_engine: FireRedAudioInference | None = None
_api_key: str | None = None
_inference_lock = threading.Lock()


def configure_engine(
    model_path: str,
    vae_decoder_path: str | None,
    device: str,
    quantization: str | None,
) -> None:
    global _engine
    _engine = FireRedAudioInference(
        model_path=model_path,
        vae_decoder_path=vae_decoder_path,
        device=device,
        quantization=quantization,
    )


def require_access(x_api_key: str | None = Header(default=None)) -> None:
    if _api_key is not None and x_api_key != _api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def engine() -> FireRedAudioInference:
    if _engine is None:
        raise HTTPException(status_code=503, detail="model is not initialized")
    return _engine


async def store_upload(upload: UploadFile, directory: str, name: str) -> str:
    suffix = Path(upload.filename or "").suffix or ".audio"
    path = os.path.join(directory, name + suffix)
    with open(path, "wb") as output:
        while chunk := await upload.read(1024 * 1024):
            output.write(chunk)
    return path


def audio_response(audio) -> StreamingResponse:
    buffer = io.BytesIO()
    sf.write(buffer, audio.detach().cpu().float().numpy().squeeze(), 24000, format="WAV")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok" if _engine is not None else "starting"})


@app.post("/v1/asr", dependencies=[Depends(require_access)])
async def asr(audio: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as directory:
        audio_path = await store_upload(audio, directory, "audio")
        with _inference_lock:
            result = engine().understand(audio_path, "Transcribe speech to text.", task="asr")
    return {"text": result.answer}


@app.post("/v1/understand", dependencies=[Depends(require_access)])
async def understand(
    audio: UploadFile = File(...),
    prompt: str = Form(...),
    enable_thinking: bool = Form(False),
    max_new_tokens: int | None = Form(None),
):
    with tempfile.TemporaryDirectory() as directory:
        audio_path = await store_upload(audio, directory, "audio")
        with _inference_lock:
            result = engine().understand(
                audio_path,
                prompt,
                task="understand",
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
    return {"answer": result.answer, "reasoning": result.reasoning}


@app.post("/v1/tts", dependencies=[Depends(require_access)])
async def tts(
    prompt_audio: UploadFile = File(...),
    prompt_text: str = Form(...),
    target_text: str = Form(...),
    language: str = Form("zh"),
):
    with tempfile.TemporaryDirectory() as directory:
        audio_path = await store_upload(prompt_audio, directory, "prompt")
        with _inference_lock:
            result = engine().tts(prompt_text, audio_path, target_text, language)
    return audio_response(result.audio)


@app.post("/v1/edit", dependencies=[Depends(require_access)])
async def edit(
    audio: UploadFile = File(...),
    instruction: str = Form(...),
    edit_type: str = Form("semantic"),
):
    with tempfile.TemporaryDirectory() as directory:
        audio_path = await store_upload(audio, directory, "audio")
        with _inference_lock:
            result = engine().edit(audio_path, instruction, edit_type)
    response = audio_response(result.audio)
    response.headers["X-FireRedAudio-Text"] = quote(result.text or "", safe="")
    return response


@app.post("/v1/voice-design", dependencies=[Depends(require_access)])
def voice_design(
    instruction: str = Form(...),
    text: str = Form(...),
):
    with _inference_lock:
        result = engine().voice_design(instruction, text)
    return audio_response(result.audio)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument(
        "--vae-decoder",
        default="pretrained_models/RedAE_decoder/model.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantization", choices=["int4"], default="int4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    _api_key = os.getenv("FIREREDAUDIO_API_KEY")
    configure_engine(args.model, args.vae_decoder, args.device, args.quantization)
    uvicorn.run(app, host=args.host, port=args.port)
