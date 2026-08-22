"""Audio loading."""

import soundfile as sf
import torch
import torchaudio

# The audio encoder consumes 16 kHz mel; RedAE operates at 24 kHz.
UNDERSTAND_SAMPLE_RATE = 16000
GENERATION_SAMPLE_RATE = 24000


def read_audio(path: str, target_sample_rate: int) -> torch.Tensor:
    """Load as a 1-D mono waveform resampled to `target_sample_rate`."""
    audio, ori_sr = sf.read(path, always_2d=True, dtype="float32")  # (T, C)
    audio = torch.from_numpy(audio).mean(dim=1)
    if ori_sr != target_sample_rate:
        audio = torchaudio.functional.resample(audio, ori_sr, target_sample_rate)
    return audio


def save_audio(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    """Save a waveform without relying on TorchCodec's FFmpeg DLLs."""
    waveform = audio.detach().cpu().float()
    if waveform.ndim == 2:
        waveform = waveform.transpose(0, 1)
    sf.write(path, waveform.numpy(), sample_rate)
