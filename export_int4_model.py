"""Export FireRedAudio as a local bitsandbytes NF4 model.

The exported directory contains quantized model shards plus the tokenizer and
processor files needed by the FireRedAudio inference entry points.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from fireredaudio.audio_encoder.processor import FireRedAudioProcessor
from fireredaudio.loading import load_fireredaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="pretrained_models/FireRedAudio",
        help="source FireRedAudio model directory",
    )
    parser.add_argument(
        "--output",
        default="pretrained_models/FireRedAudio-int4",
        help="directory for the exported local model",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used while loading and exporting, e.g. cuda:0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.device(args.device).type != "cuda":
        raise ValueError("int4 export requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("int4 export requires an available CUDA GPU")

    source = Path(args.model)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading NF4 model from {source} on {args.device}...")
    model = load_fireredaudio(
        str(source),
        device=args.device,
        quantization="int4",
    )

    print(f"Saving quantized model to {output}...")
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(source).save_pretrained(output)
    FireRedAudioProcessor.from_pretrained(source).save_pretrained(output)
    print("Done. Use this directory with --model and --quantization int4.")


if __name__ == "__main__":
    main()