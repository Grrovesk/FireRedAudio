import time

from fireredaudio.utils.audio import save_audio
from inference import FireRedAudioInference

# Use None for BF16 or "int4" for bitsandbytes NF4.
quantization = "int4"  # 可选：None、"int4"
# Init the model. Understanding tasks need only --model; generation tasks
# additionally need the RedAE decoder weights.
start_time = time.time()
engine = FireRedAudioInference(
    model_path="pretrained_models/FireRedAudio",
    vae_decoder_path="pretrained_models/RedAE_decoder/model.pt",   # only for tts / edit / voice_design
    device="cuda:0",
    quantization=quantization,
)
print(f"Model initialization time taken: {time.time() - start_time:.2f}s")

# ---- 1) Speech recognition (ASR) -----------------------------------------
start_time = time.time()
res = engine.understand("assets/examples/asr_zh_fleurs.wav", "Transcribe speech to text.", task="asr")
print(res.answer)
print(f"ASR time taken: {time.time() - start_time:.2f}s")

# ---- 2) Audio understanding (with optional chain-of-thought) --------------
start_time = time.time()
res = engine.understand(
    "assets/examples/two_speakers.wav",
    "这个音频中有几个说话人",
    task="understand", enable_thinking=True, max_new_tokens=10240,
)
print("CoT:")
print(res.reasoning)   # CoT reasoning, or None
print("Answer")
print(res.answer)
print(f"Audio understanding time taken: {time.time() - start_time:.2f}s")

# ---- 3) Zero-shot TTS (ICL voice cloning) ---------------------------------
start_time = time.time()
res = engine.tts(
    prompt_text="同时，他强调微调要科学有序。",
    prompt_audio="assets/examples/tts_zh_prompt.wav",
    target_text="安徽淮南秦师傅发现，停在小区的爱车右前驾驶窗玻璃被砸。",
    language="zh",
)
save_audio("tts.wav", res.audio, sample_rate=24000)
print(f"TTS time taken: {time.time() - start_time:.2f}s")

# ---- 4) Speech editing -----------------------------------------------------
# semantic: rewrite / substitute / insert / delete content. The model first writes
#          <|sot|>{rewritten text}<|eot|> then renders the audio.
start_time = time.time()
res = engine.edit("assets/examples/edit_semantic_zh_ref.wav", "delete '比普通的茶叶要'", edit_type="semantic")
print(res.text)
save_audio("edit_semantic.wav", res.audio, sample_rate=24000)
print(f"Speech editing time taken: {time.time() - start_time:.2f}s")

# acoustic: change pitch / speed / volume. The instruction must follow the exact
#           templates below (the model is trained on these, not free-form phrasing):
#   pitch   ->  "shift the pitch by N step(s)"       N in {-6, ..., -1, 1, ..., +6}
#   speed   ->  "adjust the speed to X"              X in [0.5, 2.0], step 0.1
#   volume  ->  "adjust the volume to X"             X in [0.3, 2.0], step 0.1
start_time = time.time()
res = engine.edit("assets/examples/edit_acoustic_zh_ref.wav", "shift the pitch by 3 steps", edit_type="acoustic")
save_audio("edit_acoustic.wav", res.audio, sample_rate=24000)
print(f"Acoustic editing time taken: {time.time() - start_time:.2f}s")

# ---- 5) Voice design (synthesis from a timbre description) -----------------
start_time = time.time()
res = engine.voice_design(
    instruction="以女性高音区的清亮音色,表现出青年阶段的特质,音量略强,语速适中稍快,语调带有解释意味和急切的情感流露,确保语音流畅自然。",
    text="是我请他来的，可他什么也不知道，他来只是想打听一下，你们厂是不是有旧锅炉？",
)
save_audio("voice_design.wav", res.audio, sample_rate=24000)
print(f"Voice design time taken: {time.time() - start_time:.2f}s")
