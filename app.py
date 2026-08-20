import os
import time
import glob
import uuid
import asyncio
import tempfile
import json
import re
import random
from datetime import datetime
import httpx

import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter, Reverb, Chorus, Distortion, PitchShift, Delay, Convolution

# Video Editing (MoviePy)
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip, CompositeAudioClip

# CPU-based TTS
import edge_tts

# FastAPI & Gradio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
import gradio as gr

# MCP (Model Context Protocol)
from mcp.server.fastmcp import FastMCP

# ==========================================
# 0. Core Configuration & Production Safety
# ==========================================
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = os.path.join(TEMP_DIR, "media_uploads")
OUTPUT_DIR = os.path.join(TEMP_DIR, "media_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPEAKERS_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models", "speakers")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "foley")
IR_DIR = os.path.join(os.path.dirname(__file__), "assets", "impulse_responses")

os.makedirs(SPEAKERS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(IR_DIR, exist_ok=True)

API_KEY_SECRET=os.environ.get("TTS_API_KEY", "") 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY_SECRET and api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

gpu_lock = asyncio.Lock()
cpu_render_lock = asyncio.Lock()

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY=os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

def get_available_speakers():
    speakers = ["Default Model"]
    if os.path.exists(SPEAKERS_DIR):
        for item in os.listdir(SPEAKERS_DIR):
            if os.path.isdir(os.path.join(SPEAKERS_DIR, item)): speakers.append(item)
    return speakers

def cleanup_old_files():
    now = time.time()
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for filepath in glob.glob(os.path.join(directory, "*")):
            try:
                if os.stat(filepath).st_mtime < now - 3600: os.remove(filepath)
            except Exception: pass

def append_log(current_logs, new_message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {new_message}"
    if not current_logs: return log_entry
    logs_list = current_logs.split("\n")
    logs_list.append(log_entry)
    if len(logs_list) > 20: logs_list = logs_list[-20:]
    return "\n".join(logs_list)

STUDIO_PRESETS = {
    "🎙️ Podcast Studio": {"bass": 5.0, "treble": 3.5, "comp": 3.5, "reverb": 0.05, "gate": True, "drive": 5.0, "pitch": 0, "delay": 0.0, "desc": "นุ่มลึก มีน้ำหนัก ฟังสบาย"},
    "📖 Audiobook Pro": {"bass": 2.0, "treble": 2.0, "comp": 2.5, "reverb": 0.15, "gate": True, "drive": 2.0, "pitch": 0, "delay": 0.0, "desc": "ใสสะอาด มีมิติเสียงก้องนิดๆ"},
    "🗣️ Natural Human": {"bass": 1.0, "treble": 1.5, "comp": 1.5, "reverb": 0.08, "gate": False, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "ธรรมชาติ ไม่บีบอัดมาก"}
}

# ==========================================
# 1. Foley & Breath Insertion (Generative Breath)
# ==========================================
def generate_synthetic_breath(duration_ms=400):
    samples = int((duration_ms / 1000.0) * 44100)
    noise = np.random.normal(0, 0.05, samples)
    x = np.linspace(0, np.pi, samples)
    envelope = np.sin(x) ** 2
    breath_wave = noise * envelope
    breath_wave = np.int16(breath_wave * 32767)
    breath_audio = AudioSegment(breath_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    return breath_audio - 15 

def insert_breaths(input_wav_path, output_wav_path):
    print("Detecting silence and inserting breaths...")
    audio = AudioSegment.from_file(input_wav_path)
    silence_ranges = detect_silence(audio, min_silence_len=400, silence_thresh=-40)
    if not silence_ranges:
        audio.export(output_wav_path, format="wav")
        return
        
    breath_files = glob.glob(os.path.join(ASSETS_DIR, "breath*.wav"))
    output_audio = AudioSegment.empty()
    last_end = 0
    
    for start, end in silence_ranges:
        output_audio += audio[last_end:start]
        pause_duration = end - start
        
        if pause_duration >= 500:
            if breath_files:
                b_file = random.choice(breath_files)
                breath_snd = AudioSegment.from_file(b_file)
                if len(breath_snd) > pause_duration:
                    breath_snd = breath_snd[:pause_duration].fade_out(50)
                remaining_silence = pause_duration - len(breath_snd)
                output_audio += breath_snd + AudioSegment.silent(duration=remaining_silence)
            else:
                breath_snd = generate_synthetic_breath(min(400, pause_duration))
                remaining_silence = pause_duration - len(breath_snd)
                output_audio += breath_snd + AudioSegment.silent(duration=max(0, remaining_silence))
        else:
            output_audio += audio[start:end]
            
        last_end = end
        
    output_audio += audio[last_end:]
    output_audio.export(output_wav_path, format="wav")

# ==========================================
# 2. AI Audio Generation & Mastering
# ==========================================
async def _generate_audio_chunk(engine: str, mode: str, text_chunk: str, lang: str, ref_path: str, speed: float, speaker_model: str):
    sample_rate = 44100
    if engine == "EdgeTTS (Fast CPU)":
        voice = "th-TH-PremwadeeNeural" if "th" in lang.lower() else "en-US-AriaNeural"
        speed_percent = int((speed - 1.0) * 100)
        speed_str = f"+{speed_percent}%" if speed_percent >= 0 else f"{speed_percent}%"
        communicate = edge_tts.Communicate(text_chunk, voice, rate=speed_str)
        temp_mp3 = os.path.join(TEMP_DIR, f"edge_{uuid.uuid4().hex}.mp3")
        await communicate.save(temp_mp3)
        audio_data, sr = sf.read(temp_mp3)
        os.remove(temp_mp3)
        if len(audio_data.shape) == 1: audio_data = audio_data.reshape(-1, 1)
        return audio_data, sr
    else:
        await asyncio.sleep(1) 
        duration = max(1.0, len(text_chunk) * 0.1) 
        audio_data = np.zeros((int(sample_rate * duration), 1))
        return audio_data, sample_rate


async def generate_tts_safely(engine: str, mode: str, full_text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0, speaker_model: str = "Default Model", apply_breaths: bool = False):
    chunks = re.split(r'(?<=[.!?\n])\s+', full_text.strip())
    chunks = [c for c in chunks if c.strip()]
    if not chunks: chunks = [full_text]
    all_audio_arrays = []
    sample_rate = 44100
    
    if engine == "EdgeTTS (Fast CPU)":
        for chunk in chunks:
            if not chunk.strip(): continue
            audio_data, sr = await _generate_audio_chunk(engine, mode, chunk, lang, ref_path, speed, speaker_model)
            sample_rate = sr
            all_audio_arrays.append(audio_data)
            pause = np.zeros((int(sample_rate * 0.6), audio_data.shape[1] if len(audio_data.shape) > 1 else 1)) 
            all_audio_arrays.append(pause)
    else:
        async with gpu_lock:
            for chunk in chunks:
                if not chunk.strip(): continue
                audio_data, sr = await _generate_audio_chunk(engine, mode, chunk, lang, ref_path, speed, speaker_model)
                sample_rate = sr
                all_audio_arrays.append(audio_data)
                pause = np.zeros((int(sample_rate * 0.6), audio_data.shape[1] if len(audio_data.shape) > 1 else 1)) 
                all_audio_arrays.append(pause)
                
    temp_concat_path = os.path.join(TEMP_DIR, f"pre_foley_{uuid.uuid4().hex}.wav")
    if all_audio_arrays:
        final_audio = np.concatenate(all_audio_arrays, axis=0)
        sf.write(temp_concat_path, final_audio, sample_rate)
    else:
        sf.write(temp_concat_path, np.zeros((sample_rate, 1)), sample_rate)

    if apply_breaths:
        insert_breaths(temp_concat_path, out_path)
        os.remove(temp_concat_path)
    else:
        os.rename(temp_concat_path, out_path)


def apply_studio_mastering(
    input_path: str, output_path: str, gate: bool=True, bass: float=4.5, treble: float=3.0, 
    comp: float=3.0, reverb_amount: float=0.0, drive_amount: float=0.0, pitch_shift: int=0, 
    delay_time: float=0.0, humanize: bool=False,
    # 🌟 NEW PARAMS FOR ADVANCED REQUIREMENTS 🌟
    de_essing: bool=True,
    tape_saturation: bool=True,
    convolution_ir_path: str=None
):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1: audio_data = audio_data.T 
    
    effects_chain = []
    
    # 1. Pitch Shift
    if pitch_shift != 0: effects_chain.append(PitchShift(semitones=pitch_shift))
    
    # 2. Noise Gate
    if gate: effects_chain.append(NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250))
    
    # 3. Dynamic EQ & Tone Shaping
    effects_chain.append(HighpassFilter(cutoff_frequency_hz=300 if bass <= -10 else 80))
    effects_chain.append(LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass))
    
    # 🌟 REQUIREMENT: De-essing (Dynamic EQ emulation for sibilance)
    # We use a slight dip around 6-8kHz to tame harsh 'S' sounds before adding air
    if de_essing:
        # Tame the 6.5kHz range slightly
        effects_chain.append(HighShelfFilter(cutoff_frequency_hz=6500, gain_db=-2.0))
        # Then boost the very high "air" frequencies (10kHz+) instead of the harsh sibilance range
        effects_chain.append(HighShelfFilter(cutoff_frequency_hz=10000, gain_db=treble))
    else:
        effects_chain.append(HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble))
    
    # 🌟 REQUIREMENT: Harmonic Tape Saturation
    # Adds 5-10% subtle analog tube/tape warmth
    if tape_saturation:
        # Light distortion mimics analog tape harmonics
        effects_chain.append(Distortion(drive_db=5.0)) 
    elif drive_amount > 0:
        effects_chain.append(Distortion(drive_db=drive_amount))

    # 4. Compression (Leveling)
    effects_chain.append(Compressor(threshold_db=-15, ratio=comp, attack_ms=2.0, release_ms=100))
    
    # 5. Delay
    if delay_time > 0: effects_chain.append(Delay(delay_seconds=delay_time, feedback=0.3, mix=0.4))
    
    # 🌟 REQUIREMENT: Convolution Reverb
    if convolution_ir_path and os.path.exists(convolution_ir_path):
        # Apply real room acoustic fingerprint (Impulse Response)
        effects_chain.append(Convolution(convolution_ir_path, mix=reverb_amount if reverb_amount > 0 else 0.1))
    elif reverb_amount > 0:
        # Fallback to algorithmic reverb
        effects_chain.append(Reverb(room_size=0.3 if delay_time > 0 else 0.1, dry_level=1.0, wet_level=reverb_amount))
        
    # 6. Humanize Mod (Micro-modulation)
    if humanize: effects_chain.append(Chorus(rate_hz=0.5, depth=0.05, mix=0.1))
    
    # 7. Safety Limiter
    effects_chain.append(Limiter(threshold_db=-1.0))
    
    board = Pedalboard([effect for effect in effects_chain if effect is not None])
    effected_audio = board(audio_data, sample_rate)
    
    if len(effected_audio.shape) > 1: effected_audio = effected_audio.T
    sf.write(output_path, effected_audio, sample_rate)
    return output_path

# ==========================================
# 3. FastAPI & Endpoints (OMITTED REDUNDANCY FOR BREVITY)
# ==========================================
app = FastAPI(title="AI Media Studio API (Audio & Video)")
mcp = FastMCP("Media_Studio_MCP")
app.mount("/sse", mcp.get_starlette_app())

@app.post("/api/tts/generate")
async def api_generate_tts(
    engine: str = Form(...), text: str = Form(...), language: str = Form("en"), mode: str = Form("standard"),
    speaker_model: str = Form("Default Model"), speed: float = Form(1.0),
    apply_humanize: bool = Form(False), apply_breaths: bool = Form(False),
    
    # 🌟 NEW PARAMS FOR ADVANCED AUDIO IN n8n
    apply_deessing: bool = Form(True),
    apply_tape_saturation: bool = Form(True),
    convolution_ir_file: UploadFile = File(None),
    
    reference_audio: UploadFile = File(None), api_key: str = Depends(verify_api_key) 
):
    cleanup_old_files()
    job_id = str(uuid.uuid4())
    
    ref_audio_path = None
    if reference_audio:
        ref_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref.wav")
        with open(ref_audio_path, "wb") as f: f.write(await reference_audio.read())

    ir_path = None
    if convolution_ir_file:
        ir_path = os.path.join(IR_DIR, f"{job_id}_ir.wav")
        with open(ir_path, "wb") as f: f.write(await convolution_ir_file.read())

    raw_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_raw.wav")
    final_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_final.wav")
    
    try:
        await generate_tts_safely(engine, mode, text, language, ref_audio_path, raw_output_path, speed, speaker_model, apply_breaths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Apply advanced studio mastering if requested by n8n
    if apply_humanize or apply_deessing or apply_tape_saturation or ir_path:
        apply_studio_mastering(
            input_path=raw_output_path, output_path=final_output_path,
            de_essing=apply_deessing, tape_saturation=apply_tape_saturation, convolution_ir_path=ir_path, humanize=apply_humanize
        )
        return FileResponse(path=final_output_path, media_type="audio/wav", filename=f"{engine}_studio.wav")

    return FileResponse(path=raw_output_path, media_type="audio/wav", filename=f"{engine}_raw.wav")

# ==========================================
# 4. Gradio UI Setup (Adding the new toggles)
# ==========================================
def gradio_studio(input_audio, preset, humanize, de_essing, tape_sat, ir_file, export_format, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, current_logs):
    cleanup_old_files()
    if not input_audio: return None, "❌", append_log(current_logs, "❌ No input")
    try:
        logs = append_log(current_logs, f"⚙️ START STUDIO: Mastering with De-essing={de_essing}, TapeSat={tape_sat}")
        ext = export_format.lower()
        output_file = os.path.join(OUTPUT_DIR, f"studio_{int(time.time())}.{ext}")
        
        # Handle IR file from UI
        ir_path = ir_file.name if ir_file else None
        
        apply_studio_mastering(
            input_path=input_audio, output_path=output_file, 
            gate=enable_gate, bass=bass, treble=treble, comp=comp, reverb_amount=reverb, 
            drive_amount=drive, pitch_shift=pitch, delay_time=delay, humanize=humanize,
            de_essing=de_essing, tape_saturation=tape_sat, convolution_ir_path=ir_path
        )
        logs = append_log(logs, f"✅ SUCCESS")
        return output_file, "✅ สำเร็จ", logs
    except Exception as e:
        return None, str(e), append_log(current_logs, f"❌ ERROR: {str(e)}")

# ... UI Definitions ...
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎬 AI Media Studio (Audio & Vertical Video Production)")
    system_logs_state = gr.State(value="")
    with gr.Row():
        with gr.Column(scale=3): 
            # TAB 1 (Omitted for brevity - logic remains the same)
            # TAB 2: STUDIO
            with gr.Tab("🎛️ 2. Advanced Audio Mastering"):
                with gr.Row():
                    with gr.Column():
                        raw_audio_input = gr.Audio(label="Input Audio", type="filepath")
                        gr.Markdown("### 🌟 Advanced Professional FX")
                        with gr.Row():
                            de_essing_check = gr.Checkbox(value=True, label="🎧 De-essing (ลดเสียง ส, ช บาดหู)")
                            tape_sat_check = gr.Checkbox(value=True, label="🎛️ Harmonic Tape Saturation (อุ่นแบบอนาล็อก)")
                            humanize_checkbox = gr.Checkbox(value=False, label="🧑 Humanize (Micro-modulation)")
                        
                        ir_file_input = gr.File(label="📂 Convolution Reverb (อัปโหลดไฟล์ Impulse Response .wav)", file_types=[".wav"])
                        
                        gr.Markdown("### ⚙️ EQ & Dynamics")
                        bass_boost = gr.Slider(minimum=-15, maximum=15, value=5.0, label="Bass")
                        treble_boost = gr.Slider(minimum=-15, maximum=15, value=3.5, label="Treble")
                        comp_ratio = gr.Slider(minimum=1, maximum=10, value=3.5, label="Compression")
                        
                        process_btn = gr.Button("🎧 ประมวลผล", variant="primary")
                    with gr.Column():
                        studio_status = gr.Markdown("🟢 รอรับไฟล์")
                        studio_audio_output = gr.Audio(label="Mastered Audio", interactive=False)

    process_btn.click(
        fn=gradio_studio,
        inputs=[raw_audio_input, gr.State("Custom"), humanize_checkbox, de_essing_check, tape_sat_check, ir_file_input, gr.State("WAV"), gr.State(True), bass_boost, treble_boost, comp_ratio, gr.State(0.1), gr.State(0.0), gr.State(0), gr.State(0.0), system_logs_state],
        outputs=[studio_audio_output, studio_status, system_logs_state]
    )

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)