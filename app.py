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
import shutil
import urllib.request

import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter, Reverb, Chorus, Distortion, PitchShift, Delay, Convolution

# Video Editing (MoviePy)
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip, CompositeAudioClip

# CPU-based TTS
import edge_tts
from piper.voice import PiperVoice

# FastAPI & Gradio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
import gradio as gr

# MCP (Model Context Protocol)
from mcp.server.fastmcp import FastMCP

# ==========================================
# 0. Core Configuration & Project Structure
# ==========================================
BASE_DIR = os.path.dirname(__file__)

TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = os.path.join(TEMP_DIR, "media_uploads")
OUTPUT_DIR = os.path.join(TEMP_DIR, "media_outputs")

ASSETS_DIR = os.path.join(BASE_DIR, "assets", "foley")                   
IR_DIR = os.path.join(BASE_DIR, "assets", "impulse_responses")           
SPEAKERS_DIR = os.path.join(BASE_DIR, "pretrained_models", "speakers")   
PIPER_DIR = os.path.join(BASE_DIR, "pretrained_models", "piper_voices")  

for d in [UPLOAD_DIR, OUTPUT_DIR, ASSETS_DIR, IR_DIR, SPEAKERS_DIR, PIPER_DIR]:
    os.makedirs(d, exist_ok=True)

API_KEY_SECRET=os.environ.get("API_KEY_SECRET", "") 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY_SECRET and api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

gpu_lock = asyncio.Lock()
cpu_render_lock = asyncio.Lock()
piper_lock = asyncio.Lock() 

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY=os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

def get_available_speakers():
    speakers = ["Default Model"]
    if os.path.exists(SPEAKERS_DIR):
        for item in os.listdir(SPEAKERS_DIR):
            if os.path.isdir(os.path.join(SPEAKERS_DIR, item)): speakers.append(item)
    return speakers

def get_available_piper_models():
    models = []
    if os.path.exists(PIPER_DIR):
        for file in os.listdir(PIPER_DIR):
            if file.endswith(".onnx"):
                models.append(file)
    return models if models else ["(No Piper models found)"]

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
# 1. Foley & Breath Insertion 
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
async def _generate_audio_chunk(engine: str, mode: str, text_chunk: str, lang: str, ref_path: str, speed: float, speaker_model: str, piper_model: str):
    sample_rate = 44100
    
    if engine == "EdgeTTS (Fast CPU / Online)":
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
        
    elif engine == "PiperTTS (Fast CPU / Offline)":
        available_models = get_available_piper_models()
        if not piper_model or piper_model not in available_models or piper_model.startswith("(No Piper"):
            if available_models and not available_models[0].startswith("(No Piper"):
                piper_model = available_models[0]
            else:
                raise ValueError("No valid Piper .onnx model selected or found. Please ensure models exist in pretrained_models/piper_voices/.")
            
        model_path = os.path.join(PIPER_DIR, piper_model)
        if not os.path.exists(model_path): 
            raise ValueError(f"Piper model not found at {model_path}")
            
        temp_wav = os.path.join(TEMP_DIR, f"piper_{uuid.uuid4().hex}.wav")
        voice = PiperVoice.load(model_path)
        
        def run_piper():
            with open(temp_wav, "wb") as wav_file:
                voice.synthesize(text_chunk, wav_file, length_scale=1.0/speed)
                
        await asyncio.to_thread(run_piper)
        audio_data, sr = sf.read(temp_wav)
        os.remove(temp_wav)
        if len(audio_data.shape) == 1: audio_data = audio_data.reshape(-1, 1)
        return audio_data, sr
        
    else:
        available_speakers = get_available_speakers()
        if not speaker_model or speaker_model not in available_speakers:
            speaker_model = "Default Model"
            
        await asyncio.sleep(1) 
        duration = max(1.0, len(text_chunk) * 0.1) 
        audio_data = np.zeros((int(sample_rate * duration), 1))
        return audio_data, sample_rate


async def generate_tts_safely(engine: str, mode: str, full_text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0, speaker_model: str = "Default Model", piper_model: str = "", apply_breaths: bool = False):
    chunks = re.split(r'(?<=[.!?\n])\s+', full_text.strip())
    chunks = [c for c in chunks if c.strip()]
    if not chunks: chunks = [full_text]
    all_audio_arrays = []
    sample_rate = 44100
    
    if engine == "EdgeTTS (Fast CPU / Online)":
        for chunk in chunks:
            if not chunk.strip(): continue
            audio_data, sr = await _generate_audio_chunk(engine, mode, chunk, lang, ref_path, speed, speaker_model, piper_model)
            sample_rate = sr
            all_audio_arrays.append(audio_data)
            pause = np.zeros((int(sample_rate * 0.6), audio_data.shape[1] if len(audio_data.shape) > 1 else 1)) 
            all_audio_arrays.append(pause)
            
    elif engine == "PiperTTS (Fast CPU / Offline)":
        async with piper_lock:
            for chunk in chunks:
                if not chunk.strip(): continue
                audio_data, sr = await _generate_audio_chunk(engine, mode, chunk, lang, ref_path, speed, speaker_model, piper_model)
                sample_rate = sr
                all_audio_arrays.append(audio_data)
                pause = np.zeros((int(sample_rate * 0.6), audio_data.shape[1] if len(audio_data.shape) > 1 else 1)) 
                all_audio_arrays.append(pause)
                
    else: 
        async with gpu_lock:
            for chunk in chunks:
                if not chunk.strip(): continue
                audio_data, sr = await _generate_audio_chunk(engine, mode, chunk, lang, ref_path, speed, speaker_model, piper_model)
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
    delay_time: float=0.0, humanize: bool=False, de_essing: bool=True, tape_saturation: bool=True, convolution_ir_path: str=None
):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1: audio_data = audio_data.T 
    
    effects_chain = []
    if pitch_shift != 0: effects_chain.append(PitchShift(semitones=pitch_shift))
    if gate: effects_chain.append(NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250))
    effects_chain.append(HighpassFilter(cutoff_frequency_hz=300 if bass <= -10 else 80))
    effects_chain.append(LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass))
    
    if de_essing:
        effects_chain.append(HighShelfFilter(cutoff_frequency_hz=6500, gain_db=-2.0))
        effects_chain.append(HighShelfFilter(cutoff_frequency_hz=10000, gain_db=treble))
    else:
        effects_chain.append(HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble))
    
    if tape_saturation:
        effects_chain.append(Distortion(drive_db=5.0)) 
    elif drive_amount > 0:
        effects_chain.append(Distortion(drive_db=drive_amount))

    effects_chain.append(Compressor(threshold_db=-15, ratio=comp, attack_ms=2.0, release_ms=100))
    if delay_time > 0: effects_chain.append(Delay(delay_seconds=delay_time, feedback=0.3, mix=0.4))
    
    if convolution_ir_path and os.path.exists(convolution_ir_path):
        effects_chain.append(Convolution(convolution_ir_path, mix=reverb_amount if reverb_amount > 0 else 0.1))
    elif reverb_amount > 0:
        effects_chain.append(Reverb(room_size=0.3 if delay_time > 0 else 0.1, dry_level=1.0, wet_level=reverb_amount))
        
    if humanize: effects_chain.append(Chorus(rate_hz=0.5, depth=0.05, mix=0.1))
    effects_chain.append(Limiter(threshold_db=-1.0))
    
    board = Pedalboard([effect for effect in effects_chain if effect is not None])
    effected_audio = board(audio_data, sample_rate)
    
    if len(effected_audio.shape) > 1: effected_audio = effected_audio.T
    sf.write(output_path, effected_audio, sample_rate)
    return output_path


# ==========================================
# 3. Video Operations
# ==========================================
async def process_video_edit(
    video_path: str, audio_path: str, output_path: str, trim_start: float = 0.0, trim_end: float = None, 
    mute_original_audio: bool = False, short_video_format: bool = True, add_watermark: str = "", text_lines: str = "" 
):
    async with cpu_render_lock:
        video = VideoFileClip(video_path)
        if trim_end is None or trim_end <= 0: trim_end = video.duration
        video = video.subclip(trim_start, trim_end)

        if mute_original_audio: video = video.without_audio()

        if short_video_format:
            target_w, target_h = 1080, 1920
            video_aspect = video.w / video.h
            target_aspect = target_w / target_h
            if video_aspect > target_aspect:
                new_w = int(video.h * target_aspect)
                x_center = video.w / 2
                video = video.crop(x_center=x_center, width=new_w, height=video.h)
            else:
                new_h = int(video.w / target_aspect)
                y_center = video.h / 2
                video = video.crop(y_center=y_center, width=video.w, height=new_h)
            video = video.resize(newsize=(target_w, target_h))

        final_audio = None
        if audio_path and os.path.exists(audio_path):
            new_audio = AudioFileClip(audio_path)
            new_audio = new_audio.set_duration(min(new_audio.duration, video.duration))
            if mute_original_audio or video.audio is None:
                final_audio = new_audio
            else:
                final_audio = CompositeAudioClip([video.audio.volumex(0.3), new_audio])
                
        if final_audio: video = video.set_audio(final_audio)

        clips_to_composite = [video]
        if text_lines and text_lines.strip():
            lines = text_lines.split("\n")
            lines = [l.strip() for l in lines if l.strip()]
            try:
                box_width = int(video.w * 0.85) 
                start_y = int(video.h * 0.25)   
                spacing = 30                    
                current_y = start_y
                for line in lines:
                    txt_clip = TextClip(line, fontsize=45, color='black', bg_color='white', method='caption', align='center', size=(box_width, None))
                    txt_clip = txt_clip.set_position(('center', current_y)).set_duration(video.duration)
                    clips_to_composite.append(txt_clip)
                    current_y += txt_clip.h + spacing
            except Exception as e: print(f"Warning: Text overlay failed: {e}")

        if add_watermark and add_watermark.strip():
            try:
                wm_clip = TextClip(add_watermark, fontsize=40, color='white', bg_color='transparent')
                wm_clip = wm_clip.set_position(('right','bottom')).set_duration(video.duration).margin(bottom=50, right=50, opacity=0)
                clips_to_composite.append(wm_clip)
            except: pass

        if len(clips_to_composite) > 1: video = CompositeVideoClip(clips_to_composite)

        video.write_videofile(
            output_path, codec="libx264", audio_codec="aac",
            temp_audiofile=os.path.join(TEMP_DIR, f"temp-audio-{uuid.uuid4().hex[:6]}.m4a"),
            remove_temp=True, fps=30, logger=None
        )
        
        video.close()
        if 'final_audio' in locals() and final_audio: final_audio.close()


# ==========================================
# 4. FastAPI Setup
# ==========================================
app = FastAPI(title="AI Media Studio API (Audio & Video)")
mcp = FastMCP("Media_Studio_MCP")
app.mount("/sse", mcp.get_starlette_app())

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AI Media Studio App", "auth_required": bool(API_KEY_SECRET)}

@app.post("/api/tts/generate")
async def api_generate_tts(
    engine: str = Form("PiperTTS (Fast CPU / Offline)"), 
    text: str = Form(""), 
    language: str = Form("th"), 
    mode: str = Form("standard"),
    speaker_model: str = Form("Default Model"), 
    piper_model: str = Form(""), 
    speed: float = Form(1.0),
    apply_humanize: bool = Form(False), 
    apply_breaths: bool = Form(False),
    apply_deessing: bool = Form(False), 
    apply_tape_saturation: bool = Form(False),
    convolution_ir_file: UploadFile = File(None), 
    reference_audio: UploadFile = File(None), 
    api_key: str = Depends(verify_api_key) 
):
    """
    Generate Text-to-Speech audio.
    By default, it uses PiperTTS and auto-selects the first available ONNX model.
    It requires only the 'text' parameter to work perfectly out-of-the-box.
    """
    cleanup_old_files()
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Missing required 'text' parameter.")

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
        await generate_tts_safely(engine, mode, text, language, ref_audio_path, raw_output_path, speed, speaker_model, piper_model, apply_breaths)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

    if apply_humanize or apply_deessing or apply_tape_saturation or ir_path:
        apply_studio_mastering(
            input_path=raw_output_path, output_path=final_output_path,
            de_essing=apply_deessing, tape_saturation=apply_tape_saturation, convolution_ir_path=ir_path, humanize=apply_humanize
        )
        return FileResponse(path=final_output_path, media_type="audio/wav", filename=f"tts_studio.wav")

    return FileResponse(path=raw_output_path, media_type="audio/wav", filename=f"tts_raw.wav")

@app.post("/api/video/edit")
async def api_video_edit(
    video_file: UploadFile = File(None), video_local_path: str = Form(""),
    audio_file: UploadFile = File(None), audio_local_path: str = Form(""),
    trim_start: float = Form(0.0), trim_end: float = Form(0.0), mute_original_audio: bool = Form(True), 
    short_video_format: bool = Form(True), text_lines: str = Form(""), watermark_text: str = Form(""),
    return_local_path: bool = Form(False), api_key: str = Depends(verify_api_key)
):
    cleanup_old_files()
    job_id = str(uuid.uuid4())
    vid_path = video_local_path if (video_local_path and os.path.exists(video_local_path)) else None
    if not vid_path and video_file:
        vid_path = os.path.join(UPLOAD_DIR, f"{job_id}_vid.mp4")
        with open(vid_path, "wb") as f: f.write(await video_file.read())
    if not vid_path: raise HTTPException(status_code=400, detail="Must provide video_file or valid video_local_path")
        
    aud_path = audio_local_path if (audio_local_path and os.path.exists(audio_local_path)) else None
    if not aud_path and audio_file:
        aud_path = os.path.join(UPLOAD_DIR, f"{job_id}_aud.wav")
        with open(aud_path, "wb") as f: f.write(await audio_file.read())

    out_vid_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.mp4")
    try:
        await process_video_edit(vid_path, aud_path, out_vid_path, trim_start, trim_end, mute_original_audio, short_video_format, watermark_text, text_lines)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

    if return_local_path: return JSONResponse(content={"status": "success", "output_path": out_vid_path})
    return FileResponse(path=out_vid_path, media_type="video/mp4", filename="edited_video.mp4")

# ==========================================
# Gradio UI Handlers
# ==========================================
def upload_model_file(files, model_type, current_logs):
    if not files: return append_log(current_logs, "❌ ERROR: No files uploaded.")
    logs = current_logs
    target_dir = PIPER_DIR if "Piper" in model_type else SPEAKERS_DIR
    for file_obj in files:
        filename = os.path.basename(file_obj.name)
        if "Piper" in model_type and not (filename.endswith(".onnx") or filename.endswith(".json")):
            logs = append_log(logs, f"⚠️ SKIPPED: {filename} (Piper requires .onnx or .json)")
            continue
        dest_path = os.path.join(target_dir, filename)
        shutil.copy(file_obj.name, dest_path)
        logs = append_log(logs, f"✅ UPLOADED: {filename} -> {os.path.basename(target_dir)}/")
    return logs

def delete_model_file(model_name, model_type, current_logs):
    target_dir = PIPER_DIR if model_type == "Piper" else SPEAKERS_DIR
    if not model_name or model_name.startswith("(No"):
        return append_log(current_logs, "❌ ERROR: No valid model selected to delete.")
    target_path = os.path.join(target_dir, model_name)
    if os.path.exists(target_path):
        if os.path.isdir(target_path): shutil.rmtree(target_path)
        else: os.remove(target_path)
        if target_path.endswith(".onnx"):
            json_path = target_path + ".json"
            if os.path.exists(json_path): os.remove(json_path)
        return append_log(current_logs, f"🗑️ DELETED: {model_name}")
    else: return append_log(current_logs, f"❌ ERROR: File not found {model_name}")

def gradio_tts(tts_mode, engine, custom_speaker, piper_model, language, speed, text, ref_audio, apply_breaths, current_logs):
    cleanup_old_files()
    if not text.strip(): return None, None, "❌", append_log(current_logs, "❌ ERROR: No text")
    logs = append_log(current_logs, f"🚀 START: Engine={engine}")
    yield None, None, "⏳ กำลังประมวลผล...", logs
    out_path = os.path.join(OUTPUT_DIR, f"gradio_tts_{int(time.time())}.wav")
    try:
        asyncio.run(generate_tts_safely(engine, tts_mode, text, language, ref_audio, out_path, speed, custom_speaker, piper_model, apply_breaths))
        logs = append_log(logs, "✅ SUCCESS: สร้างเสียงสำเร็จ")
        yield out_path, out_path, "✅ สำเร็จ", logs
    except Exception as e:
        logs = append_log(logs, f"❌ ERROR: {str(e)}")
        yield None, None, f"❌ {str(e)}", logs

def gradio_studio(input_audio, preset, humanize, de_essing, tape_sat, ir_file, export_format, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, current_logs):
    cleanup_old_files()
    if not input_audio: return None, "❌", append_log(current_logs, "❌ No input")
    try:
        logs = append_log(current_logs, f"⚙️ START STUDIO: {preset}")
        ext = export_format.lower()
        output_file = os.path.join(OUTPUT_DIR, f"studio_{int(time.time())}.{ext}")
        ir_path = ir_file.name if ir_file else None
        apply_studio_mastering(input_audio, output_file, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, humanize, de_essing, tape_sat, ir_path)
        logs = append_log(logs, f"✅ SUCCESS")
        return output_file, "✅ สำเร็จ", logs
    except Exception as e:
        return None, str(e), append_log(current_logs, f"❌ ERROR: {str(e)}")

def gradio_video_edit(video_in, audio_in, trim_start, trim_end, mute_orig, force_916, text_lines, watermark, current_logs):
    cleanup_old_files()
    if not video_in: return None, append_log(current_logs, "❌ ERROR: No video")
    logs = append_log(current_logs, "🎬 START VIDEO EDIT")
    yield None, logs
    out_vid_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
    try:
        asyncio.run(process_video_edit(video_in, audio_in, out_vid_path, trim_start, trim_end if trim_end > 0 else None, mute_orig, force_916, watermark, text_lines))
        logs = append_log(logs, "✅ SUCCESS")
        yield out_vid_path, logs
    except Exception as e:
        logs = append_log(logs, f"❌ VIDEO ERROR: {str(e)}")
        yield None, logs

def update_sliders_from_preset(preset_name):
    if preset_name in STUDIO_PRESETS:
        p = STUDIO_PRESETS[preset_name]
        return (gr.update(value=p['desc']), gr.update(value=p['gate']), gr.update(value=p['bass']), 
                gr.update(value=p['treble']), gr.update(value=p['comp']), gr.update(value=p['reverb']),
                gr.update(value=p['drive']), gr.update(value=p['pitch']), gr.update(value=p['delay']))
    return [gr.update()] * 9

def toggle_engine_visibility(engine):
    is_piper = "PiperTTS" in engine
    is_gpu = engine in ["CosyVoice 3.0", "OmniVoice"]
    return gr.update(visible=is_gpu), gr.update(visible=is_piper)

# ==========================================
# Gradio UI Building
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎬 AI Media Studio (Pro Edition)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. Audio Tools & Foley"):
                with gr.Row():
                    with gr.Column():
                        # 🌟 CHANGED DEFAULT TO PIPER
                        engine_dropdown = gr.Radio(choices=["CosyVoice 3.0", "OmniVoice", "EdgeTTS (Fast CPU / Online)", "PiperTTS (Fast CPU / Offline)"], value="PiperTTS (Fast CPU / Offline)", label="Engine")
                        tts_mode = gr.Radio(choices=["Standard", "Instruct (Emotion)", "Zero-Shot (Voice Cloning)"], value="Standard", label="Mode")
                        
                        gpu_speaker_dropdown = gr.Dropdown(choices=get_available_speakers(), value=get_available_speakers()[0], label="GPU Model Checkpoint", visible=False)
                        
                        # 🌟 CHANGED DEFAULT TO BE VISIBLE SINCE PIPER IS DEFAULT
                        piper_speaker_dropdown = gr.Dropdown(choices=get_available_piper_models(), value=get_available_piper_models()[0], label="Piper Offline Model (.onnx)", visible=True)
                        
                        lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)"], value="Thai (th)", label="Language")
                        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                        apply_breaths = gr.Checkbox(value=False, label="🫁 แทรกเสียงสูดลมหายใจอัตโนมัติ") # Turned off by default to keep simple
                        text_input = gr.Textbox(label="Text Prompt", lines=4)
                        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath", visible=False)
                        submit_btn = gr.Button("🚀 Generate Speech", variant="primary")
                        
                    with gr.Column():
                        status_output = gr.Markdown("🟢 พร้อมใช้งาน")
                        output_audio = gr.Audio(label="Output Audio", interactive=False)

            with gr.Tab("🗄️ 2. Model Manager (จัดการไฟล์โมเดล)"):
                gr.Markdown("### 📥 อัปโหลด หรือ ลบ โมเดลเสียง (Speaker Checkpoints / Piper ONNX)")
                with gr.Row():
                    with gr.Column():
                        upload_type = gr.Radio(choices=["GPU Speaker Checkpoint (Cosy/Omni)", "Piper Offline Model (.onnx, .json)"], value="Piper Offline Model (.onnx, .json)", label="ประเภทโมเดลที่จะอัปโหลด")
                        model_upload = gr.File(label="ลากไฟล์โมเดลมาวางที่นี่ (อัปโหลดได้หลายไฟล์พร้อมกัน)", file_count="multiple")
                        upload_btn = gr.Button("📤 Upload to Server", variant="primary")
                    with gr.Column():
                        gr.Markdown("### 🗑️ ลบโมเดลออกจากระบบ")
                        del_type = gr.Radio(choices=["GPU", "Piper"], value="Piper", label="ประเภทโมเดล")
                        del_dropdown = gr.Dropdown(choices=get_available_piper_models(), label="เลือกโมเดลที่จะลบ")
                        del_btn = gr.Button("🗑️ Delete Selected Model", variant="stop")
                        
            with gr.Tab("🎛️ 3. Advanced Audio Mastering"):
                with gr.Row():
                    with gr.Column():
                        raw_audio_input = gr.Audio(label="Input Audio", type="filepath")
                        preset_dropdown = gr.Dropdown(choices=list(STUDIO_PRESETS.keys()), value=list(STUDIO_PRESETS.keys())[0], label="Preset")
                        preset_desc = gr.Markdown(f"*{STUDIO_PRESETS[list(STUDIO_PRESETS.keys())[0]]['desc']}*")
                        
                        gr.Markdown("### 🌟 Advanced Professional FX")
                        with gr.Row():
                            de_essing_check = gr.Checkbox(value=True, label="🎧 De-essing (ลดเสียง ส, ช บาดหู)")
                            tape_sat_check = gr.Checkbox(value=True, label="🎛️ Tape Saturation (อุ่นแบบอนาล็อก)")
                            humanize_checkbox = gr.Checkbox(value=False, label="🧑 Humanize (Micro-modulation)")
                        
                        ir_file_input = gr.File(label="📂 Convolution Reverb (IR .wav)", file_types=[".wav"])
                        
                        with gr.Accordion("⚙️ ปรับแต่งแบบละเอียด", open=False):
                            bass_boost = gr.Slider(minimum=-15, maximum=15, value=5.0, label="Bass")
                            treble_boost = gr.Slider(minimum=-15, maximum=15, value=3.5, label="Treble")
                            comp_ratio = gr.Slider(minimum=1, maximum=10, value=3.5, label="Compression")
                            enable_gate = gr.Checkbox(value=True, label="Noise Gate")
                            reverb_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, label="Reverb")
                            delay_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Delay")
                            drive_amount = gr.Slider(minimum=0.0, maximum=30.0, value=0.0, label="Distortion")
                            pitch_shift = gr.Slider(minimum=-12, maximum=12, value=0, label="Pitch")

                        export_format = gr.Radio(choices=["WAV", "FLAC"], value="WAV", label="Format")
                        process_btn = gr.Button("🎧 ประมวลผล", variant="primary")
                    with gr.Column():
                        studio_status = gr.Markdown("🟢 รอรับไฟล์")
                        studio_audio_output = gr.Audio(label="Mastered Audio", interactive=False)

            with gr.Tab("📱 4. Social Media Video Automator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="🎥 Footage", sources=["upload"])
                        audio_input_video = gr.Audio(label="🎵 BGM/Voice", type="filepath")
                        with gr.Row():
                            force_916 = gr.Checkbox(value=True, label="📱 บังคับ 9:16")
                            mute_orig = gr.Checkbox(value=True, label="🔇 ลบเสียงคลิปเดิม")
                        with gr.Row():
                            trim_start = gr.Number(value=0, label="ตัดหัว")
                            trim_end = gr.Number(value=0, label="ตัดท้าย (0=ไม่ตัด)")
                        text_lines = gr.Textbox(label="ข้อความ (บรรทัดละกล่อง)", lines=4)
                        watermark_text = gr.Textbox(label="ลายน้ำ")
                        video_process_btn = gr.Button("🎬 Render Video", variant="primary")
                    with gr.Column(scale=1):
                        video_output = gr.Video(label="✅ วิดีโอพร้อมโพสต์", interactive=False)

        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=30, interactive=False, value="[System] Initialized.")
            refresh_models_btn = gr.Button("🔄 Refresh Dropdowns", variant="secondary")
            clear_log_btn = gr.Button("🗑️ Clear")

    # --- UI Logic Wiring ---
    engine_dropdown.change(fn=toggle_engine_visibility, inputs=[engine_dropdown], outputs=[gpu_speaker_dropdown, piper_speaker_dropdown])

    def refresh_speakers():
        gpu_list = get_available_speakers()
        piper_list = get_available_piper_models()
        return gr.update(choices=gpu_list, value=gpu_list[0]), gr.update(choices=piper_list, value=piper_list[0]), gr.update(choices=piper_list), "🔄 Refreshed models from disk."

    refresh_models_btn.click(fn=refresh_speakers, inputs=None, outputs=[gpu_speaker_dropdown, piper_speaker_dropdown, del_dropdown, logs_display])
    
    # Model Manager Wiring
    upload_btn.click(fn=upload_model_file, inputs=[model_upload, upload_type, system_logs_state], outputs=[system_logs_state]).then(fn=refresh_speakers, outputs=[gpu_speaker_dropdown, piper_speaker_dropdown, del_dropdown, logs_display])
    
    def update_del_dropdown(dtype):
        choices = get_available_piper_models() if dtype == "Piper" else get_available_speakers()
        return gr.update(choices=choices, value=choices[0] if choices else None)
        
    del_type.change(fn=update_del_dropdown, inputs=[del_type], outputs=[del_dropdown])
    del_btn.click(fn=delete_model_file, inputs=[del_dropdown, del_type, system_logs_state], outputs=[system_logs_state]).then(fn=refresh_speakers, outputs=[gpu_speaker_dropdown, piper_speaker_dropdown, del_dropdown, logs_display])


    preset_dropdown.change(fn=update_sliders_from_preset, inputs=[preset_dropdown], outputs=[preset_desc, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount])
    submit_btn.click(fn=gradio_tts, inputs=[tts_mode, engine_dropdown, gpu_speaker_dropdown, piper_speaker_dropdown, lang_dropdown, speed_slider, text_input, ref_audio_input, apply_breaths, system_logs_state], outputs=[output_audio, output_audio, status_output, system_logs_state]).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    process_btn.click(fn=gradio_studio, inputs=[raw_audio_input, preset_dropdown, humanize_checkbox, de_essing_check, tape_sat_check, ir_file_input, export_format, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount, system_logs_state], outputs=[studio_audio_output, studio_status, system_logs_state]).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    video_process_btn.click(fn=gradio_video_edit, inputs=[video_input, audio_input_video, trim_start, trim_end, mute_orig, force_916, text_lines, watermark_text, system_logs_state], outputs=[video_output, system_logs_state]).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    clear_log_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[system_logs_state, logs_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)